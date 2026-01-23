import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# CONFIG STREAMLIT
# =====================================================
st.set_page_config(page_title="Dashboard Estratégico", layout="wide")
st.title("📊 Dashboard Estratégico y de Control")

# ---------- THEME / CSS (ejecutivo, legible) ----------
st.markdown(
    """
<style>
/* Fondo general gris claro */
.stApp { background: #f4f6f9; color: #111 !important; }

/* Container spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Títulos y texto */
h1, h2, h3, h4, h5, h6, p, span, div { color: #111 !important; }

/* Tarjetas KPI (mejor contraste) */
div[data-testid="stMetric"]{
  background: #ffffff;
  border: 1px solid #e6e8ee;
  padding: 14px 14px;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(16,24,40,0.06);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: #ffffff;
  border-right: 1px solid #e6e8ee;
}

/* Plotly container para que se vea “tarjeta” */
.plot-card{
  background:#ffffff;
  border:1px solid #e6e8ee;
  border-radius:14px;
  padding:12px 12px 0px 12px;
  box-shadow: 0 2px 10px rgba(16,24,40,0.06);
  margin-bottom: 12px;
}

.small-note{ color:#444 !important; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# GOOGLE SHEETS AUTH (MISMA CONEXIÓN QUE TENÍAS)
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

service_account_info = dict(st.secrets["gcp_service_account"])
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
CREDS = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
client = gspread.authorize(CREDS)

SHEET_NAME = "DATAESTRATEGIA"

# =====================================================
# CONFIG DATA
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {"VERDE": 1, "AMARILLO": 0.5, "ROJO": 0, "MORADO": 0}

COLOR_ESTADO = {
    "VERDE": "#00a65a",
    "AMARILLO": "#f1c40f",
    "ROJO": "#e74c3c",
    "MORADO": "#8e44ad"
}

COLOR_EJEC = {
    "CUMPLIDO": "#00a65a",
    "EN SEGUIMIENTO": "#f1c40f",
    "RIESGO": "#e74c3c",
    "CRÍTICO": "#8b0000",
    "NO SUBIDO": "#8e44ad",
}

frecuencia_map = {"Mensual": 12, "Bimestral": 6, "Trimestral": 4, "Cuatrimestral": 3, "Semestral": 2, "Anual": 1}

estado_opts = ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]
estado_color_order = ["VERDE","AMARILLO","ROJO","MORADO"]

# =====================================================
# HELPERS
# =====================================================
def safe_strip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", " ")
    return df

def unify_dic(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza casos donde hay Dic y Diciembre."""
    df = df.copy()
    if "Diciembre" in df.columns and "Dic" in df.columns:
        df["Dic"] = df["Dic"].fillna(df["Diciembre"])
    elif "Diciembre" in df.columns and "Dic" not in df.columns:
        df.rename(columns={"Diciembre": "Dic"}, inplace=True)
    return df

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    return (
        df.melt(id_vars=id_cols, value_vars=meses_presentes, var_name="Mes", value_name="Estado")
          .dropna(subset=["Estado"])
    )

def apply_filter(df: pd.DataFrame, col: str, selected: list):
    """Filtros opcionales: si selected está vacío -> no filtra."""
    if df is None or df.empty:
        return df
    if not selected:
        return df
    if col not in df.columns:
        return df
    return df[df[col].isin(selected)]

def estado_exec(row) -> str:
    if row.get("morados", 0) > 0:
        return "NO SUBIDO"
    if row.get("rojos", 0) > 0:
        return "RIESGO"
    if row.get("cumplimiento_%", 0) >= 90:
        return "CUMPLIDO"
    if row.get("cumplimiento_%", 0) >= 60:
        return "EN SEGUIMIENTO"
    return "CRÍTICO"

def plotly_layout(fig, title=None):
    """Layout uniforme: fondo blanco dentro de la tarjeta, texto oscuro."""
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111", size=13),
        legend=dict(title=None),
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f6", zeroline=False)
    return fig

def chart_card(fig):
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# LOAD DATA (Google Sheets)
# =====================================================
@st.cache_data(ttl=300)
def get_years_available():
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    years = sorted([int(t) for t in titles if t.isdigit()])
    return years

@st.cache_data(ttl=300)
def load_year(year: int):
    sh = client.open(SHEET_NAME)
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    df_area = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())

    df_obj = safe_strip(df_obj)
    df_area = safe_strip(df_area)
    df_area = unify_dic(df_area)

    # normalizaciones comunes
    if "PUESTO" in df_area.columns and "PUESTO RESPONSABLE" not in df_area.columns:
        df_area.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

    if "Área" in df_area.columns and "AREA" not in df_area.columns:
        df_area.rename(columns={"Área": "AREA"}, inplace=True)

    if "Realizada?" in df_area.columns and "¿Realizada?" not in df_area.columns:
        df_area.rename(columns={"Realizada?": "¿Realizada?"}, inplace=True)

    if "AREA" not in df_area.columns and "DEPARTAMENTO" in df_area.columns:
        df_area["AREA"] = df_area["DEPARTAMENTO"]

    if "DEPARTAMENTO" not in df_area.columns and "AREA" in df_area.columns:
        df_area["DEPARTAMENTO"] = df_area["AREA"]

    return df_obj, df_area

# =====================================================
# SIDEBAR
# =====================================================
years = get_years_available()
if not years:
    st.error("No encontré hojas tipo '2024', '2025' en tu Google Sheets.")
    st.stop()

st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar",
    options=years,
    default=[y for y in [2024, 2025] if y in years]
)

df_obj, df_area = load_year(year_data)

# =====================================================
# FILTROS (opcionales, no bloquean)
# =====================================================
st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# Objetivos: posibles campos extra si existen
f_tipo_plan = st.sidebar.multiselect("Tipo (POA/PEC)", sorted(df_obj["Tipo"].dropna().unique())) if "Tipo" in df_obj.columns else []
f_persp    = st.sidebar.multiselect("Perspectiva", sorted(df_obj["Perspectiva"].dropna().unique())) if "Perspectiva" in df_obj.columns else []
f_eje      = st.sidebar.multiselect("Eje", sorted(df_obj["Eje"].dropna().unique())) if "Eje" in df_obj.columns else []
f_depto    = st.sidebar.multiselect("Departamento", sorted(df_obj["Departamento"].dropna().unique())) if "Departamento" in df_obj.columns else []

# Áreas
f_area     = st.sidebar.multiselect("Área/Departamento", sorted(df_area["AREA"].dropna().unique())) if "AREA" in df_area.columns else []
f_puesto   = st.sidebar.multiselect("Puesto Responsable", sorted(df_area["PUESTO RESPONSABLE"].dropna().unique())) if "PUESTO RESPONSABLE" in df_area.columns else []

st.sidebar.caption("Si no seleccionas filtros, se muestra todo por defecto.")

# =====================================================
# PROCESAR OBJETIVOS (AÑO BASE)
# =====================================================
obj_id_cols = ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
for c in ["Tipo","Perspectiva","Eje","Departamento"]:
    if c in df_obj.columns and c not in obj_id_cols:
        obj_id_cols.insert(0, c)

obj_long = normalizar_meses(df_obj, obj_id_cols)
obj_long["valor"] = obj_long["Estado"].map(estado_map).fillna(0)

# aplica filtros opcionales
obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "Departamento", f_depto)

group_cols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in obj_long.columns]
obj_resumen = obj_long.groupby(group_cols, as_index=False).agg(
    score_total=("valor","sum"),
    verdes=("Estado", lambda x: (x=="VERDE").sum()),
    amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum()),
    meses_reportados=("Mes","count"),
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map).fillna(12)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1)*100
obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec, axis=1)

# =====================================================
# PROCESAR ÁREAS (AÑO BASE)
# =====================================================
area_id_cols = [c for c in ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?","DEPARTAMENTO","TIPO","PERSPECTIVA","EJE"] if c in df_area.columns]
area_long = normalizar_meses(df_area, area_id_cols)
area_long["valor"] = area_long["Estado"].map(estado_map).fillna(0)

area_long = apply_filter(area_long, "AREA", f_area)
area_long = apply_filter(area_long, "PUESTO RESPONSABLE", f_puesto)

area_res_area = area_long.groupby(["AREA"], as_index=False).agg(
    cumplimiento=("valor","mean"),
    tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum()),
)
area_res_area["cumplimiento_%"] = area_res_area["cumplimiento"]*100

# =====================================================
# TABS PRINCIPALES (orden)
# =====================================================
tabs = st.tabs(["📌 Resumen", "🎯 Objetivos", "🏢 Áreas", "📊 Comparativo", "🚨 Alertas", "📄 Exportar", "📋 Datos"])

# ----------------------------
# TAB 0: Resumen
# ----------------------------
with tabs[0]:
    st.subheader(f"📌 Resumen Ejecutivo – Año {year_data}")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Objetivos", len(obj_resumen))
    k2.metric("Cumplidos", int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()))
    k3.metric("En Riesgo", int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()))
    k4.metric("Críticos/No Subido", int((obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"])).sum()))
    k5.metric("Cumplimiento Promedio", f"{obj_resumen['cumplimiento_%'].mean():.1f}%")

    g1, g2 = st.columns(2)
    fig_g1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(obj_resumen["cumplimiento_%"].mean()) if len(obj_resumen) else 0,
        gauge={"axis": {"range": [0, 100]},
               "steps": [{"range":[0,60],"color":"#e74c3c"},
                         {"range":[60,90],"color":"#f1c40f"},
                         {"range":[90,100],"color":"#00a65a"}]},
        title={"text": f"{year_data} – Cumplimiento Estratégico (Objetivos)"}
    ))
    fig_g1.update_layout(paper_bgcolor="white", font=dict(color="#111"))
    g1.markdown('<div class="plot-card">', unsafe_allow_html=True)
    g1.plotly_chart(fig_g1, use_container_width=True)
    g1.markdown("</div>", unsafe_allow_html=True)

    fig_g2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(area_long["valor"].mean()*100) if len(area_long) else 0,
        gauge={"axis": {"range": [0, 100]},
               "steps": [{"range":[0,60],"color":"#e74c3c"},
                         {"range":[60,90],"color":"#f1c40f"},
                         {"range":[90,100],"color":"#00a65a"}]},
        title={"text": f"{year_data} – Cumplimiento Operativo (Áreas)"}
    ))
    fig_g2.update_layout(paper_bgcolor="white", font=dict(color="#111"))
    g2.markdown('<div class="plot-card">', unsafe_allow_html=True)
    g2.plotly_chart(fig_g2, use_container_width=True)
    g2.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Conteo por estado ejecutivo (Objetivos) - arreglado sin ValueError
        counts = obj_resumen["estado_ejecutivo"].value_counts().reindex(estado_opts).fillna(0).reset_index()
        counts.columns = ["estado_ejecutivo", "cantidad"]
        fig = px.bar(counts, x="estado_ejecutivo", y="cantidad", text="cantidad",
                     color="estado_ejecutivo", color_discrete_map=COLOR_EJEC)
        fig = plotly_layout(fig, "Mix de Estados Ejecutivos (Objetivos)")
        fig.update_traces(textposition="outside")
        chart_card(fig)

    with c2:
        tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
        tr.columns = ["Mes","valor"]
        tr["cumplimiento_%"] = tr["valor"]*100
        fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
        fig = plotly_layout(fig, "Tendencia Mensual (promedio)")
        fig.update_yaxes(range=[0,100])
        chart_card(fig)

# ----------------------------
# TAB 1: Objetivos
# ----------------------------
with tabs[1]:
    st.subheader("🎯 Objetivos – Análisis Avanzado")

    estado_sel = st.multiselect("Filtrar por Estado Ejecutivo (opcional)", estado_opts, default=[])
    obj_view = obj_resumen.copy()
    if estado_sel:
        obj_view = obj_view[obj_view["estado_ejecutivo"].isin(estado_sel)]

    r1, r2 = st.columns(2)
    with r1:
        top_bad = obj_view.sort_values("cumplimiento_%").head(20)
        fig = px.bar(top_bad, x="cumplimiento_%", y="Objetivo", orientation="h",
                     color="estado_ejecutivo", color_discrete_map=COLOR_EJEC,
                     text=top_bad["cumplimiento_%"].round(1))
        fig = plotly_layout(fig, "Top Objetivos Críticos (peor → mejor)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        chart_card(fig)

    with r2:
        # Desviación vs 100
        df_dev = obj_view.copy()
        df_dev["desviación_%"] = df_dev["cumplimiento_%"] - 100
        df_dev = df_dev.sort_values("desviación_%").head(20)
        fig = px.bar(df_dev, x="desviación_%", y="Objetivo", orientation="h",
                     color="desviación_%", color_continuous_scale="RdYlGn")
        fig = plotly_layout(fig, "Desviación vs 100% (peor → mejor)")
        chart_card(fig)

    r3, r4 = st.columns(2)
    with r3:
        if "Perspectiva" in obj_view.columns:
            p = obj_view.groupby("Perspectiva")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
            fig = px.bar(p, x="cumplimiento_%", y="Perspectiva", orientation="h",
                         color="cumplimiento_%", color_continuous_scale="RdYlGn",
                         text=p["cumplimiento_%"].round(1))
            fig = plotly_layout(fig, "Cumplimiento por Perspectiva")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            chart_card(fig)
        else:
            st.info("No existe columna Perspectiva en este año.")

    with r4:
        if "Tipo" in obj_view.columns:
            t = obj_view.groupby("Tipo")["cumplimiento_%"].mean().reset_index()
            fig = px.bar(t, x="Tipo", y="cumplimiento_%", text=t["cumplimiento_%"].round(1))
            fig = plotly_layout(fig, "Cumplimiento por Tipo (POA vs PEC)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            chart_card(fig)
        else:
            st.info("No existe columna Tipo en este año.")

# ----------------------------
# TAB 2: Áreas
# ----------------------------
with tabs[2]:
    st.subheader("🏢 Áreas – Control Operativo")

    c1, c2 = st.columns(2)
    with c1:
        rk = area_res_area.sort_values("cumplimiento_%").head(20)
        fig = px.bar(rk, x="cumplimiento_%", y="AREA", orientation="h",
                     color="cumplimiento_%", color_continuous_scale="RdYlGn",
                     text=rk["cumplimiento_%"].round(1))
        fig = plotly_layout(fig, "Ranking crítico de áreas (peor → mejor)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        chart_card(fig)

    with c2:
        sc = area_res_area.copy()
        fig = px.scatter(sc, x="tareas", y="cumplimiento_%", hover_name="AREA",
                         size="tareas")
        fig = plotly_layout(fig, "Cumplimiento vs Carga (# tareas)")
        chart_card(fig)

    # Heatmap grande
    heat = area_long.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
    fig = px.imshow(heat, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
    fig = plotly_layout(fig, "Heatmap Operativo (Área vs Mes)")
    fig.update_layout(height=520)
    chart_card(fig)

# ----------------------------
# TAB 3: Comparativo
# ----------------------------
with tabs[3]:
    st.subheader("📊 Comparativo (Objetivos y Áreas)")

    st.markdown(
        "<div class='small-note'>Este comparativo usa los mismos filtros opcionales del año base "
        "(Tipo/Perspectiva/Eje/Departamento y Área/Puesto) para comparar “lo comparable”.</div>",
        unsafe_allow_html=True
    )

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el comparativo (sidebar).")
    else:
        comp_objs = []
        comp_areas = []

        for y in compare_years:
            o, a = load_year(y)

            # OBJETIVOS
            oid = ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            for c in ["Tipo","Perspectiva","Eje","Departamento"]:
                if c in o.columns and c not in oid:
                    oid.insert(0, c)

            ol = normalizar_meses(o, oid)
            ol["AÑO"] = y
            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)
            comp_objs.append(ol)

            # AREAS
            a = unify_dic(safe_strip(a))
            if "PUESTO" in a.columns and "PUESTO RESPONSABLE" not in a.columns:
                a.rename(columns={"PUESTO":"PUESTO RESPONSABLE"}, inplace=True)
            if "AREA" not in a.columns and "DEPARTAMENTO" in a.columns:
                a["AREA"] = a["DEPARTAMENTO"]
            if "DEPARTAMENTO" not in a.columns and "AREA" in a.columns:
                a["DEPARTAMENTO"] = a["AREA"]

            aid = [c for c in ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"] if c in a.columns]
            al = normalizar_meses(a, aid)
            al["AÑO"] = y
            al = apply_filter(al, "AREA", f_area)
            al = apply_filter(al, "PUESTO RESPONSABLE", f_puesto)
            comp_areas.append(al)

        comp_obj_long = pd.concat(comp_objs, ignore_index=True)
        comp_area_long = pd.concat(comp_areas, ignore_index=True)

        # 1) % por color (Objetivos)
        c1, c2 = st.columns(2)

        with c1:
            mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            mix["%"] = mix["conteo"] / mix.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(mix, x="AÑO", y="%", color="Estado", barmode="group",
                         color_discrete_map=COLOR_ESTADO,
                         category_orders={"Estado": estado_color_order},
                         text=mix["%"].round(1))
            fig = plotly_layout(fig, "Objetivos: % por color (VERDE/AMARILLO/ROJO/MORADO)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            chart_card(fig)

        with c2:
            mix2 = comp_area_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            mix2["%"] = mix2["conteo"] / mix2.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(mix2, x="AÑO", y="%", color="Estado", barmode="group",
                         color_discrete_map=COLOR_ESTADO,
                         category_orders={"Estado": estado_color_order},
                         text=mix2["%"].round(1))
            fig = plotly_layout(fig, "Áreas: % por color (VERDE/AMARILLO/ROJO/MORADO)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            chart_card(fig)

        # 2) Cumplimiento promedio + delta
        st.markdown("### 📌 Cumplimiento promedio y cambio (Δ)")

        comp_obj_long["valor"] = comp_obj_long["Estado"].map(estado_map).fillna(0)
        comp_area_long["valor"] = comp_area_long["Estado"].map(estado_map).fillna(0)

        s1, s2 = st.columns(2)
        with s1:
            avg = comp_obj_long.groupby("AÑO")["valor"].mean().reset_index()
            avg["cumplimiento_%"] = avg["valor"]*100
            avg["Δ_vs_prev"] = avg["cumplimiento_%"].diff()
            fig = px.line(avg, x="AÑO", y="cumplimiento_%", markers=True, text=avg["cumplimiento_%"].round(1))
            fig = plotly_layout(fig, "Objetivos: cumplimiento promedio (%)")
            fig.update_traces(texttemplate="%{text:.1f}%")
            chart_card(fig)

        with s2:
            avg2 = comp_area_long.groupby("AÑO")["valor"].mean().reset_index()
            avg2["cumplimiento_%"] = avg2["valor"]*100
            avg2["Δ_vs_prev"] = avg2["cumplimiento_%"].diff()
            fig = px.line(avg2, x="AÑO", y="cumplimiento_%", markers=True, text=avg2["cumplimiento_%"].round(1))
            fig = plotly_layout(fig, "Áreas: cumplimiento promedio (%)")
            fig.update_traces(texttemplate="%{text:.1f}%")
            chart_card(fig)

        # 3) “Qué trae nuevo 2025”: objetivos/áreas únicos por año
        st.markdown("### 🧩 Cambios en cobertura (cosas nuevas vs años anteriores)")

        u1, u2 = st.columns(2)
        with u1:
            u = comp_obj_long.groupby("AÑO")["Objetivo"].nunique().reset_index(name="Objetivos únicos")
            fig = px.bar(u, x="AÑO", y="Objetivos únicos", text="Objetivos únicos")
            fig = plotly_layout(fig, "Cantidad de objetivos distintos por año (cobertura)")
            fig.update_traces(textposition="outside")
            chart_card(fig)

        with u2:
            u = comp_area_long.groupby("AÑO")["AREA"].nunique().reset_index(name="Áreas únicas")
            fig = px.bar(u, x="AÑO", y="Áreas únicas", text="Áreas únicas")
            fig = plotly_layout(fig, "Cantidad de áreas distintas por año (cobertura)")
            fig.update_traces(textposition="outside")
            chart_card(fig)

# ----------------------------
# TAB 4: Alertas
# ----------------------------
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
    for _, r in crit_obj.iterrows():
        sev = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
        alert_rows.append({
            "Año": year_data,
            "Nivel": sev,
            "Tipo": "Objetivo",
            "Nombre": r.get("Objetivo",""),
            "Estado": r["estado_ejecutivo"],
            "Cumplimiento_%": round(float(r["cumplimiento_%"]), 1),
        })

    bad_areas = area_res_area[area_res_area["cumplimiento_%"] < 60].copy()
    for _, r in bad_areas.iterrows():
        sev = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
        alert_rows.append({
            "Año": year_data,
            "Nivel": sev,
            "Tipo": "Área",
            "Nombre": r["AREA"],
            "Estado": "BAJO CUMPLIMIENTO",
            "Cumplimiento_%": round(float(r["cumplimiento_%"]), 1),
        })

    alerts_df = pd.DataFrame(alert_rows)

    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        # orden: primero críticas, luego normales; dentro por cumplimiento
        order_level = {"CRÍTICA": 0, "NORMAL": 1}
        alerts_df["_lvl"] = alerts_df["Nivel"].map(order_level).fillna(99)
        alerts_df = alerts_df.sort_values(["_lvl","Cumplimiento_%"], ascending=[True, True]).drop(columns=["_lvl"])

        def semaforo(row):
            bg = "#ffd6d6" if row["Nivel"] == "CRÍTICA" else "#fff3cd"
            return [f"background-color: {bg}; color:#111;"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

# ----------------------------
# TAB 5: Exportar
# ----------------------------
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    # Figuras clave para export
    fig_estado_exec = px.pie(obj_resumen, names="estado_ejecutivo", title=f"{year_data} – Estados Ejecutivos (Objetivos)")
    fig_estado_exec.update_layout(paper_bgcolor="white", font=dict(color="#111"))

    fig_rank_areas = px.bar(
        area_res_area.sort_values("cumplimiento_%").head(20),
        x="cumplimiento_%", y="AREA", orientation="h",
        title=f"{year_data} – Ranking crítico de áreas"
    )
    fig_rank_areas = plotly_layout(fig_rank_areas, f"{year_data} – Ranking crítico de áreas")

    def build_report_html():
        parts = []
        parts.append(f"""
        <html><head><meta charset="utf-8"/>
        <title>Reporte Estratégico</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 22px; background:#ffffff; color:#111; }}
          .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px; }}
          .kpi {{ border:1px solid #eee; padding:10px 12px; border-radius:10px; min-width:170px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
          th {{ background: #f5f5f5; }}
        </style>
        </head><body>
        <h1>Reporte Estratégico y de Control</h1>
        <div style="color:#666;">Año: {year_data} · Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

        <h2>KPIs</h2>
        <div class="kpis">
          <div class="kpi"><b>Objetivos</b><br>{len(obj_resumen)}</div>
          <div class="kpi"><b>Cumplidos</b><br>{int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum())}</div>
          <div class="kpi"><b>En Riesgo</b><br>{int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum())}</div>
          <div class="kpi"><b>Críticos/No Subido</b><br>{int((obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"])).sum())}</div>
          <div class="kpi"><b>Cumplimiento Promedio</b><br>{obj_resumen["cumplimiento_%"].mean():.1f}%</div>
        </div>

        <h2>Gráficas</h2>
        """)
        parts.append(fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn"))
        parts.append(fig_rank_areas.to_html(full_html=False, include_plotlyjs=False))

        parts.append("<h2>Alertas</h2>")
        if 'alerts_df' in globals() and not alerts_df.empty:
            parts.append(alerts_df.to_html(index=False))
        else:
            parts.append("<p>Sin alertas.</p>")

        parts.append("<h2>Tabla: Objetivos (resumen)</h2>")
        parts.append(obj_resumen.head(200).to_html(index=False))

        parts.append("<h2>Tabla: Áreas (resumen)</h2>")
        parts.append(area_res_area.head(200).to_html(index=False))

        parts.append("</body></html>")
        return "\n".join(parts)

    html_report = build_report_html()
    st.download_button(
        "⬇️ Descargar Reporte HTML",
        data=html_report,
        file_name=f"Reporte_Estrategico_{year_data}.html",
        mime="text/html"
    )
    st.info("Tip: abre el HTML en Chrome/Edge → Ctrl+P → Guardar como PDF.")

# ----------------------------
# TAB 6: Datos
# ----------------------------
with tabs[6]:
    st.subheader("📋 Datos (auditoría)")

    with st.expander("Objetivos – Resumen"):
        st.dataframe(obj_resumen, use_container_width=True)

    with st.expander("Objetivos – Long"):
        st.dataframe(obj_long, use_container_width=True)

    with st.expander("Áreas – Resumen"):
        st.dataframe(area_res_area, use_container_width=True)

    with st.expander("Áreas – Long"):
        st.dataframe(area_long, use_container_width=True)

st.caption("Fuente: Google Sheets · Dashboard Estratégico")
