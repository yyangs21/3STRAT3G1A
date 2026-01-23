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

# =====================================================
# ESTILO EJECUTIVO (FONDO BLANCO + TEXTO NEGRO + PANELES PARA GRÁFICAS)
# =====================================================
st.markdown("""
<style>
/* Fondo general (blanco) + texto negro */
.stApp { background: #ffffff; color: #111827; }

/* Contenedor */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

/* Sidebar (oscuro) con texto claro */
section[data-testid="stSidebar"] { background: #0b1220; }
section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
section[data-testid="stSidebar"] label { color: #f9fafb !important; font-weight: 700 !important; }

/* Títulos y texto general */
h1, h2, h3, h4, h5, h6, p, div, span, label { color: #111827; }

/* Cards KPI */
div[data-testid="stMetric"] {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  padding: 14px 16px !important;
  border-radius: 14px !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
}
div[data-testid="stMetric"] label { color: #111827 !important; font-weight: 800 !important; }
div[data-testid="stMetric"] div { color: #111827 !important; }

/* Panel para gráficas (mini fondo para legibilidad) */
.panel {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 14px 8px 14px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  margin-bottom: 12px;
}

/* Encabezado dentro del panel */
.panel-title {
  font-weight: 800;
  color: #111827;
  margin: 0 0 8px 0;
}

/* Tabs */
button[data-baseweb="tab"] { font-weight: 800 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

def panel_title(txt: str):
    st.markdown(f"<div class='panel-title'>{txt}</div>", unsafe_allow_html=True)

# =====================================================
# GOOGLE SHEETS AUTH
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
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
    "VERDE": "#16a34a",
    "AMARILLO": "#f59e0b",
    "ROJO": "#ef4444",
    "MORADO": "#7c3aed"
}
COLOR_EJEC = {
    "CUMPLIDO": "#16a34a",
    "EN SEGUIMIENTO": "#f59e0b",
    "RIESGO": "#ef4444",
    "CRÍTICO": "#7f1d1d",
    "NO SUBIDO": "#7c3aed"
}

frecuencia_map = {
    "Mensual": 12,
    "Bimestral": 6,
    "Trimestral": 4,
    "Cuatrimestral": 3,
    "Semestral": 2,
    "Anual": 1
}

# =====================================================
# HELPERS
# =====================================================
def safe_strip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", " ")
    return df

def standardize_areas(df: pd.DataFrame) -> pd.DataFrame:
    df = safe_strip(df.copy())
    if "PUESTO" in df.columns and "PUESTO RESPONSABLE" not in df.columns:
        df.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)
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

def estado_exec(r) -> str:
    if r.get("morados", 0) > 0:
        return "NO SUBIDO"
    if r.get("rojos", 0) > 0:
        return "RIESGO"
    if r.get("cumplimiento_%", 0) >= 90:
        return "CUMPLIDO"
    if r.get("cumplimiento_%", 0) >= 60:
        return "EN SEGUIMIENTO"
    return "CRÍTICO"

def apply_filter(df: pd.DataFrame, col: str, selected: list):
    if df is None or df.empty:
        return df
    if not selected:
        return df
    if col not in df.columns:
        return df
    return df[df[col].isin(selected)]

def fig_layout(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=18, r=18, t=55, b=18),
        legend_title_text="",
        paper_bgcolor="#f9fafb",   # mini-fondo del panel
        plot_bgcolor="#f9fafb",
        font=dict(color="#111827")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
    return fig

def make_100pct_stacked(df, year_col, cat_col, value_col, cat_order=None):
    if df.empty:
        return df
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    totals = d.groupby(year_col)[value_col].transform("sum").replace(0, 1)
    d["%"] = (d[value_col] / totals) * 100
    if cat_order:
        d[cat_col] = pd.Categorical(d[cat_col], categories=cat_order, ordered=True)
        d = d.sort_values([year_col, cat_col])
    return d

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=300)
def load_year(year: int):
    sh = client.open(SHEET_NAME)
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    df_area = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())
    return safe_strip(df_obj), standardize_areas(df_area)

@st.cache_data(ttl=300)
def get_years_available():
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    return sorted([int(t) for t in titles if t.isdigit()])

years = get_years_available()
if not years:
    st.error("No encontré hojas tipo '2024', '2025' en tu Google Sheets.")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar (ej: 2024 vs 2025)",
    options=years,
    default=[y for y in [2024, 2025] if y in years] or years[-2:]
)

df_obj, df_area = load_year(year_data)

if "AREA" not in df_area.columns and "DEPARTAMENTO" in df_area.columns:
    df_area["AREA"] = df_area["DEPARTAMENTO"]
if "PUESTO RESPONSABLE" not in df_area.columns:
    df_area["PUESTO RESPONSABLE"] = "SIN_DATO"

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# Objetivos
f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", sorted([x for x in df_obj.get("Tipo", pd.Series([], dtype=object)).dropna().unique()]))
f_persp = st.sidebar.multiselect("Perspectiva", sorted([x for x in df_obj.get("Perspectiva", pd.Series([], dtype=object)).dropna().unique()]))
f_eje = st.sidebar.multiselect("Eje", sorted([x for x in df_obj.get("Eje", pd.Series([], dtype=object)).dropna().unique()]))
f_depto = st.sidebar.multiselect("Departamento", sorted([x for x in df_obj.get("Departamento", pd.Series([], dtype=object)).dropna().unique()]))

# Áreas
f_area = st.sidebar.multiselect("Área", sorted([x for x in df_area.get("AREA", pd.Series([], dtype=object)).dropna().unique()]))
f_puesto = st.sidebar.multiselect("Puesto Responsable", sorted([x for x in df_area.get("PUESTO RESPONSABLE", pd.Series([], dtype=object)).dropna().unique()]))

st.sidebar.caption("✅ Si no seleccionas filtros, se muestra TODO por default.")

# =====================================================
# OBJETIVOS (AÑO BASE)
# =====================================================
obj_id_cols = ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
for c in ["Tipo","Perspectiva","Eje","Departamento"]:
    if c in df_obj.columns and c not in obj_id_cols:
        obj_id_cols.insert(0, c)

obj_long = normalizar_meses(df_obj, obj_id_cols)
obj_long["valor"] = obj_long["Estado"].map(estado_map).fillna(0)

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
    meses_reportados=("Mes","count")
)
obj_resumen["meses_esperados"] = obj_resumen.get("Frecuencia Medición", pd.Series(["Mensual"]*len(obj_resumen))).map(frecuencia_map).fillna(12)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1)*100
obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec, axis=1)
estado_opts = ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]

# =====================================================
# ÁREAS (AÑO BASE)
# =====================================================
area_id_cols = ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO"]:
    if c in df_area.columns and c not in area_id_cols:
        area_id_cols.insert(0, c)

area_long = normalizar_meses(df_area, [c for c in area_id_cols if c in df_area.columns])
area_long["valor"] = area_long["Estado"].map(estado_map).fillna(0)

area_long = apply_filter(area_long, "AREA", f_area)
area_long = apply_filter(area_long, "PUESTO RESPONSABLE", f_puesto)

area_res_area = area_long.groupby(["AREA"], as_index=False).agg(
    cumplimiento=("valor","mean"),
    tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)
area_res_area["cumplimiento_%"] = area_res_area["cumplimiento"]*100

# =====================================================
# TABS
# =====================================================
tabs = st.tabs(["📌 Resumen", "🎯 Objetivos", "🏢 Áreas", "📊 Comparativo", "🚨 Alertas", "📄 Exportar", "📋 Datos"])

with tabs[0]:
    st.subheader(f"📌 Resumen Ejecutivo – Año {year_data}")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Objetivos", len(obj_resumen))
    k2.metric("Cumplidos", int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()))
    k3.metric("En Riesgo", int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()))
    k4.metric("Críticos/No Subido", int((obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"])).sum()))
    k5.metric("Cumplimiento Promedio", f"{obj_resumen['cumplimiento_%'].mean():.1f}%")

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title(f"{year_data} – Cumplimiento Estratégico (Objetivos)")
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(obj_resumen["cumplimiento_%"].mean()) if len(obj_resumen) else 0,
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#111827"},
                   "steps": [{"range":[0,60],"color":"#fee2e2"},
                             {"range":[60,90],"color":"#fef3c7"},
                             {"range":[90,100],"color":"#dcfce7"}]},
        ))
        st.plotly_chart(fig_layout(fig_g1), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title(f"{year_data} – Cumplimiento Operativo (Áreas)")
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(area_long["valor"].mean()*100) if len(area_long) else 0,
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#111827"},
                   "steps": [{"range":[0,60],"color":"#fee2e2"},
                             {"range":[60,90],"color":"#fef3c7"},
                             {"range":[90,100],"color":"#dcfce7"}]},
        ))
        st.plotly_chart(fig_layout(fig_g2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("Mix de estados ejecutivos (conteo)")
        df_estado = (
            obj_resumen["estado_ejecutivo"].value_counts()
            .reindex(estado_opts, fill_value=0)
            .rename_axis("estado_ejecutivo")
            .reset_index(name="cantidad")
        )
        fig = px.bar(df_estado, x="estado_ejecutivo", y="cantidad",
                     text="cantidad", color="estado_ejecutivo",
                     color_discrete_map=COLOR_EJEC)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("Tendencia mensual (promedio %)")
        tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
        tr["cumplimiento_%"] = tr["valor"]*100
        fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("🎯 Objetivos – Análisis Ejecutivo")

    estado_sel = st.multiselect("Filtrar por Estado Ejecutivo (opcional)", estado_opts, default=[])
    obj_view = obj_resumen.copy()
    if estado_sel:
        obj_view = obj_view[obj_view["estado_ejecutivo"].isin(estado_sel)]

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("Top 15 objetivos críticos (peor → mejor)")
        top_bad = obj_view.sort_values("cumplimiento_%").head(15)
        fig = px.bar(top_bad, x="cumplimiento_%", y="Objetivo",
                     orientation="h", color="estado_ejecutivo",
                     color_discrete_map=COLOR_EJEC, text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("Desviación vs 100% (impacto)")
        dev = obj_view.copy()
        dev["desviación"] = dev["cumplimiento_%"] - 100
        dev = dev.sort_values("desviación").head(20)
        fig = px.bar(dev, x="desviación", y="Objetivo",
                     orientation="h", color="desviación",
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.subheader("🏢 Áreas – Control Operativo")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("Ranking crítico de áreas (peor → mejor)")
        rk = area_res_area.sort_values("cumplimiento_%").head(20)
        fig = px.bar(rk, x="cumplimiento_%", y="AREA",
                     orientation="h", color="cumplimiento_%",
                     color_continuous_scale="RdYlGn", text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("Cumplimiento vs Carga (# tareas)")
        sc = area_res_area.copy()
        fig = px.scatter(sc, x="tareas", y="cumplimiento_%", hover_name="AREA", size="tareas")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    panel_title("Heatmap operativo (Área vs Mes)")
    heat = area_long.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
    fig = px.imshow(heat, color_continuous_scale=["#ef4444","#f59e0b","#16a34a"])
    st.plotly_chart(fig_layout(fig), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.subheader("📊 Comparativo (100% apilado) – VERDE/AMARILLO/ROJO/MORADO")

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el comparativo (sidebar).")
    else:
        comp_objs, comp_areas = [], []

        for y in compare_years:
            o, a = load_year(y)

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

            a = standardize_areas(a)
            if "AREA" not in a.columns and "DEPARTAMENTO" in a.columns:
                a["AREA"] = a["DEPARTAMENTO"]
            if "PUESTO RESPONSABLE" not in a.columns and "PUESTO" in a.columns:
                a.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

            aid = [c for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"] if c in a.columns]
            al = normalizar_meses(a, aid)
            al["AÑO"] = y
            al = apply_filter(al, "AREA", f_area)
            al = apply_filter(al, "PUESTO RESPONSABLE", f_puesto)
            comp_areas.append(al)

        comp_obj_long = pd.concat(comp_objs, ignore_index=True)
        comp_area_long = pd.concat(comp_areas, ignore_index=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("🎯 Objetivos: distribución por color (100%)")
        obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
        obj_mix = make_100pct_stacked(obj_mix, "AÑO", "Estado", "conteo", cat_order=["VERDE","AMARILLO","ROJO","MORADO"])
        fig = px.bar(obj_mix, x="AÑO", y="%", color="Estado",
                     barmode="stack", color_discrete_map=COLOR_ESTADO, text="%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        fig.update_layout(yaxis=dict(range=[0,100], title="%"))
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        panel_title("🏢 Áreas: distribución por color (100%)")
        area_mix = comp_area_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
        area_mix = make_100pct_stacked(area_mix, "AÑO", "Estado", "conteo", cat_order=["VERDE","AMARILLO","ROJO","MORADO"])
        fig = px.bar(area_mix, x="AÑO", y="%", color="Estado",
                     barmode="stack", color_discrete_map=COLOR_ESTADO, text="%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        fig.update_layout(yaxis=dict(range=[0,100], title="%"))
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

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
            "Cumplimiento %": round(float(r["cumplimiento_%"]), 1)
        })

    bad_areas = area_res_area[area_res_area["cumplimiento_%"] < 60].copy()
    for _, r in bad_areas.iterrows():
        alert_rows.append({
            "Año": year_data,
            "Nivel": "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL",
            "Tipo": "Área",
            "Nombre": r["AREA"],
            "Estado": "BAJO CUMPLIMIENTO",
            "Cumplimiento %": round(float(r["cumplimiento_%"]), 1)
        })

    alerts_df = pd.DataFrame(alert_rows)
    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        alerts_df["Nivel_Orden"] = alerts_df["Nivel"].map({"CRÍTICA": 0, "NORMAL": 1}).fillna(9)
        alerts_df = alerts_df.sort_values(["Nivel_Orden","Cumplimiento %"], ascending=[True, True]).drop(columns=["Nivel_Orden"])

        def semaforo(row):
            bg = "#fee2e2" if row["Nivel"] == "CRÍTICA" else "#fef3c7"
            return [f"background-color: {bg}; color: #111827;"]*len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    fig_estado_exec = px.pie(obj_resumen, names="estado_ejecutivo", title=f"{year_data} – Estados Ejecutivos (Objetivos)")
    fig_rank_areas = px.bar(area_res_area.sort_values("cumplimiento_%").head(20),
                            x="cumplimiento_%", y="AREA", orientation="h",
                            title=f"{year_data} – Ranking crítico de áreas")
    fig_estado_exec = fig_layout(fig_estado_exec)
    fig_rank_areas = fig_layout(fig_rank_areas)

    def build_report_html():
        parts = []
        parts.append(f"""
        <html><head><meta charset="utf-8"/>
        <title>Reporte Estratégico</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 22px; background:#ffffff; color:#111827; }}
          .kpis {{ display:flex; gap:12px; flex-wrap:wrap; }}
          .kpi {{ border:1px solid #e5e7eb; padding:10px 12px; border-radius:10px; min-width:170px; background:#fff; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #e5e7eb; padding: 6px; font-size: 12px; }}
          th {{ background: #f3f4f6; }}
        </style>
        </head><body>
        <h1>Reporte Estratégico y de Control</h1>
        <div style="color:#6b7280;">Año: {year_data} · Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

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

        parts.append("<h2>Tabla: Objetivos (resumen)</h2>")
        parts.append(obj_resumen.head(200).to_html(index=False))

        parts.append("<h2>Tabla: Áreas (resumen)</h2>")
        parts.append(area_res_area.head(200).to_html(index=False))

        parts.append("</body></html>")
        return "\n".join(parts)

    html_report = build_report_html()
    st.download_button("⬇️ Descargar Reporte HTML", data=html_report, file_name="Reporte_Estrategico.html", mime="text/html")
    st.info("Tip: abre el HTML en Chrome/Edge → Ctrl+P → Guardar como PDF.")

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
