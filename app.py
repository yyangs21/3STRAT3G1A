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

# --- CSS ligero para ordenar visualmente ---
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #eee; padding: 12px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

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

# Color por estado
COLOR_ESTADO = {
    "VERDE": "#00a65a",
    "AMARILLO": "#f1c40f",
    "ROJO": "#e74c3c",
    "MORADO": "#8e44ad"
}

# Estados ejecutivos (para objetivos)
COLOR_EJEC = {
    "CUMPLIDO": "green",
    "EN SEGUIMIENTO": "gold",
    "RIESGO": "red",
    "CRÍTICO": "darkred",
    "NO SUBIDO": "purple"
}

frecuencia_map = {"Mensual": 12, "Bimestral": 6, "Trimestral": 4, "Cuatrimestral": 3, "Semestral": 2, "Anual": 1}

# =====================================================
# HELPERS
# =====================================================
def safe_strip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", " ")
    return df

def standardize_areas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = safe_strip(df)

    # normaliza nombres (2024 trae PUESTO RESPONSABLE, 2025 trae PUESTO)
    if "PUESTO" in df.columns and "PUESTO RESPONSABLE" not in df.columns:
        df.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

    # normaliza Diciembre duplicado (2025 AREAS trae Dic y Diciembre)
    if "Diciembre" in df.columns and "Dic" in df.columns:
        # si Dic está vacío y Diciembre no, rellena Dic
        df["Dic"] = df["Dic"].fillna(df["Diciembre"])
    elif "Diciembre" in df.columns and "Dic" not in df.columns:
        df.rename(columns={"Diciembre": "Dic"}, inplace=True)

    return df

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    return (
        df.melt(id_vars=id_cols, value_vars=meses_presentes,
                var_name="Mes", value_name="Estado")
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

def pct_fmt(x):
    try:
        return f"{float(x):.1f}%"
    except:
        return ""

def apply_filter(df: pd.DataFrame, col: str, selected: list):
    if df is None or df.empty:
        return df
    if not selected:
        return df
    if col not in df.columns:
        return df
    return df[df[col].isin(selected)]

# =====================================================
# LOAD DATA (Google Sheets)
# =====================================================
@st.cache_data(ttl=300)
def load_year(year: int):
    sh = client.open(SHEET_NAME)

    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    df_area = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())

    df_obj = safe_strip(df_obj)
    df_area = standardize_areas(df_area)

    return df_obj, df_area

@st.cache_data(ttl=300)
def get_years_available():
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    years = sorted([int(t) for t in titles if t.isdigit()])
    return years

years = get_years_available()
if not years:
    st.error("No encontré hojas tipo '2024', '2025' en tu Google Sheets.")
    st.stop()

# =====================================================
# SIDEBAR (renombrado como pediste)
# =====================================================
st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar (ej: 2024 vs 2025)",
    options=years,
    default=[y for y in [2024, 2025] if y in years]
)

# Cargar data base
df_obj, df_area = load_year(year_data)

# =====================================================
# Construir opciones de filtros (del año base)
# =====================================================
st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# Objetivos: filtros nuevos
f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", sorted([x for x in df_obj.get("Tipo", pd.Series([])).dropna().unique()]))
f_persp = st.sidebar.multiselect("Perspectiva", sorted([x for x in df_obj.get("Perspectiva", pd.Series([])).dropna().unique()]))
f_eje = st.sidebar.multiselect("Eje", sorted([x for x in df_obj.get("Eje", pd.Series([])).dropna().unique()]))
f_depto = st.sidebar.multiselect("Departamento", sorted([x for x in df_obj.get("Departamento", pd.Series([])).dropna().unique()]))

# Areas: filtros
f_area = st.sidebar.multiselect("Área", sorted([x for x in df_area.get("DEPARTAMENTO", df_area.get("AREA", pd.Series([]))).dropna().unique()])) if ("AREA" in df_area.columns or "DEPARTAMENTO" in df_area.columns) else []
f_puesto = st.sidebar.multiselect("Puesto Responsable", sorted([x for x in df_area.get("PUESTO RESPONSABLE", pd.Series([])).dropna().unique()]))

st.sidebar.caption("Si no seleccionas filtros, se muestra todo (por default).")

# =====================================================
# PROCESAMIENTO OBJETIVOS (año base)
# =====================================================
# id_cols: incluye criterios nuevos si existen
obj_id_cols = ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
for c in ["Tipo","Perspectiva","Eje","Departamento"]:
    if c in df_obj.columns and c not in obj_id_cols:
        obj_id_cols.insert(0, c)

obj_long = normalizar_meses(df_obj, obj_id_cols)
obj_long["valor"] = obj_long["Estado"].map(estado_map).fillna(0)

# aplicar filtros a obj_long
obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "Departamento", f_depto)

# resumen
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

# filtro por estado ejecutivo (opcional) - lo dejo en UI dentro, no en sidebar para no saturar
estado_opts = ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]

# =====================================================
# PROCESAMIENTO AREAS (año base)
# =====================================================
# Estándares mínimos:
if "AREA" not in df_area.columns and "DEPARTAMENTO" in df_area.columns:
    df_area["AREA"] = df_area["DEPARTAMENTO"]

if "PUESTO RESPONSABLE" not in df_area.columns:
    df_area["PUESTO RESPONSABLE"] = "SIN_DATO"

area_id_cols = ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO"]:
    if c in df_area.columns and c not in area_id_cols:
        area_id_cols.insert(0, c)

area_long = normalizar_meses(df_area, [c for c in area_id_cols if c in df_area.columns])
area_long["valor"] = area_long["Estado"].map(estado_map).fillna(0)

# filtros areas
area_long = apply_filter(area_long, "AREA", f_area)
area_long = apply_filter(area_long, "PUESTO RESPONSABLE", f_puesto)

# resumen áreas
area_res_area = area_long.groupby(["AREA"], as_index=False).agg(
    cumplimiento=("valor","mean"),
    tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)
area_res_area["cumplimiento_%"] = area_res_area["cumplimiento"]*100

# por puesto
area_res_puesto = area_long.groupby(["AREA","PUESTO RESPONSABLE"], as_index=False).agg(
    cumplimiento=("valor","mean"),
    tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
)
area_res_puesto["cumplimiento_%"] = area_res_puesto["cumplimiento"]*100

# =====================================================
# KPI + GAUGES (más orden)
# =====================================================
tabs = st.tabs(["📌 Resumen", "🎯 Objetivos", "🏢 Áreas", "📊 Comparativo 2024 vs 2025", "🚨 Alertas", "📄 Exportar", "📋 Datos"])

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
               "steps": [{"range":[0,60],"color":"red"},
                         {"range":[60,90],"color":"yellow"},
                         {"range":[90,100],"color":"green"}]},
        title={"text": f"{year_data} – Cumplimiento Estratégico (Objetivos)"}
    ))
    g1.plotly_chart(fig_g1, use_container_width=True)

    fig_g2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(area_long["valor"].mean()*100) if len(area_long) else 0,
        gauge={"axis": {"range": [0, 100]},
               "steps": [{"range":[0,60],"color":"red"},
                         {"range":[60,90],"color":"yellow"},
                         {"range":[90,100],"color":"green"}]},
        title={"text": f"{year_data} – Cumplimiento Operativo (Áreas)"}
    ))
    g2.plotly_chart(fig_g2, use_container_width=True)

    # Gráfica rápida de mix de estados (Objetivos) + tendencia mensual
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mix de Estados Ejecutivos (Objetivos)**")
        fig = px.bar(
            obj_resumen["estado_ejecutivo"].value_counts().reindex(estado_opts).fillna(0).reset_index(),
            x="index", y="estado_ejecutivo",
            text="estado_ejecutivo"
        )
        fig.update_layout(xaxis_title="Estado", yaxis_title="Cantidad")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Tendencia mensual (promedio)**")
        tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
        tr["cumplimiento_%"] = tr["valor"]*100
        fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
        fig.update_layout(yaxis_title="% promedio", xaxis_title="Mes")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("🎯 Objetivos – Análisis Avanzado")

    # Filtro interno de estado ejecutivo (más cómodo que sidebar)
    estado_sel = st.multiselect("Filtrar por Estado Ejecutivo (opcional)", estado_opts, default=[])
    obj_view = obj_resumen.copy()
    if estado_sel:
        obj_view = obj_view[obj_view["estado_ejecutivo"].isin(estado_sel)]

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Top objetivos críticos (peor cumplimiento)**")
        top_bad = obj_view.sort_values("cumplimiento_%").head(15)
        fig = px.bar(top_bad, x="cumplimiento_%", y="Objetivo", orientation="h",
                     color="estado_ejecutivo",
                     color_discrete_map=COLOR_EJEC)
        fig.update_layout(xaxis_title="Cumplimiento %", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("**Distribución por Departamento (Treemap)**")
        if "Departamento" in obj_view.columns:
            dep = obj_view.groupby("Departamento")["cumplimiento_%"].mean().reset_index()
            fig = px.treemap(dep, path=["Departamento"], values="cumplimiento_%",
                             color="cumplimiento_%",
                             color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No existe columna Departamento en este año.")

    # Perspectiva / Eje
    r3, r4 = st.columns(2)
    with r3:
        st.markdown("**Cumplimiento por Perspectiva**")
        if "Perspectiva" in obj_view.columns:
            p = obj_view.groupby("Perspectiva")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
            fig = px.bar(p, x="cumplimiento_%", y="Perspectiva", orientation="h",
                         color="cumplimiento_%", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No existe columna Perspectiva en este año.")

    with r4:
        st.markdown("**Cumplimiento por Tipo (POA vs PEC)**")
        if "Tipo" in obj_view.columns:
            t = obj_view.groupby("Tipo")["cumplimiento_%"].mean().reset_index()
            fig = px.bar(t, x="Tipo", y="cumplimiento_%", text="cumplimiento_%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No existe columna Tipo en este año.")

with tabs[2]:
    st.subheader("🏢 Áreas – Control Operativo Avanzado")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ranking crítico de áreas (peor → mejor)**")
        rk = area_res_area.sort_values("cumplimiento_%").head(20)
        fig = px.bar(rk, x="cumplimiento_%", y="AREA", orientation="h",
                     color="cumplimiento_%", color_continuous_scale="RdYlGn")
        fig.update_layout(xaxis_title="Cumplimiento %", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Cumplimiento vs Carga (# tareas)**")
        sc = area_res_area.copy()
        fig = px.scatter(sc, x="tareas", y="cumplimiento_%", hover_name="AREA",
                         size="tareas")
        fig.update_layout(xaxis_title="# tareas (carga)", yaxis_title="Cumplimiento %")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Heatmap operativo (Área vs Mes)**")
    heat = area_long.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
    fig = px.imshow(heat, color_continuous_scale=["red","yellow","green"])
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("📊 Comparativo 2024 vs 2025 (VERDE/AMARILLO/ROJO/MORADO)")

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el comparativo (sidebar).")
    else:
        # Cargamos años comparativos
        comp_objs = []
        comp_areas = []

        for y in compare_years:
            o, a = load_year(y)
            # normaliza objetivos
            o = safe_strip(o)
            oid = ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            for c in ["Tipo","Perspectiva","Eje","Departamento"]:
                if c in o.columns and c not in oid:
                    oid.insert(0, c)
            ol = normalizar_meses(o, oid)
            ol["AÑO"] = y

            # aplica filtros comparativos (si los usas)
            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)

            comp_objs.append(ol)

            # areas
            a = standardize_areas(a)
            if "AREA" not in a.columns and "DEPARTAMENTO" in a.columns:
                a["AREA"] = a["DEPARTAMENTO"]
            if "PUESTO RESPONSABLE" not in a.columns and "PUESTO" in a.columns:
                a.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

            aid = [c for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"] if c in a.columns]
            al = normalizar_meses(a, aid)
            al["AÑO"] = y

            # filtros de areas
            al = apply_filter(al, "AREA", f_area)
            al = apply_filter(al, "PUESTO RESPONSABLE", f_puesto)

            comp_areas.append(al)

        comp_obj_long = pd.concat(comp_objs, ignore_index=True)
        comp_area_long = pd.concat(comp_areas, ignore_index=True)

        # --- Objetivos: % por estado por año ---
        st.markdown("### 🎯 Objetivos: % por color (VERDE/AMARILLO/ROJO/MORADO)")
        obj_mix = (comp_obj_long.groupby(["AÑO","Estado"]).size()
                   .reset_index(name="conteo"))
        total = obj_mix.groupby("AÑO")["conteo"].transform("sum")
        obj_mix["%"] = (obj_mix["conteo"] / total) * 100

        fig = px.bar(
            obj_mix,
            x="AÑO", y="%", color="Estado",
            barmode="group",
            color_discrete_map=COLOR_ESTADO,
            text="%",
            category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # --- Áreas: % por estado por año ---
        st.markdown("### 🏢 Áreas: % por color (VERDE/AMARILLO/ROJO/MORADO)")
        area_mix = (comp_area_long.groupby(["AÑO","Estado"]).size()
                    .reset_index(name="conteo"))
        total2 = area_mix.groupby("AÑO")["conteo"].transform("sum")
        area_mix["%"] = (area_mix["conteo"] / total2) * 100

        fig = px.bar(
            area_mix,
            x="AÑO", y="%", color="Estado",
            barmode="group",
            color_discrete_map=COLOR_ESTADO,
            text="%",
            category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # --- Comparativo por Departamento (objetivos) ---
        if "Departamento" in comp_obj_long.columns:
            st.markdown("### 🧩 Comparativo por Departamento (promedio %)")
            comp_dep = comp_obj_long.copy()
            comp_dep["valor"] = comp_dep["Estado"].map(estado_map).fillna(0)
            dep = comp_dep.groupby(["AÑO","Departamento"])["valor"].mean().reset_index()
            dep["cumplimiento_%"] = dep["valor"]*100
            fig = px.bar(dep, x="cumplimiento_%", y="Departamento", color="AÑO", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    # Alertas objetivos
    crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
    for _, r in crit_obj.iterrows():
        sev = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
        alert_rows.append({
            "Año": year_data,
            "Nivel": sev,
            "Tipo": "Objetivo",
            "Nombre": r.get("Objetivo",""),
            "Estado": r["estado_ejecutivo"],
            "Cumplimiento": r["cumplimiento_%"]
        })

    # Alertas áreas (<60%)
    bad_areas = area_res_area[area_res_area["cumplimiento_%"] < 60].copy()
    for _, r in bad_areas.iterrows():
        alert_rows.append({
            "Año": year_data,
            "Nivel": "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL",
            "Tipo": "Área",
            "Nombre": r["AREA"],
            "Estado": "BAJO CUMPLIMIENTO",
            "Cumplimiento": r["cumplimiento_%"]
        })

    alerts_df = pd.DataFrame(alert_rows)
    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        alerts_df = alerts_df.sort_values(["Nivel","Cumplimiento"], ascending=[True, True])
        alerts_df["Cumplimiento"] = alerts_df["Cumplimiento"].map(lambda x: round(float(x),1))

        def semaforo(row):
            # ROJO: crítica, AMARILLO: normal
            bg = "background-color: #ffdddd" if row["Nivel"] == "CRÍTICA" else "background-color: #fff3cd"
            return [bg]*len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas, imprime a PDF)")

    # Gráficas clave para export
    fig_estado_exec = px.pie(obj_resumen, names="estado_ejecutivo", title=f"{year_data} – Estados Ejecutivos (Objetivos)")
    fig_rank_areas = px.bar(area_res_area.sort_values("cumplimiento_%").head(20),
                            x="cumplimiento_%", y="AREA", orientation="h",
                            title=f"{year_data} – Ranking crítico de áreas")

    def build_report_html():
        parts = []
        parts.append(f"""
        <html><head><meta charset="utf-8"/>
        <title>Reporte Estratégico</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 22px; }}
          .kpis {{ display:flex; gap:12px; flex-wrap:wrap; }}
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

        # tabla alertas
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
