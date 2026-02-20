import streamlit as st
import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import WorksheetNotFound
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
# ESTILO (EJECUTIVO CLARO)
# =====================================================
st.markdown(
    """
<style>
/* App */
.stApp { background: #f3f4f6; color: #111111; }
.block-container { padding-top: 1.1rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
section[data-testid="stSidebar"] * { color: #111111 !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #111111 !important; }
section[data-testid="stSidebar"] input { color: #111111 !important; }

/* Baseweb selects (placeholder + value) */
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #111111 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] div { color: #111111 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] input { color: #111111 !important; }

/* Titles/text */
h1, h2, h3, h4, p { color: #111111 !important; }

/* KPI cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 14px;
    box-shadow: 0 1px 0 rgba(0,0,0,0.03);
}
div[data-testid="stMetricLabel"] > div { color: #111111 !important; font-weight: 650; }
div[data-testid="stMetricValue"] { color: #111111 !important; }
div[data-testid="stMetricDelta"] { color: #111111 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; }
[data-testid="stDataFrame"] * { color: #111111 !important; }

/* Dividers */
hr { border: none; border-top: 1px solid #e5e7eb; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# GOOGLE SHEETS AUTH (MISMA CONEXIÓN)
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

# Mapeo para puntaje: Verde=100%, Amarillo=50%, Rojo/Morado=0%
ESTADO_MAP = {"VERDE": 1.0, "AMARILLO": 0.5, "ROJO": 0.0, "MORADO": 0.0}
COLOR_ESTADO = {"VERDE":"#00a65a","AMARILLO":"#f1c40f","ROJO":"#e74c3c","MORADO":"#8e44ad"}

ESTADO_EJEC_ORDEN = ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]
COLOR_EJEC = {
    "CUMPLIDO": "#00a65a",
    "EN SEGUIMIENTO": "#f1c40f",
    "RIESGO": "#e74c3c",
    "CRÍTICO": "#8b0000",
    "NO SUBIDO": "#8e44ad",
}

FRECUENCIA_MAP = {
    "MENSUAL": 12, "BIMESTRAL": 6, "TRIMESTRAL": 4, "CUATRIMESTRAL": 3, "SEMESTRAL": 2, "ANUAL": 1
}

# =====================================================
# HELPERS
# =====================================================
def style_plotly(fig, height=520, title=None):
    """
    Aplica layout ejecutivo.
    Nota: si el fig es go.Indicator (gauge), NO tiene ejes y NO se deben tocar update_xaxes/update_yaxes.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=55 if title else 18, b=18),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111111", size=13),
        title=dict(text=title, x=0.02, xanchor="left") if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    has_indicator = any(getattr(tr, "type", "") == "indicator" for tr in fig.data)
    if not has_indicator:
        fig.update_xaxes(
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
            gridcolor="#e5e7eb",
        )
        fig.update_yaxes(
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
            gridcolor="#e5e7eb",
        )
    return fig

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\n", " ", regex=False).str.strip()

def normalize_estado_series(s: pd.Series) -> pd.Series:
    x = s.replace("", np.nan).dropna().astype(str).str.strip().str.upper()
    return x

def normalize_realizada(s: pd.Series) -> pd.Series:
    """
    Realizada/Planificada -> Realizada/No realizada
    Planificada cuenta como NO realizada.
    """
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"PLANIFICADA": "NO REALIZADA", "PLANIFICADO": "NO REALIZADA"})
    x = x.replace({"REALIZADA": "REALIZADA"})
    return x

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    long = df.melt(id_vars=id_cols, value_vars=meses_presentes, var_name="Mes", value_name="Estado").dropna(subset=["Estado"])
    long["Estado"] = normalize_estado_series(long["Estado"])
    long = long.dropna(subset=["Estado"])
    long["valor"] = long["Estado"].map(ESTADO_MAP).fillna(0.0)
    return long

def apply_filter(df: pd.DataFrame, col: str, selected: list):
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

def add_no_col(df: pd.DataFrame, name="No.") -> pd.DataFrame:
    """Para que el conteo/índice se vea desde 1 (no desde 0)."""
    if df is None or df.empty:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, name, range(1, len(out) + 1))
    return out

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_years_available():
    """
    Devuelve años disponibles por hojas numéricas.
    Nota: 2023 puede NO tener '2023 AREAS' y aún así debe aparecer.
    """
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    years = sorted([int(t) for t in titles if t.isdigit()])
    has_areas = {y: (f"{y} AREAS" in titles) for y in years}
    return years, has_areas

@st.cache_data(ttl=300, show_spinner=False)
def load_year(year: int):
    """
    Carga:
      - Hoja '{year}' (InformesGenerales) -> siempre
      - Hoja '{year} AREAS' (operativo) -> opcional (puede no existir en 2023)
    """
    sh = client.open(SHEET_NAME)

    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    df_obj.columns = normalize_text_series(df_obj.columns.to_series())

    # Operativo (puede no existir)
    try:
        df_dept = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())
        df_dept.columns = normalize_text_series(df_dept.columns.to_series())
    except WorksheetNotFound:
        df_dept = pd.DataFrame()

    # --- Normalizaciones InformesGenerales ---
    for c in ["Tipo","Perspectiva","Eje","TareasObjetivos","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
        if c in df_obj.columns:
            df_obj[c] = normalize_text_series(df_obj[c])

    # --- Normalizaciones Operativo ---
    if not df_dept.empty:
        # Dic duplicado
        if "Diciembre" in df_dept.columns:
            if "Dic" in df_dept.columns:
                df_dept["Dic"] = df_dept["Dic"].fillna(df_dept["Diciembre"])
            else:
                df_dept.rename(columns={"Diciembre": "Dic"}, inplace=True)

        # Puesto responsable
        if "PUESTO" in df_dept.columns and "PUESTO RESPONSABLE" not in df_dept.columns:
            df_dept.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

        # Operativo: TareasObjetivos (si existe "Área", la convertimos)
        if "TareasObjetivos" not in df_dept.columns and "Área" in df_dept.columns:
            df_dept.rename(columns={"Área":"TareasObjetivos"}, inplace=True)

        for c in ["TIPO","PERSPECTIVA","EJE","TareasObjetivos","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?"]:
            if c in df_dept.columns:
                df_dept[c] = normalize_text_series(df_dept[c])

        if "¿Realizada?" in df_dept.columns:
            df_dept["¿Realizada?"] = normalize_realizada(df_dept["¿Realizada?"])

    return df_obj, df_dept

# =====================================================
# SIDEBAR
# =====================================================
years, has_areas = get_years_available()
if not years:
    st.error("No encontré hojas tipo '2023', '2024', '2025' dentro de tu Google Sheets.")
    st.stop()

st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)

# default comparativo: últimos 2 años si existen, si no, los disponibles
default_compare = []
for cand in [2024, 2025]:
    if cand in years:
        default_compare.append(cand)
if len(default_compare) < 2 and len(years) >= 2:
    default_compare = years[-2:]

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar",
    options=years,
    default=default_compare
)

df_obj, df_dept = load_year(year_data)
has_operativo_year = (df_dept is not None) and (not df_dept.empty)

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# --- InformesGenerales ---
f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", sorted(df_obj["Tipo"].dropna().unique()) if "Tipo" in df_obj.columns else [])
f_persp = st.sidebar.multiselect("Perspectiva", sorted(df_obj["Perspectiva"].dropna().unique()) if "Perspectiva" in df_obj.columns else [])
f_eje = st.sidebar.multiselect("Eje", sorted(df_obj["Eje"].dropna().unique()) if "Eje" in df_obj.columns else [])
f_depto = st.sidebar.multiselect("TareasObjetivos (estratégico)", sorted(df_obj["TareasObjetivos"].dropna().unique()) if "TareasObjetivos" in df_obj.columns else [])

st.sidebar.markdown("---")

# --- Operativo (solo si existe la hoja AREAS del año base) ---
if has_operativo_year:
    f_dept_op = st.sidebar.multiselect("TareasObjetivos (operativo)", sorted(df_dept["TareasObjetivos"].dropna().unique()) if "TareasObjetivos" in df_dept.columns else [])
    f_puesto = st.sidebar.multiselect("Puesto Responsable", sorted(df_dept["PUESTO RESPONSABLE"].dropna().unique()) if "PUESTO RESPONSABLE" in df_dept.columns else [])
    f_realizada = st.sidebar.multiselect("Ejecución (Realizada / No realizada)", sorted(df_dept["¿Realizada?"].dropna().unique()) if "¿Realizada?" in df_dept.columns else [])
else:
    st.sidebar.info(f"ℹ️ El año {year_data} no tiene hoja '{year_data} AREAS'. Se mostrará solo la parte estratégica (InformesGenerales).")
    f_dept_op, f_puesto, f_realizada = [], [], []

st.sidebar.caption("✅ Si NO seleccionas filtros, se muestra TODO por default.")

# =====================================================
# PROCESAMIENTO InformesGenerales
# =====================================================
obj_id_cols = ["Tipo","Perspectiva","Eje","TareasObjetivos","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
obj_id_cols = [c for c in obj_id_cols if c in df_obj.columns]

obj_long = normalizar_meses(df_obj, obj_id_cols)

obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "TareasObjetivos", f_depto)

grp_cols = [c for c in ["Tipo","Perspectiva","Eje","TareasObjetivos","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in obj_long.columns]

obj_resumen = obj_long.groupby(grp_cols, as_index=False).agg(
    score_total=("valor","sum"),
    verdes=("Estado", lambda x: (x=="VERDE").sum()),
    amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum()),
    meses_reportados=("Mes","count")
)

if "Frecuencia Medición" in obj_resumen.columns:
    obj_resumen["Frecuencia Medición"] = obj_resumen["Frecuencia Medición"].astype(str).str.strip().str.upper()
    obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(FRECUENCIA_MAP).fillna(12)
else:
    obj_resumen["meses_esperados"] = 12

obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1) * 100
obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec, axis=1)

# =====================================================
# PROCESAMIENTO OPERATIVO (si existe)
# =====================================================
if has_operativo_year:
    dept_id_cols = ["TIPO","PERSPECTIVA","EJE","TareasObjetivos","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
    dept_id_cols = [c for c in dept_id_cols if c in df_dept.columns]

    dept_long = normalizar_meses(df_dept, dept_id_cols)

    dept_long = apply_filter(dept_long, "TareasObjetivos", f_dept_op)
    dept_long = apply_filter(dept_long, "PUESTO RESPONSABLE", f_puesto)
    dept_long = apply_filter(dept_long, "¿Realizada?", f_realizada)

    dept_res = dept_long.groupby("TareasObjetivos", as_index=False).agg(
        cumplimiento=("valor","mean"),
        tareas=("TAREA","nunique") if "TAREA" in dept_long.columns else ("Estado","count"),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
        verdes=("Estado", lambda x: (x=="VERDE").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum())
    )
    dept_res["cumplimiento_%"] = dept_res["cumplimiento"] * 100

    exec_res = None
    if "¿Realizada?" in dept_long.columns:
        exec_res = (dept_long.groupby(["TareasObjetivos","¿Realizada?"]).size()
                    .reset_index(name="conteo"))
        exec_res["%"] = exec_res["conteo"] / exec_res.groupby("TareasObjetivos")["conteo"].transform("sum") * 100

    dept_res_puesto = None
    if "PUESTO RESPONSABLE" in dept_long.columns:
        dept_res_puesto = dept_long.groupby(["TareasObjetivos","PUESTO RESPONSABLE"], as_index=False).agg(
            cumplimiento=("valor","mean"),
            tareas=("TAREA","nunique") if "TAREA" in dept_long.columns else ("Estado","count")
        )
        dept_res_puesto["cumplimiento_%"] = dept_res_puesto["cumplimiento"] * 100
else:
    dept_long = pd.DataFrame(columns=["TareasObjetivos","Mes","Estado","valor"])
    dept_res = pd.DataFrame(columns=["TareasObjetivos","cumplimiento","tareas","rojos","amarillos","verdes","morados","cumplimiento_%"])
    exec_res = None
    dept_res_puesto = None

# =====================================================
# TABS
# =====================================================
tabs = st.tabs(["📌 Resumen", "🎯 InformesGenerales", "🏢 Operativo (Deptos)", "📊 Comparativo", "🚨 Alertas", "📄 Exportar", "📋 Datos"])

# =====================================================
# TAB 0: RESUMEN
# =====================================================
with tabs[0]:
    st.subheader(f"📌 Resumen Ejecutivo — Año {year_data}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("InformesGenerales", int(len(obj_resumen)))
    c2.metric("Cumplidos", int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()))
    c3.metric("En Riesgo", int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()))
    c4.metric("Críticos / No Subido", int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()))
    c5.metric("Cumplimiento Promedio", f"{obj_resumen['cumplimiento_%'].mean():.1f}%" if len(obj_resumen) else "—")

    g1, g2 = st.columns(2)

    val_obj = float(obj_resumen["cumplimiento_%"].mean()) if len(obj_resumen) else 0
    fig_g1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_obj,
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":"#111111"},
               "steps":[{"range":[0,60],"color":"#e74c3c"},
                        {"range":[60,90],"color":"#f1c40f"},
                        {"range":[90,100],"color":"#00a65a"}]},
        title={"text": f"{year_data} — Cumplimiento Estratégico (InformesGenerales)"}
    ))
    g1.plotly_chart(style_plotly(fig_g1, height=460), use_container_width=True)

    if has_operativo_year:
        val_dept = float(dept_long["valor"].mean()*100) if len(dept_long) else 0
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_dept,
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#111111"},
                   "steps":[{"range":[0,60],"color":"#e74c3c"},
                            {"range":[60,90],"color":"#f1c40f"},
                            {"range":[90,100],"color":"#00a65a"}]},
            title={"text": f"{year_data} — Cumplimiento Operativo (TareasObjetivos)"}
        ))
        g2.plotly_chart(style_plotly(fig_g2, height=460), use_container_width=True)
    else:
        g2.info(f"No hay hoja '{year_data} AREAS' → Operativo no disponible en {year_data}.")

    left, right = st.columns(2)

    with left:
        counts = obj_resumen["estado_ejecutivo"].value_counts().reindex(ESTADO_EJEC_ORDEN).dropna()
        counts = counts[counts > 0].reset_index()
        counts.columns = ["Estado Ejecutivo", "Cantidad"]

        if counts.empty:
            st.info("No hay datos para graficar con los filtros actuales.")
        else:
            fig = px.bar(
                counts,
                x="Estado Ejecutivo", y="Cantidad",
                color="Estado Ejecutivo",
                color_discrete_map=COLOR_EJEC,
                text="Cantidad"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, height=660, title="Distribución de Estados Ejecutivos (InformesGenerales)"), use_container_width=True)

    with right:
        if obj_long.empty:
            st.info("No hay datos de meses para graficar con los filtros actuales.")
        else:
            tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            st.plotly_chart(style_plotly(fig, height=660, title="Tendencia Mensual — Cumplimiento Promedio (InformesGenerales)"), use_container_width=True)

# =====================================================
# TAB 1: InformesGenerales
# =====================================================
with tabs[1]:
    st.subheader("🎯 InformesGenerales — Análisis Avanzado")

    colA, colB = st.columns(2)

    with colA:
        if obj_resumen.empty:
            st.info("No hay InformesGenerales con los filtros actuales.")
        else:
            top_bad = obj_resumen.sort_values("cumplimiento_%").head(15)
            fig = px.bar(
                top_bad, x="cumplimiento_%", y="Objetivo",
                orientation="h",
                color="estado_ejecutivo",
                color_discrete_map=COLOR_EJEC,
                text="cumplimiento_%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=760, title="Top 15 InformesGenerales más críticos (peor cumplimiento)"), use_container_width=True)

    with colB:
        if obj_resumen.empty:
            st.info("No hay datos para el mix.")
        else:
            fig = px.pie(
                obj_resumen, names="estado_ejecutivo", hole=0.55,
                color="estado_ejecutivo", color_discrete_map=COLOR_EJEC
            )
            st.plotly_chart(style_plotly(fig, height=760, title="Mix de Estado Ejecutivo (InformesGenerales)"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if "TareasObjetivos" in obj_resumen.columns and not obj_resumen.empty:
            dep = obj_resumen.groupby("TareasObjetivos")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
            fig = px.bar(dep, x="cumplimiento_%", y="TareasObjetivos", orientation="h",
                         color="cumplimiento_%", color_continuous_scale="RdYlGn")
            st.plotly_chart(style_plotly(fig, height=660, title="Cumplimiento Promedio por TareasObjetivos (estratégico)"), use_container_width=True)
        else:
            st.info("No existe columna TareasObjetivos (o no hay datos) en InformesGenerales para este año.")

    with c2:
        if "Perspectiva" in obj_resumen.columns and not obj_resumen.empty:
            p = obj_resumen.groupby("Perspectiva")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
            fig = px.bar(p, x="cumplimiento_%", y="Perspectiva", orientation="h",
                         color="cumplimiento_%", color_continuous_scale="RdYlGn")
            st.plotly_chart(style_plotly(fig, height=660, title="Cumplimiento Promedio por Perspectiva"), use_container_width=True)
        else:
            st.info("No existe columna Perspectiva (o no hay datos) en InformesGenerales para este año.")

    st.markdown("#### 🌡️ Heatmap — Objetivo vs Mes (Top 25 más críticos)")
    if obj_long.empty or "Objetivo" not in obj_long.columns:
        st.info("No hay datos suficientes para el heatmap con los filtros actuales.")
    else:
        hm_base = obj_long.copy()
        avg_obj = hm_base.groupby("Objetivo")["valor"].mean().sort_values().head(25).index.tolist()
        hm = hm_base[hm_base["Objetivo"].isin(avg_obj)].pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
        fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
        st.plotly_chart(style_plotly(fig, height=760, title="Heatmap (Top 25 InformesGenerales más críticos)"), use_container_width=True)

# =====================================================
# TAB 2: OPERATIVO
# =====================================================
with tabs[2]:
    st.subheader("🏢 Operativo — Control por TareasObjetivos")

    if not has_operativo_year:
        st.info(f"Este año ({year_data}) no incluye operativo porque no existe la hoja '{year_data} AREAS'.")
        st.stop()

    order = st.selectbox("Orden del ranking", ["Peor → Mejor", "Mejor → Peor"], index=0)
    asc = True if order == "Peor → Mejor" else False

    left, right = st.columns(2)
    with left:
        rk = dept_res.sort_values("cumplimiento_%", ascending=asc).head(20)
        fig = px.bar(rk, x="cumplimiento_%", y="TareasObjetivos", orientation="h",
                     color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(style_plotly(fig, height=760, title="Ranking de TareasObjetivos Operativos (Top 20)"), use_container_width=True)

    with right:
        sc = dept_res.copy()
        fig = px.scatter(sc, x="tareas", y="cumplimiento_%", size="tareas", hover_name="TareasObjetivos")
        st.plotly_chart(style_plotly(fig, height=760, title="Cumplimiento vs Carga (# tareas)"), use_container_width=True)

    if exec_res is not None and not exec_res.empty:
        st.markdown("#### ✅ Ejecución (Realizada vs No realizada) por TareasObjetivos — Top 15 con menor % realizada")
        tmp = exec_res.pivot_table(index="TareasObjetivos", columns="¿Realizada?", values="%", fill_value=0).reset_index()
        if "REALIZADA" in tmp.columns:
            tmp = tmp.sort_values("REALIZADA").head(15)
            fig = px.bar(tmp, x="REALIZADA", y="TareasObjetivos", orientation="h", text="REALIZADA")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=660, title="Top 15 deptos con menor % Realizada"), use_container_width=True)
        else:
            st.info("No se encontró categoría REALIZADA en ¿Realizada?")

    st.markdown("#### 🌡️ Heatmap Operativo — TareasObjetivos vs Mes (Top 25 más críticos)")
    hm_base = dept_long.copy()
    if hm_base.empty:
        st.info("No hay datos suficientes para el heatmap operativo con los filtros actuales.")
    else:
        avg_d = hm_base.groupby("TareasObjetivos")["valor"].mean().sort_values().head(25).index.tolist()
        hm = hm_base[hm_base["TareasObjetivos"].isin(avg_d)].pivot_table(index="TareasObjetivos", columns="Mes", values="valor", fill_value=0)
        fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
        st.plotly_chart(style_plotly(fig, height=760, title="Heatmap Operativo (Top 25 deptos más críticos)"), use_container_width=True)

# =====================================================
# TAB 3: COMPARATIVO
# =====================================================
with tabs[3]:
    st.subheader("📊 Comparativo (InformesGenerales y Operativo por TareasObjetivos)")

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el sidebar para comparar.")
    else:
        comp_obj = []
        comp_dept = []

        for y in compare_years:
            o, d = load_year(y)

            # InformesGenerales
            o_id = ["Tipo","Perspectiva","Eje","TareasObjetivos","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            o_id = [c for c in o_id if c in o.columns]
            ol = normalizar_meses(o, o_id)
            ol["AÑO"] = y

            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "TareasObjetivos", f_depto)
            comp_obj.append(ol)

            # OPERATIVO (solo si existe la hoja AREAS del año y)
            if d is not None and not d.empty:
                if "TareasObjetivos" not in d.columns and "Área" in d.columns:
                    d.rename(columns={"Área":"TareasObjetivos"}, inplace=True)
                if "¿Realizada?" in d.columns:
                    d["¿Realizada?"] = normalize_realizada(d["¿Realizada?"])

                d_id = ["TIPO","PERSPECTIVA","EJE","TareasObjetivos","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
                d_id = [c for c in d_id if c in d.columns]
                dl = normalizar_meses(d, d_id)
                dl["AÑO"] = y

                dl = apply_filter(dl, "TareasObjetivos", f_dept_op)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        comp_dept_long = pd.concat(comp_dept, ignore_index=True) if comp_dept else pd.DataFrame()

        st.markdown("### 🎯 InformesGenerales — % por color (VERDE/AMARILLO/ROJO/MORADO)")
        if comp_obj_long.empty:
            st.info("No hay datos de InformesGenerales para comparar con los filtros actuales.")
        else:
            obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            obj_mix["%"] = obj_mix["conteo"] / obj_mix.groupby("AÑO")["conteo"].transform("sum") * 100

            fig = px.bar(
                obj_mix, x="AÑO", y="%", color="Estado",
                barmode="group", color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=660, title="Comparativo InformesGenerales — % por color"), use_container_width=True)

        st.markdown("### 🏢 Operativo — % por color (VERDE/AMARILLO/ROJO/MORADO)")
        if comp_dept_long.empty:
            st.info("No hay datos operativos para comparar (por ejemplo, 2023 no tiene AREAS).")
        else:
            dep_mix = comp_dept_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            dep_mix["%"] = dep_mix["conteo"] / dep_mix.groupby("AÑO")["conteo"].transform("sum") * 100

            fig = px.bar(
                dep_mix, x="AÑO", y="%", color="Estado",
                barmode="group", color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=660, title="Comparativo Operativo — % por color"), use_container_width=True)

            st.markdown("### ✅ Operativo — % Realizada vs No realizada (por año)")
            if "¿Realizada?" in comp_dept_long.columns:
                ex = comp_dept_long.groupby(["AÑO","¿Realizada?"]).size().reset_index(name="conteo")
                ex["%"] = ex["conteo"] / ex.groupby("AÑO")["conteo"].transform("sum") * 100
                fig = px.bar(ex, x="AÑO", y="%", color="¿Realizada?", barmode="group", text="%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=620, title="Comparativo Ejecución (Realizada vs No realizada)"), use_container_width=True)

        st.markdown("### 📈 Tendencia mensual comparativa (promedio %)")
        left, right = st.columns(2)
        with left:
            if not comp_obj_long.empty:
                t = comp_obj_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                t["cumplimiento_%"] = t["valor"] * 100
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=620, title="InformesGenerales — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos de InformesGenerales para tendencia.")

        with right:
            if not comp_dept_long.empty:
                t = comp_dept_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                t["cumplimiento_%"] = t["valor"] * 100
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=620, title="Operativo — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos operativos para tendencia.")

# =====================================================
# TAB 4: ALERTAS
# =====================================================
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    if not obj_resumen.empty:
        crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
        for _, r in crit_obj.iterrows():
            nivel = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Tipo": "Objetivo",
                "Nombre": r.get("Objetivo",""),
                "Estado": r["estado_ejecutivo"],
                "Cumplimiento %": float(r["cumplimiento_%"])
            })

    if has_operativo_year and not dept_res.empty:
        bad_dept = dept_res[dept_res["cumplimiento_%"] < 60].copy()
        for _, r in bad_dept.iterrows():
            nivel = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Tipo": "TareasObjetivos (operativo)",
                "Nombre": r["TareasObjetivos"],
                "Estado": "BAJO CUMPLIMIENTO",
                "Cumplimiento %": float(r["cumplimiento_%"])
            })

    alerts_df = pd.DataFrame(alert_rows)

    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        alerts_df["OrdenNivel"] = alerts_df["Nivel"].map({"CRÍTICA":0,"NORMAL":1}).fillna(2)
        alerts_df = alerts_df.sort_values(["OrdenNivel","Tipo","Cumplimiento %"], ascending=[True, True, True]).drop(columns=["OrdenNivel"])
        alerts_df["Cumplimiento %"] = alerts_df["Cumplimiento %"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")

        alerts_df = add_no_col(alerts_df, "No.")

        def semaforo(row):
            bg = "#ffe1e1" if row["Nivel"] == "CRÍTICA" else "#fff3cd"
            return [f"background-color: {bg}; color: #111111;"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

# =====================================================
# TAB 5: EXPORTAR
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    if obj_resumen.empty:
        st.info("No hay datos para exportar con los filtros actuales.")
        st.stop()

    fig_estado_exec = px.pie(
        obj_resumen, names="estado_ejecutivo", hole=0.55,
        color="estado_ejecutivo", color_discrete_map=COLOR_EJEC,
        title=f"{year_data} — Estado Ejecutivo (InformesGenerales)"
    )
    fig_estado_exec = style_plotly(fig_estado_exec, height=520)

    fig_rank_dept = None
    if has_operativo_year and not dept_res.empty:
        fig_rank_dept = px.bar(
            dept_res.sort_values("cumplimiento_%").head(20),
            x="cumplimiento_%", y="TareasObjetivos", orientation="h",
            title=f"{year_data} — Ranking crítico Operativo (Top 20 deptos)"
        )
        fig_rank_dept = style_plotly(fig_rank_dept, height=520)

    def build_report_html():
        k_obj = len(obj_resumen)
        k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum())
        k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum())
        k_crit = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum())
        k_avg = float(obj_resumen["cumplimiento_%"].mean()) if len(obj_resumen) else 0

        rep_alert_html = "<p>Sin alertas.</p>"
        if 'alerts_df' in globals() and isinstance(alerts_df, pd.DataFrame) and (not alerts_df.empty):
            rep_alert_html = alerts_df.to_html(index=False)

        dept_html = "<p>Este año no incluye hoja AREAS.</p>"
        if has_operativo_year and not dept_res.empty:
            dept_html = dept_res.head(200).to_html(index=False)

        dept_chart_html = ""
        if fig_rank_dept is not None:
            dept_chart_html = fig_rank_dept.to_html(full_html=False, include_plotlyjs=False)

        return f"""
<html>
<head>
<meta charset="utf-8"/>
<title>Reporte Estratégico</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; background:#fff; }}
  .top {{ color:#555; margin-bottom:12px; }}
  .kpis {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .kpi {{ border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; min-width:160px; background:#fff; }}
  h1,h2 {{ margin: 10px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
  th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Reporte Estratégico y de Control</h1>
<div class="top">Año base: <b>{year_data}</b> · Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

<h2>KPIs</h2>
<div class="kpis">
  <div class="kpi"><b>InformesGenerales</b><br>{k_obj}</div>
  <div class="kpi"><b>Cumplidos</b><br>{k_ok}</div>
  <div class="kpi"><b>En Riesgo</b><br>{k_riesgo}</div>
  <div class="kpi"><b>Críticos/No Subido</b><br>{k_crit}</div>
  <div class="kpi"><b>Cumplimiento Promedio</b><br>{k_avg:.1f}%</div>
</div>

<h2>Gráficas</h2>
{fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn")}
{dept_chart_html}

<h2>Alertas</h2>
{rep_alert_html}

<h2>Tabla: InformesGenerales (resumen)</h2>
{obj_resumen.head(200).to_html(index=False)}

<h2>Tabla: Operativo (TareasObjetivos resumen)</h2>
{dept_html}

</body>
</html>
"""

    html_report = build_report_html()
    st.download_button(
        "⬇️ Descargar Reporte HTML",
        data=html_report,
        file_name=f"Reporte_Estrategico_{year_data}.html",
        mime="text/html"
    )
    st.info("Tip: abre el HTML en Chrome/Edge → Ctrl+P → Guardar como PDF.")

# =====================================================
# TAB 6: DATOS
# =====================================================
with tabs[6]:
    st.subheader("📋 Datos (auditoría)")

    with st.expander("InformesGenerales — Resumen"):
        st.dataframe(add_no_col(obj_resumen), use_container_width=True)

    with st.expander("InformesGenerales — Long"):
        st.dataframe(add_no_col(obj_long), use_container_width=True)

    with st.expander("Operativo — TareasObjetivos resumen"):
        if has_operativo_year:
            st.dataframe(add_no_col(dept_res), use_container_width=True)
        else:
            st.info(f"No hay hoja '{year_data} AREAS'.")

    with st.expander("Operativo — Long"):
        if has_operativo_year:
            st.dataframe(add_no_col(dept_long), use_container_width=True)
        else:
            st.info(f"No hay hoja '{year_data} AREAS'.")

st.caption("Fuente: Google Sheets · Dashboard Estratégico")

