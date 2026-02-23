import streamlit as st
import pandas as pd
import numpy as np
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
# ESTILO (EJECUTIVO CLARO + SIDEBAR LEGIBLE)
# =====================================================
st.markdown(
    """
<style>
.stApp { background: #f3f4f6; color: #111111; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] * {
    color: #111111 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #111111 !important;
}

/* Inputs del sidebar (multiselect/selectbox) */
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111111 !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #eef2ff !important;
    color: #111111 !important;
    border: 1px solid #c7d2fe !important;
}
section[data-testid="stSidebar"] input {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="popover"] * {
    color: #111111 !important;
}

/* Texto general */
h1, h2, h3, h4, h5, p, label, span, div {
    color: #111111;
}

/* Métricas */
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

/* Tabs */
button[data-baseweb="tab"] {
    color: #111111 !important;
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    margin-right: 6px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #eef2ff !important;
    border-color: #c7d2fe !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Alerts */
div[data-testid="stAlert"] * {
    color: #111111 !important;
}

/* Expanders */
details {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 6px 10px;
}
summary {
    color: #111111 !important;
    font-weight: 600;
}
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

ESTADO_MAP = {"VERDE": 1.0, "AMARILLO": 0.5, "ROJO": 0.0, "MORADO": 0.0}
COLOR_ESTADO = {
    "VERDE":"#00a65a",
    "AMARILLO":"#f1c40f",
    "ROJO":"#e74c3c",
    "MORADO":"#8e44ad"
}

ESTADO_EJEC_ORDEN = ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]
COLOR_EJEC = {
    "CUMPLIDO": "#00a65a",
    "EN SEGUIMIENTO": "#f1c40f",
    "RIESGO": "#e74c3c",
    "CRÍTICO": "#8b0000",
    "NO SUBIDO": "#8e44ad",
}

FRECUENCIA_MAP = {
    "MENSUAL": 12,
    "BIMESTRAL": 6,
    "TRIMESTRAL": 4,
    "CUATRIMESTRAL": 3,
    "SEMESTRAL": 2,
    "ANUAL": 1
}

# =====================================================
# HELPERS
# =====================================================
def safe_mean(s):
    try:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return 0.0
        return float(s.mean())
    except:
        return 0.0

def style_plotly(fig, height=560, title=None):
    """
    Estilo ejecutivo sin romper gauges (indicator).
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60 if title else 20, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfbfc",
        font=dict(color="#111111", size=13),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(color="#111111")) if title else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#111111")
        ),
    )

    has_indicator = any(getattr(tr, "type", "") == "indicator" for tr in fig.data)
    if not has_indicator:
        fig.update_xaxes(
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
            gridcolor="#e5e7eb",
            zerolinecolor="#e5e7eb"
        )
        fig.update_yaxes(
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
            gridcolor="#e5e7eb",
            zerolinecolor="#e5e7eb"
        )
    return fig

def gauge_fig(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value) if pd.notna(value) else 0,
        number={"suffix": "%", "font": {"color": "#111111", "size": 28}},
        title={"text": title, "font": {"color": "#111111", "size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#111111", "tickfont": {"color": "#111111"}},
            "bar": {"color": "#111111"},
            "bgcolor": "#ffffff",
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 60], "color": "#e74c3c"},
                {"range": [60, 90], "color": "#f1c40f"},
                {"range": [90, 100], "color": "#00a65a"},
            ],
        }
    ))
    return style_plotly(fig, height=420)

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\n", " ", regex=False).str.strip()

def normalize_estado_series(s: pd.Series) -> pd.Series:
    x = s.replace("", np.nan).dropna().astype(str).str.strip().str.upper()
    x = x.replace({
        "VERDE ": "VERDE",
        " AMARILLO": "AMARILLO",
        "ROJO ": "ROJO",
        "MORADO ": "MORADO"
    })
    return x

def normalize_realizada(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({
        "PLANIFICADA": "NO REALIZADA",
        "PLANIFICADO": "NO REALIZADA",
        "NO": "NO REALIZADA",
        "SI": "REALIZADA",
        "SÍ": "REALIZADA",
        "REALIZADA": "REALIZADA"
    })
    return x

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    if not meses_presentes:
        return pd.DataFrame(columns=id_cols + ["Mes", "Estado", "valor"])
    long = df.melt(
        id_vars=[c for c in id_cols if c in df.columns],
        value_vars=meses_presentes,
        var_name="Mes",
        value_name="Estado"
    ).dropna(subset=["Estado"])
    if long.empty:
        long["valor"] = []
        return long
    long["Estado"] = normalize_estado_series(long["Estado"])
    long = long[long["Estado"].isin(ESTADO_MAP.keys())].copy()
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

def make_count_df_from_series(series_counts, x_name="Categoría", y_name="Cantidad", start_at_one=False):
    s = series_counts.copy()
    if isinstance(s, pd.Series):
        df = s.reset_index()
        if df.shape[1] >= 2:
            df.columns = [x_name, y_name]
        else:
            df = pd.DataFrame({x_name: [], y_name: []})
    else:
        df = pd.DataFrame(columns=[x_name, y_name])

    if start_at_one and not df.empty:
        df[y_name] = pd.to_numeric(df[y_name], errors="coerce").fillna(0)
        df["y_name_plot"] = df[y_name].apply(lambda v: v if v > 0 else 1)
    return df

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_all_sheet_titles():
    sh = client.open(SHEET_NAME)
    return [ws.title.strip() for ws in sh.worksheets()]

@st.cache_data(ttl=300, show_spinner=False)
def get_years_available():
    titles = get_all_sheet_titles()
    years = sorted([int(t) for t in titles if t.isdigit()])
    return years

@st.cache_data(ttl=300, show_spinner=False)
def load_year(year: int):
    sh = client.open(SHEET_NAME)

    # Hoja estratégica (obligatoria)
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    if df_obj.empty:
        df_obj = pd.DataFrame()

    # Hoja operativa (puede no existir, ej. 2023)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    area_sheet_name = f"{year} AREAS"
    has_operativo = area_sheet_name in titles

    if has_operativo:
        df_dept = pd.DataFrame(sh.worksheet(area_sheet_name).get_all_records())
    else:
        df_dept = pd.DataFrame()

    # Normaliza columnas
    if not df_obj.empty:
        df_obj.columns = normalize_text_series(df_obj.columns.to_series())
    if not df_dept.empty:
        df_dept.columns = normalize_text_series(df_dept.columns.to_series())

        # Diciembre / Dic
        if "Diciembre" in df_dept.columns:
            if "Dic" in df_dept.columns:
                df_dept["Dic"] = df_dept["Dic"].replace("", np.nan).fillna(df_dept["Diciembre"])
            else:
                df_dept.rename(columns={"Diciembre": "Dic"}, inplace=True)

        # Puesto responsable
        if "PUESTO" in df_dept.columns and "PUESTO RESPONSABLE" not in df_dept.columns:
            df_dept.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

        # DEPARTAMENTO vs Área
        if "DEPARTAMENTO" not in df_dept.columns and "Área" in df_dept.columns:
            df_dept.rename(columns={"Área":"DEPARTAMENTO"}, inplace=True)
        if "DEPARTAMENTO" not in df_dept.columns and "AREA" in df_dept.columns:
            df_dept.rename(columns={"AREA":"DEPARTAMENTO"}, inplace=True)

    # Limpieza texto (objetivos)
    for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
        if c in df_obj.columns:
            df_obj[c] = normalize_text_series(df_obj[c])

    # Limpieza texto (operativo)
    for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?"]:
        if c in df_dept.columns:
            df_dept[c] = normalize_text_series(df_dept[c])

    if "¿Realizada?" in df_dept.columns:
        df_dept["¿Realizada?"] = normalize_realizada(df_dept["¿Realizada?"])

    return df_obj, df_dept, has_operativo

# =====================================================
# SIDEBAR
# =====================================================
years = get_years_available()
if not years:
    st.error("No encontré hojas de años (ej. 2023, 2024, 2025) en tu Google Sheets.")
    st.stop()

st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar",
    options=years,
    default=[y for y in [2024, 2025] if y in years] if len(years) >= 2 else [years[-1]]
)

df_obj, df_dept, has_operativo_year = load_year(year_data)

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# Filtro general sobre objetivos
f_objetivo = st.sidebar.multiselect(
    "Objetivo (estratégico)",
    sorted(df_obj["Objetivo"].dropna().unique()) if "Objetivo" in df_obj.columns else []
)

f_tipo_plan = st.sidebar.multiselect(
    "Tipo (POA / PEC)",
    sorted(df_obj["Tipo"].dropna().unique()) if "Tipo" in df_obj.columns else []
)

f_persp = st.sidebar.multiselect(
    "Perspectiva",
    sorted(df_obj["Perspectiva"].dropna().unique()) if "Perspectiva" in df_obj.columns else []
)

f_eje = st.sidebar.multiselect(
    "Eje",
    sorted(df_obj["Eje"].dropna().unique()) if "Eje" in df_obj.columns else []
)

f_depto = st.sidebar.multiselect(
    "Departamento",
    sorted(df_obj["Departamento"].dropna().unique()) if "Departamento" in df_obj.columns else []
)

# Filtros operativos (si existe hoja AREAS)
if has_operativo_year and not df_dept.empty:
    f_dept_op = st.sidebar.multiselect(
        "Departamento (operativo)",
        sorted(df_dept["DEPARTAMENTO"].dropna().unique()) if "DEPARTAMENTO" in df_dept.columns else []
    )
    f_puesto = st.sidebar.multiselect(
        "Puesto Responsable",
        sorted(df_dept["PUESTO RESPONSABLE"].dropna().unique()) if "PUESTO RESPONSABLE" in df_dept.columns else []
    )
    f_realizada = st.sidebar.multiselect(
        "Ejecución (Realizada / No realizada)",
        sorted(df_dept["¿Realizada?"].dropna().unique()) if "¿Realizada?" in df_dept.columns else []
    )

    # nuevo filtro que mencionaste (si existe y te estaba dando 0)
    f_objetivo_op = st.sidebar.multiselect(
        "Objetivo (operativo)",
        sorted(df_dept["OBJETIVO"].dropna().unique()) if "OBJETIVO" in df_dept.columns else []
    )
else:
    f_dept_op, f_puesto, f_realizada, f_objetivo_op = [], [], [], []

st.sidebar.caption("✅ Si NO seleccionas filtros, se muestra TODO por default.")

# =====================================================
# PROCESAMIENTO OBJETIVOS (HOJA AÑO)
# =====================================================
obj_id_cols = [
    "Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo",
    "Fecha Inicio","Fecha Fin","Frecuencia Medición"
]
obj_id_cols = [c for c in obj_id_cols if c in df_obj.columns]

obj_long = normalizar_meses(df_obj, obj_id_cols) if not df_obj.empty else pd.DataFrame()

# Aplicar filtros (estratégicos)
obj_long = apply_filter(obj_long, "Objetivo", f_objetivo)
obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "Departamento", f_depto)

if not obj_long.empty:
    grp_cols = [c for c in [
        "Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"
    ] if c in obj_long.columns]

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
else:
    obj_resumen = pd.DataFrame(columns=[
        "Objetivo","cumplimiento_%","estado_ejecutivo","score_total","verdes","amarillos","rojos","morados","meses_reportados","meses_esperados"
    ])

# =====================================================
# PROCESAMIENTO OPERATIVO (HOJA AÑO AREAS)
# =====================================================
if has_operativo_year and not df_dept.empty:
    dept_id_cols = [
        "TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA",
        "Fecha Inicio","Fecha Fin","¿Realizada?"
    ]
    dept_id_cols = [c for c in dept_id_cols if c in df_dept.columns]

    dept_long = normalizar_meses(df_dept, dept_id_cols)

    # Aplicar filtros operativos
    dept_long = apply_filter(dept_long, "DEPARTAMENTO", f_dept_op)
    dept_long = apply_filter(dept_long, "PUESTO RESPONSABLE", f_puesto)
    dept_long = apply_filter(dept_long, "¿Realizada?", f_realizada)
    dept_long = apply_filter(dept_long, "OBJETIVO", f_objetivo_op)

    if not dept_long.empty and "DEPARTAMENTO" in dept_long.columns:
        dept_res = dept_long.groupby("DEPARTAMENTO", as_index=False).agg(
            cumplimiento=("valor","mean"),
            tareas=("TAREA","nunique") if "TAREA" in dept_long.columns else ("Estado","count"),
            rojos=("Estado", lambda x: (x=="ROJO").sum()),
            amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
            verdes=("Estado", lambda x: (x=="VERDE").sum()),
            morados=("Estado", lambda x: (x=="MORADO").sum())
        )
        dept_res["cumplimiento_%"] = dept_res["cumplimiento"] * 100
    else:
        dept_res = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento_%","cumplimiento","tareas","rojos","amarillos","verdes","morados"])

    exec_res = None
    if not dept_long.empty and "¿Realizada?" in dept_long.columns and "DEPARTAMENTO" in dept_long.columns:
        exec_res = (dept_long.groupby(["DEPARTAMENTO","¿Realizada?"]).size()
                    .reset_index(name="conteo"))
        exec_res["%"] = exec_res["conteo"] / exec_res.groupby("DEPARTAMENTO")["conteo"].transform("sum") * 100

    dept_res_puesto = None
    if not dept_long.empty and "PUESTO RESPONSABLE" in dept_long.columns and "DEPARTAMENTO" in dept_long.columns:
        dept_res_puesto = dept_long.groupby(["DEPARTAMENTO","PUESTO RESPONSABLE"], as_index=False).agg(
            cumplimiento=("valor","mean"),
            tareas=("TAREA","nunique") if "TAREA" in dept_long.columns else ("Estado","count")
        )
        dept_res_puesto["cumplimiento_%"] = dept_res_puesto["cumplimiento"] * 100
else:
    dept_long = pd.DataFrame()
    dept_res = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento_%","cumplimiento","tareas","rojos","amarillos","verdes","morados"])
    exec_res = None
    dept_res_puesto = None

# =====================================================
# MEDIDORES (CORRECTOS POR HOJA + UNIÓN)
# =====================================================
# 1) Estratégico = hoja "YYYY"
gauge_obj_val = safe_mean(obj_resumen["cumplimiento_%"]) if not obj_resumen.empty else 0.0

# 2) Operativo = hoja "YYYY AREAS"
if has_operativo_year and not dept_long.empty:
    gauge_op_val = safe_mean(dept_long["valor"]) * 100
else:
    gauge_op_val = 0.0

# 3) Unión = promedio de ambos medidores
if has_operativo_year:
    gauge_global_val = (gauge_obj_val + gauge_op_val) / 2
else:
    gauge_global_val = gauge_obj_val

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "📌 Resumen",
    "🎯 Objetivos",
    "🏢 Operativo (Deptos)",
    "📊 Comparativo",
    "🚨 Alertas",
    "📄 Exportar",
    "📋 Datos"
])

# =====================================================
# TAB 0: RESUMEN
# =====================================================
with tabs[0]:
    st.subheader(f"📌 Resumen Ejecutivo — Año {year_data}")

    c1,c2,c3,c4,c5,c6 = st.columns(6)

    c1.metric("Objetivos", int(len(obj_resumen)))
    c2.metric("Cumplidos", int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0)
    c3.metric("En Riesgo", int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0)
    c4.metric("Críticos / No Subido", int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0)
    c5.metric("Cumpl. Estratégico", f"{gauge_obj_val:.1f}%")
    c6.metric("Cumpl. Global (Unión)", f"{gauge_global_val:.1f}%")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_fig(gauge_obj_val, f"{year_data} — Hoja {year_data} (Estratégico)"), use_container_width=True)
    with g2:
        if has_operativo_year:
            st.plotly_chart(gauge_fig(gauge_op_val, f"{year_data} — Hoja {year_data} AREAS (Operativo)"), use_container_width=True)
        else:
            st.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")
    with g3:
        st.plotly_chart(gauge_fig(gauge_global_val, f"{year_data} — Unión Estratégico + Operativo"), use_container_width=True)

    left, right = st.columns(2)

    with left:
        if not obj_resumen.empty:
            counts = (
                obj_resumen["estado_ejecutivo"]
                .value_counts()
                .reindex(ESTADO_EJEC_ORDEN)
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            counts.columns = ["Estado Ejecutivo", "Cantidad"]
            # Conteo visual no desde 0 (solo visual), como pediste
            counts["Cantidad_plot"] = counts["Cantidad"].apply(lambda x: x if x > 0 else 1)

            fig = px.bar(
                counts,
                x="Estado Ejecutivo",
                y="Cantidad_plot",
                color="Estado Ejecutivo",
                color_discrete_map=COLOR_EJEC,
                text="Cantidad"
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(title="Cantidad (mínimo visual = 1)")
            st.plotly_chart(style_plotly(fig, height=640, title="Distribución de Estados Ejecutivos (Objetivos)"), use_container_width=True)
        else:
            st.info("Sin datos de objetivos con los filtros actuales.")

    with right:
        if not obj_long.empty:
            tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
            tr.columns = ["Mes", "valor"]
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_yaxes(title="% promedio", range=[0, 100])
            st.plotly_chart(style_plotly(fig, height=640, title="Tendencia Mensual — Cumplimiento Promedio (Objetivos)"), use_container_width=True)
        else:
            st.info("Sin datos suficientes para tendencia de objetivos.")

# =====================================================
# TAB 1: OBJETIVOS
# =====================================================
with tabs[1]:
    st.subheader("🎯 Objetivos — Análisis Avanzado")

    if obj_resumen.empty:
        st.warning("No hay datos de objetivos con los filtros actuales.")
    else:
        colA, colB = st.columns(2)

        with colA:
            top_bad = obj_resumen.sort_values("cumplimiento_%").head(15)
            fig = px.bar(
                top_bad,
                x="cumplimiento_%",
                y="Objetivo",
                orientation="h",
                color="estado_ejecutivo",
                color_discrete_map=COLOR_EJEC,
                text="cumplimiento_%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(style_plotly(fig, height=760, title="Top 15 Objetivos más críticos (peor cumplimiento)"), use_container_width=True)

        with colB:
            fig = px.pie(
                obj_resumen,
                names="estado_ejecutivo",
                hole=0.55,
                color="estado_ejecutivo",
                color_discrete_map=COLOR_EJEC
            )
            st.plotly_chart(style_plotly(fig, height=760, title="Mix de Estado Ejecutivo (Objetivos)"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Departamento" in obj_resumen.columns:
                dep = obj_resumen.groupby("Departamento")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                fig = px.bar(dep, x="cumplimiento_%", y="Departamento", orientation="h",
                             color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=660, title="Cumplimiento Promedio por Departamento (estratégico)"), use_container_width=True)
            else:
                st.info("No existe columna Departamento en objetivos para este año.")

        with c2:
            if "Perspectiva" in obj_resumen.columns:
                p = obj_resumen.groupby("Perspectiva")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                fig = px.bar(p, x="cumplimiento_%", y="Perspectiva", orientation="h",
                             color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=660, title="Cumplimiento Promedio por Perspectiva"), use_container_width=True)
            else:
                st.info("No existe columna Perspectiva en objetivos para este año.")

        c3, c4 = st.columns(2)
        with c3:
            if "Tipo" in obj_resumen.columns:
                t = obj_resumen.groupby("Tipo")["cumplimiento_%"].mean().reset_index()
                fig = px.bar(t, x="Tipo", y="cumplimiento_%", color="Tipo", text="cumplimiento_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=560, title="Cumplimiento por Tipo (POA / PEC)"), use_container_width=True)
            else:
                st.info("No existe columna Tipo.")

        with c4:
            if "Eje" in obj_resumen.columns:
                e = obj_resumen.groupby("Eje")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                fig = px.bar(e, x="cumplimiento_%", y="Eje", orientation="h",
                             color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=560, title="Cumplimiento por Eje"), use_container_width=True)
            else:
                st.info("No existe columna Eje.")

        st.markdown("#### 🌡️ Heatmap — Objetivo vs Mes (Top 25 más críticos)")
        if not obj_long.empty and "Objetivo" in obj_long.columns:
            hm_base = obj_long.copy()
            avg_obj = hm_base.groupby("Objetivo")["valor"].mean().sort_values().head(25).index.tolist()
            hm = hm_base[hm_base["Objetivo"].isin(avg_obj)].pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
            if not hm.empty:
                hm = hm.reindex(columns=[m for m in MESES if m in hm.columns])
                fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
                st.plotly_chart(style_plotly(fig, height=760, title="Heatmap (Top 25 objetivos más críticos)"), use_container_width=True)
            else:
                st.info("No hay datos para el heatmap.")
        else:
            st.info("No hay datos para el heatmap.")

# =====================================================
# TAB 2: OPERATIVO (DEPARTAMENTO)
# =====================================================
with tabs[2]:
    st.subheader("🏢 Operativo — Control por Departamento")

    if not has_operativo_year:
        st.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")
    elif dept_long.empty:
        st.warning("No hay datos operativos con los filtros actuales.")
    else:
        order = st.selectbox("Orden del ranking", ["Peor → Mejor", "Mejor → Peor"], index=0)
        asc = True if order == "Peor → Mejor" else False

        left, right = st.columns(2)
        with left:
            rk = dept_res.sort_values("cumplimiento_%", ascending=asc).head(20)
            fig = px.bar(
                rk,
                x="cumplimiento_%",
                y="DEPARTAMENTO",
                orientation="h",
                color="cumplimiento_%",
                color_continuous_scale="RdYlGn",
                text="cumplimiento_%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis={'categoryorder':'total ascending' if asc else 'total descending'})
            st.plotly_chart(style_plotly(fig, height=760, title="Ranking de Departamentos Operativos (Top 20)"), use_container_width=True)

        with right:
            sc = dept_res.copy()
            fig = px.scatter(sc, x="tareas", y="cumplimiento_%", size="tareas", hover_name="DEPARTAMENTO")
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(style_plotly(fig, height=760, title="Cumplimiento vs Carga (# tareas)"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if exec_res is not None and not exec_res.empty:
                tmp = exec_res.pivot_table(index="DEPARTAMENTO", columns="¿Realizada?", values="%", fill_value=0).reset_index()
                if "REALIZADA" in tmp.columns:
                    tmp = tmp.sort_values("REALIZADA").head(15)
                    fig = px.bar(tmp, x="REALIZADA", y="DEPARTAMENTO", orientation="h", text="REALIZADA")
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    st.plotly_chart(style_plotly(fig, height=620, title="Top 15 deptos con menor % Realizada"), use_container_width=True)
                else:
                    st.info("No se encontró categoría REALIZADA en ¿Realizada?")
            else:
                st.info("No hay datos de ejecución (Realizada / No realizada).")

        with c2:
            # Mix por color operativo
            mix = dept_long["Estado"].value_counts().reindex(["VERDE","AMARILLO","ROJO","MORADO"]).fillna(0).astype(int).reset_index()
            mix.columns = ["Estado","Cantidad"]
            mix["Cantidad_plot"] = mix["Cantidad"].apply(lambda x: x if x > 0 else 1)
            fig = px.bar(
                mix, x="Estado", y="Cantidad_plot", color="Estado",
                color_discrete_map=COLOR_ESTADO, text="Cantidad"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, height=620, title="Distribución de colores (Operativo)"), use_container_width=True)

        st.markdown("#### 🌡️ Heatmap Operativo — Departamento vs Mes (Top 25 más críticos)")
        hm_base = dept_long.copy()
        avg_d = hm_base.groupby("DEPARTAMENTO")["valor"].mean().sort_values().head(25).index.tolist()
        hm = hm_base[hm_base["DEPARTAMENTO"].isin(avg_d)].pivot_table(index="DEPARTAMENTO", columns="Mes", values="valor", fill_value=0)
        if not hm.empty:
            hm = hm.reindex(columns=[m for m in MESES if m in hm.columns])
            fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
            st.plotly_chart(style_plotly(fig, height=760, title="Heatmap Operativo (Top 25 deptos más críticos)"), use_container_width=True)
        else:
            st.info("No hay datos para heatmap operativo.")

# =====================================================
# TAB 3: COMPARATIVO
# =====================================================
with tabs[3]:
    st.subheader("📊 Comparativo (Objetivos y Operativo por Departamento)")

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el sidebar para comparar.")
    else:
        comp_obj = []
        comp_dept = []

        for y in compare_years:
            o, d, has_op = load_year(y)

            # OBJETIVOS
            if not o.empty:
                o_id = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
                o_id = [c for c in o_id if c in o.columns]
                ol = normalizar_meses(o, o_id)
                ol["AÑO"] = y

                # aplica mismos filtros estratégicos
                ol = apply_filter(ol, "Objetivo", f_objetivo)
                ol = apply_filter(ol, "Tipo", f_tipo_plan)
                ol = apply_filter(ol, "Perspectiva", f_persp)
                ol = apply_filter(ol, "Eje", f_eje)
                ol = apply_filter(ol, "Departamento", f_depto)
                comp_obj.append(ol)

            # OPERATIVO
            if has_op and not d.empty:
                if "¿Realizada?" in d.columns:
                    d["¿Realizada?"] = normalize_realizada(d["¿Realizada?"])

                d_id = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
                d_id = [c for c in d_id if c in d.columns]
                dl = normalizar_meses(d, d_id)
                dl["AÑO"] = y

                dl = apply_filter(dl, "DEPARTAMENTO", f_dept_op)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                dl = apply_filter(dl, "OBJETIVO", f_objetivo_op)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        comp_dept_long = pd.concat(comp_dept, ignore_index=True) if comp_dept else pd.DataFrame()

        if not comp_obj_long.empty:
            st.markdown("### 🎯 Objetivos — % por color (VERDE/AMARILLO/ROJO/MORADO)")
            obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            obj_mix["%"] = obj_mix["conteo"] / obj_mix.groupby("AÑO")["conteo"].transform("sum") * 100

            fig = px.bar(
                obj_mix, x="AÑO", y="%", color="Estado",
                barmode="group", color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}, text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=640, title="Comparativo Objetivos — % por color"), use_container_width=True)
        else:
            st.info("Sin datos comparativos de objetivos con filtros actuales.")

        if not comp_dept_long.empty:
            st.markdown("### 🏢 Operativo — % por color (VERDE/AMARILLO/ROJO/MORADO)")
            dep_mix = comp_dept_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            dep_mix["%"] = dep_mix["conteo"] / dep_mix.groupby("AÑO")["conteo"].transform("sum") * 100

            fig = px.bar(
                dep_mix, x="AÑO", y="%", color="Estado",
                barmode="group", color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}, text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=640, title="Comparativo Operativo — % por color"), use_container_width=True)

            st.markdown("### ✅ Operativo — % Realizada vs No realizada (por año)")
            if "¿Realizada?" in comp_dept_long.columns:
                ex = comp_dept_long.groupby(["AÑO","¿Realizada?"]).size().reset_index(name="conteo")
                ex["%"] = ex["conteo"] / ex.groupby("AÑO")["conteo"].transform("sum") * 100
                fig = px.bar(ex, x="AÑO", y="%", color="¿Realizada?", barmode="group", text="%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=600, title="Comparativo Ejecución (Realizada vs No realizada)"), use_container_width=True)

            st.markdown("### 📈 Tendencia mensual comparativa (promedio %)")
            left, right = st.columns(2)
            with left:
                t = comp_obj_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index() if not comp_obj_long.empty else pd.DataFrame()
                if not t.empty:
                    t["cumplimiento_%"] = t["valor"] * 100
                    fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(style_plotly(fig, height=620, title="Objetivos — tendencia mensual promedio"), use_container_width=True)
                else:
                    st.info("Sin datos de objetivos para tendencia comparativa.")

            with right:
                t = comp_dept_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                if not t.empty:
                    t["cumplimiento_%"] = t["valor"] * 100
                    fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(style_plotly(fig, height=620, title="Operativo — tendencia mensual promedio"), use_container_width=True)
                else:
                    st.info("Sin datos operativos para tendencia comparativa.")
        else:
            st.info("No hay hojas AREAS suficientes en los años comparados (o los filtros dejaron los datos vacíos).")

# =====================================================
# TAB 4: ALERTAS
# =====================================================
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    # Objetivos
    if not obj_resumen.empty:
        crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
        for _, r in crit_obj.iterrows():
            nivel = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Tipo": "Objetivo",
                "Nombre": r.get("Objetivo",""),
                "Estado": r["estado_ejecutivo"],
                "Cumplimiento %": float(r.get("cumplimiento_%", 0))
            })

    # Operativo
    if has_operativo_year and not dept_res.empty:
        bad_dept = dept_res[dept_res["cumplimiento_%"] < 60].copy()
        for _, r in bad_dept.iterrows():
            nivel = "CRÍTICA" if float(r["cumplimiento_%"]) < 40 else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Tipo": "Departamento (operativo)",
                "Nombre": r["DEPARTAMENTO"],
                "Estado": "BAJO CUMPLIMIENTO",
                "Cumplimiento %": float(r["cumplimiento_%"])
            })

    alerts_df = pd.DataFrame(alert_rows)

    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        alerts_df["OrdenNivel"] = alerts_df["Nivel"].map({"CRÍTICA":0,"NORMAL":1}).fillna(2)
        alerts_df = alerts_df.sort_values(["OrdenNivel","Tipo","Cumplimiento %"], ascending=[True, True, True]).drop(columns=["OrdenNivel"])
        alerts_df["Cumplimiento %"] = alerts_df["Cumplimiento %"].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}")

        def semaforo(row):
            bg = "#ffe1e1" if row["Nivel"] == "CRÍTICA" else "#fff3cd"
            fg = "#111111"
            return [f"background-color: {bg}; color: {fg};"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

# =====================================================
# TAB 5: EXPORTAR
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    fig_estado_exec = None
    fig_rank_dept = None

    if not obj_resumen.empty:
        fig_estado_exec = px.pie(
            obj_resumen, names="estado_ejecutivo", hole=0.55,
            color="estado_ejecutivo", color_discrete_map=COLOR_EJEC,
            title=f"{year_data} — Estado Ejecutivo (Objetivos)"
        )
        fig_estado_exec = style_plotly(fig_estado_exec, height=520)

    if has_operativo_year and not dept_res.empty:
        fig_rank_dept = px.bar(
            dept_res.sort_values("cumplimiento_%").head(20),
            x="cumplimiento_%",
            y="DEPARTAMENTO",
            orientation="h",
            title=f"{year_data} — Ranking crítico Operativo (Top 20 deptos)"
        )
        fig_rank_dept = style_plotly(fig_rank_dept, height=520)

    def build_report_html():
        k_obj = len(obj_resumen)
        k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0
        k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0
        k_crit = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0
        k_avg = float(safe_mean(obj_resumen["cumplimiento_%"])) if not obj_resumen.empty else 0
        k_op = float(gauge_op_val)
        k_global = float(gauge_global_val)

        rep_alert_html = alerts_df.to_html(index=False) if ('alerts_df' in locals() or 'alerts_df' in globals()) and not alerts_df.empty else "<p>Sin alertas.</p>"

        charts_html = ""
        if fig_estado_exec is not None:
            charts_html += fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn")
        if fig_rank_dept is not None:
            charts_html += fig_rank_dept.to_html(full_html=False, include_plotlyjs=False)

        return f"""
<html>
<head>
<meta charset="utf-8"/>
<title>Reporte Estratégico</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; background:#fff; }}
  .top {{ color:#555; margin-bottom:12px; }}
  .kpis {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .kpi {{ border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; min-width:170px; background:#fff; }}
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
  <div class="kpi"><b>Objetivos</b><br>{k_obj}</div>
  <div class="kpi"><b>Cumplidos</b><br>{k_ok}</div>
  <div class="kpi"><b>En Riesgo</b><br>{k_riesgo}</div>
  <div class="kpi"><b>Críticos/No Subido</b><br>{k_crit}</div>
  <div class="kpi"><b>Cumplimiento Estratégico</b><br>{k_avg:.1f}%</div>
  <div class="kpi"><b>Cumplimiento Operativo</b><br>{k_op:.1f}%</div>
  <div class="kpi"><b>Cumplimiento Global (Unión)</b><br>{k_global:.1f}%</div>
</div>

<h2>Gráficas</h2>
{charts_html}

<h2>Alertas</h2>
{rep_alert_html}

<h2>Tabla: Objetivos (resumen)</h2>
{obj_resumen.head(200).to_html(index=False) if not obj_resumen.empty else "<p>Sin datos.</p>"}

<h2>Tabla: Operativo (departamentos resumen)</h2>
{dept_res.head(200).to_html(index=False) if not dept_res.empty else "<p>Sin datos o no existe hoja AREAS.</p>"}

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

    with st.expander("Objetivos — Resumen"):
        st.dataframe(obj_resumen, use_container_width=True)

    with st.expander("Objetivos — Long"):
        st.dataframe(obj_long, use_container_width=True)

    with st.expander("Operativo — Departamentos resumen"):
        if has_operativo_year:
            st.dataframe(dept_res, use_container_width=True)
        else:
            st.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")

    with st.expander("Operativo — Long"):
        if has_operativo_year:
            st.dataframe(dept_long, use_container_width=True)
        else:
            st.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")

st.caption("Fuente: Google Sheets · Dashboard Estratégico")


