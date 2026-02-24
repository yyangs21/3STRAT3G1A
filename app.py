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
# ESTILO (EJECUTIVO CLARO + TEXTOS VISIBLES)
# =====================================================
st.markdown(
    """
<style>
.stApp { background: #f3f4f6; color: #111111; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
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

/* Inputs generales */
.stSelectbox label, .stMultiSelect label, .stTextInput label {
    color: #111111 !important;
    font-weight: 600 !important;
}

/* Select / MultiSelect: fondo y texto visibles siempre */
[data-baseweb="select"] > div,
[data-baseweb="select"] div[data-value],
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[data-baseweb="option"],
[data-baseweb="select"] input {
    background: #ffffff !important;
    color: #111111 !important;
    border-color: #d1d5db !important;
}

/* Dropdown abierto — fondo de lista */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"] > ul {
    background: #ffffff !important;
    color: #111111 !important;
}

/* Items de la lista al hacer hover */
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {
    background: #f3f4f6 !important;
    color: #111111 !important;
}

/* Tags seleccionados en multiselect */
[data-baseweb="tag"] {
    background: #e5e7eb !important;
    color: #111111 !important;
}
[data-baseweb="tag"] span {
    color: #111111 !important;
}

/* Texto dentro del input del select (valor seleccionado) */
[data-baseweb="select"] span,
[data-baseweb="select"] div span,
div[data-testid="stSelectbox"] span {
    color: #111111 !important;
}

/* Forzar texto negro en CUALQUIER elemento de lista dentro de popover */
[role="listbox"] *,
[role="option"] *,
[role="listbox"],
[role="option"] {
    color: #111111 !important;
    background-color: transparent;
}
[role="option"]:hover {
    background-color: #f3f4f6 !important;
}

/* Títulos y texto */
h1, h2, h3, h4, h5, h6, p, label, span, div {
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
div[data-testid="stMetricLabel"] > div {
    color: #111111 !important;
    font-weight: 650;
}
div[data-testid="stMetricValue"] {
    color: #111111 !important;
}
div[data-testid="stMetricDelta"] {
    color: #111111 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Tabs */
button[role="tab"] {
    color: #111111 !important;
    font-weight: 600 !important;
}
button[role="tab"][aria-selected="true"] {
    background: #ffffff !important;
    border-radius: 10px 10px 0 0 !important;
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
    font-weight: 600 !important;
}

/* Alerts */
[data-testid="stAlert"] * {
    color: #111111 !important;
}
hr { border: none; border-top: 1px solid #e5e7eb; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# GOOGLE SHEETS AUTH
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

try:
    service_account_info = dict(st.secrets["gcp_service_account"])
    service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
    CREDS = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(CREDS)
except Exception as e:
    st.error(f"❌ Error al conectar con Google Sheets: {e}")
    st.stop()

SHEET_NAME = "DATAESTRATEGIA"

# =====================================================
# CONFIG DATA
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
MESES_ORDEN = {m: i+1 for i, m in enumerate(MESES)}

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
def style_plotly(fig, height=520, title=None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=58 if title else 18, b=18),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
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
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#d1d5db",
            title_font=dict(color="#111111")
        )
        fig.update_yaxes(
            tickfont=dict(color="#111111"),
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#d1d5db",
            title_font=dict(color="#111111")
        )
    return fig

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\n", " ", regex=False).str.strip()

def normalize_estado_series(s: pd.Series) -> pd.Series:
    x = s.replace("", np.nan).dropna().astype(str).str.strip().str.upper()
    x = x.str.replace("  ", " ", regex=False)
    return x

def normalize_realizada(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"PLANIFICADA": "NO REALIZADA", "PLANIFICADO": "NO REALIZADA"})
    x = x.replace({"REALIZADA": "REALIZADA"})
    return x

def normalizar_meses(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    if not meses_presentes:
        return pd.DataFrame(columns=id_cols + ["Mes","Estado","valor","MesNum"])

    long = (
        df.melt(
            id_vars=id_cols,
            value_vars=meses_presentes,
            var_name="Mes",
            value_name="Estado"
        )
        .dropna(subset=["Estado"])
        .copy()
    )
    long["Estado"] = normalize_estado_series(long["Estado"])
    long = long.dropna(subset=["Estado"]).copy()
    long["valor"] = long["Estado"].map(ESTADO_MAP).fillna(0.0)
    long["MesNum"] = long["Mes"].map(MESES_ORDEN)
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

def safe_mean_percent(series):
    if series is None or len(series) == 0:
        return np.nan
    try:
        return float(pd.to_numeric(series, errors="coerce").mean())
    except Exception:
        return np.nan

def build_gauge(value, title, delta_ref=None):
    kwargs = dict(
        mode="gauge+number+delta" if delta_ref is not None else "gauge+number",
        value=float(0 if pd.isna(value) else value),
        title={"text": title},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#111111"},
            "steps":[
                {"range":[0,60],"color":"#e74c3c"},
                {"range":[60,90],"color":"#f1c40f"},
                {"range":[90,100],"color":"#00a65a"}
            ]
        }
    )
    if delta_ref is not None:
        kwargs["delta"] = {"reference": delta_ref}
    return go.Figure(go.Indicator(**kwargs))

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_sheet_titles():
    sh = client.open(SHEET_NAME)
    return [ws.title.strip() for ws in sh.worksheets()]

@st.cache_data(ttl=300, show_spinner=False)
def get_years_available():
    titles = get_sheet_titles()
    years = sorted([int(t) for t in titles if t.isdigit()])
    return years, titles

@st.cache_data(ttl=300, show_spinner=False)
def load_year(year: int):
    sh = client.open(SHEET_NAME)

    # Hoja Informes Plan (principal / obligatoria)
    df_inf = pd.DataFrame(sh.worksheet(str(year)).get_all_records())

    # Hoja Tareas Plan (opcional)
    df_tar = None
    area_sheet_name = f"{year} AREAS"
    try:
        df_tar = pd.DataFrame(sh.worksheet(area_sheet_name).get_all_records())
    except Exception:
        df_tar = None

    # --- Informes Plan: normalización ---
    df_inf.columns = normalize_text_series(df_inf.columns.to_series())
    for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
        if c in df_inf.columns:
            df_inf[c] = normalize_text_series(df_inf[c])

    # --- Tareas Plan: normalización ---
    if df_tar is not None:
        df_tar.columns = normalize_text_series(df_tar.columns.to_series())

        # Normaliza Dic duplicado
        if "Diciembre" in df_tar.columns:
            if "Dic" in df_tar.columns:
                df_tar["Dic"] = df_tar["Dic"].fillna(df_tar["Diciembre"])
            else:
                df_tar.rename(columns={"Diciembre": "Dic"}, inplace=True)

        # Normaliza columnas de nombre variable
        if "PUESTO" in df_tar.columns and "PUESTO RESPONSABLE" not in df_tar.columns:
            df_tar.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)
        if "DEPARTAMENTO" not in df_tar.columns and "Área" in df_tar.columns:
            df_tar.rename(columns={"Área": "DEPARTAMENTO"}, inplace=True)
        if "Realizada?" in df_tar.columns and "¿Realizada?" not in df_tar.columns:
            df_tar.rename(columns={"Realizada?": "¿Realizada?"}, inplace=True)

        for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?"]:
            if c in df_tar.columns:
                df_tar[c] = normalize_text_series(df_tar[c])

        if "¿Realizada?" in df_tar.columns:
            df_tar["¿Realizada?"] = normalize_realizada(df_tar["¿Realizada?"])

    return df_inf, df_tar


# =====================================================
# HELPERS — obtener valores únicos de AMBAS hojas
# =====================================================
def get_unique_values(df_inf, df_tar, col_inf, col_tar=None):
    """
    Devuelve valores únicos combinados de la columna en Informes Plan (col_inf)
    y en Tareas Plan (col_tar). Si col_tar es None usa col_inf en mayúsculas.
    """
    col_tar = col_tar or col_inf.upper()
    vals = set()
    if df_inf is not None and col_inf in df_inf.columns:
        vals.update(df_inf[col_inf].dropna().unique())
    if df_tar is not None and col_tar in df_tar.columns:
        vals.update(df_tar[col_tar].dropna().unique())
    return sorted([v for v in vals if str(v).strip() != ""])


# =====================================================
# SIDEBAR
# =====================================================
years, sheet_titles = get_years_available()
if not years:
    st.error("No encontré hojas tipo '2023', '2024', '2025' dentro de tu Google Sheets.")
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

df_inf, df_tar = load_year(year_data)
has_tareas_year = df_tar is not None and not df_tar.empty

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (aplican a Informes Plan y Tareas Plan)")

# -------------------------------------------------------
# Filtros UNIFICADOS — buscan en ambas hojas
# El filtro Departamento filtra en df_inf["Departamento"]
# Y también en df_tar["DEPARTAMENTO"] si existe.
# -------------------------------------------------------

f_tipo_plan = st.sidebar.multiselect(
    "Tipo (POA / PEC)",
    get_unique_values(df_inf, df_tar, "Tipo", "TIPO")
)
f_persp = st.sidebar.multiselect(
    "Perspectiva",
    get_unique_values(df_inf, df_tar, "Perspectiva", "PERSPECTIVA")
)
f_eje = st.sidebar.multiselect(
    "Eje",
    get_unique_values(df_inf, df_tar, "Eje", "EJE")
)
# Departamento: unifica df_inf["Departamento"] + df_tar["DEPARTAMENTO"]
f_depto = st.sidebar.multiselect(
    "Departamento",
    get_unique_values(df_inf, df_tar, "Departamento", "DEPARTAMENTO")
)
# Objetivo: unifica df_inf["Objetivo"] + df_tar["OBJETIVO"]
f_obje = st.sidebar.multiselect(
    "Objetivo",
    get_unique_values(df_inf, df_tar, "Objetivo", "OBJETIVO")
)

# Filtros exclusivos de Tareas Plan
if has_tareas_year:
    f_puesto = st.sidebar.multiselect(
        "Puesto Responsable",
        sorted(df_tar["PUESTO RESPONSABLE"].dropna().unique()) if "PUESTO RESPONSABLE" in df_tar.columns else []
    )
    f_realizada = st.sidebar.multiselect(
        "Ejecución (Realizada / No realizada)",
        sorted(df_tar["¿Realizada?"].dropna().unique()) if "¿Realizada?" in df_tar.columns else []
    )
else:
    st.sidebar.info(f"El año {year_data} no tiene hoja '{year_data} AREAS' (Tareas Plan).")
    f_puesto, f_realizada = [], []

st.sidebar.caption("✅ Si NO seleccionas filtros, se muestra TODO por default.")

# =====================================================
# PROCESAMIENTO Informes Plan
# =====================================================
inf_id_cols = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
inf_id_cols = [c for c in inf_id_cols if c in df_inf.columns]

inf_long = normalizar_meses(df_inf, inf_id_cols)

# Filtros unificados aplicados a Informes Plan
inf_long = apply_filter(inf_long, "Tipo", f_tipo_plan)
inf_long = apply_filter(inf_long, "Perspectiva", f_persp)
inf_long = apply_filter(inf_long, "Eje", f_eje)
inf_long = apply_filter(inf_long, "Departamento", f_depto)
inf_long = apply_filter(inf_long, "Objetivo", f_obje)

grp_cols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in inf_long.columns]

if inf_long.empty:
    obj_resumen = pd.DataFrame(columns=grp_cols + ["score_total","verdes","amarillos","rojos","morados","meses_reportados","meses_esperados","cumplimiento_%","estado_ejecutivo"])
else:
    obj_resumen = inf_long.groupby(grp_cols, as_index=False).agg(
        score_total=("valor","sum"),
        verdes=("Estado", lambda x: (x=="VERDE").sum()),
        amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum()),
        meses_reportados=("Mes","count")
    )

    if "Frecuencia Medición" in obj_resumen.columns:
        obj_resumen["Frecuencia Medición"] = obj_resumen["Frecuencia Medición"].astype(str).str.strip().str.upper()
        freq_no_mapeadas = obj_resumen[~obj_resumen["Frecuencia Medición"].isin(FRECUENCIA_MAP.keys())]["Frecuencia Medición"].unique()
        if len(freq_no_mapeadas) > 0:
            st.sidebar.warning(f"⚠️ Frecuencias no reconocidas (se asumen MENSUAL): {list(freq_no_mapeadas)}")
        obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(FRECUENCIA_MAP).fillna(12)
    else:
        obj_resumen["meses_esperados"] = 12

    obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1) * 100
    obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec, axis=1)

# =====================================================
# PROCESAMIENTO Tareas Plan — OPCIONAL
# =====================================================
tar_long = pd.DataFrame()
dept_res = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento","tareas","rojos","amarillos","verdes","morados","cumplimiento_%"])
exec_res = None
dept_res_puesto = None

# Inicializar alerts_df vacío desde el principio
alerts_df = pd.DataFrame()

if has_tareas_year:
    tar_id_cols = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
    tar_id_cols = [c for c in tar_id_cols if c in df_tar.columns]

    tar_long = normalizar_meses(df_tar, tar_id_cols)

    # Filtros unificados aplicados a Tareas Plan (usando nombres en mayúsculas)
    tar_long = apply_filter(tar_long, "TIPO", f_tipo_plan)
    tar_long = apply_filter(tar_long, "PERSPECTIVA", f_persp)
    tar_long = apply_filter(tar_long, "EJE", f_eje)
    tar_long = apply_filter(tar_long, "DEPARTAMENTO", f_depto)     # mismo filtro que Informes Plan
    tar_long = apply_filter(tar_long, "OBJETIVO", f_obje)          # mismo filtro que Informes Plan
    tar_long = apply_filter(tar_long, "PUESTO RESPONSABLE", f_puesto)
    tar_long = apply_filter(tar_long, "¿Realizada?", f_realizada)

    if not tar_long.empty and "DEPARTAMENTO" in tar_long.columns:
        dept_res = tar_long.groupby("DEPARTAMENTO", as_index=False).agg(
            cumplimiento=("valor","mean"),
            tareas=("TAREA","nunique") if "TAREA" in tar_long.columns else ("Estado","count"),
            rojos=("Estado", lambda x: (x=="ROJO").sum()),
            amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
            verdes=("Estado", lambda x: (x=="VERDE").sum()),
            morados=("Estado", lambda x: (x=="MORADO").sum())
        )
        dept_res["cumplimiento_%"] = dept_res["cumplimiento"] * 100

        if "¿Realizada?" in tar_long.columns:
            exec_res = (
                tar_long.groupby(["DEPARTAMENTO","¿Realizada?"]).size()
                .reset_index(name="conteo")
            )
            exec_res["%"] = exec_res["conteo"] / exec_res.groupby("DEPARTAMENTO")["conteo"].transform("sum") * 100

        if "PUESTO RESPONSABLE" in tar_long.columns:
            dept_res_puesto = tar_long.groupby(["DEPARTAMENTO","PUESTO RESPONSABLE"], as_index=False).agg(
                cumplimiento=("valor","mean"),
                tareas=("TAREA","nunique") if "TAREA" in tar_long.columns else ("Estado","count")
            )
            dept_res_puesto["cumplimiento_%"] = dept_res_puesto["cumplimiento"] * 100

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "📌 Resumen",
    "🎯 Informes Plan",
    "🏢 Tareas Plan (Deptos)",
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

    k_obj_total = int(len(obj_resumen))
    k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0
    k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0
    k_crit_ns = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0
    k_prom_obj = safe_mean_percent(obj_resumen["cumplimiento_%"]) if "cumplimiento_%" in obj_resumen.columns else np.nan
    k_prom_op = safe_mean_percent(tar_long["valor"] * 100) if not tar_long.empty else np.nan

    if has_tareas_year and not pd.isna(k_prom_op):
        k_global = float(np.nanmean([k_prom_obj, k_prom_op]))
    else:
        k_global = float(k_prom_obj) if not pd.isna(k_prom_obj) else np.nan

    c1.metric("Objetivos", k_obj_total)
    c2.metric("Cumplidos", k_ok)
    c3.metric("En Riesgo", k_riesgo)
    c4.metric("Críticos / No Subido", k_crit_ns)
    c5.metric("Cumpl. Informes Plan", f"{k_prom_obj:.1f}%" if not pd.isna(k_prom_obj) else "—")
    c6.metric("Cumplimiento Global", f"{k_global:.1f}%" if not pd.isna(k_global) else "—")

    g1, g2, g3 = st.columns(3)

    val_obj = float(k_prom_obj) if not pd.isna(k_prom_obj) else 0.0
    fig_g1 = build_gauge(val_obj, f"{year_data} — Cumplimiento Informes Plan", delta_ref=90)
    g1.plotly_chart(style_plotly(fig_g1, height=420), use_container_width=True)

    if has_tareas_year:
        val_dept = float(k_prom_op) if not pd.isna(k_prom_op) else 0.0
        fig_g2 = build_gauge(val_dept, f"{year_data} — Cumplimiento Tareas Plan", delta_ref=90)
        g2.plotly_chart(style_plotly(fig_g2, height=420), use_container_width=True)
    else:
        val_dept = np.nan
        g2.info(f"ℹ️ El año {year_data} no tiene hoja '{year_data} AREAS' (Tareas Plan)")

    if has_tareas_year and not pd.isna(val_dept):
        val_global = float(np.nanmean([val_obj, val_dept]))
        subt = "Promedio: Informes Plan + Tareas Plan"
    else:
        val_global = float(val_obj)
        subt = "Solo Informes Plan (sin Tareas Plan)"

    fig_g3 = build_gauge(val_global, f"{year_data} — Cumplimiento Global Consolidado", delta_ref=90)
    fig_g3.add_annotation(
        text=subt,
        x=0.5, y=0.02, xref="paper", yref="paper",
        showarrow=False, font=dict(size=11, color="#444444")
    )
    g3.plotly_chart(style_plotly(fig_g3, height=420), use_container_width=True)

    left, right = st.columns(2)

    with left:
        counts = obj_resumen["estado_ejecutivo"].value_counts().reindex(ESTADO_EJEC_ORDEN).fillna(0).reset_index()
        counts.columns = ["Estado Ejecutivo", "Cantidad"]
        fig = px.bar(
            counts,
            x="Estado Ejecutivo",
            y="Cantidad",
            color="Estado Ejecutivo",
            color_discrete_map=COLOR_EJEC,
            text="Cantidad"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(
            style_plotly(fig, height=640, title="Distribución de Estados Ejecutivos (Informes Plan)"),
            use_container_width=True
        )

    with right:
        if not inf_long.empty:
            tr = inf_long.groupby(["Mes","MesNum"], as_index=False)["valor"].mean()
            tr = tr.sort_values("MesNum")
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(
                style_plotly(fig, height=640, title="Tendencia Mensual — Cumplimiento Promedio (Informes Plan)"),
                use_container_width=True
            )
        else:
            st.info("Sin datos de Informes Plan para mostrar tendencia.")

# =====================================================
# TAB 1: Informes Plan
# =====================================================
with tabs[1]:
    st.subheader("🎯 Informes Plan — Análisis Avanzado")

    if obj_resumen.empty:
        st.warning("No hay datos de Informes Plan con los filtros actuales.")
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
            st.plotly_chart(
                style_plotly(fig, height=760, title="Top 15 Objetivos más críticos (peor cumplimiento)"),
                use_container_width=True
            )

        with colB:
            fig = px.pie(
                obj_resumen,
                names="estado_ejecutivo",
                hole=0.55,
                color="estado_ejecutivo",
                color_discrete_map=COLOR_EJEC
            )
            st.plotly_chart(
                style_plotly(fig, height=760, title="Mix de Estado Ejecutivo (Informes Plan)"),
                use_container_width=True
            )

        r1, r2 = st.columns(2)
        with r1:
            if "Departamento" in obj_resumen.columns:
                dep = (
                    obj_resumen.groupby("Departamento", as_index=False)["cumplimiento_%"]
                    .mean()
                    .sort_values("cumplimiento_%")
                )
                fig = px.bar(
                    dep,
                    x="cumplimiento_%",
                    y="Departamento",
                    orientation="h",
                    color="cumplimiento_%",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(
                    style_plotly(fig, height=640, title="Cumplimiento Promedio por Departamento (Informes Plan)"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Departamento en Informes Plan para este año.")

        with r2:
            if "Perspectiva" in obj_resumen.columns:
                p = (
                    obj_resumen.groupby("Perspectiva", as_index=False)["cumplimiento_%"]
                    .mean()
                    .sort_values("cumplimiento_%")
                )
                fig = px.bar(
                    p,
                    x="cumplimiento_%",
                    y="Perspectiva",
                    orientation="h",
                    color="cumplimiento_%",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(
                    style_plotly(fig, height=640, title="Cumplimiento Promedio por Perspectiva"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Perspectiva en Informes Plan para este año.")

        r3, r4 = st.columns(2)
        with r3:
            if "Tipo" in obj_resumen.columns:
                t = obj_resumen.groupby("Tipo", as_index=False)["cumplimiento_%"].mean()
                fig = px.bar(
                    t,
                    x="Tipo",
                    y="cumplimiento_%",
                    text="cumplimiento_%",
                    color="Tipo"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(
                    style_plotly(fig, height=560, title="Cumplimiento por Tipo (POA / PEC)"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Tipo para este año.")

        with r4:
            dv = obj_resumen[["Objetivo","cumplimiento_%","estado_ejecutivo"]].copy()
            dv["desviación_%"] = dv["cumplimiento_%"] - 100
            dv = dv.sort_values("desviación_%").head(15)
            fig = px.bar(
                dv,
                x="desviación_%",
                y="Objetivo",
                orientation="h",
                color="estado_ejecutivo",
                color_discrete_map=COLOR_EJEC,
                text="desviación_%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(
                style_plotly(fig, height=560, title="Desviación vs Meta (100%) — Top 15"),
                use_container_width=True
            )

        st.markdown("#### 🌡️ Heatmap — Objetivo vs Mes (Top 25 más críticos)")
        if not inf_long.empty and "Objetivo" in inf_long.columns:
            hm_base = inf_long.copy()
            avg_obj = hm_base.groupby("Objetivo")["valor"].mean().sort_values().head(25).index.tolist()
            hm = hm_base[hm_base["Objetivo"].isin(avg_obj)].pivot_table(
                index="Objetivo",
                columns="Mes",
                values="valor",
                fill_value=0
            )
            hm = hm.reindex(columns=[m for m in MESES if m in hm.columns])
            fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
            st.plotly_chart(
                style_plotly(fig, height=760, title="Heatmap (Top 25 objetivos más críticos)"),
                use_container_width=True
            )
        else:
            st.info("No hay datos para heatmap de Informes Plan.")

# =====================================================
# TAB 2: Tareas Plan (DEPARTAMENTO)
# =====================================================
with tabs[2]:
    st.subheader("🏢 Tareas Plan — Control por Departamento")

    if not has_tareas_year:
        st.info(f"El año {year_data} no tiene hoja '{year_data} AREAS' (Tareas Plan).")
    elif tar_long.empty or dept_res.empty:
        st.warning("No hay datos de Tareas Plan con los filtros actuales.")
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
            st.plotly_chart(
                style_plotly(fig, height=760, title="Ranking de Departamentos Tareas Plan (Top 20)"),
                use_container_width=True
            )

        with right:
            sc = dept_res.copy()
            fig = px.scatter(
                sc,
                x="tareas",
                y="cumplimiento_%",
                size="tareas",
                hover_name="DEPARTAMENTO",
                color="cumplimiento_%",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(
                style_plotly(fig, height=760, title="Cumplimiento vs Carga (# tareas)"),
                use_container_width=True
            )

        r1, r2 = st.columns(2)
        with r1:
            mix_op = tar_long["Estado"].value_counts().reindex(["VERDE","AMARILLO","ROJO","MORADO"]).fillna(0).reset_index()
            mix_op.columns = ["Estado", "Cantidad"]
            fig = px.bar(
                mix_op,
                x="Estado",
                y="Cantidad",
                color="Estado",
                color_discrete_map=COLOR_ESTADO,
                text="Cantidad"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(
                style_plotly(fig, height=560, title="Distribución de estados (Tareas Plan)"),
                use_container_width=True
            )

        with r2:
            tr = tar_long.groupby(["Mes","MesNum"], as_index=False)["valor"].mean()
            tr = tr.sort_values("MesNum")
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(
                style_plotly(fig, height=560, title="Tendencia mensual — Cumplimiento promedio (Tareas Plan)"),
                use_container_width=True
            )

        if exec_res is not None and not exec_res.empty:
            st.markdown("#### ✅ Ejecución (Realizada vs No realizada) por Departamento — Top 15 con menor % realizada")
            tmp = exec_res.pivot_table(index="DEPARTAMENTO", columns="¿Realizada?", values="%", fill_value=0).reset_index()
            if "REALIZADA" in tmp.columns:
                tmp = tmp.sort_values("REALIZADA").head(15)
                fig = px.bar(tmp, x="REALIZADA", y="DEPARTAMENTO", orientation="h", text="REALIZADA")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(
                    style_plotly(fig, height=640, title="Top 15 deptos con menor % Realizada"),
                    use_container_width=True
                )
            else:
                st.info("No se encontró categoría REALIZADA en ¿Realizada?")

        st.markdown("#### 🌡️ Heatmap Tareas Plan — Departamento vs Mes (Top 25 más críticos)")
        hm_base = tar_long.copy()
        avg_d = hm_base.groupby("DEPARTAMENTO")["valor"].mean().sort_values().head(25).index.tolist()
        hm = hm_base[hm_base["DEPARTAMENTO"].isin(avg_d)].pivot_table(
            index="DEPARTAMENTO",
            columns="Mes",
            values="valor",
            fill_value=0
        )
        hm = hm.reindex(columns=[m for m in MESES if m in hm.columns])
        fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
        st.plotly_chart(
            style_plotly(fig, height=760, title="Heatmap Tareas Plan (Top 25 deptos más críticos)"),
            use_container_width=True
        )

# =====================================================
# TAB 3: COMPARATIVO
# =====================================================
with tabs[3]:
    st.subheader("📊 Comparativo (Informes Plan y Tareas Plan por Departamento)")

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el sidebar para comparar.")
    else:
        comp_obj = []
        comp_dept = []

        for y in compare_years:
            o, d = load_year(y)

            # Informes Plan
            o_id = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            o_id = [c for c in o_id if c in o.columns]
            ol = normalizar_meses(o, o_id)
            ol["AÑO"] = y

            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)
            ol = apply_filter(ol, "Objetivo", f_obje)   # ✅ CORREGIDO: era 'o1'
            comp_obj.append(ol)

            # Tareas Plan (opcional por año)
            if d is not None and not d.empty:
                if "DEPARTAMENTO" not in d.columns and "Área" in d.columns:
                    d = d.copy()
                    d.rename(columns={"Área":"DEPARTAMENTO"}, inplace=True)
                if "Realizada?" in d.columns and "¿Realizada?" not in d.columns:
                    d = d.copy()
                    d.rename(columns={"Realizada?":"¿Realizada?"}, inplace=True)
                if "¿Realizada?" in d.columns:
                    d = d.copy()
                    d["¿Realizada?"] = normalize_realizada(d["¿Realizada?"])

                d_id = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
                d_id = [c for c in d_id if c in d.columns]
                dl = normalizar_meses(d, d_id)
                dl["AÑO"] = y

                # Filtros unificados también en comparativo
                dl = apply_filter(dl, "TIPO", f_tipo_plan)
                dl = apply_filter(dl, "PERSPECTIVA", f_persp)
                dl = apply_filter(dl, "EJE", f_eje)
                dl = apply_filter(dl, "DEPARTAMENTO", f_depto)
                dl = apply_filter(dl, "OBJETIVO", f_obje)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        comp_dept_long = pd.concat(comp_dept, ignore_index=True) if comp_dept else pd.DataFrame()

        # ── GAUGE CONSOLIDADO COMPARATIVO ──────────────────────────────────
        st.markdown("### 🎯 Medidor Consolidado — Promedio de años comparados")

        gauge_data = []
        for y in compare_years:
            prom_inf = float(comp_obj_long[comp_obj_long["AÑO"]==y]["valor"].mean() * 100) \
                if not comp_obj_long.empty and y in comp_obj_long["AÑO"].values else np.nan
            prom_tar = float(comp_dept_long[comp_dept_long["AÑO"]==y]["valor"].mean() * 100) \
                if not comp_dept_long.empty and y in comp_dept_long["AÑO"].values else np.nan
            prom_global = float(np.nanmean([v for v in [prom_inf, prom_tar] if not np.isnan(v)])) \
                if any(not np.isnan(v) for v in [prom_inf, prom_tar]) else np.nan
            gauge_data.append({"año": y, "inf": prom_inf, "tar": prom_tar, "global": prom_global})

        # Promedio CONSOLIDADO de todos los años
        vals_inf    = [d["inf"]    for d in gauge_data if not np.isnan(d["inf"])]
        vals_tar    = [d["tar"]    for d in gauge_data if not np.isnan(d["tar"])]
        vals_global = [d["global"] for d in gauge_data if not np.isnan(d["global"])]

        prom_total_inf    = float(np.mean(vals_inf))    if vals_inf    else 0.0
        prom_total_tar    = float(np.mean(vals_tar))    if vals_tar    else 0.0
        prom_total_global = float(np.mean(vals_global)) if vals_global else 0.0

        years_label = " + ".join(str(y) for y in sorted(compare_years))

        # Una columna por año + 1 columna consolidada al final
        gauge_cols = st.columns(len(compare_years) + 1)

        for i, gd in enumerate(gauge_data):
            fig_gy = build_gauge(
                gd["global"],
                f"{gd['año']} — Global",
                delta_ref=90
            )
            fig_gy.add_annotation(
                text=(
                    f"Informes Plan: {gd['inf']:.1f}%  |  Tareas Plan: {gd['tar']:.1f}%"
                    if not np.isnan(gd["tar"])
                    else f"Solo Informes Plan: {gd['inf']:.1f}%"
                ),
                x=0.5, y=0.02, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="#555555")
            )
            gauge_cols[i].plotly_chart(style_plotly(fig_gy, height=400), use_container_width=True)

        # Gauge consolidado final
        fig_consol = build_gauge(
            prom_total_global,
            f"Consolidado ({years_label})",
            delta_ref=90
        )
        fig_consol.add_annotation(
            text=(
                f"Informes Plan: {prom_total_inf:.1f}%  |  Tareas Plan: {prom_total_tar:.1f}%"
                if vals_tar
                else f"Solo Informes Plan: {prom_total_inf:.1f}%"
            ),
            x=0.5, y=0.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#555555")
        )
        gauge_cols[-1].plotly_chart(style_plotly(fig_consol, height=400), use_container_width=True)

        # KPIs resumen comparativo
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Promedio Informes Plan ({years_label})", f"{prom_total_inf:.1f}%" if vals_inf else "—")
        k2.metric(f"Promedio Tareas Plan ({years_label})",   f"{prom_total_tar:.1f}%"  if vals_tar  else "—")
        k3.metric(f"Promedio Global Consolidado",            f"{prom_total_global:.1f}%" if vals_global else "—")

        st.divider()
        # ── FIN GAUGE CONSOLIDADO ───────────────────────────────────────────

        st.markdown("### 🎯 Informes Plan — % por color (VERDE/AMARILLO/ROJO/MORADO)")
        if not comp_obj_long.empty:
            obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            obj_mix["%"] = obj_mix["conteo"] / obj_mix.groupby("AÑO")["conteo"].transform("sum") * 100

            fig = px.bar(
                obj_mix,
                x="AÑO",
                y="%",
                color="Estado",
                barmode="group",
                color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=640, title="Comparativo Informes Plan — % por color"), use_container_width=True)
        else:
            st.info("Sin datos de Informes Plan para comparativo con los filtros actuales.")

        st.markdown("### 🏢 Tareas Plan — % por color (VERDE/AMARILLO/ROJO/MORADO)")
        if not comp_dept_long.empty:
            dep_mix = comp_dept_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            dep_mix["%"] = dep_mix["conteo"] / dep_mix.groupby("AÑO")["conteo"].transform("sum") * 100

            fig = px.bar(
                dep_mix,
                x="AÑO",
                y="%",
                color="Estado",
                barmode="group",
                color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=640, title="Comparativo Tareas Plan — % por color"), use_container_width=True)
        else:
            st.info("No hay hojas AREAS en los años comparados o quedaron sin datos por filtros.")

        st.markdown("### ✅ Tareas Plan — % Realizada vs No realizada (por año)")
        if not comp_dept_long.empty and "¿Realizada?" in comp_dept_long.columns:
            ex = comp_dept_long.groupby(["AÑO","¿Realizada?"]).size().reset_index(name="conteo")
            ex["%"] = ex["conteo"] / ex.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(ex, x="AÑO", y="%", color="¿Realizada?", barmode="group", text="%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=600, title="Comparativo Ejecución (Realizada vs No realizada)"), use_container_width=True)
        else:
            st.info("No hay datos de ejecución de Tareas Plan comparables.")

        st.markdown("### 📈 Tendencia mensual comparativa (promedio %)")
        left, right = st.columns(2)

        with left:
            if not comp_obj_long.empty:
                t = comp_obj_long.groupby(["AÑO","Mes","MesNum"], as_index=False)["valor"].mean()
                t["cumplimiento_%"] = t["valor"] * 100
                t = t.sort_values(["AÑO","MesNum"])
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=600, title="Informes Plan — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos de Informes Plan para tendencia comparativa.")

        with right:
            if not comp_dept_long.empty:
                t = comp_dept_long.groupby(["AÑO","Mes","MesNum"], as_index=False)["valor"].mean()
                t["cumplimiento_%"] = t["valor"] * 100
                t = t.sort_values(["AÑO","MesNum"])
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=600, title="Tareas Plan — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos de Tareas Plan para tendencia comparativa.")

        st.markdown("### 🧩 Cobertura comparativa (elementos que existen por año)")
        c1, c2 = st.columns(2)

        with c1:
            if not comp_obj_long.empty and "Objetivo" in comp_obj_long.columns:
                cov_obj = comp_obj_long.groupby("AÑO")["Objetivo"].nunique().reset_index(name="Objetivos únicos")
                fig = px.bar(cov_obj, x="AÑO", y="Objetivos únicos", text="Objetivos únicos")
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_plotly(fig, height=520, title="Cobertura de objetivos únicos por año"), use_container_width=True)

        with c2:
            if not comp_dept_long.empty and "DEPARTAMENTO" in comp_dept_long.columns:
                cov_dep = comp_dept_long.groupby("AÑO")["DEPARTAMENTO"].nunique().reset_index(name="Deptos únicos")
                fig = px.bar(cov_dep, x="AÑO", y="Deptos únicos", text="Deptos únicos")
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_plotly(fig, height=520, title="Cobertura de deptos Tareas Plan únicos por año"), use_container_width=True)

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
                "Tipo": "Informes Plan",
                "Nombre": r.get("Objetivo",""),
                "Estado": r["estado_ejecutivo"],
                "Cumplimiento %": float(r["cumplimiento_%"])
            })

    if not dept_res.empty:
        bad_dept = dept_res[dept_res["cumplimiento_%"] < 60].copy()
        for _, r in bad_dept.iterrows():
            nivel = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Tipo": "Tareas Plan (Departamento)",
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
            return [f"background-color: {bg}; color: #111111"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

        a1, a2, a3 = st.columns(3)
        a1.metric("Total Alertas", int(len(alerts_df)))
        a2.metric("Críticas", int((alerts_df["Nivel"]=="CRÍTICA").sum()))
        a3.metric("Normales", int((alerts_df["Nivel"]=="NORMAL").sum()))

# =====================================================
# TAB 5: EXPORTAR
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    if obj_resumen.empty:
        st.warning("No hay datos para exportar con los filtros actuales.")
    else:
        fig_estado_exec = px.pie(
            obj_resumen,
            names="estado_ejecutivo",
            hole=0.55,
            color="estado_ejecutivo",
            color_discrete_map=COLOR_EJEC,
            title=f"{year_data} — Estado Ejecutivo (Informes Plan)"
        )

        if not dept_res.empty:
            fig_rank_dept = px.bar(
                dept_res.sort_values("cumplimiento_%").head(20),
                x="cumplimiento_%",
                y="DEPARTAMENTO",
                orientation="h",
                title=f"{year_data} — Ranking crítico Tareas Plan (Top 20 deptos)"
            )
        else:
            fig_rank_dept = px.bar(
                pd.DataFrame({"Mensaje":["Sin datos de Tareas Plan"]}),
                x="Mensaje",
                y=[1],
                title=f"{year_data} — Tareas Plan"
            )

        fig_estado_exec = style_plotly(fig_estado_exec, height=520)
        fig_rank_dept = style_plotly(fig_rank_dept, height=520)

        def build_report_html():
            k_obj = len(obj_resumen)
            k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0
            k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0
            k_crit = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0
            k_avg_obj = safe_mean_percent(obj_resumen["cumplimiento_%"])
            k_avg_op = safe_mean_percent(tar_long["valor"] * 100) if not tar_long.empty else np.nan
            k_avg_global = float(np.nanmean([k_avg_obj, k_avg_op])) if not pd.isna(k_avg_op) else k_avg_obj

            rep_alert_html = alerts_df.to_html(index=False) if not alerts_df.empty else "<p>Sin alertas.</p>"

            dept_res_html = (
                dept_res.head(200).to_html(index=False)
                if not dept_res.empty
                else "<p>Este año no tiene hoja AREAS o no hay datos de Tareas Plan con los filtros.</p>"
            )

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
  <div class="kpi"><b>Objetivos</b><br>{k_obj}</div>
  <div class="kpi"><b>Cumplidos</b><br>{k_ok}</div>
  <div class="kpi"><b>En Riesgo</b><br>{k_riesgo}</div>
  <div class="kpi"><b>Críticos/No Subido</b><br>{k_crit}</div>
  <div class="kpi"><b>Cumpl. Informes Plan</b><br>{0 if pd.isna(k_avg_obj) else k_avg_obj:.1f}%</div>
  <div class="kpi"><b>Cumpl. Global</b><br>{0 if pd.isna(k_avg_global) else k_avg_global:.1f}%</div>
</div>

<h2>Gráficas</h2>
{fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn")}
{fig_rank_dept.to_html(full_html=False, include_plotlyjs=False)}

<h2>Alertas</h2>
{rep_alert_html}

<h2>Tabla: Informes Plan (resumen)</h2>
{obj_resumen.head(200).to_html(index=False)}

<h2>Tabla: Tareas Plan (departamentos resumen)</h2>
{dept_res_html}

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

    with st.expander("Informes Plan — Resumen"):
        st.dataframe(obj_resumen, use_container_width=True)

    with st.expander("Informes Plan — Long"):
        st.dataframe(inf_long, use_container_width=True)

    with st.expander("Tareas Plan — Departamentos resumen"):
        if not dept_res.empty:
            st.dataframe(dept_res, use_container_width=True)
        else:
            st.info("Sin datos de Tareas Plan para este año o con los filtros actuales.")

    with st.expander("Tareas Plan — Long"):
        if not tar_long.empty:
            st.dataframe(tar_long, use_container_width=True)
        else:
            st.info("Sin datos de Tareas Plan para este año o con los filtros actuales.")

st.caption("Fuente: Google Sheets · Dashboard Estratégico")

