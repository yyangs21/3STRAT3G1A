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
# ESTILO (EJECUTIVO CLARO + FIX FILTROS SIDEBAR)
# =====================================================
st.markdown(
    """
<style>
/* Base */
.stApp {
    background: #f3f4f6 !important;
    color: #111111 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* Labels / textos */
h1, h2, h3, h4, h5, p, span, label, div {
    color: #111111;
}

/* Widgets sidebar visibles (select/multiselect) */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #111111 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #eef2ff !important;
    color: #111111 !important;
    border: 1px solid #c7d2fe !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] * {
    color: #111111 !important;
}
section[data-testid="stSidebar"] [role="listbox"] {
    background: #ffffff !important;
}
section[data-testid="stSidebar"] [role="option"] {
    background: #ffffff !important;
    color: #111111 !important;
}
section[data-testid="stSidebar"] [role="option"][aria-selected="true"] {
    background: #e5e7eb !important;
    color: #111111 !important;
}

/* Métricas */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    padding: 14px 14px !important;
    box-shadow: 0 1px 0 rgba(0,0,0,0.03) !important;
}
div[data-testid="stMetricLabel"] > div { color: #111111 !important; font-weight: 650 !important; }
div[data-testid="stMetricValue"] { color: #111111 !important; }
div[data-testid="stMetricDelta"] { color: #111111 !important; }

/* Dataframes */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #111111 !important;
}

/* Expander */
details {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 4px 10px !important;
}

/* Plot containers */
.element-container:has(.js-plotly-plot) {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 8px;
    margin-bottom: 8px;
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
    "MENSUAL": 12, "BIMESTRAL": 6, "TRIMESTRAL": 4,
    "CUATRIMESTRAL": 3, "SEMESTRAL": 2, "ANUAL": 1
}

# =====================================================
# HELPERS
# =====================================================
def style_plotly(fig, height=560, title=None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=60 if title else 18, b=18),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111111", size=13),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(color="#111111")) if title else None,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color="#111111")
        ),
    )
    has_indicator = any(getattr(tr, "type", "") == "indicator" for tr in fig.data)
    if not has_indicator:
        # usar title (no titlefont) para evitar error de plotly
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

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\n", " ", regex=False).str.strip()

def normalize_estado_series(s: pd.Series) -> pd.Series:
    x = s.replace("", np.nan).dropna().astype(str).str.strip().str.upper()
    return x

def normalize_realizada(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"PLANIFICADA": "NO REALIZADA", "PLANIFICADO": "NO REALIZADA"})
    x = x.replace({"REALIZADA": "REALIZADA"})
    return x

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=id_cols + ["Mes","Estado","valor"])
    meses_presentes = [m for m in MESES if m in df.columns]
    if not meses_presentes:
        return pd.DataFrame(columns=id_cols + ["Mes","Estado","valor"])

    long = (
        df.melt(
            id_vars=[c for c in id_cols if c in df.columns],
            value_vars=meses_presentes,
            var_name="Mes",
            value_name="Estado"
        )
        .dropna(subset=["Estado"])
    )
    if long.empty:
        long["valor"] = []
        return long

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

def unique_union(*series_list):
    vals = []
    for s in series_list:
        if s is None:
            continue
        try:
            vals.extend([x for x in pd.Series(s).dropna().astype(str).str.strip().tolist() if x != ""])
        except Exception:
            pass
    if not vals:
        return []
    return sorted(pd.unique(pd.Series(vals)).tolist())

def safe_mean(series, default=0.0):
    try:
        if series is None:
            return default
        v = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
        return float(v.mean()) if len(v) else default
    except Exception:
        return default

def make_count_df_from_series(series_counts, x_name="Categoría", y_name="Cantidad", start_at_one=False):
    s = series_counts.copy() if hasattr(series_counts, "copy") else series_counts
    if isinstance(s, pd.Series):
        df = s.reset_index()
        if df.shape[1] >= 2:
            df.columns = [x_name, y_name]
        else:
            df = pd.DataFrame({x_name: [], y_name: []})
    else:
        df = pd.DataFrame(columns=[x_name, y_name])

    if y_name in df.columns:
        df[y_name] = pd.to_numeric(df[y_name], errors="coerce").fillna(0)

    # Solo para graficar (evita barras "invisibles" en 0)
    if start_at_one and not df.empty and y_name in df.columns:
        df[f"{y_name}_plot"] = df[y_name].apply(lambda v: v if v > 0 else 1)

    return df

def gauge_fig(valor, titulo):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(valor),
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#111111"},
            "steps": [
                {"range": [0, 60], "color": "#e74c3c"},
                {"range": [60, 90], "color": "#f1c40f"},
                {"range": [90, 100], "color": "#00a65a"},
            ],
        },
        title={"text": titulo},
    ))
    return style_plotly(fig, height=430)

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
    return years

@st.cache_data(ttl=300, show_spinner=False)
def load_year(year: int):
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]

    # Hoja principal (obligatoria)
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())

    # Hoja AREAS (opcional: 2023 no tiene)
    area_sheet_name = f"{year} AREAS"
    if area_sheet_name in titles:
        df_dept = pd.DataFrame(sh.worksheet(area_sheet_name).get_all_records())
    else:
        df_dept = pd.DataFrame()

    # Normalizar columnas
    if not df_obj.empty:
        df_obj.columns = normalize_text_series(df_obj.columns.to_series())
    if not df_dept.empty:
        df_dept.columns = normalize_text_series(df_dept.columns.to_series())

        # Dic / Diciembre
        if "Diciembre" in df_dept.columns:
            if "Dic" in df_dept.columns:
                df_dept["Dic"] = df_dept["Dic"].fillna(df_dept["Diciembre"])
            else:
                df_dept.rename(columns={"Diciembre": "Dic"}, inplace=True)

        # Puesto responsable
        if "PUESTO" in df_dept.columns and "PUESTO RESPONSABLE" not in df_dept.columns:
            df_dept.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

        # Estandarizar columna de área/departamento operativo
        if "DEPARTAMENTO" not in df_dept.columns and "Área" in df_dept.columns:
            df_dept.rename(columns={"Área": "DEPARTAMENTO"}, inplace=True)
        if "DEPARTAMENTO" not in df_dept.columns and "AREA" in df_dept.columns:
            df_dept.rename(columns={"AREA": "DEPARTAMENTO"}, inplace=True)

    # Limpieza texto objetivos
    for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
        if c in df_obj.columns:
            df_obj[c] = normalize_text_series(df_obj[c])

    # Limpieza texto operativo
    if not df_dept.empty:
        for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?"]:
            if c in df_dept.columns:
                df_dept[c] = normalize_text_series(df_dept[c])

        if "¿Realizada?" in df_dept.columns:
            df_dept["¿Realizada?"] = normalize_realizada(df_dept["¿Realizada?"])

    return df_obj, df_dept

# =====================================================
# SIDEBAR: AÑO + COMPARATIVO
# =====================================================
years = get_years_available()
if not years:
    st.error("No encontré hojas tipo 2023/2024/2025 en tu Google Sheets.")
    st.stop()

st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar",
    options=years,
    default=[y for y in [2024, 2025] if y in years] or years[-2:] if len(years) >= 2 else years
)

# Carga año base
df_obj, df_dept = load_year(year_data)

# =====================================================
# PROCESAMIENTO BASE (SIN FILTRO) -> LONGS
# =====================================================
obj_id_cols = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
obj_id_cols = [c for c in obj_id_cols if c in df_obj.columns]
obj_long = normalizar_meses(df_obj, obj_id_cols)

dept_id_cols = []
if df_dept is not None and not df_dept.empty:
    dept_id_cols = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
    dept_id_cols = [c for c in dept_id_cols if c in df_dept.columns]
dept_long = normalizar_meses(df_dept, dept_id_cols) if (df_dept is not None and not df_dept.empty) else pd.DataFrame(columns=["Mes","Estado","valor"])

# =====================================================
# FILTROS GENERALES (APLICAN A TODO)
# =====================================================
st.sidebar.divider()
st.sidebar.header("🔎 Filtros generales")

# Opciones generales
opt_tipo = unique_union(
    df_obj["Tipo"] if "Tipo" in df_obj.columns else None,
    df_dept["TIPO"] if (df_dept is not None and not df_dept.empty and "TIPO" in df_dept.columns) else None
)
opt_perspectiva = unique_union(
    df_obj["Perspectiva"] if "Perspectiva" in df_obj.columns else None,
    df_dept["PERSPECTIVA"] if (df_dept is not None and not df_dept.empty and "PERSPECTIVA" in df_dept.columns) else None
)
opt_eje = unique_union(
    df_obj["Eje"] if "Eje" in df_obj.columns else None,
    df_dept["EJE"] if (df_dept is not None and not df_dept.empty and "EJE" in df_dept.columns) else None
)
opt_departamento = unique_union(
    df_obj["Departamento"] if "Departamento" in df_obj.columns else None,
    df_dept["DEPARTAMENTO"] if (df_dept is not None and not df_dept.empty and "DEPARTAMENTO" in df_dept.columns) else None
)
opt_objetivo = unique_union(
    df_obj["Objetivo"] if "Objetivo" in df_obj.columns else None,
    df_dept["OBJETIVO"] if (df_dept is not None and not df_dept.empty and "OBJETIVO" in df_dept.columns) else None
)
opt_tipo_objetivo = unique_union(
    df_obj["Tipo Objetivo"] if "Tipo Objetivo" in df_obj.columns else None
)
opt_puesto = unique_union(
    df_dept["PUESTO RESPONSABLE"] if (df_dept is not None and not df_dept.empty and "PUESTO RESPONSABLE" in df_dept.columns) else None
)
opt_realizada = unique_union(
    df_dept["¿Realizada?"] if (df_dept is not None and not df_dept.empty and "¿Realizada?" in df_dept.columns) else None
)

# Filtros (todos opcionales)
f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", opt_tipo)
f_persp = st.sidebar.multiselect("Perspectiva", opt_perspectiva)
f_eje = st.sidebar.multiselect("Eje", opt_eje)
f_depto = st.sidebar.multiselect("Departamento", opt_departamento)
f_objetivo = st.sidebar.multiselect("Objetivo", opt_objetivo)
f_tipo_objetivo = st.sidebar.multiselect("Tipo de Objetivo", opt_tipo_objetivo)
f_puesto = st.sidebar.multiselect("Puesto Responsable", opt_puesto)
f_realizada = st.sidebar.multiselect("Ejecución (Realizada / No realizada)", opt_realizada)

st.sidebar.caption("✅ Si no seleccionas filtros, se muestra TODO por defecto.")

# =====================================================
# APLICAR FILTROS A INFORMESPLAN (obj_long -> obj_resumen_f)
# =====================================================
obj_long_f = obj_long.copy()

obj_long_f = apply_filter(obj_long_f, "Tipo", f_tipo_plan)
obj_long_f = apply_filter(obj_long_f, "Perspectiva", f_persp)
obj_long_f = apply_filter(obj_long_f, "Eje", f_eje)
obj_long_f = apply_filter(obj_long_f, "Departamento", f_depto)
obj_long_f = apply_filter(obj_long_f, "Objetivo", f_objetivo)
obj_long_f = apply_filter(obj_long_f, "Tipo Objetivo", f_tipo_objetivo)

grp_cols_f = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in obj_long_f.columns]

if not obj_long_f.empty and grp_cols_f:
    obj_resumen_f = obj_long_f.groupby(grp_cols_f, as_index=False).agg(
        score_total=("valor","sum"),
        verdes=("Estado", lambda x: (x=="VERDE").sum()),
        amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum()),
        meses_reportados=("Mes","count"),
    )

    if "Frecuencia Medición" in obj_resumen_f.columns:
        obj_resumen_f["Frecuencia Medición"] = obj_resumen_f["Frecuencia Medición"].astype(str).str.strip().str.upper()
        obj_resumen_f["meses_esperados"] = obj_resumen_f["Frecuencia Medición"].map(FRECUENCIA_MAP).fillna(12)
    else:
        obj_resumen_f["meses_esperados"] = 12

    obj_resumen_f["cumplimiento_%"] = (obj_resumen_f["score_total"] / obj_resumen_f["meses_esperados"]).clip(0,1) * 100
    obj_resumen_f["estado_ejecutivo"] = obj_resumen_f.apply(estado_exec, axis=1)
else:
    obj_resumen_f = pd.DataFrame(columns=["cumplimiento_%","estado_ejecutivo","Objetivo"])

# =====================================================
# APLICAR FILTROS A TAREASPLAN (dept_long -> dept_res_f)
# =====================================================
dept_long_f = dept_long.copy()

if not dept_long_f.empty:
    dept_long_f = apply_filter(dept_long_f, "TIPO", f_tipo_plan)
    dept_long_f = apply_filter(dept_long_f, "PERSPECTIVA", f_persp)
    dept_long_f = apply_filter(dept_long_f, "EJE", f_eje)
    dept_long_f = apply_filter(dept_long_f, "DEPARTAMENTO", f_depto)
    dept_long_f = apply_filter(dept_long_f, "OBJETIVO", f_objetivo)
    dept_long_f = apply_filter(dept_long_f, "PUESTO RESPONSABLE", f_puesto)
    dept_long_f = apply_filter(dept_long_f, "¿Realizada?", f_realizada)

if not dept_long_f.empty and "DEPARTAMENTO" in dept_long_f.columns:
    dept_res_f = dept_long_f.groupby("DEPARTAMENTO", as_index=False).agg(
        cumplimiento=("valor","mean"),
        tareas=("TAREA","nunique") if "TAREA" in dept_long_f.columns else ("Estado","count"),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
        verdes=("Estado", lambda x: (x=="VERDE").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum())
    )
    dept_res_f["cumplimiento_%"] = dept_res_f["cumplimiento"] * 100
else:
    dept_res_f = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento_%","tareas"])

# % ejecución por depto filtrado
exec_res_f = None
if not dept_long_f.empty and "¿Realizada?" in dept_long_f.columns and "DEPARTAMENTO" in dept_long_f.columns:
    exec_res_f = dept_long_f.groupby(["DEPARTAMENTO","¿Realizada?"]).size().reset_index(name="conteo")
    exec_res_f["%"] = exec_res_f["conteo"] / exec_res_f.groupby("DEPARTAMENTO")["conteo"].transform("sum") * 100

# =====================================================
# MEDIDORES (3 dinámicos según filtros)
# =====================================================
valor_informesplan = safe_mean(obj_resumen_f["cumplimiento_%"] if "cumplimiento_%" in obj_resumen_f.columns else pd.Series(dtype=float), 0.0)
valor_tareasplan = safe_mean((dept_long_f["valor"] * 100) if "valor" in dept_long_f.columns else pd.Series(dtype=float), 0.0)

obj_vals_total = (obj_long_f["valor"] * 100) if "valor" in obj_long_f.columns else pd.Series(dtype=float)
dep_vals_total = (dept_long_f["valor"] * 100) if "valor" in dept_long_f.columns else pd.Series(dtype=float)
vals_total = pd.concat([obj_vals_total, dep_vals_total], ignore_index=True) if (len(obj_vals_total) > 0 or len(dep_vals_total) > 0) else pd.Series(dtype=float)
valor_total_plan = safe_mean(vals_total, 0.0)

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "📌 Resumen",
    "🎯 InformesPlan",
    "🗂️ TareasPlan",
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

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("InformesPlan (items)", int(len(obj_resumen_f)))
    k2.metric("TareasPlan (registros)", int(len(dept_long_f)))
    k3.metric("Cumplidos", int((obj_resumen_f["estado_ejecutivo"]=="CUMPLIDO").sum()) if "estado_ejecutivo" in obj_resumen_f.columns else 0)
    k4.metric("Riesgo", int((obj_resumen_f["estado_ejecutivo"]=="RIESGO").sum()) if "estado_ejecutivo" in obj_resumen_f.columns else 0)
    k5.metric("Crít./No subido", int(obj_resumen_f["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if "estado_ejecutivo" in obj_resumen_f.columns else 0)
    k6.metric("Total Plan %", f"{valor_total_plan:.1f}%")

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge_fig(valor_informesplan, f"{year_data} — InformesPlan"), use_container_width=True)
    g2.plotly_chart(gauge_fig(valor_tareasplan, f"{year_data} — TareasPlan"), use_container_width=True)
    g3.plotly_chart(gauge_fig(valor_total_plan, f"{year_data} — Total Plan (Unión)"), use_container_width=True)

    c_left, c_right = st.columns(2)

    with c_left:
        counts = make_count_df_from_series(
            obj_resumen_f["estado_ejecutivo"].value_counts().reindex(ESTADO_EJEC_ORDEN).fillna(0),
            x_name="Estado Ejecutivo", y_name="Cantidad", start_at_one=True
        )
        if counts.empty:
            st.info("Sin datos para distribución de estados con los filtros actuales.")
        else:
            fig = px.bar(
                counts, x="Estado Ejecutivo", y="Cantidad",
                color="Estado Ejecutivo", color_discrete_map=COLOR_EJEC, text="Cantidad"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, height=620, title="Distribución de Estados Ejecutivos (InformesPlan)"), use_container_width=True)

    with c_right:
        if obj_long_f.empty:
            st.info("Sin datos para tendencia mensual con los filtros actuales.")
        else:
            tr = obj_long_f.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(style_plotly(fig, height=620, title="Tendencia Mensual — InformesPlan"), use_container_width=True)

# =====================================================
# TAB 1: INFORMESPLAN (antes estratégico)
# =====================================================
with tabs[1]:
    st.subheader("🎯 InformesPlan — Análisis Avanzado")

    if obj_resumen_f.empty:
        st.warning("No hay datos de InformesPlan con los filtros actuales.")
    else:
        colA, colB = st.columns(2)

        with colA:
            top_bad = obj_resumen_f.sort_values("cumplimiento_%").head(15)
            fig = px.bar(
                top_bad, x="cumplimiento_%", y="Objetivo", orientation="h",
                color="estado_ejecutivo", color_discrete_map=COLOR_EJEC, text="cumplimiento_%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=760, title="Top 15 objetivos más críticos"), use_container_width=True)

        with colB:
            fig = px.pie(
                obj_resumen_f, names="estado_ejecutivo", hole=0.55,
                color="estado_ejecutivo", color_discrete_map=COLOR_EJEC
            )
            st.plotly_chart(style_plotly(fig, height=760, title="Mix de Estado Ejecutivo (InformesPlan)"), use_container_width=True)

        r1, r2 = st.columns(2)

        with r1:
            if "Departamento" in obj_resumen_f.columns:
                dep = obj_resumen_f.groupby("Departamento")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                fig = px.bar(dep, x="cumplimiento_%", y="Departamento", orientation="h",
                             color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=650, title="Cumplimiento por Departamento"), use_container_width=True)
            else:
                st.info("No existe columna Departamento en InformesPlan para este año.")

        with r2:
            if "Perspectiva" in obj_resumen_f.columns:
                p = obj_resumen_f.groupby("Perspectiva")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                fig = px.bar(p, x="cumplimiento_%", y="Perspectiva", orientation="h",
                             color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=650, title="Cumplimiento por Perspectiva"), use_container_width=True)
            else:
                st.info("No existe columna Perspectiva.")

        r3, r4 = st.columns(2)
        with r3:
            if "Tipo" in obj_resumen_f.columns:
                t = obj_resumen_f.groupby("Tipo")["cumplimiento_%"].mean().reset_index()
                fig = px.bar(t, x="Tipo", y="cumplimiento_%", text="cumplimiento_%",
                             color="Tipo")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=560, title="Cumplimiento por Tipo (POA / PEC)"), use_container_width=True)
            else:
                st.info("No existe columna Tipo.")

        with r4:
            df_dev = obj_resumen_f.copy()
            df_dev["desviación_%"] = df_dev["cumplimiento_%"] - 100
            df_dev = df_dev.sort_values("desviación_%").head(15)
            fig = px.bar(df_dev, x="desviación_%", y="Objetivo", orientation="h",
                         color="desviación_%", color_continuous_scale="RdYlGn", text="desviación_%")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=560, title="Desviación vs 100% (Top 15)"), use_container_width=True)

        st.markdown("#### 🌡️ Heatmap — Objetivo vs Mes (Top 25 más críticos)")
        if not obj_long_f.empty and "Objetivo" in obj_long_f.columns:
            avg_obj = obj_long_f.groupby("Objetivo")["valor"].mean().sort_values().head(25).index.tolist()
            hm = obj_long_f[obj_long_f["Objetivo"].isin(avg_obj)].pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
            if not hm.empty:
                fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
                st.plotly_chart(style_plotly(fig, height=760, title="Heatmap InformesPlan (Top 25 críticos)"), use_container_width=True)
            else:
                st.info("Sin datos para heatmap.")
        else:
            st.info("Sin datos para heatmap.")

# =====================================================
# TAB 2: TAREASPLAN (antes operativo)
# =====================================================
with tabs[2]:
    st.subheader("🗂️ TareasPlan — Control Operativo")

    if dept_long_f.empty:
        st.warning("No hay datos de TareasPlan para este año o con los filtros actuales.")
    else:
        order = st.selectbox("Orden del ranking", ["Peor → Mejor", "Mejor → Peor"], index=0)
        asc = True if order == "Peor → Mejor" else False

        left, right = st.columns(2)

        with left:
            if not dept_res_f.empty:
                rk = dept_res_f.sort_values("cumplimiento_%", ascending=asc).head(20)
                fig = px.bar(
                    rk, x="cumplimiento_%", y="DEPARTAMENTO", orientation="h",
                    color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=760, title="Ranking de Departamentos (Top 20)"), use_container_width=True)
            else:
                st.info("Sin resumen por departamento.")

        with right:
            if not dept_res_f.empty and "tareas" in dept_res_f.columns:
                sc = dept_res_f.copy()
                fig = px.scatter(sc, x="tareas", y="cumplimiento_%", size="tareas", hover_name="DEPARTAMENTO")
                st.plotly_chart(style_plotly(fig, height=760, title="Cumplimiento vs Carga (# tareas)"), use_container_width=True)
            else:
                st.info("No hay datos suficientes para dispersión.")

        c1, c2 = st.columns(2)
        with c1:
            # Mix de colores en TareasPlan
            mix_task = make_count_df_from_series(
                dept_long_f["Estado"].value_counts().reindex(["VERDE","AMARILLO","ROJO","MORADO"]).fillna(0),
                x_name="Estado", y_name="Cantidad", start_at_one=True
            )
            if not mix_task.empty:
                fig = px.bar(mix_task, x="Estado", y="Cantidad", color="Estado", text="Cantidad",
                             color_discrete_map=COLOR_ESTADO)
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_plotly(fig, height=560, title="Distribución de Colores (TareasPlan)"), use_container_width=True)

        with c2:
            # Tendencia mensual TareasPlan
            tr2 = dept_long_f.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
            tr2["cumplimiento_%"] = tr2["valor"] * 100
            if not tr2.empty:
                fig = px.line(tr2, x="Mes", y="cumplimiento_%", markers=True)
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(style_plotly(fig, height=560, title="Tendencia Mensual — TareasPlan"), use_container_width=True)

        if exec_res_f is not None and not exec_res_f.empty:
            st.markdown("#### ✅ Ejecución (Realizada vs No realizada) por Departamento — Top 15 menor % realizada")
            tmp = exec_res_f.pivot_table(index="DEPARTAMENTO", columns="¿Realizada?", values="%", fill_value=0).reset_index()
            if "REALIZADA" in tmp.columns:
                tmp = tmp.sort_values("REALIZADA").head(15)
                fig = px.bar(tmp, x="REALIZADA", y="DEPARTAMENTO", orientation="h", text="REALIZADA")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(style_plotly(fig, height=620, title="Top 15 deptos con menor % Realizada"), use_container_width=True)
            else:
                st.info("No se encontró categoría REALIZADA.")

        st.markdown("#### 🌡️ Heatmap — Departamento vs Mes (Top 25 más críticos)")
        if "DEPARTAMENTO" in dept_long_f.columns:
            avg_d = dept_long_f.groupby("DEPARTAMENTO")["valor"].mean().sort_values().head(25).index.tolist()
            hm = dept_long_f[dept_long_f["DEPARTAMENTO"].isin(avg_d)].pivot_table(
                index="DEPARTAMENTO", columns="Mes", values="valor", fill_value=0
            )
            if not hm.empty:
                fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"])
                st.plotly_chart(style_plotly(fig, height=760, title="Heatmap TareasPlan (Top 25 críticos)"), use_container_width=True)

# =====================================================
# TAB 3: COMPARATIVO
# =====================================================
with tabs[3]:
    st.subheader("📊 Comparativo (Años)")

    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el sidebar para comparar.")
    else:
        comp_obj = []
        comp_dept = []

        for y in compare_years:
            o, d = load_year(y)

            # InformesPlan
            o_id = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            o_id = [c for c in o_id if c in o.columns]
            ol = normalizar_meses(o, o_id)
            ol["AÑO"] = y
            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)
            ol = apply_filter(ol, "Objetivo", f_objetivo)
            ol = apply_filter(ol, "Tipo Objetivo", f_tipo_objetivo)
            comp_obj.append(ol)

            # TareasPlan (si existe)
            if d is not None and not d.empty:
                d_id = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
                d_id = [c for c in d_id if c in d.columns]
                dl = normalizar_meses(d, d_id)
                dl["AÑO"] = y
                dl = apply_filter(dl, "TIPO", f_tipo_plan)
                dl = apply_filter(dl, "PERSPECTIVA", f_persp)
                dl = apply_filter(dl, "EJE", f_eje)
                dl = apply_filter(dl, "DEPARTAMENTO", f_depto)
                dl = apply_filter(dl, "OBJETIVO", f_objetivo)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        comp_dept_long = pd.concat(comp_dept, ignore_index=True) if comp_dept else pd.DataFrame()

        # --- Bloque 1: % por color InformesPlan ---
        st.markdown("### 🎯 InformesPlan — % por color")
        if not comp_obj_long.empty:
            obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            obj_mix["%"] = obj_mix["conteo"] / obj_mix.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(
                obj_mix, x="AÑO", y="%", color="Estado",
                barmode="group", color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=620, title="Comparativo InformesPlan — % por color"), use_container_width=True)
        else:
            st.info("Sin datos comparativos de InformesPlan.")

        # --- Bloque 2: % por color TareasPlan ---
        st.markdown("### 🗂️ TareasPlan — % por color")
        if not comp_dept_long.empty:
            dep_mix = comp_dept_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            dep_mix["%"] = dep_mix["conteo"] / dep_mix.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(
                dep_mix, x="AÑO", y="%", color="Estado",
                barmode="group", color_discrete_map=COLOR_ESTADO,
                category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                text="%"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=620, title="Comparativo TareasPlan — % por color"), use_container_width=True)
        else:
            st.info("No hay hojas AREAS para los años seleccionados o no hay datos filtrados.")

        # --- Bloque 3: Tendencia mensual comparativa ---
        st.markdown("### 📈 Tendencia mensual comparativa")
        cL, cR = st.columns(2)
        with cL:
            if not comp_obj_long.empty:
                t = comp_obj_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                t["cumplimiento_%"] = t["valor"] * 100
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=580, title="InformesPlan — promedio mensual"), use_container_width=True)
        with cR:
            if not comp_dept_long.empty:
                t = comp_dept_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                t["cumplimiento_%"] = t["valor"] * 100
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=580, title="TareasPlan — promedio mensual"), use_container_width=True)

        # --- Bloque 4: Comparativo cobertura de objetivos (comunes vs nuevos) ---
        st.markdown("### 🧩 Cobertura de objetivos por año (cantidad de objetivos únicos)")
        try:
            rows_cov = []
            for y in compare_years:
                o, _ = load_year(y)
                if "Objetivo" in o.columns:
                    o2 = o.copy()
                    o2 = apply_filter(o2, "Tipo", f_tipo_plan)
                    o2 = apply_filter(o2, "Perspectiva", f_persp)
                    o2 = apply_filter(o2, "Eje", f_eje)
                    o2 = apply_filter(o2, "Departamento", f_depto)
                    o2 = apply_filter(o2, "Objetivo", f_objetivo)
                    o2 = apply_filter(o2, "Tipo Objetivo", f_tipo_objetivo)
                    rows_cov.append({"AÑO": y, "Objetivos únicos": int(o2["Objetivo"].nunique()) if not o2.empty else 0})
            cov_df = pd.DataFrame(rows_cov)
            if not cov_df.empty:
                fig = px.bar(cov_df, x="AÑO", y="Objetivos únicos", text="Objetivos únicos")
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_plotly(fig, height=520, title="Cobertura de objetivos por año"), use_container_width=True)
        except Exception:
            pass

# =====================================================
# TAB 4: ALERTAS (tabla semáforo)
# =====================================================
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    # InformesPlan
    if not obj_resumen_f.empty:
        crit_obj = obj_resumen_f[obj_resumen_f["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
        for _, r in crit_obj.iterrows():
            nivel = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Fuente": "InformesPlan",
                "Nombre": r.get("Objetivo",""),
                "Estado": r["estado_ejecutivo"],
                "Cumplimiento %": float(r.get("cumplimiento_%", np.nan))
            })

    # TareasPlan
    if not dept_res_f.empty:
        bad_dept = dept_res_f[dept_res_f["cumplimiento_%"] < 60].copy()
        for _, r in bad_dept.iterrows():
            nivel = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Fuente": "TareasPlan",
                "Nombre": r.get("DEPARTAMENTO",""),
                "Estado": "BAJO CUMPLIMIENTO",
                "Cumplimiento %": float(r.get("cumplimiento_%", np.nan))
            })

    alerts_df = pd.DataFrame(alert_rows)

    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        alerts_df["OrdenNivel"] = alerts_df["Nivel"].map({"CRÍTICA":0, "NORMAL":1}).fillna(2)
        alerts_df = alerts_df.sort_values(["OrdenNivel","Fuente","Cumplimiento %"], ascending=[True, True, True]).drop(columns=["OrdenNivel"])
        alerts_df["Cumplimiento %"] = alerts_df["Cumplimiento %"].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}")

        def semaforo(row):
            bg = "#ffe1e1" if row["Nivel"] == "CRÍTICA" else "#fff3cd"
            txt = "#111111"
            return [f"background-color: {bg}; color: {txt};"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

        # Resumen visual de alertas
        alert_count = make_count_df_from_series(alerts_df["Nivel"].value_counts(), "Nivel", "Cantidad", start_at_one=True)
        if not alert_count.empty:
            fig = px.bar(alert_count, x="Nivel", y="Cantidad", color="Nivel", text="Cantidad",
                         color_discrete_map={"CRÍTICA":"#e74c3c", "NORMAL":"#f1c40f"})
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, height=420, title="Resumen de alertas"), use_container_width=True)

# =====================================================
# TAB 5: EXPORTAR
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    fig_estado_exec = None
    fig_rank_dept = None

    if not obj_resumen_f.empty:
        fig_estado_exec = px.pie(
            obj_resumen_f, names="estado_ejecutivo", hole=0.55,
            color="estado_ejecutivo", color_discrete_map=COLOR_EJEC,
            title=f"{year_data} — Estado Ejecutivo (InformesPlan)"
        )
        fig_estado_exec = style_plotly(fig_estado_exec, height=520)

    if not dept_res_f.empty:
        fig_rank_dept = px.bar(
            dept_res_f.sort_values("cumplimiento_%").head(20),
            x="cumplimiento_%", y="DEPARTAMENTO", orientation="h",
            title=f"{year_data} — Ranking crítico TareasPlan (Top 20)"
        )
        fig_rank_dept = style_plotly(fig_rank_dept, height=520)

    def build_report_html():
        k_info = len(obj_resumen_f)
        k_tareas = len(dept_long_f)
        k_ok = int((obj_resumen_f["estado_ejecutivo"]=="CUMPLIDO").sum()) if "estado_ejecutivo" in obj_resumen_f.columns else 0
        k_riesgo = int((obj_resumen_f["estado_ejecutivo"]=="RIESGO").sum()) if "estado_ejecutivo" in obj_resumen_f.columns else 0
        k_crit = int(obj_resumen_f["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if "estado_ejecutivo" in obj_resumen_f.columns else 0

        rep_alert_html = alerts_df.to_html(index=False) if 'alerts_df' in locals() and isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty else "<p>Sin alertas.</p>"

        html_parts = [f"""
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
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
  th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Reporte Estratégico y de Control</h1>
<div class="top">Año base: <b>{year_data}</b> · Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

<h2>KPIs</h2>
<div class="kpis">
  <div class="kpi"><b>InformesPlan (items)</b><br>{k_info}</div>
  <div class="kpi"><b>TareasPlan (registros)</b><br>{k_tareas}</div>
  <div class="kpi"><b>Cumplidos</b><br>{k_ok}</div>
  <div class="kpi"><b>En Riesgo</b><br>{k_riesgo}</div>
  <div class="kpi"><b>Crít./No subido</b><br>{k_crit}</div>
  <div class="kpi"><b>InformesPlan %</b><br>{valor_informesplan:.1f}%</div>
  <div class="kpi"><b>TareasPlan %</b><br>{valor_tareasplan:.1f}%</div>
  <div class="kpi"><b>Total Plan %</b><br>{valor_total_plan:.1f}%</div>
</div>

<h2>Gráficas</h2>
"""]

        if fig_estado_exec is not None:
            html_parts.append(fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn"))
        if fig_rank_dept is not None:
            html_parts.append(fig_rank_dept.to_html(full_html=False, include_plotlyjs=False))

        html_parts.append("<h2>Alertas</h2>")
        html_parts.append(rep_alert_html)

        html_parts.append("<h2>Tabla: InformesPlan (resumen)</h2>")
        html_parts.append(obj_resumen_f.head(200).to_html(index=False) if not obj_resumen_f.empty else "<p>Sin datos.</p>")

        html_parts.append("<h2>Tabla: TareasPlan (resumen)</h2>")
        html_parts.append(dept_res_f.head(200).to_html(index=False) if not dept_res_f.empty else "<p>Sin datos.</p>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

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

    with st.expander("InformesPlan — Resumen"):
        st.dataframe(obj_resumen_f, use_container_width=True)

    with st.expander("InformesPlan — Long"):
        st.dataframe(obj_long_f, use_container_width=True)

    with st.expander("TareasPlan — Resumen"):
        st.dataframe(dept_res_f, use_container_width=True)

    with st.expander("TareasPlan — Long"):
        st.dataframe(dept_long_f, use_container_width=True)

st.caption("Fuente: Google Sheets · Dashboard Estratégico")

