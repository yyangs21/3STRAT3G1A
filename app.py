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

/* Inputs */
.stSelectbox label, .stMultiSelect label, .stTextInput label {
    color: #111111 !important;
    font-weight: 600 !important;
}
[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #d1d5db !important;
}
[data-baseweb="tag"] {
    background: #e5e7eb !important;
    color: #111111 !important;
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
MESES_ORDEN = {m: i+1 for i, m in enumerate(MESES)}  # empieza en 1

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
    """
    Aplica layout ejecutivo. Si el fig es go.Indicator (gauge),
    NO toca x/y axes para evitar errores.
    """
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
            title_font=dict(color="#111111")  # <- corregido (antes titlefont daba error)
        )
        fig.update_yaxes(
            tickfont=dict(color="#111111"),
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#d1d5db",
            title_font=dict(color="#111111")  # <- corregido
        )
    return fig

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\n", " ", regex=False).str.strip()

def normalize_estado_series(s: pd.Series) -> pd.Series:
    x = s.replace("", np.nan).dropna().astype(str).str.strip().str.upper()
    # limpia posibles espacios dobles / variantes
    x = x.str.replace("  ", " ", regex=False)
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
    long["MesNum"] = long["Mes"].map(MESES_ORDEN)  # 1..12
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

    # Hoja principal (obligatoria)
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())

    # Hoja AREAS (opcional, porque 2023 no tendrá)
    df_dept = None
    area_sheet_name = f"{year} AREAS"
    try:
        df_dept = pd.DataFrame(sh.worksheet(area_sheet_name).get_all_records())
    except Exception:
        df_dept = None

    # --- Objetivos ---
    df_obj.columns = normalize_text_series(df_obj.columns.to_series())

    for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
        if c in df_obj.columns:
            df_obj[c] = normalize_text_series(df_obj[c])

    # --- Operativo (si existe) ---
    if df_dept is not None:
        df_dept.columns = normalize_text_series(df_dept.columns.to_series())

        # Normaliza Dic duplicado
        if "Diciembre" in df_dept.columns:
            if "Dic" in df_dept.columns:
                df_dept["Dic"] = df_dept["Dic"].fillna(df_dept["Diciembre"])
            else:
                df_dept.rename(columns={"Diciembre": "Dic"}, inplace=True)

        # Normaliza puesto responsable
        if "PUESTO" in df_dept.columns and "PUESTO RESPONSABLE" not in df_dept.columns:
            df_dept.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

        # Operativo: si viene "Área", convertir a DEPARTAMENTO
        if "DEPARTAMENTO" not in df_dept.columns and "Área" in df_dept.columns:
            df_dept.rename(columns={"Área":"DEPARTAMENTO"}, inplace=True)

        # Limpieza texto
        for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?","Realizada?"]:
            if c in df_dept.columns:
                df_dept[c] = normalize_text_series(df_dept[c])

        # Normaliza nombre Realizada
        if "Realizada?" in df_dept.columns and "¿Realizada?" not in df_dept.columns:
            df_dept.rename(columns={"Realizada?":"¿Realizada?"}, inplace=True)

        # Normaliza realizada/planificada
        if "¿Realizada?" in df_dept.columns:
            df_dept["¿Realizada?"] = normalize_realizada(df_dept["¿Realizada?"])

    return df_obj, df_dept

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

df_obj, df_dept = load_year(year_data)
has_operativo_year = df_dept is not None and not df_dept.empty

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# --- Filtros objetivos ---
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
    "Departamento (estratégico)",
    sorted(df_obj["Departamento"].dropna().unique()) if "Departamento" in df_obj.columns else []
)
f_obje = st.sidebar.multiselect("Objetivo",sorted(df_obj{"Objetivo"].dropna().unique()) if "Objetivo in df_obj.columns else []

)

# --- Filtros operativo (si existe hoja AREAS) ---
if has_operativo_year:
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
else:
    st.sidebar.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")
    f_dept_op, f_puesto, f_realizada = [], [], []

st.sidebar.caption("✅ Si NO seleccionas filtros, se muestra TODO por default.")

# =====================================================
# PROCESAMIENTO OBJETIVOS
# =====================================================
obj_id_cols = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
obj_id_cols = [c for c in obj_id_cols if c in df_obj.columns]

obj_long = normalizar_meses(df_obj, obj_id_cols)

obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "Departamento", f_depto)
obj_long = apply_filter(obj_long, "Objetivo", f_obje)

grp_cols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in obj_long.columns]

if obj_long.empty:
    obj_resumen = pd.DataFrame(columns=grp_cols + ["score_total","verdes","amarillos","rojos","morados","meses_reportados","meses_esperados","cumplimiento_%","estado_ejecutivo"])
else:
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
# PROCESAMIENTO OPERATIVO (DEPARTAMENTO) — OPCIONAL
# =====================================================
dept_long = pd.DataFrame()
dept_res = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento","tareas","rojos","amarillos","verdes","morados","cumplimiento_%"])
exec_res = None
dept_res_puesto = None

if has_operativo_year:
    dept_id_cols = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
    dept_id_cols = [c for c in dept_id_cols if c in df_dept.columns]

    dept_long = normalizar_meses(df_dept, dept_id_cols)

    dept_long = apply_filter(dept_long, "DEPARTAMENTO", f_dept_op)
    dept_long = apply_filter(dept_long, "PUESTO RESPONSABLE", f_puesto)
    dept_long = apply_filter(dept_long, "¿Realizada?", f_realizada)

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

        if "¿Realizada?" in dept_long.columns:
            exec_res = (
                dept_long.groupby(["DEPARTAMENTO","¿Realizada?"]).size()
                .reset_index(name="conteo")
            )
            exec_res["%"] = exec_res["conteo"] / exec_res.groupby("DEPARTAMENTO")["conteo"].transform("sum") * 100

        if "PUESTO RESPONSABLE" in dept_long.columns:
            dept_res_puesto = dept_long.groupby(["DEPARTAMENTO","PUESTO RESPONSABLE"], as_index=False).agg(
                cumplimiento=("valor","mean"),
                tareas=("TAREA","nunique") if "TAREA" in dept_long.columns else ("Estado","count")
            )
            dept_res_puesto["cumplimiento_%"] = dept_res_puesto["cumplimiento"] * 100

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

    # KPIs (agregando KPI global consolidado)
    c1,c2,c3,c4,c5,c6 = st.columns(6)

    k_obj_total = int(len(obj_resumen))
    k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0
    k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0
    k_crit_ns = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0
    k_prom_obj = safe_mean_percent(obj_resumen["cumplimiento_%"]) if "cumplimiento_%" in obj_resumen.columns else np.nan
    k_prom_op = safe_mean_percent(dept_long["valor"] * 100) if not dept_long.empty else np.nan

    if has_operativo_year and not pd.isna(k_prom_op):
        k_global = float(np.nanmean([k_prom_obj, k_prom_op]))
    else:
        k_global = float(k_prom_obj) if not pd.isna(k_prom_obj) else np.nan

    c1.metric("Objetivos", k_obj_total)
    c2.metric("Cumplidos", k_ok)
    c3.metric("En Riesgo", k_riesgo)
    c4.metric("Críticos / No Subido", k_crit_ns)
    c5.metric("Cumpl. Estratégico", f"{k_prom_obj:.1f}%" if not pd.isna(k_prom_obj) else "—")
    c6.metric("Cumplimiento Global", f"{k_global:.1f}%" if not pd.isna(k_global) else "—")

    # 3 gauges (estratégico, operativo, consolidado)
    g1, g2, g3 = st.columns(3)

    val_obj = float(k_prom_obj) if not pd.isna(k_prom_obj) else 0.0
    fig_g1 = build_gauge(val_obj, f"{year_data} — Cumplimiento Estratégico", delta_ref=90)
    g1.plotly_chart(style_plotly(fig_g1, height=420), use_container_width=True)

    if has_operativo_year:
        val_dept = float(k_prom_op) if not pd.isna(k_prom_op) else 0.0
        fig_g2 = build_gauge(val_dept, f"{year_data} — Cumplimiento Operativo", delta_ref=90)
        g2.plotly_chart(style_plotly(fig_g2, height=420), use_container_width=True)
    else:
        val_dept = np.nan
        g2.info(f"ℹ️ El año {year_data} no tiene hoja '{year_data} AREAS'")

    if has_operativo_year and not pd.isna(val_dept):
        val_global = float(np.nanmean([val_obj, val_dept]))
        subt = "Promedio: Estratégico + Operativo"
    else:
        val_global = float(val_obj)
        subt = "Solo Estratégico (sin AREAS)"

    fig_g3 = build_gauge(val_global, f"{year_data} — Cumplimiento Global Consolidado", delta_ref=90)
    fig_g3.add_annotation(
        text=subt,
        x=0.5, y=0.02, xref="paper", yref="paper",
        showarrow=False, font=dict(size=11, color="#444444")
    )
    g3.plotly_chart(style_plotly(fig_g3, height=420), use_container_width=True)

    # Bloque visuales resumen
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
            style_plotly(fig, height=640, title="Distribución de Estados Ejecutivos (Objetivos)"),
            use_container_width=True
        )

    with right:
        if not obj_long.empty:
            tr = obj_long.groupby(["Mes","MesNum"], as_index=False)["valor"].mean()
            tr = tr.sort_values("MesNum")
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(
                style_plotly(fig, height=640, title="Tendencia Mensual — Cumplimiento Promedio (Objetivos)"),
                use_container_width=True
            )
        else:
            st.info("Sin datos de objetivos para mostrar tendencia.")

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
                style_plotly(fig, height=760, title="Mix de Estado Ejecutivo (Objetivos)"),
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
                    style_plotly(fig, height=640, title="Cumplimiento Promedio por Departamento (estratégico)"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Departamento en objetivos para este año.")

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
                st.info("No existe columna Perspectiva en objetivos para este año.")

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
            # Desviación vs 100%
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
        if not obj_long.empty and "Objetivo" in obj_long.columns:
            hm_base = obj_long.copy()
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
            st.info("No hay datos para heatmap de objetivos.")

# =====================================================
# TAB 2: OPERATIVO (DEPARTAMENTO)
# =====================================================
with tabs[2]:
    st.subheader("🏢 Operativo — Control por Departamento")

    if not has_operativo_year:
        st.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")
    elif dept_long.empty or dept_res.empty:
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
            st.plotly_chart(
                style_plotly(fig, height=760, title="Ranking de Departamentos Operativos (Top 20)"),
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

        # Más gráficas operativas
        r1, r2 = st.columns(2)
        with r1:
            # Mix de colores operativo
            mix_op = dept_long["Estado"].value_counts().reindex(["VERDE","AMARILLO","ROJO","MORADO"]).fillna(0).reset_index()
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
                style_plotly(fig, height=560, title="Distribución de estados (Operativo)"),
                use_container_width=True
            )

        with r2:
            # Tendencia mensual operativo
            tr = dept_long.groupby(["Mes","MesNum"], as_index=False)["valor"].mean()
            tr = tr.sort_values("MesNum")
            tr["cumplimiento_%"] = tr["valor"] * 100
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(
                style_plotly(fig, height=560, title="Tendencia mensual — Cumplimiento promedio (Operativo)"),
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

        st.markdown("#### 🌡️ Heatmap Operativo — Departamento vs Mes (Top 25 más críticos)")
        hm_base = dept_long.copy()
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
            style_plotly(fig, height=760, title="Heatmap Operativo (Top 25 deptos más críticos)"),
            use_container_width=True
        )

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
            o, d = load_year(y)

            # OBJETIVOS
            o_id = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            o_id = [c for c in o_id if c in o.columns]
            ol = normalizar_meses(o, o_id)
            ol["AÑO"] = y

            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)
            o1 = apply_filter(o1, "Objetivo", f_obje)
            comp_obj.append(ol)

            # OPERATIVO (opcional por año)
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

                dl = apply_filter(dl, "DEPARTAMENTO", f_dept_op)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        comp_dept_long = pd.concat(comp_dept, ignore_index=True) if comp_dept else pd.DataFrame()

        # --- Objetivos: % por color ---
        st.markdown("### 🎯 Objetivos — % por color (VERDE/AMARILLO/ROJO/MORADO)")
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
            st.plotly_chart(style_plotly(fig, height=640, title="Comparativo Objetivos — % por color"), use_container_width=True)
        else:
            st.info("Sin datos de objetivos para comparativo con los filtros actuales.")

        # --- Operativo: % por color ---
        st.markdown("### 🏢 Operativo — % por color (VERDE/AMARILLO/ROJO/MORADO)")
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
            st.plotly_chart(style_plotly(fig, height=640, title="Comparativo Operativo — % por color"), use_container_width=True)
        else:
            st.info("No hay hojas AREAS en los años comparados o quedaron sin datos por filtros.")

        # --- Comparativo ejecución ---
        st.markdown("### ✅ Operativo — % Realizada vs No realizada (por año)")
        if not comp_dept_long.empty and "¿Realizada?" in comp_dept_long.columns:
            ex = comp_dept_long.groupby(["AÑO","¿Realizada?"]).size().reset_index(name="conteo")
            ex["%"] = ex["conteo"] / ex.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(ex, x="AÑO", y="%", color="¿Realizada?", barmode="group", text="%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=600, title="Comparativo Ejecución (Realizada vs No realizada)"), use_container_width=True)
        else:
            st.info("No hay datos de ejecución operativa comparables.")

        # --- Tendencias comparativas ---
        st.markdown("### 📈 Tendencia mensual comparativa (promedio %)")
        left, right = st.columns(2)

        with left:
            if not comp_obj_long.empty:
                t = comp_obj_long.groupby(["AÑO","Mes","MesNum"], as_index=False)["valor"].mean()
                t["cumplimiento_%"] = t["valor"] * 100
                t = t.sort_values(["AÑO","MesNum"])
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=600, title="Objetivos — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos de objetivos para tendencia comparativa.")

        with right:
            if not comp_dept_long.empty:
                t = comp_dept_long.groupby(["AÑO","Mes","MesNum"], as_index=False)["valor"].mean()
                t["cumplimiento_%"] = t["valor"] * 100
                t = t.sort_values(["AÑO","MesNum"])
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                st.plotly_chart(style_plotly(fig, height=600, title="Operativo — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos operativos para tendencia comparativa.")

        # --- Comparativo de cobertura (qué existe / qué no existe) ---
        st.markdown("### 🧩 Cobertura comparativa (elementos que existen por año)")
        c1, c2 = st.columns(2)

        with c1:
            # Objetivos únicos por año (cantidad)
            if not comp_obj_long.empty and "Objetivo" in comp_obj_long.columns:
                cov_obj = comp_obj_long.groupby("AÑO")["Objetivo"].nunique().reset_index(name="Objetivos únicos")
                fig = px.bar(cov_obj, x="AÑO", y="Objetivos únicos", text="Objetivos únicos")
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_plotly(fig, height=520, title="Cobertura de objetivos únicos por año"), use_container_width=True)

        with c2:
            # Deptos únicos por año (si aplica)
            if not comp_dept_long.empty and "DEPARTAMENTO" in comp_dept_long.columns:
                cov_dep = comp_dept_long.groupby("AÑO")["DEPARTAMENTO"].nunique().reset_index(name="Deptos únicos")
                fig = px.bar(cov_dep, x="AÑO", y="Deptos únicos", text="Deptos únicos")
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_plotly(fig, height=520, title="Cobertura de deptos operativos únicos por año"), use_container_width=True)

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
                "Cumplimiento %": float(r["cumplimiento_%"])
            })

    # Operativo
    if not dept_res.empty:
        bad_dept = dept_res[dept_res["cumplimiento_%"] < 60].copy()
        for _, r in bad_dept.iterrows():
            nivel = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
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
            return [f"background-color: {bg}; color: #111111"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

        # Mini KPIs alertas
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
            title=f"{year_data} — Estado Ejecutivo (Objetivos)"
        )

        if not dept_res.empty:
            fig_rank_dept = px.bar(
                dept_res.sort_values("cumplimiento_%").head(20),
                x="cumplimiento_%",
                y="DEPARTAMENTO",
                orientation="h",
                title=f"{year_data} — Ranking crítico Operativo (Top 20 deptos)"
            )
        else:
            fig_rank_dept = px.bar(
                pd.DataFrame({"Mensaje":["Sin datos operativos"]}),
                x="Mensaje",
                y=[1],
                title=f"{year_data} — Operativo"
            )

        fig_estado_exec = style_plotly(fig_estado_exec, height=520)
        fig_rank_dept = style_plotly(fig_rank_dept, height=520)

        # generar alerts_df si no existe (por seguridad)
        if 'alerts_df' not in locals():
            alerts_df = pd.DataFrame()

        def build_report_html():
            k_obj = len(obj_resumen)
            k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0
            k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0
            k_crit = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0
            k_avg_obj = safe_mean_percent(obj_resumen["cumplimiento_%"])
            k_avg_op = safe_mean_percent(dept_long["valor"] * 100) if not dept_long.empty else np.nan
            k_avg_global = float(np.nanmean([k_avg_obj, k_avg_op])) if not pd.isna(k_avg_op) else k_avg_obj

            rep_alert_html = alerts_df.to_html(index=False) if not alerts_df.empty else "<p>Sin alertas.</p>"

            dept_res_html = (
                dept_res.head(200).to_html(index=False)
                if not dept_res.empty
                else "<p>Este año no tiene hoja AREAS o no hay datos operativos con los filtros.</p>"
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
  <div class="kpi"><b>Cumpl. Estratégico</b><br>{0 if pd.isna(k_avg_obj) else k_avg_obj:.1f}%</div>
  <div class="kpi"><b>Cumpl. Global</b><br>{0 if pd.isna(k_avg_global) else k_avg_global:.1f}%</div>
</div>

<h2>Gráficas</h2>
{fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn")}
{fig_rank_dept.to_html(full_html=False, include_plotlyjs=False)}

<h2>Alertas</h2>
{rep_alert_html}

<h2>Tabla: Objetivos (resumen)</h2>
{obj_resumen.head(200).to_html(index=False)}

<h2>Tabla: Operativo (departamentos resumen)</h2>
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

    with st.expander("Objetivos — Resumen"):
        st.dataframe(obj_resumen, use_container_width=True)

    with st.expander("Objetivos — Long"):
        st.dataframe(obj_long, use_container_width=True)

    with st.expander("Operativo — Departamentos resumen"):
        if not dept_res.empty:
            st.dataframe(dept_res, use_container_width=True)
        else:
            st.info("Sin datos operativos para este año o con los filtros actuales.")

    with st.expander("Operativo — Long"):
        if not dept_long.empty:
            st.dataframe(dept_long, use_container_width=True)
        else:
            st.info("Sin datos operativos para este año o con los filtros actuales.")

st.caption("Fuente: Google Sheets · Dashboard Estratégico")



