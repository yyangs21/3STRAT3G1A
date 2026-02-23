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
# ESTILO (EJECUTIVO CLARO)
# =====================================================
st.markdown(
    """
<style>
.stApp { background: #f3f4f6; color: #111111; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
section[data-testid="stSidebar"] * { color: #111111 !important; }

/* textos visibles */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #111111;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {
    color: #111111 !important;
}

/* inputs sidebar visibles */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #111111 !important;
}
section[data-testid="stSidebar"] input {
    color: #111111 !important;
    background: #ffffff !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #eef2ff !important;
    color: #111111 !important;
    border: 1px solid #dbeafe !important;
}

/* métricas */
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

/* tabs / contenedores */
div[data-testid="stTabs"] button {
    color: #111111 !important;
}
[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}
hr { border: none; border-top: 1px solid #e5e7eb; }

/* alertas legibles */
div[data-testid="stAlert"] * {
    color: #111111 !important;
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
def style_plotly(fig, height=560, title=None):
    """
    Layout ejecutivo. Si el gráfico es gauge/indicator, no toca ejes.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60 if title else 20, b=20),
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
        # OJO: en plotly moderno titlefont puede fallar en update_xaxes/update_yaxes
        fig.update_xaxes(
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
            gridcolor="#e5e7eb",
            zerolinecolor="#e5e7eb",
        )
        fig.update_yaxes(
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
            gridcolor="#e5e7eb",
            zerolinecolor="#e5e7eb",
        )
    return fig

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\n", " ", regex=False).str.strip()

def normalize_estado_series(s: pd.Series) -> pd.Series:
    x = s.replace("", np.nan).dropna().astype(str).str.strip().str.upper()
    return x

def normalize_realizada(s: pd.Series) -> pd.Series:
    """
    Planificada/Planificado cuentan como NO REALIZADA.
    """
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"PLANIFICADA": "NO REALIZADA", "PLANIFICADO": "NO REALIZADA"})
    x = x.replace({"REALIZADA": "REALIZADA"})
    return x

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    if not meses_presentes:
        return pd.DataFrame(columns=id_cols + ["Mes", "Estado", "valor"])

    long = (
        df.melt(
            id_vars=[c for c in id_cols if c in df.columns],
            value_vars=meses_presentes,
            var_name="Mes",
            value_name="Estado"
        )
        .dropna(subset=["Estado"])
        .copy()
    )
    long["Estado"] = normalize_estado_series(long["Estado"])
    long = long[long["Estado"].isin(list(ESTADO_MAP.keys()))].copy()
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

def safe_mean(series, default=0.0):
    try:
        if series is None or len(series) == 0:
            return default
        v = pd.to_numeric(series, errors="coerce").dropna()
        if len(v) == 0:
            return default
        return float(v.mean())
    except Exception:
        return default

def gauge_fig(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value) if pd.notna(value) else 0,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#111111"},
            "bar": {"color": "#111111"},
            "steps": [
                {"range": [0, 60], "color": "#e74c3c"},
                {"range": [60, 90], "color": "#f1c40f"},
                {"range": [90, 100], "color": "#00a65a"},
            ],
            "bordercolor": "#e5e7eb",
            "bgcolor": "#ffffff",
        },
        title={"text": title, "font": {"color": "#111111"}}
    ))
    return style_plotly(fig, height=430)

def union_values(df1, col1, df2=None, col2=None):
    vals = []
    if df1 is not None and not df1.empty and col1 in df1.columns:
        vals.extend(df1[col1].dropna().astype(str).str.strip().tolist())
    if df2 is not None and not df2.empty and col2 and col2 in df2.columns:
        vals.extend(df2[col2].dropna().astype(str).str.strip().tolist())
    if not vals:
        return []
    out = pd.Series(vals).dropna().astype(str).str.strip()
    out = out[out != ""].unique().tolist()
    return sorted(out)

def build_count_df_from_series(s: pd.Series, category_col="Categoría", value_col="Cantidad", order=None):
    if s is None or len(s) == 0:
        return pd.DataFrame(columns=[category_col, value_col])
    x = s.value_counts()
    if order:
        x = x.reindex(order).fillna(0)
    df = x.reset_index()
    df.columns = [category_col, value_col]
    return df

def format_pct_text(fig, colname):
    try:
        fig.update_traces(texttemplate=f"%{{text:.1f}}%", textposition="outside")
    except Exception:
        pass
    return fig

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_years_available():
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    years = sorted([int(t) for t in titles if t.isdigit()])
    return years, titles

@st.cache_data(ttl=300, show_spinner=False)
def load_year(year: int):
    """
    Devuelve:
      df_obj (siempre intenta cargar hoja 'YYYY')
      df_dept (si existe 'YYYY AREAS'; si no existe, DataFrame vacío)
    """
    sh = client.open(SHEET_NAME)

    # Hoja estratégica
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    if not df_obj.empty:
        df_obj.columns = normalize_text_series(df_obj.columns.to_series())

    # Hoja operativa opcional (ej: 2023 no tiene 2023 AREAS)
    ws_titles = [ws.title.strip() for ws in sh.worksheets()]
    area_title = f"{year} AREAS"
    if area_title in ws_titles:
        df_dept = pd.DataFrame(sh.worksheet(area_title).get_all_records())
        if not df_dept.empty:
            df_dept.columns = normalize_text_series(df_dept.columns.to_series())
    else:
        df_dept = pd.DataFrame()

    # Normalizaciones operativas
    if not df_dept.empty:
        # Diciembre duplicado
        if "Diciembre" in df_dept.columns:
            if "Dic" in df_dept.columns:
                df_dept["Dic"] = df_dept["Dic"].replace("", np.nan)
                df_dept["Dic"] = df_dept["Dic"].fillna(df_dept["Diciembre"])
            else:
                df_dept.rename(columns={"Diciembre": "Dic"}, inplace=True)

        # PUESTO
        if "PUESTO" in df_dept.columns and "PUESTO RESPONSABLE" not in df_dept.columns:
            df_dept.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

        # Área -> DEPARTAMENTO
        if "DEPARTAMENTO" not in df_dept.columns and "Área" in df_dept.columns:
            df_dept.rename(columns={"Área": "DEPARTAMENTO"}, inplace=True)
        if "DEPARTAMENTO" not in df_dept.columns and "AREA" in df_dept.columns:
            df_dept.rename(columns={"AREA": "DEPARTAMENTO"}, inplace=True)

    # Limpieza texto (objetivos)
    if not df_obj.empty:
        for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
            if c in df_obj.columns:
                df_obj[c] = normalize_text_series(df_obj[c])

    # Limpieza texto (operativo)
    if not df_dept.empty:
        for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?"]:
            if c in df_dept.columns:
                df_dept[c] = normalize_text_series(df_dept[c])

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

# -------- Filtros GLOBALes por criterio (aplican a todo el tablero) --------
f_tipo_plan = st.sidebar.multiselect(
    "Tipo (POA / PEC)",
    union_values(df_obj, "Tipo", df_dept, "TIPO")
)

f_persp = st.sidebar.multiselect(
    "Perspectiva",
    union_values(df_obj, "Perspectiva", df_dept, "PERSPECTIVA")
)

f_eje = st.sidebar.multiselect(
    "Eje",
    union_values(df_obj, "Eje", df_dept, "EJE")
)

f_departamento = st.sidebar.multiselect(
    "Departamento",
    union_values(df_obj, "Departamento", df_dept, "DEPARTAMENTO")
)

f_objetivo = st.sidebar.multiselect(
    "Objetivo",
    union_values(df_obj, "Objetivo", df_dept, "OBJETIVO")
)

# -------- Filtros adicionales (solo operativos) --------
if has_operativo_year:
    f_puesto = st.sidebar.multiselect(
        "Puesto Responsable",
        sorted(df_dept["PUESTO RESPONSABLE"].dropna().unique()) if "PUESTO RESPONSABLE" in df_dept.columns else []
    )
    f_realizada = st.sidebar.multiselect(
        "Ejecución (Realizada / No realizada)",
        sorted(df_dept["¿Realizada?"].dropna().unique()) if "¿Realizada?" in df_dept.columns else []
    )
else:
    f_puesto, f_realizada = [], []
    st.sidebar.info(f"El año {year_data} no tiene hoja '{year_data} AREAS'.")

st.sidebar.caption("✅ Si no seleccionas filtros, se muestra toda la información por default.")

# =====================================================
# PROCESAMIENTO OBJETIVOS (estratégico)
# =====================================================
obj_id_cols = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
obj_id_cols = [c for c in obj_id_cols if c in df_obj.columns]

obj_long = normalizar_meses(df_obj, obj_id_cols) if not df_obj.empty else pd.DataFrame()

# filtros GLOBALes en objetivos
obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "Departamento", f_departamento)
obj_long = apply_filter(obj_long, "Objetivo", f_objetivo)

grp_cols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in obj_long.columns] if not obj_long.empty else []

if not obj_long.empty and grp_cols:
    obj_resumen = obj_long.groupby(grp_cols, as_index=False).agg(
        score_total=("valor","sum"),
        verdes=("Estado", lambda x: (x=="VERDE").sum()),
        amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum()),
        meses_reportados=("Mes","count")
    )
else:
    obj_resumen = pd.DataFrame(columns=grp_cols + ["score_total","verdes","amarillos","rojos","morados","meses_reportados"])

if not obj_resumen.empty:
    if "Frecuencia Medición" in obj_resumen.columns:
        obj_resumen["Frecuencia Medición"] = obj_resumen["Frecuencia Medición"].astype(str).str.strip().str.upper()
        obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(FRECUENCIA_MAP).fillna(12)
    else:
        obj_resumen["meses_esperados"] = 12

    obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1) * 100
    obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec, axis=1)
else:
    obj_resumen["meses_esperados"] = []
    obj_resumen["cumplimiento_%"] = []
    obj_resumen["estado_ejecutivo"] = []

# =====================================================
# PROCESAMIENTO OPERATIVO (departamentos)
# =====================================================
if has_operativo_year:
    dept_id_cols = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
    dept_id_cols = [c for c in dept_id_cols if c in df_dept.columns]
    dept_long = normalizar_meses(df_dept, dept_id_cols)

    # filtros GLOBALes por criterio
    dept_long = apply_filter(dept_long, "TIPO", f_tipo_plan)
    dept_long = apply_filter(dept_long, "PERSPECTIVA", f_persp)
    dept_long = apply_filter(dept_long, "EJE", f_eje)
    dept_long = apply_filter(dept_long, "DEPARTAMENTO", f_departamento)
    dept_long = apply_filter(dept_long, "OBJETIVO", f_objetivo)

    # filtros propios operativos
    dept_long = apply_filter(dept_long, "PUESTO RESPONSABLE", f_puesto)
    dept_long = apply_filter(dept_long, "¿Realizada?", f_realizada)
else:
    dept_long = pd.DataFrame()

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
    dept_res = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento","tareas","rojos","amarillos","verdes","morados","cumplimiento_%"])

exec_res = None
if not dept_long.empty and "¿Realizada?" in dept_long.columns and "DEPARTAMENTO" in dept_long.columns:
    exec_res = (dept_long.groupby(["DEPARTAMENTO","¿Realizada?"]).size().reset_index(name="conteo"))
    exec_res["%"] = exec_res["conteo"] / exec_res.groupby("DEPARTAMENTO")["conteo"].transform("sum") * 100

dept_res_puesto = None
if not dept_long.empty and "DEPARTAMENTO" in dept_long.columns and "PUESTO RESPONSABLE" in dept_long.columns:
    dept_res_puesto = dept_long.groupby(["DEPARTAMENTO","PUESTO RESPONSABLE"], as_index=False).agg(
        cumplimiento=("valor","mean"),
        tareas=("TAREA","nunique") if "TAREA" in dept_long.columns else ("Estado","count")
    )
    dept_res_puesto["cumplimiento_%"] = dept_res_puesto["cumplimiento"] * 100

# =====================================================
# MEDIDORES (recalculan con filtros globales)
# =====================================================
gauge_obj_val = safe_mean(obj_resumen["cumplimiento_%"]) if not obj_resumen.empty else 0.0
gauge_op_val = (safe_mean(dept_long["valor"]) * 100) if not dept_long.empty else 0.0

# medidor combinado (promedio de los disponibles)
vals_for_global = []
if not obj_resumen.empty:
    vals_for_global.append(gauge_obj_val)
if not dept_long.empty:
    vals_for_global.append(gauge_op_val)
gauge_global_val = float(np.mean(vals_for_global)) if vals_for_global else 0.0

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
    c5.metric("Cumplimiento Estratégico", f"{gauge_obj_val:.1f}%")
    c6.metric("Cumplimiento Operativo", f"{gauge_op_val:.1f}%")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_fig(gauge_obj_val, f"{year_data} — Estratégico (Objetivos)"), use_container_width=True)
    with g2:
        if has_operativo_year:
            st.plotly_chart(gauge_fig(gauge_op_val, f"{year_data} — Operativo (Departamentos)"), use_container_width=True)
        else:
            st.info(f"El año {year_data} no tiene hoja AREAS.")
    with g3:
        st.plotly_chart(gauge_fig(gauge_global_val, f"{year_data} — Cumplimiento Global (Promedio)"), use_container_width=True)

    left, right = st.columns(2)

    with left:
        counts = build_count_df_from_series(
            obj_resumen["estado_ejecutivo"] if not obj_resumen.empty else pd.Series(dtype=str),
            category_col="Estado Ejecutivo",
            value_col="Cantidad",
            order=ESTADO_EJEC_ORDEN
        )
        fig = px.bar(
            counts,
            x="Estado Ejecutivo",
            y="Cantidad",
            color="Estado Ejecutivo",
            color_discrete_map=COLOR_EJEC,
            text="Cantidad"
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(
            style_plotly(fig, height=620, title="Distribución de Estados Ejecutivos (Objetivos)"),
            use_container_width=True
        )

    with right:
        if not obj_long.empty:
            tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
            tr["cumplimiento_%"] = tr["valor"] * 100
            # recuento visual desde 1 (índice) si quieres mostrar enumeración
            tr["No."] = range(1, len(tr) + 1)
            fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(
                style_plotly(fig, height=620, title="Tendencia Mensual — Cumplimiento Promedio (Objetivos)"),
                use_container_width=True
            )
        else:
            st.info("Sin datos para tendencia mensual con los filtros actuales.")

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
            top_bad = obj_resumen.sort_values("cumplimiento_%").head(15).copy()
            top_bad["No."] = range(1, len(top_bad) + 1)  # recuento empieza en 1
            fig = px.bar(
                top_bad,
                x="cumplimiento_%",
                y="Objetivo",
                orientation="h",
                color="estado_ejecutivo",
                color_discrete_map=COLOR_EJEC,
                text="cumplimiento_%",
                hover_data=["No."] + [c for c in ["Tipo","Perspectiva","Eje","Departamento"] if c in top_bad.columns]
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_xaxes(range=[0, 100])
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

        c1, c2 = st.columns(2)
        with c1:
            if "Departamento" in obj_resumen.columns:
                dep = obj_resumen.groupby("Departamento")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                dep["No."] = range(1, len(dep) + 1)
                fig = px.bar(
                    dep,
                    x="cumplimiento_%",
                    y="Departamento",
                    orientation="h",
                    color="cumplimiento_%",
                    color_continuous_scale="RdYlGn",
                    text="cumplimiento_%",
                    hover_data=["No."]
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_xaxes(range=[0, 100])
                st.plotly_chart(
                    style_plotly(fig, height=650, title="Cumplimiento Promedio por Departamento (estratégico)"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Departamento en objetivos para este año.")

        with c2:
            if "Perspectiva" in obj_resumen.columns:
                p = obj_resumen.groupby("Perspectiva")["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
                p["No."] = range(1, len(p) + 1)
                fig = px.bar(
                    p,
                    x="cumplimiento_%",
                    y="Perspectiva",
                    orientation="h",
                    color="cumplimiento_%",
                    color_continuous_scale="RdYlGn",
                    text="cumplimiento_%",
                    hover_data=["No."]
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_xaxes(range=[0, 100])
                st.plotly_chart(
                    style_plotly(fig, height=650, title="Cumplimiento Promedio por Perspectiva"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Perspectiva en objetivos para este año.")

        c3, c4 = st.columns(2)
        with c3:
            if "Tipo" in obj_resumen.columns:
                t = obj_resumen.groupby("Tipo")["cumplimiento_%"].mean().reset_index()
                t["No."] = range(1, len(t) + 1)
                fig = px.bar(
                    t,
                    x="Tipo",
                    y="cumplimiento_%",
                    text="cumplimiento_%",
                    color="Tipo"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(
                    style_plotly(fig, height=600, title="Cumplimiento Promedio por Tipo (POA / PEC)"),
                    use_container_width=True
                )
            else:
                st.info("No existe columna Tipo en objetivos para este año.")

        with c4:
            # Desviación vs 100%
            df_dev = obj_resumen.copy()
            df_dev["desviación_%"] = df_dev["cumplimiento_%"] - 100
            df_dev = df_dev.sort_values("desviación_%").head(15)
            df_dev["No."] = range(1, len(df_dev) + 1)
            fig = px.bar(
                df_dev,
                x="desviación_%",
                y="Objetivo",
                orientation="h",
                color="desviación_%",
                color_continuous_scale="RdYlGn",
                text="desviación_%",
                hover_data=["No."]
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(
                style_plotly(fig, height=600, title="Desviación de Cumplimiento vs Meta (100%) — Top 15"),
                use_container_width=True
            )

        st.markdown("#### 🌡️ Heatmap — Objetivo vs Mes (Top 25 más críticos)")
        hm_base = obj_long.copy()
        if not hm_base.empty and "Objetivo" in hm_base.columns:
            avg_obj = hm_base.groupby("Objetivo")["valor"].mean().sort_values().head(25).index.tolist()
            hm = hm_base[hm_base["Objetivo"].isin(avg_obj)].pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
            hm = hm.reindex(columns=[m for m in MESES if m in hm.columns])
            fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"], aspect="auto")
            st.plotly_chart(
                style_plotly(fig, height=760, title="Heatmap (Top 25 objetivos más críticos)"),
                use_container_width=True
            )
        else:
            st.info("No hay datos suficientes para heatmap de objetivos.")

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
            rk = dept_res.sort_values("cumplimiento_%", ascending=asc).head(20).copy()
            rk["No."] = range(1, len(rk) + 1)
            fig = px.bar(
                rk,
                x="cumplimiento_%",
                y="DEPARTAMENTO",
                orientation="h",
                color="cumplimiento_%",
                color_continuous_scale="RdYlGn",
                text="cumplimiento_%",
                hover_data=["No.","tareas"]
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(
                style_plotly(fig, height=760, title="Ranking de Departamentos Operativos (Top 20)"),
                use_container_width=True
            )

        with right:
            sc = dept_res.copy()
            sc["No."] = range(1, len(sc) + 1)
            fig = px.scatter(
                sc,
                x="tareas",
                y="cumplimiento_%",
                size="tareas",
                hover_name="DEPARTAMENTO",
                hover_data=["No."]
            )
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(
                style_plotly(fig, height=760, title="Cumplimiento vs Carga (# tareas)"),
                use_container_width=True
            )

        c1, c2 = st.columns(2)

        with c1:
            # Mix de colores operativo
            mix = build_count_df_from_series(
                dept_long["Estado"],
                category_col="Estado",
                value_col="Cantidad",
                order=["VERDE","AMARILLO","ROJO","MORADO"]
            )
            fig = px.bar(
                mix,
                x="Estado",
                y="Cantidad",
                color="Estado",
                color_discrete_map=COLOR_ESTADO,
                text="Cantidad"
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(
                style_plotly(fig, height=620, title="Distribución de Estados (Operativo)"),
                use_container_width=True
            )

        with c2:
            # Tendencia mensual operativo
            t = dept_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
            t["cumplimiento_%"] = t["valor"] * 100
            t["No."] = range(1, len(t) + 1)
            fig = px.line(t, x="Mes", y="cumplimiento_%", markers=True)
            fig.update_yaxes(range=[0,100])
            st.plotly_chart(
                style_plotly(fig, height=620, title="Tendencia Mensual — Cumplimiento Promedio (Operativo)"),
                use_container_width=True
            )

        if exec_res is not None and not exec_res.empty:
            st.markdown("#### ✅ Ejecución (Realizada vs No realizada) por Departamento — Top 15 con menor % realizada")
            tmp = exec_res.pivot_table(index="DEPARTAMENTO", columns="¿Realizada?", values="%", fill_value=0).reset_index()
            if "REALIZADA" in tmp.columns:
                tmp = tmp.sort_values("REALIZADA").head(15).copy()
                tmp["No."] = range(1, len(tmp) + 1)
                fig = px.bar(
                    tmp,
                    x="REALIZADA",
                    y="DEPARTAMENTO",
                    orientation="h",
                    text="REALIZADA",
                    hover_data=["No."]
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_xaxes(range=[0, 100])
                st.plotly_chart(
                    style_plotly(fig, height=650, title="Top 15 deptos con menor % Realizada"),
                    use_container_width=True
                )
            else:
                st.info("No se encontró categoría REALIZADA en ¿Realizada?")

        st.markdown("#### 🌡️ Heatmap Operativo — Departamento vs Mes (Top 25 más críticos)")
        hm_base = dept_long.copy()
        avg_d = hm_base.groupby("DEPARTAMENTO")["valor"].mean().sort_values().head(25).index.tolist()
        hm = hm_base[hm_base["DEPARTAMENTO"].isin(avg_d)].pivot_table(index="DEPARTAMENTO", columns="Mes", values="valor", fill_value=0)
        hm = hm.reindex(columns=[m for m in MESES if m in hm.columns])
        fig = px.imshow(hm, color_continuous_scale=["#e74c3c","#f1c40f","#00a65a"], aspect="auto")
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
            if o is not None and not o.empty:
                o_id = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
                o_id = [c for c in o_id if c in o.columns]
                ol = normalizar_meses(o, o_id)
                ol["AÑO"] = y

                # filtros globales
                ol = apply_filter(ol, "Tipo", f_tipo_plan)
                ol = apply_filter(ol, "Perspectiva", f_persp)
                ol = apply_filter(ol, "Eje", f_eje)
                ol = apply_filter(ol, "Departamento", f_departamento)
                ol = apply_filter(ol, "Objetivo", f_objetivo)
                comp_obj.append(ol)

            # OPERATIVO (si existe)
            if d is not None and not d.empty:
                d_id = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
                d_id = [c for c in d_id if c in d.columns]
                dl = normalizar_meses(d, d_id)
                dl["AÑO"] = y

                # filtros globales + propios
                dl = apply_filter(dl, "TIPO", f_tipo_plan)
                dl = apply_filter(dl, "PERSPECTIVA", f_persp)
                dl = apply_filter(dl, "EJE", f_eje)
                dl = apply_filter(dl, "DEPARTAMENTO", f_departamento)
                dl = apply_filter(dl, "OBJETIVO", f_objetivo)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        comp_dept_long = pd.concat(comp_dept, ignore_index=True) if comp_dept else pd.DataFrame()

        ctop1, ctop2 = st.columns(2)
        with ctop1:
            if not comp_obj_long.empty:
                st.markdown("### 🎯 Objetivos — % por color (VERDE/AMARILLO/ROJO/MORADO)")
                obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
                obj_mix["%"] = obj_mix["conteo"] / obj_mix.groupby("AÑO")["conteo"].transform("sum") * 100

                fig = px.bar(
                    obj_mix,
                    x="AÑO", y="%", color="Estado",
                    barmode="group",
                    color_discrete_map=COLOR_ESTADO,
                    category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                    text="%"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(style_plotly(fig, height=620, title="Comparativo Objetivos — % por color"), use_container_width=True)
            else:
                st.info("Sin datos comparativos de objetivos con los filtros actuales.")

        with ctop2:
            if not comp_dept_long.empty:
                st.markdown("### 🏢 Operativo — % por color (VERDE/AMARILLO/ROJO/MORADO)")
                dep_mix = comp_dept_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
                dep_mix["%"] = dep_mix["conteo"] / dep_mix.groupby("AÑO")["conteo"].transform("sum") * 100

                fig = px.bar(
                    dep_mix,
                    x="AÑO", y="%", color="Estado",
                    barmode="group",
                    color_discrete_map=COLOR_ESTADO,
                    category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]},
                    text="%"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(style_plotly(fig, height=620, title="Comparativo Operativo — % por color"), use_container_width=True)
            else:
                st.info("No todos los años seleccionados tienen hoja AREAS o no hay datos con filtros.")

        c2a, c2b = st.columns(2)

        with c2a:
            if not comp_dept_long.empty and "¿Realizada?" in comp_dept_long.columns:
                st.markdown("### ✅ Operativo — % Realizada vs No realizada (por año)")
                ex = comp_dept_long.groupby(["AÑO","¿Realizada?"]).size().reset_index(name="conteo")
                ex["%"] = ex["conteo"] / ex.groupby("AÑO")["conteo"].transform("sum") * 100
                fig = px.bar(ex, x="AÑO", y="%", color="¿Realizada?", barmode="group", text="%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(style_plotly(fig, height=620, title="Comparativo Ejecución (Realizada vs No realizada)"), use_container_width=True)
            else:
                st.info("No hay datos comparativos de ejecución operativa disponibles.")

        with c2b:
            # comparativo % promedio (gauge-like en barras)
            rows = []
            for y in compare_years:
                o, d = load_year(y)

                # objetivos
                v_obj = np.nan
                if o is not None and not o.empty:
                    o_id = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"] if c in o.columns]
                    ol = normalizar_meses(o, o_id)
                    ol = apply_filter(ol, "Tipo", f_tipo_plan)
                    ol = apply_filter(ol, "Perspectiva", f_persp)
                    ol = apply_filter(ol, "Eje", f_eje)
                    ol = apply_filter(ol, "Departamento", f_departamento)
                    ol = apply_filter(ol, "Objetivo", f_objetivo)
                    v_obj = safe_mean(ol["valor"]) * 100 if not ol.empty else np.nan

                # operativo
                v_op = np.nan
                if d is not None and not d.empty:
                    d_id = [c for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"] if c in d.columns]
                    dl = normalizar_meses(d, d_id)
                    dl = apply_filter(dl, "TIPO", f_tipo_plan)
                    dl = apply_filter(dl, "PERSPECTIVA", f_persp)
                    dl = apply_filter(dl, "EJE", f_eje)
                    dl = apply_filter(dl, "DEPARTAMENTO", f_departamento)
                    dl = apply_filter(dl, "OBJETIVO", f_objetivo)
                    dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                    dl = apply_filter(dl, "¿Realizada?", f_realizada)
                    v_op = safe_mean(dl["valor"]) * 100 if not dl.empty else np.nan

                rows.append({"AÑO": y, "Tipo": "Estratégico", "Cumplimiento %": v_obj})
                if pd.notna(v_op):
                    rows.append({"AÑO": y, "Tipo": "Operativo", "Cumplimiento %": v_op})

            comp_mean = pd.DataFrame(rows).dropna()
            if not comp_mean.empty:
                fig = px.bar(comp_mean, x="AÑO", y="Cumplimiento %", color="Tipo", barmode="group", text="Cumplimiento %")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(style_plotly(fig, height=620, title="Comparativo de cumplimiento promedio por año"), use_container_width=True)
            else:
                st.info("No hay datos suficientes para comparativo de cumplimiento promedio.")

        st.markdown("### 📈 Tendencia mensual comparativa (promedio %)")
        left, right = st.columns(2)
        with left:
            if not comp_obj_long.empty:
                t = comp_obj_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                t["cumplimiento_%"] = t["valor"] * 100
                t["Mes"] = pd.Categorical(t["Mes"], categories=MESES, ordered=True)
                t = t.sort_values(["AÑO","Mes"])
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(style_plotly(fig, height=620, title="Objetivos — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos comparativos de objetivos.")

        with right:
            if not comp_dept_long.empty:
                t = comp_dept_long.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
                t["cumplimiento_%"] = t["valor"] * 100
                t["Mes"] = pd.Categorical(t["Mes"], categories=MESES, ordered=True)
                t = t.sort_values(["AÑO","Mes"])
                fig = px.line(t, x="Mes", y="cumplimiento_%", color="AÑO", markers=True)
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(style_plotly(fig, height=620, title="Operativo — tendencia mensual promedio"), use_container_width=True)
            else:
                st.info("Sin datos comparativos operativos.")

# =====================================================
# TAB 4: ALERTAS
# =====================================================
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    # Objetivos
    if not obj_resumen.empty:
        crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
        crit_obj = crit_obj.sort_values("cumplimiento_%")
        crit_obj["No."] = range(1, len(crit_obj) + 1)  # empieza en 1
        for _, r in crit_obj.iterrows():
            nivel = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
            alert_rows.append({
                "No.": int(r["No."]),
                "Nivel": nivel,
                "Tipo": "Objetivo",
                "Nombre": r.get("Objetivo",""),
                "Estado": r["estado_ejecutivo"],
                "Cumplimiento %": float(r["cumplimiento_%"])
            })

    # Operativo
    if not dept_res.empty:
        bad_dept = dept_res[dept_res["cumplimiento_%"] < 60].copy().sort_values("cumplimiento_%")
        bad_dept["No."] = range(1, len(bad_dept) + 1)
        for _, r in bad_dept.iterrows():
            nivel = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
            alert_rows.append({
                "No.": int(r["No."]),
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
            bg = "#ffdede" if row["Nivel"] == "CRÍTICA" else "#fff3cd"
            return [f"background-color: {bg}; color: #111111;"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

        # mini resumen visual alertas
        a1, a2 = st.columns(2)
        with a1:
            a_count = alerts_df.groupby("Nivel").size().reset_index(name="Cantidad")
            fig = px.bar(a_count, x="Nivel", y="Cantidad", color="Nivel", text="Cantidad",
                         color_discrete_map={"CRÍTICA":"#e74c3c","NORMAL":"#f1c40f"})
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, height=420, title="Alertas por nivel"), use_container_width=True)
        with a2:
            a_type = alerts_df.groupby("Tipo").size().reset_index(name="Cantidad")
            fig = px.pie(a_type, names="Tipo", values="Cantidad", hole=0.45)
            st.plotly_chart(style_plotly(fig, height=420, title="Distribución de alertas por tipo"), use_container_width=True)

# =====================================================
# TAB 5: EXPORTAR
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    # Figuras base de export
    if not obj_resumen.empty:
        fig_estado_exec = px.pie(
            obj_resumen,
            names="estado_ejecutivo",
            hole=0.55,
            color="estado_ejecutivo",
            color_discrete_map=COLOR_EJEC,
            title=f"{year_data} — Estado Ejecutivo (Objetivos)"
        )
        fig_estado_exec = style_plotly(fig_estado_exec, height=520)
    else:
        fig_estado_exec = None

    if not dept_res.empty:
        fig_rank_dept = px.bar(
            dept_res.sort_values("cumplimiento_%").head(20),
            x="cumplimiento_%",
            y="DEPARTAMENTO",
            orientation="h",
            title=f"{year_data} — Ranking crítico Operativo (Top 20 deptos)"
        )
        fig_rank_dept = style_plotly(fig_rank_dept, height=520)
    else:
        fig_rank_dept = None

    def build_report_html():
        k_obj = len(obj_resumen)
        k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if not obj_resumen.empty else 0
        k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if not obj_resumen.empty else 0
        k_crit = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if not obj_resumen.empty else 0
        k_avg = gauge_obj_val
        k_op = gauge_op_val
        k_glb = gauge_global_val

        rep_alert_html = alerts_df.to_html(index=False) if 'alerts_df' in locals() and not alerts_df.empty else "<p>Sin alertas.</p>"

        parts = [f"""
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
  th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; color:#111; }}
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
  <div class="kpi"><b>Cump. Estratégico</b><br>{k_avg:.1f}%</div>
  <div class="kpi"><b>Cump. Operativo</b><br>{k_op:.1f}%</div>
  <div class="kpi"><b>Cump. Global</b><br>{k_glb:.1f}%</div>
</div>

<h2>Gráficas</h2>
"""]

        if fig_estado_exec is not None:
            parts.append(fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn"))
        if fig_rank_dept is not None:
            parts.append(fig_rank_dept.to_html(full_html=False, include_plotlyjs=False))

        parts.append(f"""
<h2>Alertas</h2>
{rep_alert_html}

<h2>Tabla: Objetivos (resumen)</h2>
{obj_resumen.head(200).to_html(index=False) if not obj_resumen.empty else '<p>Sin datos.</p>'}

<h2>Tabla: Operativo (departamentos resumen)</h2>
{dept_res.head(200).to_html(index=False) if not dept_res.empty else '<p>Sin datos (sin hoja AREAS o sin filtros).</p>'}

</body>
</html>
""")
        return "\n".join(parts)

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
        if not obj_resumen.empty:
            df_show = obj_resumen.copy().reset_index(drop=True)
            df_show.index = df_show.index + 1  # empieza en 1
            st.dataframe(df_show, use_container_width=True)
        else:
            st.info("Sin datos.")

    with st.expander("Objetivos — Long"):
        if not obj_long.empty:
            df_show = obj_long.copy().reset_index(drop=True)
            df_show.index = df_show.index + 1
            st.dataframe(df_show, use_container_width=True)
        else:
            st.info("Sin datos.")

    with st.expander("Operativo — Departamentos resumen"):
        if not dept_res.empty:
            df_show = dept_res.copy().reset_index(drop=True)
            df_show.index = df_show.index + 1
            st.dataframe(df_show, use_container_width=True)
        else:
            st.info("Sin datos operativos para este año/filtros.")

    with st.expander("Operativo — Long"):
        if not dept_long.empty:
            df_show = dept_long.copy().reset_index(drop=True)
            df_show.index = df_show.index + 1
            st.dataframe(df_show, use_container_width=True)
        else:
            st.info("Sin datos operativos.")

st.caption("Fuente: Google Sheets · Dashboard Estratégico")




