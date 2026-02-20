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
.stApp { background: #f3f4f6; color: #111111; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
section[data-testid="stSidebar"] * { color: #111111 !important; }

h1, h2, h3, h4, p, label { color: #111111 !important; }

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

[data-testid="stDataFrame"] { background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; }
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

FRECUENCIA_MAP = {"MENSUAL": 12, "BIMESTRAL": 6, "TRIMESTRAL": 4, "CUATRIMESTRAL": 3, "SEMESTRAL": 2, "ANUAL": 1}

# =====================================================
# HELPERS
# =====================================================
def style_plotly(fig, height=520, title=None):
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
        fig.update_xaxes(tickfont=dict(color="#111111"), titlefont=dict(color="#111111"), gridcolor="#e5e7eb")
        fig.update_yaxes(tickfont=dict(color="#111111"), titlefont=dict(color="#111111"), gridcolor="#e5e7eb")
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

def empty_dept_df():
    # Estructura mínima para que el app no falle aunque no exista hoja AREAS
    cols = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"] + MESES
    return pd.DataFrame(columns=cols)

# =====================================================
# LOAD DATA (AUTO-DETECT AÑOS)
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_titles():
    sh = client.open(SHEET_NAME)
    return [ws.title.strip() for ws in sh.worksheets()]

@st.cache_data(ttl=300, show_spinner=False)
def get_years_meta():
    titles = get_titles()
    years = []
    meta = {}
    for t in titles:
        if t.isdigit():
            y = int(t)
            years.append(y)
    years = sorted(list(set(years)))
    for y in years:
        meta[y] = {"has_areas": (f"{y} AREAS" in titles)}
    return years, meta

@st.cache_data(ttl=300, show_spinner=False)
def load_year(year: int, has_areas: bool):
    sh = client.open(SHEET_NAME)

    # Objetivos SIEMPRE
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    df_obj.columns = normalize_text_series(df_obj.columns.to_series())

    for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"]:
        if c in df_obj.columns:
            df_obj[c] = normalize_text_series(df_obj[c])

    # Operativo SOLO si existe hoja
    if not has_areas:
        return df_obj, empty_dept_df()

    try:
        df_dept = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())
    except WorksheetNotFound:
        return df_obj, empty_dept_df()

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

    # estandariza DEPARTAMENTO
    if "DEPARTAMENTO" not in df_dept.columns and "Área" in df_dept.columns:
        df_dept.rename(columns={"Área":"DEPARTAMENTO"}, inplace=True)

    for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","¿Realizada?"]:
        if c in df_dept.columns:
            df_dept[c] = normalize_text_series(df_dept[c])

    if "¿Realizada?" in df_dept.columns:
        df_dept["¿Realizada?"] = normalize_realizada(df_dept["¿Realizada?"])

    return df_obj, df_dept

# =====================================================
# SIDEBAR
# =====================================================
years, meta = get_years_meta()
if not years:
    st.error("No encontré hojas tipo '2023', '2024', '2025' en tu Google Sheets.")
    st.stop()

st.sidebar.header("🗂️ Seleccionar año de data")
year_data = st.sidebar.selectbox("Año base", options=years, index=len(years)-1)
has_areas_base = meta[year_data]["has_areas"]

st.sidebar.divider()
st.sidebar.header("📊 Comparativo")
compare_years = st.sidebar.multiselect(
    "Años a comparar",
    options=years,
    default=[y for y in [2023, 2024, 2025] if y in years]
)

df_obj, df_dept = load_year(year_data, has_areas_base)

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# Objetivos
f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", sorted(df_obj["Tipo"].dropna().unique()) if "Tipo" in df_obj.columns else [])
f_persp = st.sidebar.multiselect("Perspectiva", sorted(df_obj["Perspectiva"].dropna().unique()) if "Perspectiva" in df_obj.columns else [])
f_eje = st.sidebar.multiselect("Eje", sorted(df_obj["Eje"].dropna().unique()) if "Eje" in df_obj.columns else [])
f_depto = st.sidebar.multiselect("Departamento (estratégico)", sorted(df_obj["Departamento"].dropna().unique()) if "Departamento" in df_obj.columns else [])

# Operativo (solo si hay AREAS; si no, dejamos filtros vacíos)
if has_areas_base and not df_dept.empty:
    f_dept_op = st.sidebar.multiselect("Departamento (operativo)", sorted(df_dept["DEPARTAMENTO"].dropna().unique()) if "DEPARTAMENTO" in df_dept.columns else [])
    f_puesto = st.sidebar.multiselect("Puesto Responsable", sorted(df_dept["PUESTO RESPONSABLE"].dropna().unique()) if "PUESTO RESPONSABLE" in df_dept.columns else [])
    f_realizada = st.sidebar.multiselect("Ejecución (Realizada / No realizada)", sorted(df_dept["¿Realizada?"].dropna().unique()) if "¿Realizada?" in df_dept.columns else [])
else:
    f_dept_op, f_puesto, f_realizada = [], [], []
    st.sidebar.info("ℹ️ Este año no tiene hoja AREAS (operativo no disponible).")

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
# PROCESAMIENTO OPERATIVO (si existe)
# =====================================================
dept_res = pd.DataFrame(columns=["DEPARTAMENTO","cumplimiento","tareas","rojos","amarillos","verdes","morados","cumplimiento_%"])
dept_long = pd.DataFrame(columns=["DEPARTAMENTO","Mes","Estado","valor"])
exec_res = None

if has_areas_base and not df_dept.empty:
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

    if "¿Realizada?" in dept_long.columns and not dept_long.empty:
        exec_res = (dept_long.groupby(["DEPARTAMENTO","¿Realizada?"]).size().reset_index(name="conteo"))
        exec_res["%"] = exec_res["conteo"] / exec_res.groupby("DEPARTAMENTO")["conteo"].transform("sum") * 100

# =====================================================
# TABS
# =====================================================
tabs = st.tabs(["📌 Resumen", "🎯 Objetivos", "🏢 Operativo (Deptos)", "📊 Comparativo", "🚨 Alertas", "📄 Exportar", "📋 Datos"])

# =====================================================
# TAB 0: RESUMEN
# =====================================================
with tabs[0]:
    st.subheader(f"📌 Resumen Ejecutivo — Año {year_data}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Objetivos", int(len(obj_resumen)))
    c2.metric("Cumplidos", int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if "estado_ejecutivo" in obj_resumen.columns else 0)
    c3.metric("En Riesgo", int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if "estado_ejecutivo" in obj_resumen.columns else 0)
    c4.metric("Críticos / No Subido", int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if "estado_ejecutivo" in obj_resumen.columns else 0)
    c5.metric("Cumplimiento Promedio", f"{obj_resumen['cumplimiento_%'].mean():.1f}%" if "cumplimiento_%" in obj_resumen.columns and len(obj_resumen) else "0.0%")

    g1, g2 = st.columns(2)

    val_obj = float(obj_resumen["cumplimiento_%"].mean()) if "cumplimiento_%" in obj_resumen.columns and len(obj_resumen) else 0
    fig_g1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_obj,
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":"#111111"},
               "steps":[{"range":[0,60],"color":"#e74c3c"},
                        {"range":[60,90],"color":"#f1c40f"},
                        {"range":[90,100],"color":"#00a65a"}]},
        title={"text": f"{year_data} — Cumplimiento Estratégico (Objetivos)"}
    ))
    g1.plotly_chart(style_plotly(fig_g1, height=460), use_container_width=True)

    if has_areas_base:
        val_dept = float(dept_long["valor"].mean()*100) if len(dept_long) else 0
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_dept,
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#111111"},
                   "steps":[{"range":[0,60],"color":"#e74c3c"},
                            {"range":[60,90],"color":"#f1c40f"},
                            {"range":[90,100],"color":"#00a65a"}]},
            title={"text": f"{year_data} — Cumplimiento Operativo (Departamentos)"}
        ))
        g2.plotly_chart(style_plotly(fig_g2, height=460), use_container_width=True)
    else:
        g2.info("ℹ️ Operativo no disponible para este año (no existe hoja AREAS).")

# =====================================================
# TAB 1: OBJETIVOS
# =====================================================
with tabs[1]:
    st.subheader("🎯 Objetivos — Análisis Avanzado")
    if obj_resumen.empty:
        st.info("No hay datos de objetivos con los filtros actuales.")
    else:
        colA, colB = st.columns(2)

        with colA:
            top_bad = obj_resumen.sort_values("cumplimiento_%").head(15)
            fig = px.bar(top_bad, x="cumplimiento_%", y="Objetivo", orientation="h",
                         color="estado_ejecutivo", color_discrete_map=COLOR_EJEC, text="cumplimiento_%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=760, title="Top 15 Objetivos más críticos (peor cumplimiento)"), use_container_width=True)

        with colB:
            fig = px.pie(obj_resumen, names="estado_ejecutivo", hole=0.55,
                         color="estado_ejecutivo", color_discrete_map=COLOR_EJEC)
            st.plotly_chart(style_plotly(fig, height=760, title="Mix de Estado Ejecutivo (Objetivos)"), use_container_width=True)

# =====================================================
# TAB 2: OPERATIVO
# =====================================================
with tabs[2]:
    st.subheader("🏢 Operativo — Control por Departamento")
    if not has_areas_base:
        st.info("Este año no tiene hoja AREAS, por lo tanto no hay operativo para mostrar.")
    elif dept_res.empty:
        st.info("No hay datos operativos con los filtros actuales.")
    else:
        order = st.selectbox("Orden del ranking", ["Peor → Mejor", "Mejor → Peor"], index=0)
        asc = True if order == "Peor → Mejor" else False

        rk = dept_res.sort_values("cumplimiento_%", ascending=asc).head(20)
        fig = px.bar(rk, x="cumplimiento_%", y="DEPARTAMENTO", orientation="h",
                     color="cumplimiento_%", color_continuous_scale="RdYlGn", text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(style_plotly(fig, height=760, title="Ranking de Departamentos Operativos (Top 20)"), use_container_width=True)

# =====================================================
# TAB 3: COMPARATIVO
# =====================================================
with tabs[3]:
    st.subheader("📊 Comparativo")
    if len(compare_years) < 2:
        st.info("Selecciona al menos 2 años en el sidebar para comparar.")
    else:
        comp_obj = []
        comp_dept = []

        for y in compare_years:
            has_areas_y = meta.get(y, {}).get("has_areas", False)
            o, d = load_year(y, has_areas_y)

            # OBJETIVOS
            o_id = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
            o_id = [c for c in o_id if c in o.columns]
            ol = normalizar_meses(o, o_id)
            ol["AÑO"] = y
            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)
            comp_obj.append(ol)

            # OPERATIVO (solo si existe AREAS)
            if has_areas_y and not d.empty:
                d_id = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
                d_id = [c for c in d_id if c in d.columns]
                dl = normalizar_meses(d, d_id)
                dl["AÑO"] = y
                dl = apply_filter(dl, "DEPARTAMENTO", f_dept_op)
                dl = apply_filter(dl, "PUESTO RESPONSABLE", f_puesto)
                dl = apply_filter(dl, "¿Realizada?", f_realizada)
                comp_dept.append(dl)

        comp_obj_long = pd.concat(comp_obj, ignore_index=True) if comp_obj else pd.DataFrame()
        st.markdown("### 🎯 Objetivos — % por color")
        if comp_obj_long.empty:
            st.info("No hay data de objetivos para el comparativo.")
        else:
            obj_mix = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            obj_mix["%"] = obj_mix["conteo"] / obj_mix.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(obj_mix, x="AÑO", y="%", color="Estado",
                         barmode="group", color_discrete_map=COLOR_ESTADO,
                         category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}, text="%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=660, title="Comparativo Objetivos — % por color"), use_container_width=True)

        st.markdown("### 🏢 Operativo — % por color (solo años con AREAS)")
        if not comp_dept:
            st.info("Ninguno de los años seleccionados tiene hoja AREAS para comparar operativo (ej. 2023 no tiene).")
        else:
            comp_dept_long = pd.concat(comp_dept, ignore_index=True)
            dep_mix = comp_dept_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
            dep_mix["%"] = dep_mix["conteo"] / dep_mix.groupby("AÑO")["conteo"].transform("sum") * 100
            fig = px.bar(dep_mix, x="AÑO", y="%", color="Estado",
                         barmode="group", color_discrete_map=COLOR_ESTADO,
                         category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}, text="%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plotly(fig, height=660, title="Comparativo Operativo — % por color"), use_container_width=True)

# =====================================================
# TAB 4: ALERTAS
# =====================================================
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    if not obj_resumen.empty and "estado_ejecutivo" in obj_resumen.columns:
        crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
        for _, r in crit_obj.iterrows():
            nivel = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
            alert_rows.append({
                "Nivel": nivel,
                "Tipo": "Objetivo",
                "Nombre": r.get("Objetivo",""),
                "Estado": r["estado_ejecutivo"],
                "Cumplimiento %": float(r["cumplimiento_%"]) if "cumplimiento_%" in r else np.nan
            })

    if has_areas_base and not dept_res.empty:
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
            return [f"background-color: {bg}"] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

# =====================================================
# TAB 5: EXPORTAR
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas)")

    fig_estado_exec = None
    if not obj_resumen.empty and "estado_ejecutivo" in obj_resumen.columns:
        fig_estado_exec = px.pie(obj_resumen, names="estado_ejecutivo", hole=0.55,
                                 color="estado_ejecutivo", color_discrete_map=COLOR_EJEC,
                                 title=f"{year_data} — Estado Ejecutivo (Objetivos)")
        fig_estado_exec = style_plotly(fig_estado_exec, height=560)

    fig_rank_dept = None
    if has_areas_base and not dept_res.empty:
        fig_rank_dept = px.bar(dept_res.sort_values("cumplimiento_%").head(20),
                               x="cumplimiento_%", y="DEPARTAMENTO", orientation="h",
                               title=f"{year_data} — Ranking crítico Operativo (Top 20 deptos)")
        fig_rank_dept = style_plotly(fig_rank_dept, height=560)

    def build_report_html():
        k_obj = len(obj_resumen)
        k_ok = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()) if "estado_ejecutivo" in obj_resumen.columns else 0
        k_riesgo = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()) if "estado_ejecutivo" in obj_resumen.columns else 0
        k_crit = int(obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"]).sum()) if "estado_ejecutivo" in obj_resumen.columns else 0
        k_avg = float(obj_resumen["cumplimiento_%"].mean()) if "cumplimiento_%" in obj_resumen.columns and len(obj_resumen) else 0

        rep_alert_html = alerts_df.to_html(index=False) if 'alerts_df' in globals() and not alerts_df.empty else "<p>Sin alertas.</p>"

        charts_html = ""
        if fig_estado_exec is not None:
            charts_html += fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn")
        if fig_rank_dept is not None:
            charts_html += fig_rank_dept.to_html(full_html=False, include_plotlyjs=False)
        if charts_html.strip() == "":
            charts_html = "<p>No hay gráficas para exportar con los filtros actuales.</p>"

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
  <div class="kpi"><b>Cumplimiento Promedio</b><br>{k_avg:.1f}%</div>
</div>

<h2>Gráficas</h2>
{charts_html}

<h2>Alertas</h2>
{rep_alert_html}

<h2>Tabla: Objetivos (resumen)</h2>
{obj_resumen.head(200).to_html(index=False) if not obj_resumen.empty else "<p>Sin datos</p>"}

<h2>Tabla: Operativo (departamentos resumen)</h2>
{dept_res.head(200).to_html(index=False) if (has_areas_base and not dept_res.empty) else "<p>Operativo no disponible</p>"}

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

    if has_areas_base:
        with st.expander("Operativo — Departamentos resumen"):
            st.dataframe(dept_res, use_container_width=True)
        with st.expander("Operativo — Long"):
            st.dataframe(dept_long, use_container_width=True)
    else:
        st.info("Operativo no disponible para este año (no hay hoja AREAS).")

st.caption("Fuente: Google Sheets · Dashboard Estratégico")
