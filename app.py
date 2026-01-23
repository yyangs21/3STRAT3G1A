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
# ESTILO (FONDO GRIS CLARO + TEXTO NEGRO + CARDS)
# =====================================================
st.markdown(
    """
    <style>
      /* Fondo general */
      html, body, [class*="css"]  { color: #111 !important; }
      .stApp { background-color: #F3F5F7; }
      section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #EAECEF; }

      /* Contenedor */
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Cards KPI */
      div[data-testid="stMetric"]{
        background: #FFFFFF;
        border: 1px solid #EAECEF;
        padding: 14px 14px;
        border-radius: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      }
      div[data-testid="stMetric"] * { color: #111 !important; }

      /* Títulos */
      h1, h2, h3 { color: #111 !important; }

      /* Separadores */
      hr { border: none; border-top: 1px solid #EAECEF; }

      /* Dataframe */
      .stDataFrame { background: #FFFFFF; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# CONSTANTES
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {
    "VERDE": 1.0,
    "AMARILLO": 0.5,
    "ROJO": 0.0,
    "MORADO": 0.0,  # No subido
}

COLOR_ESTADO = {
    "VERDE": "#16A34A",
    "AMARILLO": "#F59E0B",
    "ROJO": "#EF4444",
    "MORADO": "#7C3AED"
}

COLOR_EJEC = {
    "CUMPLIDO": "#16A34A",
    "EN SEGUIMIENTO": "#F59E0B",
    "RIESGO": "#EF4444",
    "CRÍTICO": "#991B1B",
    "NO SUBIDO": "#7C3AED"
}

frecuencia_map = {
    "Mensual": 12,
    "Bimestral": 6,
    "Trimestral": 4,
    "Cuatrimestral": 3,
    "Semestral": 2,
    "Anual": 1
}

FIG_H_BIG = 520
FIG_H_MED = 440

# =====================================================
# HELPERS (ROBUSTOS PARA 2024/2025)
# =====================================================
def safe_strip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", " ", regex=False)
    return df

def unify_dic(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica 'Diciembre' -> 'Dic' y rellena si existen ambos."""
    df = df.copy()
    if "Diciembre" in df.columns and "Dic" in df.columns:
        df["Dic"] = df["Dic"].fillna(df["Diciembre"])
    elif "Diciembre" in df.columns and "Dic" not in df.columns:
        df.rename(columns={"Diciembre": "Dic"}, inplace=True)
    return df

def theme_plotly(fig: go.Figure, height: int = FIG_H_MED) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111"),
        margin=dict(l=40, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def normalizar_meses(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    df = df.copy()
    df = unify_dic(df)
    meses_presentes = [m for m in MESES if m in df.columns]
    long = (
        df.melt(
            id_vars=[c for c in id_cols if c in df.columns],
            value_vars=meses_presentes,
            var_name="Mes",
            value_name="Estado"
        )
        .dropna(subset=["Estado"])
    )
    long["Estado"] = long["Estado"].astype(str).str.strip().str.upper()
    return long

def apply_filter(df: pd.DataFrame, col: str, selected: list):
    if df is None or df.empty:
        return df
    if not selected:
        return df
    if col not in df.columns:
        return df
    return df[df[col].isin(selected)]

def estado_exec_row(r) -> str:
    if r.get("morados", 0) > 0:
        return "NO SUBIDO"
    if r.get("rojos", 0) > 0:
        return "RIESGO"
    if r.get("cumplimiento_%", 0) >= 90:
        return "CUMPLIDO"
    if r.get("cumplimiento_%", 0) >= 60:
        return "EN SEGUIMIENTO"
    return "CRÍTICO"

def pct(x):
    try:
        return float(x)
    except:
        return 0.0

# =====================================================
# CARGA DE GOOGLE SHEETS
# =====================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

@st.cache_data(ttl=300)
def gs_client():
    service_account_info = dict(st.secrets["gcp_service_account"])
    service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return gspread.authorize(creds)

https://docs.google.com/spreadsheets/d/1uY30GMPPzR754z2xIRdO2rEnJzG4lOCc04-Q2eELNbk/edit?hl=es&gid=1451350731#gid=1451350731 = "DATAESTRATEGIA"

@st.cache_data(ttl=300)
def get_years_available():
    cl = gs_client()
    sh = cl.open(https://docs.google.com/spreadsheets/d/1uY30GMPPzR754z2xIRdO2rEnJzG4lOCc04-Q2eELNbk/edit?hl=es&gid=1451350731#gid=1451350731)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    years = sorted([int(t) for t in titles if t.isdigit()])
    return years

@st.cache_data(ttl=300)
def load_year(year: int):
    cl = gs_client()
    sh = cl.open(https://docs.google.com/spreadsheets/d/1uY30GMPPzR754z2xIRdO2rEnJzG4lOCc04-Q2eELNbk/edit?hl=es&gid=1451350731#gid=1451350731)
    df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    df_area = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())
    df_obj = safe_strip(df_obj)
    df_area = safe_strip(df_area)
    df_area = unify_dic(df_area)

    # Normalizaciones típicas (según tu data real 2024/2025)
    if "PUESTO" in df_area.columns and "PUESTO RESPONSABLE" not in df_area.columns:
        df_area.rename(columns={"PUESTO": "PUESTO RESPONSABLE"}, inplace=True)

    # Si no existe AREA, usamos DEPARTAMENTO como área operativa
    if "AREA" not in df_area.columns and "DEPARTAMENTO" in df_area.columns:
        df_area["AREA"] = df_area["DEPARTAMENTO"]

    # Si no existe DEPARTAMENTO pero existe AREA, espejo
    if "DEPARTAMENTO" not in df_area.columns and "AREA" in df_area.columns:
        df_area["DEPARTAMENTO"] = df_area["AREA"]

    return df_obj, df_area

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
    "Años a comparar",
    options=years,
    default=[y for y in [2024, 2025] if y in years]
)

# Cargar data base
df_obj, df_area = load_year(year_data)

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# ---- Objetivos (según tu Excel, existen en 2024/2025)
def opts(df, col):
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().unique().tolist()])

f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", opts(df_obj, "Tipo"))
f_persp     = st.sidebar.multiselect("Perspectiva", opts(df_obj, "Perspectiva"))
f_eje       = st.sidebar.multiselect("Eje", opts(df_obj, "Eje"))
f_depto     = st.sidebar.multiselect("Departamento", opts(df_obj, "Departamento"))
f_tipo_obj  = st.sidebar.multiselect("Tipo Objetivo", opts(df_obj, "Tipo Objetivo"))
f_freq      = st.sidebar.multiselect("Frecuencia Medición", opts(df_obj, "Frecuencia Medición"))

# ---- Áreas (según tu Excel, existen en 2024/2025 AREAS)
f_area      = st.sidebar.multiselect("Área (operativa)", opts(df_area, "AREA"))
f_puesto    = st.sidebar.multiselect("Puesto Responsable", opts(df_area, "PUESTO RESPONSABLE"))
f_dep_area  = st.sidebar.multiselect("Departamento (áreas)", opts(df_area, "DEPARTAMENTO"))
f_tipo_area = st.sidebar.multiselect("Tipo (áreas)", opts(df_area, "TIPO")) if "TIPO" in df_area.columns else []
f_pers_area = st.sidebar.multiselect("Perspectiva (áreas)", opts(df_area, "PERSPECTIVA")) if "PERSPECTIVA" in df_area.columns else []

st.sidebar.caption("✅ Si no seleccionas filtros, el tablero muestra TODO por defecto.")

# =====================================================
# OBJETIVOS: LONG + RESUMEN + ESTADO EJECUTIVO (AÑO BASE)
# =====================================================
obj_id_cols = ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
obj_long = normalizar_meses(df_obj, obj_id_cols)
obj_long["valor"] = obj_long["Estado"].map(estado_map).fillna(0)

# aplicar filtros (opcionales)
obj_long = apply_filter(obj_long, "Tipo", f_tipo_plan)
obj_long = apply_filter(obj_long, "Perspectiva", f_persp)
obj_long = apply_filter(obj_long, "Eje", f_eje)
obj_long = apply_filter(obj_long, "Departamento", f_depto)
obj_long = apply_filter(obj_long, "Tipo Objetivo", f_tipo_obj)
obj_long = apply_filter(obj_long, "Frecuencia Medición", f_freq)

group_cols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in obj_long.columns]
obj_resumen = obj_long.groupby(group_cols, as_index=False).agg(
    score_total=("valor","sum"),
    verdes=("Estado", lambda x: (x=="VERDE").sum()),
    amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum()),
    celdas=("Estado","count")
)

# cálculo de meses esperados según Frecuencia (como tu lógica original)
if "Frecuencia Medición" in obj_resumen.columns:
    obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map).fillna(12)
else:
    obj_resumen["meses_esperados"] = 12

obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1)*100
obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec_row, axis=1)

estado_opts = ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]

# =====================================================
# ÁREAS: LONG + RESÚMENES (AÑO BASE)
# =====================================================
area_id_cols = ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
area_long = normalizar_meses(df_area, area_id_cols)
area_long["valor"] = area_long["Estado"].map(estado_map).fillna(0)

# filtros áreas (opcionales)
area_long = apply_filter(area_long, "AREA", f_area)
area_long = apply_filter(area_long, "PUESTO RESPONSABLE", f_puesto)
area_long = apply_filter(area_long, "DEPARTAMENTO", f_dep_area)
if "TIPO" in area_long.columns:
    area_long = apply_filter(area_long, "TIPO", f_tipo_area)
if "PERSPECTIVA" in area_long.columns:
    area_long = apply_filter(area_long, "PERSPECTIVA", f_pers_area)

# resumen por área
area_res_area = area_long.groupby(["AREA"], as_index=False).agg(
    cumplimiento=("valor","mean"),
    tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum()),
    celdas=("Estado","count")
)
area_res_area["cumplimiento_%"] = area_res_area["cumplimiento"] * 100

# resumen por puesto responsable
area_res_puesto = area_long.groupby(["AREA","PUESTO RESPONSABLE"], as_index=False).agg(
    cumplimiento=("valor","mean"),
    tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
)
area_res_puesto["cumplimiento_%"] = area_res_puesto["cumplimiento"] * 100

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "📌 Resumen",
    "🎯 Objetivos",
    "🏢 Áreas",
    "📊 Comparativo",
    "🚨 Alertas",
    "📄 Exportar",
    "📋 Datos"
])

# =====================================================
# TAB 0: RESUMEN
# =====================================================
with tabs[0]:
    st.subheader(f"📌 Resumen Ejecutivo – Año {year_data}")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Objetivos", int(len(obj_resumen)))
    k2.metric("Cumplidos", int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum()))
    k3.metric("En Riesgo", int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum()))
    k4.metric("Críticos / No subido", int((obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"])).sum()))
    k5.metric("Cumplimiento Promedio", f"{obj_resumen['cumplimiento_%'].mean():.1f}%")

    st.markdown("---")

    g1, g2 = st.columns(2)

    fig_g1 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(obj_resumen["cumplimiento_%"].mean()) if len(obj_resumen) else 0,
        delta={'reference': 90, 'increasing': {'color': '#16A34A'}, 'decreasing': {'color': '#EF4444'}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#111827"},
            "steps":[
                {"range":[0,60],"color":"#FEE2E2"},
                {"range":[60,90],"color":"#FEF3C7"},
                {"range":[90,100],"color":"#DCFCE7"}
            ]
        },
        title={"text": f"{year_data} – Cumplimiento Estratégico (Objetivos)"}
    ))
    g1.plotly_chart(theme_plotly(fig_g1, height=380), use_container_width=True)

    fig_g2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(area_long["valor"].mean()*100) if len(area_long) else 0,
        delta={'reference': 90, 'increasing': {'color': '#16A34A'}, 'decreasing': {'color': '#EF4444'}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#111827"},
            "steps":[
                {"range":[0,60],"color":"#FEE2E2"},
                {"range":[60,90],"color":"#FEF3C7"},
                {"range":[90,100],"color":"#DCFCE7"}
            ]
        },
        title={"text": f"{year_data} – Cumplimiento Operativo (Áreas)"}
    ))
    g2.plotly_chart(theme_plotly(fig_g2, height=380), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🧭 Mix de estados ejecutivos (Objetivos)")
        vc = (obj_resumen["estado_ejecutivo"]
              .value_counts()
              .reindex(estado_opts)
              .fillna(0)
              .rename_axis("Estado")
              .reset_index(name="Cantidad"))

        fig = px.bar(vc, x="Estado", y="Cantidad", text="Cantidad",
                     color="Estado", color_discrete_map=COLOR_EJEC)
        fig.update_traces(textposition="outside")
        fig = theme_plotly(fig, height=FIG_H_MED)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 📈 Tendencia mensual (promedio)")
        tr = obj_long.groupby("Mes")["valor"].mean().reindex(MESES).reset_index()
        tr["cumplimiento_%"] = tr["valor"] * 100
        fig = px.line(tr, x="Mes", y="cumplimiento_%", markers=True)
        fig.update_yaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_MED)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 1: OBJETIVOS
# =====================================================
with tabs[1]:
    st.subheader("🎯 Objetivos – Análisis avanzado")

    r1, r2 = st.columns([1.2, 1])
    with r1:
        st.markdown("### 🔥 Top objetivos críticos (peor cumplimiento)")
        top_bad = obj_resumen.sort_values("cumplimiento_%").head(15)
        fig = px.bar(top_bad, x="cumplimiento_%", y="Objetivo",
                     orientation="h",
                     color="estado_ejecutivo",
                     color_discrete_map=COLOR_EJEC,
                     text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_xaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("### 🧩 Distribución (donut) estados ejecutivos")
        pie_df = obj_resumen["estado_ejecutivo"].value_counts().reindex(estado_opts).fillna(0).reset_index()
        pie_df.columns = ["Estado", "Cantidad"]
        fig = px.pie(pie_df, names="Estado", values="Cantidad", hole=0.55,
                     color="Estado", color_discrete_map=COLOR_EJEC)
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("### 🌡️ Heatmap (Objetivo vs Mes)")
        heat_df = obj_long.pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
        fig = px.imshow(heat_df, aspect="auto", color_continuous_scale=["#EF4444","#F59E0B","#16A34A"])
        fig = theme_plotly(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        st.markdown("### 🏷️ Cumplimiento por criterio (promedio)")
        # Se adapta según lo que exista en tu año
        dims = []
        for c in ["Tipo", "Perspectiva", "Eje", "Departamento", "Tipo Objetivo"]:
            if c in obj_resumen.columns:
                dims.append(c)

        if not dims:
            st.info("No se encontraron columnas de criterios para agrupar.")
        else:
            dim = st.selectbox("Agrupar por", options=dims, index=0)
            g = obj_resumen.groupby(dim)["cumplimiento_%"].mean().reset_index().sort_values("cumplimiento_%")
            fig = px.bar(g, x="cumplimiento_%", y=dim, orientation="h",
                         color="cumplimiento_%", color_continuous_scale="RdYlGn",
                         text="cumplimiento_%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_xaxes(range=[0, 100])
            fig = theme_plotly(fig, height=560)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2: ÁREAS
# =====================================================
with tabs[2]:
    st.subheader("🏢 Áreas – Control operativo")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📉 Ranking crítico de áreas (peor → mejor)")
        rk = area_res_area.sort_values("cumplimiento_%").head(20)
        fig = px.bar(rk, x="cumplimiento_%", y="AREA", orientation="h",
                     color="cumplimiento_%", color_continuous_scale="RdYlGn",
                     text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_xaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### ⚖️ Cumplimiento vs carga (# tareas)")
        sc = area_res_area.copy()
        fig = px.scatter(sc, x="tareas", y="cumplimiento_%", size="tareas",
                         hover_name="AREA")
        fig.update_yaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    d1, d2 = st.columns([1.25, 0.75])
    with d1:
        st.markdown("### 🌡️ Heatmap operativo (Área vs Mes)")
        heat = area_long.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale=["#EF4444","#F59E0B","#16A34A"])
        fig = theme_plotly(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        st.markdown("### 👥 Top puestos con peor promedio")
        worst_p = area_res_puesto.sort_values("cumplimiento_%").head(15)
        fig = px.bar(worst_p, x="cumplimiento_%", y="PUESTO RESPONSABLE",
                     orientation="h", color="AREA",
                     text="cumplimiento_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_xaxes(range=[0, 100])
        fig = theme_plotly(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3: COMPARATIVO
# =====================================================
with tabs[3]:
    st.subheader("📊 Comparativo (años) – explicado y más potente")

    st.info(
        "Este comparativo compara 2024 vs 2025 en dos niveles:\n"
        "1) **Objetivos**: estados ejecutivos (CUMPLIDO/SEGUIMIENTO/RIESGO/CRÍTICO/NO SUBIDO)\n"
        "2) **Colores mensuales**: % de celdas VERDE/AMARILLO/ROJO/MORADO en Objetivos y en Áreas\n"
        "3) **Novedades**: qué criterios/objetivos/áreas aparecen en 2025 y no en 2024\n"
        "4) **Delta por Área**: cambios de cumplimiento (si comparas exactamente 2 años)"
    )

    if len(compare_years) < 2:
        st.warning("Selecciona al menos 2 años en el sidebar.")
    else:
        comp_obj_res = []
        comp_obj_long = []
        comp_area_long = []
        comp_area_res = []

        for y in compare_years:
            o, a = load_year(y)

            # --------- OBJETIVOS ---------
            o = safe_strip(o)
            ol = normalizar_meses(o, ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"])
            # aplica filtros (mismos del año base)
            ol = apply_filter(ol, "Tipo", f_tipo_plan)
            ol = apply_filter(ol, "Perspectiva", f_persp)
            ol = apply_filter(ol, "Eje", f_eje)
            ol = apply_filter(ol, "Departamento", f_depto)
            ol = apply_filter(ol, "Tipo Objetivo", f_tipo_obj)
            ol = apply_filter(ol, "Frecuencia Medición", f_freq)

            ol["valor"] = ol["Estado"].map(estado_map).fillna(0)
            ol["AÑO"] = y
            comp_obj_long.append(ol)

            gcols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in ol.columns]
            ores = ol.groupby(gcols, as_index=False).agg(
                score_total=("valor","sum"),
                verdes=("Estado", lambda x: (x=="VERDE").sum()),
                amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
                rojos=("Estado", lambda x: (x=="ROJO").sum()),
                morados=("Estado", lambda x: (x=="MORADO").sum())
            )
            ores["meses_esperados"] = ores.get("Frecuencia Medición", pd.Series(["Mensual"]*len(ores))).map(frecuencia_map).fillna(12)
            ores["cumplimiento_%"] = (ores["score_total"]/ores["meses_esperados"]).clip(0,1)*100
            ores["estado_ejecutivo"] = ores.apply(estado_exec_row, axis=1)
            ores["AÑO"] = y
            comp_obj_res.append(ores)

            # --------- ÁREAS ---------
            a = safe_strip(a)
            a = unify_dic(a)
            if "PUESTO" in a.columns and "PUESTO RESPONSABLE" not in a.columns:
                a.rename(columns={"PUESTO":"PUESTO RESPONSABLE"}, inplace=True)
            if "AREA" not in a.columns and "DEPARTAMENTO" in a.columns:
                a["AREA"] = a["DEPARTAMENTO"]
            if "DEPARTAMENTO" not in a.columns and "AREA" in a.columns:
                a["DEPARTAMENTO"] = a["AREA"]

            al = normalizar_meses(a, ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO","OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","¿Realizada?"])
            # filtros (mismos del año base)
            al = apply_filter(al, "AREA", f_area)
            al = apply_filter(al, "PUESTO RESPONSABLE", f_puesto)
            al = apply_filter(al, "DEPARTAMENTO", f_dep_area)
            if "TIPO" in al.columns:
                al = apply_filter(al, "TIPO", f_tipo_area)
            if "PERSPECTIVA" in al.columns:
                al = apply_filter(al, "PERSPECTIVA", f_pers_area)

            al["valor"] = al["Estado"].map(estado_map).fillna(0)
            al["AÑO"] = y
            comp_area_long.append(al)

            ar = al.groupby(["AREA"], as_index=False).agg(
                cumplimiento=("valor","mean"),
                tareas=("TAREA","nunique") if "TAREA" in al.columns else ("Estado","count")
            )
            ar["cumplimiento_%"] = ar["cumplimiento"]*100
            ar["AÑO"] = y
            comp_area_res.append(ar)

        comp_obj_long = pd.concat(comp_obj_long, ignore_index=True)
        comp_obj_res  = pd.concat(comp_obj_res, ignore_index=True)
        comp_area_long = pd.concat(comp_area_long, ignore_index=True)
        comp_area_res  = pd.concat(comp_area_res, ignore_index=True)

        # 1) Objetivos: estado ejecutivo por año (conteo y %)
        st.markdown("### 1) 🎯 Objetivos: Estados ejecutivos por año")
        mix = comp_obj_res.groupby(["AÑO","estado_ejecutivo"]).size().reset_index(name="conteo")
        mix["%"] = mix["conteo"] / mix.groupby("AÑO")["conteo"].transform("sum") * 100

        fig = px.bar(
            mix,
            x="AÑO", y="%", color="estado_ejecutivo",
            barmode="group",
            color_discrete_map=COLOR_EJEC,
            text="%",
            category_orders={"estado_ejecutivo": estado_opts}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

        # 2) Objetivos: % por color mensual
        st.markdown("### 2) 🎨 Objetivos: % por color mensual (celdas)")
        obj_color = comp_obj_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
        obj_color["%"] = obj_color["conteo"] / obj_color.groupby("AÑO")["conteo"].transform("sum") * 100

        fig = px.bar(
            obj_color,
            x="AÑO", y="%", color="Estado",
            barmode="group",
            color_discrete_map=COLOR_ESTADO,
            text="%",
            category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

        # 3) Áreas: % por color mensual
        st.markdown("### 3) 🏢 Áreas: % por color mensual (celdas)")
        area_color = comp_area_long.groupby(["AÑO","Estado"]).size().reset_index(name="conteo")
        area_color["%"] = area_color["conteo"] / area_color.groupby("AÑO")["conteo"].transform("sum") * 100

        fig = px.bar(
            area_color,
            x="AÑO", y="%", color="Estado",
            barmode="group",
            color_discrete_map=COLOR_ESTADO,
            text="%",
            category_orders={"Estado":["VERDE","AMARILLO","ROJO","MORADO"]}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(range=[0, 100])
        fig = theme_plotly(fig, height=FIG_H_BIG)
        st.plotly_chart(fig, use_container_width=True)

        # 4) Qué es nuevo en 2025 vs 2024 (si ambos existen en selección)
        st.markdown("### 4) 🆕 Novedades (qué aparece en el año más reciente y no en el anterior)")
        if len(compare_years) >= 2:
            y_new = max(compare_years)
            y_old = min(compare_years)

            o_old, a_old = load_year(y_old)
            o_new, a_new = load_year(y_new)

            # Objetivos nuevos (por nombre de Objetivo)
            set_old_obj = set(o_old.get("Objetivo", pd.Series([])).dropna().astype(str).tolist())
            set_new_obj = set(o_new.get("Objetivo", pd.Series([])).dropna().astype(str).tolist())
            nuevos_obj = sorted(list(set_new_obj - set_old_obj))

            # Dimensiones nuevas
            def new_values(col):
                old = set(o_old.get(col, pd.Series([])).dropna().astype(str).tolist())
                new = set(o_new.get(col, pd.Series([])).dropna().astype(str).tolist())
                return sorted(list(new - old))

            nuevos_dep = new_values("Departamento")
            nuevos_persp = new_values("Perspectiva")
            nuevos_eje = new_values("Eje")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Nuevos Objetivos {y_new}", len(nuevos_obj))
            col2.metric(f"Nuevos Departamentos {y_new}", len(nuevos_dep))
            col3.metric(f"Nuevas Perspectivas {y_new}", len(nuevos_persp))
            col4.metric(f"Nuevos Ejes {y_new}", len(nuevos_eje))

            with st.expander("Ver detalle de novedades"):
                st.markdown(f"**Objetivos nuevos en {y_new} (no estaban en {y_old}):**")
                st.write(nuevos_obj if nuevos_obj else "—")

                st.markdown(f"**Departamentos nuevos en {y_new}:**")
                st.write(nuevos_dep if nuevos_dep else "—")

                st.markdown(f"**Perspectivas nuevas en {y_new}:**")
                st.write(nuevos_persp if nuevos_persp else "—")

                st.markdown(f"**Ejes nuevos en {y_new}:**")
                st.write(nuevos_eje if nuevos_eje else "—")

        # 5) Delta por Área (solo si comparas 2 años)
        st.markdown("### 5) 📈 Delta de cumplimiento por Área (si comparas 2 años)")
        if len(compare_years) == 2:
            y1, y2 = sorted(compare_years)
            a1 = comp_area_res[comp_area_res["AÑO"] == y1][["AREA","cumplimiento_%"]].rename(columns={"cumplimiento_%": f"cumpl_{y1}"})
            a2 = comp_area_res[comp_area_res["AÑO"] == y2][["AREA","cumplimiento_%"]].rename(columns={"cumplimiento_%": f"cumpl_{y2}"})
            m = a1.merge(a2, on="AREA", how="inner")
            m["delta"] = m[f"cumpl_{y2}"] - m[f"cumpl_{y1}"]
            m = m.sort_values("delta").head(25)

            fig = px.bar(m, x="delta", y="AREA", orientation="h",
                         color="delta", color_continuous_scale="RdYlGn",
                         title=f"Cambio de cumplimiento {y1} → {y2}")
            fig = theme_plotly(fig, height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Selecciona exactamente 2 años para ver delta por área.")

# =====================================================
# TAB 4: ALERTAS (TABLA SEMÁFORO)
# =====================================================
with tabs[4]:
    st.subheader("🚨 Alertas automáticas (tabla semáforo)")

    alert_rows = []

    # Objetivos críticos / riesgo / no subido
    crit_obj = obj_resumen[obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])].copy()
    for _, r in crit_obj.iterrows():
        sev = "CRÍTICA" if r["estado_ejecutivo"] in ["CRÍTICO","NO SUBIDO"] else "NORMAL"
        alert_rows.append({
            "Año": year_data,
            "Nivel": sev,
            "Tipo": "Objetivo",
            "Nombre": r.get("Objetivo",""),
            "Estado": r["estado_ejecutivo"],
            "Cumplimiento %": round(pct(r["cumplimiento_%"]), 1)
        })

    # Áreas con bajo cumplimiento
    bad_areas = area_res_area[area_res_area["cumplimiento_%"] < 60].copy()
    for _, r in bad_areas.iterrows():
        sev = "CRÍTICA" if r["cumplimiento_%"] < 40 else "NORMAL"
        alert_rows.append({
            "Año": year_data,
            "Nivel": sev,
            "Tipo": "Área",
            "Nombre": r["AREA"],
            "Estado": "BAJO CUMPLIMIENTO",
            "Cumplimiento %": round(pct(r["cumplimiento_%"]), 1)
        })

    alerts_df = pd.DataFrame(alert_rows)

    if alerts_df.empty:
        st.success("✅ Sin alertas con los filtros actuales.")
    else:
        alerts_df = alerts_df.sort_values(["Nivel","Cumplimiento %"], ascending=[True, True])

        def semaforo(row):
            if row["Nivel"] == "CRÍTICA":
                bg = "background-color: #FEE2E2;"  # rojo claro
            else:
                bg = "background-color: #FEF3C7;"  # amarillo claro
            return [bg] * len(row)

        st.dataframe(alerts_df.style.apply(semaforo, axis=1), use_container_width=True)

# =====================================================
# TAB 5: EXPORTAR (HTML para imprimir a PDF)
# =====================================================
with tabs[5]:
    st.subheader("📄 Exportar reporte (HTML con gráficas → imprimir a PDF)")

    # Gráficas clave para export (se regeneran acá)
    pie_df = obj_resumen["estado_ejecutivo"].value_counts().reindex(estado_opts).fillna(0).reset_index()
    pie_df.columns = ["Estado", "Cantidad"]
    fig_estado_exec = px.pie(pie_df, names="Estado", values="Cantidad", hole=0.55,
                             color="Estado", color_discrete_map=COLOR_EJEC,
                             title=f"{year_data} – Estados ejecutivos (Objetivos)")
    fig_estado_exec = theme_plotly(fig_estado_exec, height=520)

    rank_areas = area_res_area.sort_values("cumplimiento_%").head(20)
    fig_rank_areas = px.bar(rank_areas, x="cumplimiento_%", y="AREA", orientation="h",
                            text="cumplimiento_%", color="cumplimiento_%",
                            color_continuous_scale="RdYlGn",
                            title=f"{year_data} – Ranking crítico de áreas")
    fig_rank_areas.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_rank_areas.update_xaxes(range=[0, 100])
    fig_rank_areas = theme_plotly(fig_rank_areas, height=520)

    def build_report_html():
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        k_obj = int(len(obj_resumen))
        k_ok  = int((obj_resumen["estado_ejecutivo"]=="CUMPLIDO").sum())
        k_r   = int((obj_resumen["estado_ejecutivo"]=="RIESGO").sum())
        k_c   = int((obj_resumen["estado_ejecutivo"].isin(["CRÍTICO","NO SUBIDO"])).sum())
        k_avg = float(obj_resumen["cumplimiento_%"].mean()) if len(obj_resumen) else 0

        html = f"""
        <html><head><meta charset="utf-8"/>
        <title>Reporte Estratégico</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 22px; background:#fff; color:#111; }}
          .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin: 14px 0 18px; }}
          .kpi {{ border:1px solid #EAECEF; padding:12px 14px; border-radius:12px; min-width:170px; }}
          .muted {{ color:#666; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
          th {{ background: #f5f5f5; }}
          h1,h2 {{ margin-bottom: 8px; }}
          hr {{ border:none; border-top:1px solid #EAECEF; margin: 16px 0; }}
        </style>
        </head><body>
        <h1>Reporte Estratégico y de Control</h1>
        <div class="muted">Año: {year_data} · Generado: {now}</div>

        <hr/>
        <h2>KPIs</h2>
        <div class="kpis">
          <div class="kpi"><b>Objetivos</b><br>{k_obj}</div>
          <div class="kpi"><b>Cumplidos</b><br>{k_ok}</div>
          <div class="kpi"><b>En riesgo</b><br>{k_r}</div>
          <div class="kpi"><b>Críticos/No subido</b><br>{k_c}</div>
          <div class="kpi"><b>Cumplimiento promedio</b><br>{k_avg:.1f}%</div>
        </div>

        <hr/>
        <h2>Gráficas</h2>
        {fig_estado_exec.to_html(full_html=False, include_plotlyjs="cdn")}
        <br/>
        {fig_rank_areas.to_html(full_html=False, include_plotlyjs=False)}

        <hr/>
        <h2>Alertas</h2>
        """
        if 'alerts_df' in globals() and isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty:
            html += alerts_df.to_html(index=False)
        else:
            html += "<p>Sin alertas.</p>"

        html += f"""
        <hr/>
        <h2>Tabla: Objetivos (resumen)</h2>
        {obj_resumen.head(250).to_html(index=False)}

        <hr/>
        <h2>Tabla: Áreas (resumen)</h2>
        {area_res_area.head(250).to_html(index=False)}

        <p class="muted">Tip: Abre este HTML en Chrome/Edge → Ctrl+P → Guardar como PDF.</p>
        </body></html>
        """
        return html

    html_report = build_report_html()
    st.download_button(
        "⬇️ Descargar Reporte HTML",
        data=html_report,
        file_name=f"Reporte_Estrategico_{year_data}.html",
        mime="text/html"
    )
    st.info("Luego abre el HTML en Chrome/Edge → Ctrl+P → Guardar como PDF (queda como informe formal).")

# =====================================================
# TAB 6: DATOS (AUDITORÍA)
# =====================================================
with tabs[6]:
    st.subheader("📋 Datos (auditoría / trazabilidad)")
    with st.expander("Objetivos – Resumen"):
        st.dataframe(obj_resumen, use_container_width=True)
    with st.expander("Objetivos – Long"):
        st.dataframe(obj_long, use_container_width=True)
    with st.expander("Áreas – Resumen"):
        st.dataframe(area_res_area, use_container_width=True)
    with st.expander("Áreas – Long"):
        st.dataframe(area_long, use_container_width=True)

st.caption("Fuente: Google Sheets · Dashboard Estratégico")


