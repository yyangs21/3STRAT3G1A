import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Dashboard Estratégico 2023",
    layout="wide"
)

st.title("📊 Dashboard Estratégico y de Control 2023")

# =====================================================
# AUTENTICACIÓN GOOGLE SHEETS (STREAMLIT SECRETS)
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# REEMPLAZA ST.SECRETS CON REEMPLAZO DE SALTOS DE LÍNEA
service_account_info = dict(st.secrets["gcp_service_account"])
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

CREDS = Credentials.from_service_account_info(
    service_account_info,
    scopes=SCOPES
)

client = gspread.authorize(CREDS)

# 👉 EDITA SOLO ESTO (nombre exacto del Google Sheets)
SHEET_NAME = "DATAESTRATEGIA"

# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data(ttl=300)
def load_data():
    sh = client.open(SHEET_NAME)
    df_obj = pd.DataFrame(sh.worksheet("2023").get_all_records())
    df_area = pd.DataFrame(sh.worksheet("2023 AREAS").get_all_records())
    
    # LIMPIEZA BÁSICA DE COLUMNAS: quita espacios y reemplaza saltos de línea
    df_obj.columns = df_obj.columns.str.strip().str.replace("\n", " ")
    df_area.columns = df_area.columns.str.strip().str.replace("\n", " ")
    
    # RENOMBRAR columnas críticas para evitar KeyError
    df_area.rename(columns={
        "Área": "AREA",
        "Realizada?": "¿Realizada?",
        "Puesto Responsable": "PUESTO RESPONSABLE"
    }, inplace=True)
    
    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {
    "VERDE": 1,
    "AMARILLO": 0.5,
    "ROJO": 0,
    "MORADO": 0  # Nuevo: MORADO = No subido
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
# NORMALIZACIÓN DE MESES
# =====================================================
def normalizar_meses(df, id_cols):
    # Validar columnas
    faltantes = [c for c in id_cols if c not in df.columns]
    if faltantes:
        raise KeyError(f"Columnas faltantes en df: {faltantes}")
    
    # Filtrar meses que existen en df
    meses_presentes = [m for m in MESES if m in df.columns]
    
    return df.melt(
        id_vars=id_cols,
        value_vars=meses_presentes,
        var_name="Mes",
        value_name="Estado"
    ).dropna(subset=["Estado"])

# Normalizar datos de objetivos y áreas
obj_long = normalizar_meses(
    df_obj,
    ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
)

area_long = normalizar_meses(
    df_area,
    ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
)

# =====================================================
# MAPEO DE ESTADOS
# =====================================================
obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# CUMPLIMIENTO ESTRATÉGICO POR OBJETIVO
# =====================================================
obj_resumen = (
    obj_long
    .groupby(
        ["Objetivo", "Tipo Objetivo", "Frecuencia Medición"],
        as_index=False
    )
    .agg(
        meses_reportados=("Mes", "count"),
        score_total=("valor", "sum"),
        verdes=("Estado", lambda x: (x == "VERDE").sum()),
        amarillos=("Estado", lambda x: (x == "AMARILLO").sum()),
        rojos=("Estado", lambda x: (x == "ROJO").sum()),
        morados=("Estado", lambda x: (x == "MORADO").sum())  # nuevo
    )
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)

obj_resumen["cumplimiento_%"] = (
    obj_resumen["score_total"] / obj_resumen["meses_esperados"]
).clip(0, 1) * 100

obj_resumen["riesgo"] = obj_resumen["rojos"] > 0

# =====================================================
# CLASIFICACIÓN EJECUTIVA
# =====================================================
def clasificar_estado(row):
    if row["morados"] > 0:
        return "NO SUBIDO"
    if row["riesgo"]:
        return "RIESGO"
    if row["cumplimiento_%"] >= 90:
        return "CUMPLIDO"
    if row["cumplimiento_%"] >= 60:
        return "EN SEGUIMIENTO"
    return "CRÍTICO"

obj_resumen["estado_ejecutivo"] = obj_resumen.apply(clasificar_estado, axis=1)

# =====================================================
# KPIs GENERALES
# =====================================================
st.subheader("📌 Indicadores Generales")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Objetivos", obj_resumen.shape[0])
c2.metric("Cumplidos", (obj_resumen["estado_ejecutivo"] == "CUMPLIDO").sum())
c3.metric("En Riesgo", (obj_resumen["estado_ejecutivo"] == "RIESGO").sum())
c4.metric("Cumplimiento Promedio", f"{obj_resumen['cumplimiento_%'].mean():.1f}%")

# =====================================================
# TABLA ESTRATÉGICA
# =====================================================
st.header("🎯 Cumplimiento Estratégico por Objetivo")
st.dataframe(
    obj_resumen.sort_values("cumplimiento_%", ascending=False),
    use_container_width=True
)

# =====================================================
# GRÁFICAS INTERACTIVAS
# =====================================================
st.header("📈 Análisis Visual del Cumplimiento Estratégico")

# 1️⃣ Conteo de objetivos por estado
fig1 = px.bar(
    obj_resumen,
    x="estado_ejecutivo",
    y="meses_reportados",
    color="estado_ejecutivo",
    color_discrete_map={
        "CUMPLIDO": "green",
        "EN SEGUIMIENTO": "yellow",
        "RIESGO": "red",
        "CRÍTICO": "darkred",
        "NO SUBIDO": "purple"
    },
    text="meses_reportados",
    title="Objetivos por Estado Ejecutivo"
)
fig1.update_layout(yaxis_title="Objetivos", xaxis_title="Estado")
st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ Heatmap de cumplimiento por objetivo y mes
heat_df = obj_long.pivot_table(
    index="Objetivo",
    columns="Mes",
    values="valor",
    fill_value=0
)

fig2 = px.imshow(
    heat_df,
    labels=dict(x="Mes", y="Objetivo", color="Valor"),
    color_continuous_scale=["red","yellow","green"],
    title="Heatmap de Cumplimiento Mensual por Objetivo"
)
st.plotly_chart(fig2, use_container_width=True)

# 3️⃣ Cumplimiento promedio mensual
cum_mensual = obj_long.groupby("Mes")["valor"].mean().reindex(MESES)

fig3 = px.line(
    cum_mensual,
    y="valor",
    x=cum_mensual.index,
    title="Cumplimiento Promedio por Mes",
    markers=True
)
fig3.update_yaxes(range=[0,1], tickformat=".0%")
st.plotly_chart(fig3, use_container_width=True)

# 4️⃣ Cumplimiento por Área
area_summary = area_long.groupby("AREA")["valor"].mean().sort_values(ascending=False).reset_index()

fig4 = px.bar(
    area_summary,
    x="AREA",
    y="valor",
    color="valor",
    color_continuous_scale="Viridis",
    title="Cumplimiento Promedio por Área"
)
st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# TABLAS DE CONTROL / TRANSPARENCIA
# =====================================================
with st.expander("📋 Datos normalizados - Objetivos"):
    st.dataframe(obj_long, use_container_width=True)

with st.expander("📋 Datos normalizados - Áreas y Tareas"):
    st.dataframe(area_long, use_container_width=True)

st.caption("Fuente: Google Sheets · Actualización automática cada 5 minutos")



