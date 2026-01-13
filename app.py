import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Dashboard Estratégico 2023",
    layout="wide"
)

st.title("📊 Dashboard Estratégico y de Control 2023")

# =====================================================
# AUTENTICACIÓN GOOGLE SHEETS
# =====================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

CREDS = Credentials.from_service_account_file(
    "credentials/service_account.json",
    scopes=SCOPES
)

client = gspread.authorize(CREDS)

# 👉 EDITA SOLO ESTO
SHEET_NAME = "DATAESTRATEGIA"

# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data(ttl=300)
def load_data():
    sh = client.open(SHEET_NAME)
    df_obj = pd.DataFrame(sh.worksheet("2023").get_all_records())
    df_area = pd.DataFrame(sh.worksheet("2023 AREAS").get_all_records())
    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {
    "VERDE": 1,
    "AMARILLO": 0.5,
    "ROJO": 0
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
    return df.melt(
        id_vars=id_cols,
        value_vars=MESES,
        var_name="Mes",
        value_name="Estado"
    ).dropna(subset=["Estado"])

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
        rojos=("Estado", lambda x: (x == "ROJO").sum())
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
# TABLAS DE CONTROL (DEBUG / TRANSPARENCIA)
# =====================================================
with st.expander("📋 Datos normalizados - Objetivos"):
    st.dataframe(obj_long, use_container_width=True)

with st.expander("📋 Datos normalizados - Áreas y Tareas"):
    st.dataframe(area_long, use_container_width=True)

st.caption("Fuente: Google Sheets · Actualización automática cada 5 minutos")
