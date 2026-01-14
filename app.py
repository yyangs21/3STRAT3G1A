import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(page_title="Tablero de Control Estratégico 2023", layout="wide")
st.title("📊 Tablero de Control Estratégico y Operativo 2023")

# =====================================================
# AUTENTICACIÓN GOOGLE SHEETS
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

service_account_info = dict(st.secrets["gcp_service_account"])
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
CREDS = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
client = gspread.authorize(CREDS)

SHEET_NAME = "DATAESTRATEGIA"

# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data(ttl=300)
def load_data():
    sh = client.open(SHEET_NAME)

    df_obj = pd.DataFrame(sh.worksheet("2023").get_all_records())
    df_area = pd.DataFrame(sh.worksheet("2023 AREAS").get_all_records())

    df_obj.columns = df_obj.columns.str.strip()
    df_area.columns = df_area.columns.str.strip()

    df_area.rename(columns={
        "Área": "AREA",
        "Realizada?": "¿Realizada?",
        "Puesto Responsable": "PUESTO RESPONSABLE"
    }, inplace=True)

    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIGURACIONES
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {
    "VERDE": 1,
    "AMARILLO": 0.5,
    "ROJO": 0,
    "MORADO": 0
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
# NORMALIZAR MESES
# =====================================================
def normalizar_meses(df, id_cols):
    meses_presentes = [m for m in MESES if m in df.columns]
    return (
        df.melt(
            id_vars=id_cols,
            value_vars=meses_presentes,
            var_name="Mes",
            value_name="Estado"
        )
        .dropna(subset=["Estado"])
    )

obj_long = normalizar_meses(
    df_obj,
    ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
)

area_long = normalizar_meses(
    df_area,
    ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
)

obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# RESUMEN ESTRATÉGICO (OBJETIVOS)
# =====================================================
obj_resumen = obj_long.groupby(
    ["Objetivo","Tipo Objetivo","Frecuencia Medición"], as_index=False
).agg(
    score_total=("valor","sum"),
    meses_reportados=("Mes","count"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1) * 100

def clasificar_estado(row):
    if row["morados"] > 0:
        return "NO SUBIDO"
    if row["rojos"] > 0:
        return "RIESGO"
    if row["cumplimiento_%"] >= 90:
        return "CUMPLIDO"
    if row["cumplimiento_%"] >= 60:
        return "EN SEGUIMIENTO"
    return "CRÍTICO"

obj_resumen["estado_ejecutivo"] = obj_resumen.apply(clasificar_estado, axis=1)

# =====================================================
# RESUMEN OPERATIVO (ÁREAS)
# =====================================================
area_resumen = area_long.groupby(
    ["AREA"], as_index=False
).agg(
    cumplimiento_%=("valor","mean")
)

area_resumen["cumplimiento_%"] *= 100

# =====================================================
# KPIs + GAUGES (DOBLES)
# =====================================================
st.subheader("🎯 Indicadores Clave")

c1, c2 = st.columns(2)

# --- Gauge Estratégico
cum_obj = obj_resumen["cumplimiento_%"].mean()

fig_obj = go.Figure(go.Indicator(
    mode="gauge+number",
    value=cum_obj,
    gauge={
        "axis": {"range":[0,100]},
        "bar": {"color":"green"},
        "steps":[
            {"range":[0,60],"color":"red"},
            {"range":[60,90],"color":"yellow"},
            {"range":[90,100],"color":"green"}
        ]
    },
    title={"text":"Cumplimiento Estratégico (Objetivos)"}
))
c1.plotly_chart(fig_obj, use_container_width=True)

# --- Gauge Operativo
cum_area = area_resumen["cumplimiento_%"].mean()

fig_area = go.Figure(go.Indicator(
    mode="gauge+number",
    value=cum_area,
    gauge={
        "axis": {"range":[0,100]},
        "bar": {"color":"blue"},
        "steps":[
            {"range":[0,60],"color":"red"},
            {"range":[60,90],"color":"yellow"},
            {"range":[90,100],"color":"green"}
        ]
    },
    title={"text":"Cumplimiento Operativo (Áreas)"}
))
c2.plotly_chart(fig_area, use_container_width=True)

# =====================================================
# ALERTAS AUTOMÁTICAS
# =====================================================
st.subheader("🚨 Alertas Automáticas")

alertas = []

if (obj_resumen["estado_ejecutivo"]=="CRÍTICO").any():
    alertas.append("🔴 Existen objetivos en estado CRÍTICO")

if (obj_resumen["estado_ejecutivo"]=="NO SUBIDO").any():
    alertas.append("🟣 Existen objetivos NO SUBIDOS (MORADO)")

if (area_resumen["cumplimiento_%"] < 60).any():
    alertas.append("🔴 Existen áreas con ejecución menor al 60%")

if alertas:
    for a in alertas:
        st.error(a)
else:
    st.success("✅ No se detectan alertas críticas")

# =====================================================
# VISUALIZACIONES (NO SATURADAS)
# =====================================================
st.header("📈 Visualizaciones Ejecutivas")

with st.expander("📊 Estado Ejecutivo de Objetivos"):
    fig_estado = px.bar(
        obj_resumen,
        x="estado_ejecutivo",
        title="Objetivos por Estado Ejecutivo",
        color="estado_ejecutivo",
        color_discrete_map={
            "CUMPLIDO":"green",
            "EN SEGUIMIENTO":"yellow",
            "RIESGO":"red",
            "CRÍTICO":"darkred",
            "NO SUBIDO":"purple"
        }
    )
    st.plotly_chart(fig_estado, use_container_width=True)

with st.expander("📊 Ranking de Áreas Críticas"):
    fig_rank = px.bar(
        area_resumen.sort_values("cumplimiento_%"),
        x="cumplimiento_%",
        y="AREA",
        orientation="h",
        title="Ranking de Cumplimiento por Área"
    )
    st.plotly_chart(fig_rank, use_container_width=True)

with st.expander("📊 Tendencia Mensual Estratégica"):
    tendencia = obj_long.groupby("Mes")["valor"].mean().reindex(MESES)
    fig_trend = px.line(
        tendencia,
        markers=True,
        title="Tendencia Mensual de Cumplimiento Estratégico"
    )
    fig_trend.update_yaxes(range=[0,1], tickformat=".0%")
    st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# TABLAS DETALLE
# =====================================================
st.header("📋 Detalle Operativo")

with st.expander("Objetivos – Detalle"):
    st.dataframe(obj_resumen, use_container_width=True)

with st.expander("Áreas – Detalle"):
    st.dataframe(area_resumen, use_container_width=True)

st.caption("Fuente: Google Sheets · Actualización automática cada 5 minutos")
