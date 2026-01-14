import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Dashboard Estratégico 2023",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Dashboard Estratégico y de Control 2023")

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

    df_obj.columns = df_obj.columns.str.strip().str.replace("\n", " ")
    df_area.columns = df_area.columns.str.strip().str.replace("\n", " ")

    df_area.rename(columns={
        "Área": "AREA",
        "Puesto Responsable": "PUESTO RESPONSABLE",
        "Realizada?": "¿Realizada?"
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
# RESUMEN ESTRATÉGICO (HOJA 2023)
# =====================================================
obj_resumen = obj_long.groupby(
    ["Objetivo","Tipo Objetivo","Frecuencia Medición"],
    as_index=False
).agg(
    meses_reportados=("Mes","count"),
    score_total=("valor","sum"),
    verdes=("Estado", lambda x:(x=="VERDE").sum()),
    amarillos=("Estado", lambda x:(x=="AMARILLO").sum()),
    rojos=("Estado", lambda x:(x=="ROJO").sum()),
    morados=("Estado", lambda x:(x=="MORADO").sum())
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"]/obj_resumen["meses_esperados"]).clip(0,1)*100

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
# RESUMEN OPERATIVO (HOJA 2023 AREAS)
# =====================================================
area_resumen = (
    area_long
    .groupby("AREA", as_index=False)
    .agg(
        score=("valor","mean"),
        rojos=("Estado", lambda x:(x=="ROJO").sum()),
        morados=("Estado", lambda x:(x=="MORADO").sum())
    )
)

# =====================================================
# FILTROS
# =====================================================
st.sidebar.header("🔎 Filtros")

filtro_area = st.sidebar.multiselect(
    "Área",
    options=area_long["AREA"].unique(),
    default=area_long["AREA"].unique()
)

filtro_tipo = st.sidebar.multiselect(
    "Tipo Objetivo",
    options=obj_resumen["Tipo Objetivo"].unique(),
    default=obj_resumen["Tipo Objetivo"].unique()
)

obj_resumen_f = obj_resumen[obj_resumen["Tipo Objetivo"].isin(filtro_tipo)]
area_long_f = area_long[area_long["AREA"].isin(filtro_area)]

# =====================================================
# KPIs GENERALES
# =====================================================
st.subheader("📌 Indicadores Clave")

k1,k2,k3,k4 = st.columns(4)

k1.metric("Objetivos", obj_resumen_f.shape[0])
k2.metric("Cumplidos", (obj_resumen_f["estado_ejecutivo"]=="CUMPLIDO").sum())
k3.metric("En Riesgo", (obj_resumen_f["estado_ejecutivo"]=="RIESGO").sum())
k4.metric("No Subidos", (obj_resumen_f["estado_ejecutivo"]=="NO SUBIDO").sum())

# =====================================================
# MEDIDORES (LO QUE TE GUSTÓ)
# =====================================================
st.subheader("🎯 Medidores Ejecutivos")

g1, g2 = st.columns(2)

cum_estrategico = obj_resumen_f["cumplimiento_%"].mean()
cum_operativo = area_resumen["score"].mean() * 100

with g1:
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cum_estrategico,
        title={"text":"Cumplimiento Estratégico 2023"},
        gauge={
            "axis":{"range":[0,100]},
            "steps":[
                {"range":[0,60],"color":"#ff4d4d"},
                {"range":[60,90],"color":"#ffd966"},
                {"range":[90,100],"color":"#70ad47"}
            ]
        }
    ))
    st.plotly_chart(fig1,use_container_width=True)

with g2:
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cum_operativo,
        title={"text":"Cumplimiento Operativo 2023 AREAS"},
        gauge={
            "axis":{"range":[0,100]},
            "steps":[
                {"range":[0,60],"color":"#ff4d4d"},
                {"range":[60,90],"color":"#ffd966"},
                {"range":[90,100],"color":"#70ad47"}
            ]
        }
    ))
    st.plotly_chart(fig2,use_container_width=True)

# =====================================================
# VISUALIZACIONES ESTRATÉGICAS
# =====================================================
st.header("📘 Análisis Estratégico – Hoja 2023")

with st.expander("📊 Estado y Tendencias Estratégicas", expanded=True):

    st.plotly_chart(
        px.bar(
            obj_resumen_f,
            x="estado_ejecutivo",
            y="meses_reportados",
            color="estado_ejecutivo",
            title="Objetivos por Estado Ejecutivo"
        ),
        use_container_width=True
    )

    heat_df = obj_long.pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
    st.plotly_chart(
        px.imshow(
            heat_df,
            color_continuous_scale=["red","yellow","green"],
            title="Heatmap de Cumplimiento Estratégico"
        ),
        use_container_width=True
    )

    mensual = obj_long.groupby("Mes")["valor"].mean().reindex(MESES)
    st.plotly_chart(
        px.line(
            mensual,
            markers=True,
            title="Tendencia Mensual Estratégica"
        ),
        use_container_width=True
    )

# =====================================================
# VISUALIZACIONES OPERATIVAS
# =====================================================
st.header("📗 Análisis Operativo – Hoja 2023 AREAS")

with st.expander("📊 Ejecución y Ranking de Áreas", expanded=True):

    st.plotly_chart(
        px.bar(
            area_resumen.sort_values("score"),
            x="score",
            y="AREA",
            orientation="h",
            title="Ranking de Áreas Críticas"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(
            area_long_f.groupby("PUESTO RESPONSABLE")["valor"].mean().reset_index(),
            x="PUESTO RESPONSABLE",
            y="valor",
            title="Cumplimiento por Responsable"
        ),
        use_container_width=True
    )

# =====================================================
# ALERTAS
# =====================================================
st.header("🚨 Alertas Ejecutivas")

alertas = obj_resumen_f[
    obj_resumen_f["estado_ejecutivo"].isin(["RIESGO","CRÍTICO","NO SUBIDO"])
]

st.dataframe(alertas, use_container_width=True)

# =====================================================
# TABLAS DETALLE
# =====================================================
st.header("📋 Tablas Detalle")

with st.expander("Datos Normalizados – Objetivos"):
    st.dataframe(obj_long,use_container_width=True)

with st.expander("Datos Normalizados – Áreas y Tareas"):
    st.dataframe(area_long,use_container_width=True)

st.caption("Fuente: Google Sheets · Actualización automática cada 5 minutos")


