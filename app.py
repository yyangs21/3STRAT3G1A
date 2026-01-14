import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(page_title="Dashboard Estratégico 2023", layout="wide")
st.title("📊 Dashboard Estratégico y de Control 2023 – ULTRA PRO")

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
# NORMALIZACIÓN
# =====================================================
def normalizar_meses(df, id_cols):
    meses = [m for m in MESES if m in df.columns]
    return df.melt(
        id_vars=id_cols,
        value_vars=meses,
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

obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# RESUMEN OBJETIVOS
# =====================================================
obj_resumen = obj_long.groupby(
    ["Objetivo","Tipo Objetivo","Frecuencia Medición"],
    as_index=False
).agg(
    score_total=("valor","sum"),
    verdes=("Estado", lambda x: (x=="VERDE").sum()),
    amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1)*100

def clasificar_estado(r):
    if r["morados"] > 0: return "NO SUBIDO"
    if r["rojos"] > 0: return "RIESGO"
    if r["cumplimiento_%"] >= 90: return "CUMPLIDO"
    if r["cumplimiento_%"] >= 60: return "EN SEGUIMIENTO"
    return "CRÍTICO"

obj_resumen["estado_ejecutivo"] = obj_resumen.apply(clasificar_estado, axis=1)

# =====================================================
# FILTROS (NO BLOQUEAN DATA)
# =====================================================
st.sidebar.header("🔎 Filtros (opcionales)")
filtro_estado = st.sidebar.multiselect(
    "Estado Ejecutivo",
    obj_resumen["estado_ejecutivo"].unique(),
    default=list(obj_resumen["estado_ejecutivo"].unique())
)

filtro_area = st.sidebar.multiselect(
    "Área",
    area_long["AREA"].unique(),
    default=list(area_long["AREA"].unique())
)

obj_f = obj_resumen[obj_resumen["estado_ejecutivo"].isin(filtro_estado)]
area_f = area_long[area_long["AREA"].isin(filtro_area)]

# =====================================================
# KPIs + DOS GAUGES
# =====================================================
st.subheader("📌 Indicadores Estratégicos")

k1,k2,k3,k4,k5 = st.columns(5)

k1.metric("Objetivos", obj_f.shape[0])
k2.metric("Cumplidos", (obj_f["estado_ejecutivo"]=="CUMPLIDO").sum())
k3.metric("En Riesgo", (obj_f["estado_ejecutivo"]=="RIESGO").sum())
k4.metric("No Subidos", (obj_f["estado_ejecutivo"]=="NO SUBIDO").sum())
k5.metric("Cumplimiento Promedio", f"{obj_f['cumplimiento_%'].mean():.1f}%")

g1,g2 = st.columns(2)

# Gauge Objetivos
fig_g1 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=obj_f["cumplimiento_%"].mean(),
    gauge={'axis': {'range':[0,100]},
           'steps':[
               {'range':[0,60],'color':'red'},
               {'range':[60,90],'color':'yellow'},
               {'range':[90,100],'color':'green'}
           ]},
    title={'text':'Cumplimiento Estratégico – Objetivos'}
))
g1.plotly_chart(fig_g1, use_container_width=True)

# Gauge Áreas
fig_g2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=area_f["valor"].mean()*100,
    gauge={'axis': {'range':[0,100]},
           'steps':[
               {'range':[0,60],'color':'red'},
               {'range':[60,90],'color':'yellow'},
               {'range':[90,100],'color':'green'}
           ]},
    title={'text':'Cumplimiento Operativo – Áreas'}
))
g2.plotly_chart(fig_g2, use_container_width=True)

# =====================================================
# ALERTAS AUTOMÁTICAS
# =====================================================
st.subheader("🚨 Alertas Automáticas")

alertas = []

for _, r in obj_f.iterrows():
    if r["estado_ejecutivo"] in ["CRÍTICO","RIESGO","NO SUBIDO"]:
        alertas.append(f"⚠️ Objetivo **{r['Objetivo']}** en estado **{r['estado_ejecutivo']}**")

for area in area_f.groupby("AREA")["valor"].mean().items():
    if area[1] < 0.6:
        alertas.append(f"🔴 Área **{area[0]}** con bajo cumplimiento")

if alertas:
    for a in alertas:
        st.warning(a)
else:
    st.success("✅ No se detectan alertas críticas")

# =====================================================
# VISUALIZACIONES (SE MANTIENEN + MÁS)
# =====================================================
st.header("📈 Análisis Visual Ejecutivo")

with st.expander("📊 Distribución de Estados (Objetivos)"):
    fig = px.pie(obj_f, names="estado_ejecutivo", title="Estados Ejecutivos")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📉 Desviación vs Esperado"):
    df_dev = obj_f.copy()
    df_dev["desviacion"] = df_dev["cumplimiento_%"] - 100
    fig = px.bar(df_dev.sort_values("desviacion"),
                 x="desviacion", y="Objetivo",
                 orientation="h",
                 color="desviacion",
                 color_continuous_scale="RdYlGn")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🔥 Objetivos Críticos"):
    crit = obj_f[obj_f["estado_ejecutivo"].isin(["CRÍTICO","RIESGO"])]
    fig = px.bar(crit, x="cumplimiento_%", y="Objetivo",
                 orientation="h", color="estado_ejecutivo")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Ranking de Áreas"):
    area_rank = area_f.groupby("AREA")["valor"].mean().sort_values()
    fig = px.bar(area_rank, x=area_rank.values, y=area_rank.index,
                 orientation="h", title="Ranking de Cumplimiento por Área")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🌡️ Heatmap Operativo"):
    heat = area_f.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
    fig = px.imshow(heat, color_continuous_scale=["red","yellow","green"])
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TABLAS
# =====================================================
st.header("📋 Detalle de Datos")

with st.expander("Objetivos"):
    st.dataframe(obj_f)

with st.expander("Áreas y Tareas"):
    st.dataframe(area_f)

st.caption("Fuente: Google Sheets · Dashboard Estratégico 2023")
