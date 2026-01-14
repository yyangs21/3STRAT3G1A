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
st.title("📊 Dashboard Estratégico y de Control 2023 - ULTRA PRO")

# =====================================================
# AUTENTICACIÓN GOOGLE SHEETS
# =====================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

service_account_info = dict(st.secrets["gcp_service_account"])
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n","\n")
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
    
    df_obj.columns = df_obj.columns.str.strip().str.replace("\n"," ")
    df_area.columns = df_area.columns.str.strip().str.replace("\n"," ")
    
    df_area.rename(columns={
        "Área":"AREA",
        "Realizada?":"¿Realizada?",
        "Puesto Responsable":"PUESTO RESPONSABLE"
    }, inplace=True)
    
    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
estado_map = {"VERDE":1,"AMARILLO":0.5,"ROJO":0,"MORADO":0}
frecuencia_map = {"Mensual":12,"Bimestral":6,"Trimestral":4,"Cuatrimestral":3,"Semestral":2,"Anual":1}

# =====================================================
# NORMALIZAR MESES
# =====================================================
def normalizar_meses(df, id_cols):
    faltantes = [c for c in id_cols if c not in df.columns]
    if faltantes:
        raise KeyError(f"Columnas faltantes: {faltantes}")
    meses_presentes = [m for m in MESES if m in df.columns]
    return df.melt(id_vars=id_cols, value_vars=meses_presentes, var_name="Mes", value_name="Estado").dropna(subset=["Estado"])

obj_long = normalizar_meses(df_obj, ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"])
area_long = normalizar_meses(df_area, ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"])

obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# RESUMEN OBJETIVOS
# =====================================================
obj_resumen = obj_long.groupby(["Objetivo","Tipo Objetivo","Frecuencia Medición"], as_index=False).agg(
    meses_reportados=("Mes","count"),
    score_total=("valor","sum"),
    verdes=("Estado", lambda x: (x=="VERDE").sum()),
    amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"]/obj_resumen["meses_esperados"]).clip(0,1)*100
obj_resumen["riesgo"] = obj_resumen["rojos"]>0

def clasificar_estado(row):
    if row["morados"]>0: return "NO SUBIDO"
    if row["riesgo"]: return "RIESGO"
    if row["cumplimiento_%"]>=90: return "CUMPLIDO"
    if row["cumplimiento_%"]>=60: return "EN SEGUIMIENTO"
    return "CRÍTICO"

obj_resumen["estado_ejecutivo"] = obj_resumen.apply(clasificar_estado, axis=1)

# =====================================================
# FILTROS SIDEBAR
# =====================================================
st.sidebar.header("🔎 Filtros opcionales")
filtro_area = st.sidebar.multiselect("Filtrar por Área", options=area_long["AREA"].unique(), default=area_long["AREA"].unique())
filtro_estado = st.sidebar.multiselect("Filtrar por Estado Ejecutivo", options=obj_resumen["estado_ejecutivo"].unique(), default=obj_resumen["estado_ejecutivo"].unique())
filtro_tipo = st.sidebar.multiselect("Filtrar por Tipo Objetivo", options=obj_resumen["Tipo Objetivo"].unique(), default=obj_resumen["Tipo Objetivo"].unique())

obj_resumen_filtrado = obj_resumen[
    (obj_resumen["estado_ejecutivo"].isin(filtro_estado)) &
    (obj_resumen["Tipo Objetivo"].isin(filtro_tipo))
]

area_long_filtrado = area_long[area_long["AREA"].isin(filtro_area)]

# =====================================================
# KPIs PROFESIONALES + MEDIDORES
# =====================================================
st.subheader("📌 KPIs Estratégicos")
c1,c2,c3,c4,c5 = st.columns(5)

def color_kpi(valor, tipo="%"):
    if tipo=="%":
        if valor>=90: return "green"
        if valor>=60: return "yellow"
        return "red"
    return "blue"

c1.metric("Total Objetivos", obj_resumen_filtrado.shape[0])
c2.metric("Cumplidos",(obj_resumen_filtrado["estado_ejecutivo"]=="CUMPLIDO").sum())
c3.metric("En Riesgo",(obj_resumen_filtrado["estado_ejecutivo"]=="RIESGO").sum())
c4.metric("No Subidos",(obj_resumen_filtrado["estado_ejecutivo"]=="NO SUBIDO").sum())
c5.metric("Cumplimiento Promedio",f"{obj_resumen_filtrado['cumplimiento_%'].mean():.1f}%")

# Medidor circular de cumplimiento general
cum_general = obj_resumen_filtrado['cumplimiento_%'].mean()
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=cum_general,
    delta={'reference':90, 'increasing': {'color':'green'}, 'decreasing': {'color':'red'}},
    gauge={'axis': {'range':[0,100]},
           'bar': {'color':'green'},
           'steps':[{'range':[0,60],'color':'red'},
                    {'range':[60,90],'color':'yellow'},
                    {'range':[90,100],'color':'green'}]},
    title={'text':"Cumplimiento General %"}
))
st.plotly_chart(fig_gauge,use_container_width=True)

# =====================================================
# GRAFICAS INTERACTIVAS ORDENADAS POR BLOQUES EXPANDER
# =====================================================
st.header("📈 Visualizaciones Ejecutivas")

with st.expander("📊 Estado Ejecutivo de Objetivos"):
    fig1 = px.bar(obj_resumen_filtrado, x="estado_ejecutivo", y="meses_reportados",
                 color="estado_ejecutivo", text="meses_reportados",
                 color_discrete_map={"CUMPLIDO":"green","EN SEGUIMIENTO":"yellow","RIESGO":"red","CRÍTICO":"darkred","NO SUBIDO":"purple"},
                 title="Objetivos por Estado Ejecutivo")
    st.plotly_chart(fig1,use_container_width=True)
    
    fig_pie = px.pie(obj_resumen_filtrado, names="estado_ejecutivo", values="meses_reportados",
                     color="estado_ejecutivo",
                     color_discrete_map={"CUMPLIDO":"green","EN SEGUIMIENTO":"yellow","RIESGO":"red","CRÍTICO":"darkred","NO SUBIDO":"purple"},
                     title="Distribución de Objetivos por Estado")
    st.plotly_chart(fig_pie,use_container_width=True)

with st.expander("📊 Cumplimiento Mensual y Heatmap"):
    heat_df = obj_long.pivot_table(index="Objetivo", columns="Mes", values="valor", fill_value=0)
    fig2 = px.imshow(heat_df, labels=dict(x="Mes",y="Objetivo",color="Valor"),
                     color_continuous_scale=["red","yellow","green"], title="Heatmap Cumplimiento Mensual por Objetivo")
    st.plotly_chart(fig2,use_container_width=True)
    
    cum_mensual = obj_long.groupby("Mes")["valor"].mean().reindex(MESES)
    fig3 = px.line(cum_mensual, y="valor", x=cum_mensual.index, markers=True,
                   title="Cumplimiento Promedio Mensual")
    fig3.update_yaxes(range=[0,1], tickformat=".0%")
    st.plotly_chart(fig3,use_container_width=True)

with st.expander("📊 Cumplimiento por Área y Responsable"):
    area_summary = area_long_filtrado.groupby(["AREA","PUESTO RESPONSABLE"])["valor"].mean().reset_index()
    fig4 = px.bar(area_summary, x="PUESTO RESPONSABLE", y="valor", color="AREA",
                  title="Cumplimiento Promedio por Responsable y Área", text="valor", color_continuous_scale="Viridis")
    st.plotly_chart(fig4,use_container_width=True)
    
with st.expander("📊 Cumplimiento por Tipo de Objetivo"):
    tipo_summary = obj_resumen_filtrado.groupby("Tipo Objetivo")["cumplimiento_%"].mean().reset_index()
    fig5 = px.bar(tipo_summary, x="Tipo Objetivo", y="cumplimiento_%",
                  color="cumplimiento_%", text="cumplimiento_%",
                  color_continuous_scale="Blues", title="Cumplimiento Promedio por Tipo de Objetivo")
    st.plotly_chart(fig5,use_container_width=True)

# =====================================================
# TABLAS DETALLE
# =====================================================
st.header("📋 Detalle de Objetivos y Tareas")
with st.expander("Datos Normalizados - Objetivos"):
    st.dataframe(obj_long, use_container_width=True)
with st.expander("Datos Normalizados - Áreas y Tareas"):
    st.dataframe(area_long, use_container_width=True)

st.caption("Fuente: Google Sheets · Actualización automática cada 5 minutos")
