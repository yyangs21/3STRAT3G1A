import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(page_title="Tablero de Control 2023", layout="wide")
st.title("📊 Tablero de Control Estratégico y Operativo 2023")

# =====================================================
# GOOGLE SHEETS
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

service_account_info = dict(st.secrets["gcp_service_account"])
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n","\n")
CREDS = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
client = gspread.authorize(CREDS)

SHEET_NAME = "DATAESTRATEGIA"

@st.cache_data(ttl=300)
def load_data():
    sh = client.open(SHEET_NAME)
    df_obj = pd.DataFrame(sh.worksheet("2023").get_all_records())
    df_area = pd.DataFrame(sh.worksheet("2023 AREAS").get_all_records())

    df_obj.columns = df_obj.columns.str.strip()
    df_area.columns = df_area.columns.str.strip()

    df_area.rename(columns={
        "Área":"AREA",
        "Realizada?":"¿Realizada?",
        "Puesto Responsable":"PUESTO RESPONSABLE"
    }, inplace=True)

    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIGURACIONES
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {"VERDE":1,"AMARILLO":0.5,"ROJO":0,"MORADO":0}
frecuencia_map = {"Mensual":12,"Bimestral":6,"Trimestral":4,"Cuatrimestral":3,"Semestral":2,"Anual":1}

# =====================================================
# NORMALIZAR MESES
# =====================================================
def normalizar_meses(df, id_cols):
    meses = [m for m in MESES if m in df.columns]
    return df.melt(id_vars=id_cols, value_vars=meses,
                   var_name="Mes", value_name="Estado").dropna()

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
# RESUMEN ESTRATÉGICO
# =====================================================
obj_resumen = obj_long.groupby(
    ["Objetivo","Frecuencia Medición"], as_index=False
).agg(
    score=("valor","sum"),
    meses=("Mes","count"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

obj_resumen["esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)
obj_resumen["cumplimiento_%"] = (obj_resumen["score"]/obj_resumen["esperados"]).clip(0,1)*100

def estado_exec(r):
    if r["morados"]>0: return "NO SUBIDO"
    if r["rojos"]>0: return "RIESGO"
    if r["cumplimiento_%"]>=90: return "CUMPLIDO"
    if r["cumplimiento_%"]>=60: return "EN SEGUIMIENTO"
    return "CRÍTICO"

obj_resumen["estado"] = obj_resumen.apply(estado_exec, axis=1)

# =====================================================
# RESUMEN OPERATIVO
# =====================================================
area_resumen = area_long.groupby("AREA", as_index=False).agg(
    cumplimiento=("valor","mean")
)
area_resumen["cumplimiento_%"] = area_resumen["cumplimiento"]*100

# =====================================================
# GAUGES
# =====================================================
st.subheader("🎯 Indicadores Generales")

c1, c2 = st.columns(2)

c1.plotly_chart(go.Figure(go.Indicator(
    mode="gauge+number",
    value=obj_resumen["cumplimiento_%"].mean(),
    title={"text":"Cumplimiento Estratégico"},
    gauge={"axis":{"range":[0,100]},
           "steps":[{"range":[0,60],"color":"red"},
                    {"range":[60,90],"color":"yellow"},
                    {"range":[90,100],"color":"green"}]}
)), use_container_width=True)

c2.plotly_chart(go.Figure(go.Indicator(
    mode="gauge+number",
    value=area_resumen["cumplimiento_%"].mean(),
    title={"text":"Cumplimiento Operativo"},
    gauge={"axis":{"range":[0,100]},
           "steps":[{"range":[0,60],"color":"red"},
                    {"range":[60,90],"color":"yellow"},
                    {"range":[90,100],"color":"green"}]}
)), use_container_width=True)

# =====================================================
# BLOQUE ESTRATÉGICO
# =====================================================
st.header("🟦 Análisis Estratégico – Objetivos")

fig_pie = px.pie(obj_resumen, names="estado", title="Distribución de Objetivos")
st.plotly_chart(fig_pie, use_container_width=True)

fig_dev = px.bar(obj_resumen, x="Objetivo", y="cumplimiento_%",
                 title="Desviación de Cumplimiento por Objetivo")
fig_dev.add_hline(y=90, line_dash="dash", line_color="green")
st.plotly_chart(fig_dev, use_container_width=True)

criticos = obj_resumen[obj_resumen["estado"].isin(["CRÍTICO","NO SUBIDO"])]
if not criticos.empty:
    st.error("⚠️ Objetivos con alerta:")
    st.dataframe(criticos[["Objetivo","estado","cumplimiento_%"]])

# =====================================================
# BLOQUE OPERATIVO
# =====================================================
st.header("🟪 Análisis Operativo – Áreas")

fig_rank = px.bar(area_resumen.sort_values("cumplimiento_%"),
                  x="cumplimiento_%", y="AREA",
                  orientation="h",
                  title="Ranking de Áreas")
st.plotly_chart(fig_rank, use_container_width=True)

heat = area_long.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
fig_heat = px.imshow(heat, title="Heatmap de Ejecución por Área")
st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# REPORTE
# =====================================================
st.header("📄 Reporte Ejecutivo")
if st.button("Generar Reporte"):
    st.success("Reporte Ejecutivo")
    st.write(f"- Cumplimiento estratégico: {obj_resumen['cumplimiento_%'].mean():.1f}%")
    st.write(f"- Cumplimiento operativo: {area_resumen['cumplimiento_%'].mean():.1f}%")
    st.write(f"- Objetivos críticos: {criticos.shape[0]}")

