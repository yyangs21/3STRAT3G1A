import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from weasyprint import HTML
import tempfile
import os

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(page_title="Dashboard Estratégico 2023", layout="wide")
st.title("📊 Dashboard Estratégico y de Control 2023 – ULTRA PRO")

# =====================================================
# GOOGLE SHEETS
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
# CONFIG
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {"VERDE":1,"AMARILLO":0.5,"ROJO":0,"MORADO":0}
frecuencia_map = {"Mensual":12,"Bimestral":6,"Trimestral":4,"Cuatrimestral":3,"Semestral":2,"Anual":1}

# =====================================================
# NORMALIZACIÓN
# =====================================================
def normalizar_meses(df, id_cols):
    meses = [m for m in MESES if m in df.columns]
    return df.melt(id_vars=id_cols, value_vars=meses,
                   var_name="Mes", value_name="Estado").dropna(subset=["Estado"])

obj_long = normalizar_meses(df_obj,
    ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"])

area_long = normalizar_meses(df_area,
    ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"])

obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# RESUMEN OBJETIVOS
# =====================================================
obj_resumen = obj_long.groupby(
    ["Objetivo","Tipo Objetivo","Frecuencia Medición"], as_index=False
).agg(
    score_total=("valor","sum"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map)
obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"]/obj_resumen["meses_esperados"]).clip(0,1)*100

def estado_exec(r):
    if r["morados"]>0: return "NO SUBIDO"
    if r["rojos"]>0: return "RIESGO"
    if r["cumplimiento_%"]>=90: return "CUMPLIDO"
    if r["cumplimiento_%"]>=60: return "EN SEGUIMIENTO"
    return "CRÍTICO"

obj_resumen["estado_ejecutivo"] = obj_resumen.apply(estado_exec, axis=1)

# =====================================================
# FILTROS (NO OBLIGATORIOS)
# =====================================================
st.sidebar.header("🔎 Filtros (opcionales)")

f_estado = st.sidebar.multiselect("Estado Ejecutivo",
    obj_resumen["estado_ejecutivo"].unique(), default=[])

f_area = st.sidebar.multiselect("Área",
    area_long["AREA"].unique(), default=[])

obj_f = obj_resumen.copy()
area_f = area_long.copy()

if f_estado:
    obj_f = obj_f[obj_f["estado_ejecutivo"].isin(f_estado)]

if f_area:
    area_f = area_f[area_f["AREA"].isin(f_area)]

# =====================================================
# KPIs + GAUGES
# =====================================================
st.subheader("📌 KPIs Estratégicos")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Objetivos", obj_f.shape[0])
c2.metric("Cumplidos", (obj_f["estado_ejecutivo"]=="CUMPLIDO").sum())
c3.metric("En Riesgo", (obj_f["estado_ejecutivo"]=="RIESGO").sum())
c4.metric("Cumplimiento Promedio", f"{obj_f['cumplimiento_%'].mean():.1f}%")

fig_g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=obj_f["cumplimiento_%"].mean(),
    gauge={
        "axis":{"range":[0,100]},
        "steps":[
            {"range":[0,60],"color":"red"},
            {"range":[60,90],"color":"yellow"},
            {"range":[90,100],"color":"green"}],
        "bar":{"color":"green"}}
))
st.plotly_chart(fig_g, use_container_width=True)

# =====================================================
# ALERTAS
# =====================================================
st.subheader("🚨 Alertas Automáticas")
alertas = []

for _,r in obj_f.iterrows():
    if r["estado_ejecutivo"] in ["CRÍTICO","RIESGO","NO SUBIDO"]:
        alertas.append(f"⚠️ Objetivo {r['Objetivo']} en estado {r['estado_ejecutivo']}")

for a,v in area_f.groupby("AREA")["valor"].mean().items():
    if v < 0.6:
        alertas.append(f"🔴 Área {a} con bajo cumplimiento")

if alertas:
    for a in alertas: st.warning(a)
else:
    st.success("✅ Sin alertas críticas")

# =====================================================
# GRÁFICAS
# =====================================================
st.header("📈 Análisis Visual")

with st.expander("Distribución de Estados"):
    st.plotly_chart(px.pie(obj_f, names="estado_ejecutivo"), use_container_width=True)

with st.expander("Desviación de Cumplimiento"):
    df = obj_f.copy()
    df["desv"] = df["cumplimiento_%"] - 100
    st.plotly_chart(px.bar(df.sort_values("desv"),
        x="desv", y="Objetivo", orientation="h",
        color="desv", color_continuous_scale="RdYlGn"),
        use_container_width=True)

with st.expander("Ranking de Áreas"):
    rank = area_f.groupby("AREA")["valor"].mean().sort_values()
    st.plotly_chart(px.bar(rank, x=rank.values, y=rank.index,
        orientation="h"), use_container_width=True)

with st.expander("Heatmap Operativo"):
    heat = area_f.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
    st.plotly_chart(px.imshow(heat,
        color_continuous_scale=["red","yellow","green"]),
        use_container_width=True)

# =====================================================
# EXPORTAR PDF
# =====================================================
st.header("📄 Exportar Reporte PDF")

if st.button("📥 Generar PDF"):
    html = obj_f.to_html(index=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        HTML(string=html).write_pdf(f.name)
        st.download_button("⬇️ Descargar PDF",
            open(f.name,"rb"), file_name="Reporte_Estrategico_2023.pdf")

st.caption("Dashboard Estratégico 2023 · ULTRA PRO")
