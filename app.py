# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

# =====================================================
# CONFIG STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Dashboard Estratégico 2023",
    layout="wide"
)

st.title("📊 Dashboard Estratégico y Operativo 2023")
st.markdown("**Análisis ejecutivo de cumplimiento, riesgo y desviación operativa.**")

# =====================================================
# GOOGLE SHEETS AUTH
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

service_account_info = dict(st.secrets["gcp_service_account"])
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

CREDS = Credentials.from_service_account_info(
    service_account_info,
    scopes=SCOPES
)

client = gspread.authorize(CREDS)
SHEET_NAME = "DATAESTRATEGIA"

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=300)
def load_data():
    sh = client.open(SHEET_NAME)
    df_obj = pd.DataFrame(sh.worksheet("2023").get_all_records())
    df_area = pd.DataFrame(sh.worksheet("2023 AREAS").get_all_records())
    df_obj.columns = df_obj.columns.str.strip()
    df_area.columns = df_area.columns.str.strip()
    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIG
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {
    "VERDE": 1,
    "AMARILLO": 0.5,
    "ROJO": 0,
    "MORADO": 0
}

COLOR_MAP = {
    "VERDE":"#2ecc71",
    "AMARILLO":"#f1c40f",
    "ROJO":"#e74c3c",
    "MORADO":"#8e44ad"
}

# =====================================================
# NORMALIZAR
# =====================================================
def normalizar(df, id_cols):
    meses_validos = [m for m in MESES if m in df.columns]
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=meses_validos,
        var_name="Mes",
        value_name="Estado"
    )
    df_long = df_long.dropna(subset=["Estado"])
    df_long["valor"] = df_long["Estado"].map(estado_map)
    return df_long

obj_long = normalizar(
    df_obj,
    ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
)

area_long = normalizar(
    df_area,
    ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
)

# =====================================================
# KPIs
# =====================================================
st.subheader("📌 Indicadores Clave")

k1,k2,k3,k4 = st.columns(4)

k1.metric("Objetivos Estratégicos", obj_long["Objetivo"].nunique())
k2.metric("Áreas Ejecutoras", area_long["AREA"].nunique())
k3.metric("Tareas Totales", area_long["TAREA"].nunique())
k4.metric("Cumplimiento Global", f"{obj_long['valor'].mean()*100:.1f}%")

# =====================================================
# MEDIDORES (GAUGE)
# =====================================================
st.subheader("🎯 Nivel de Cumplimiento Global")

c1, c2 = st.columns(2)

fig_gauge_obj = go.Figure(go.Indicator(
    mode="gauge+number",
    value=obj_long["valor"].mean()*100,
    title={"text":"Objetivos Estratégicos 2023"},
    gauge={
        "axis":{"range":[0,100]},
        "bar":{"color":"#2ecc71"},
        "steps":[
            {"range":[0,50],"color":"#e74c3c"},
            {"range":[50,80],"color":"#f1c40f"},
            {"range":[80,100],"color":"#2ecc71"}
        ]
    }
))

fig_gauge_area = go.Figure(go.Indicator(
    mode="gauge+number",
    value=area_long["valor"].mean()*100,
    title={"text":"Ejecución Operativa 2023"},
    gauge={
        "axis":{"range":[0,100]},
        "bar":{"color":"#3498db"},
        "steps":[
            {"range":[0,50],"color":"#e74c3c"},
            {"range":[50,80],"color":"#f1c40f"},
            {"range":[80,100],"color":"#2ecc71"}
        ]
    }
))

c1.plotly_chart(fig_gauge_obj, use_container_width=True)
c2.plotly_chart(fig_gauge_area, use_container_width=True)

# =====================================================
# DESVIACIÓN MENSUAL
# =====================================================
with st.expander("📉 Análisis de Desviación Mensual"):
    st.markdown("**Mide la estabilidad del cumplimiento durante el año. Alta desviación = gestión inestable.**")

    c1,c2 = st.columns(2)

    obj_std = obj_long.groupby("Mes")["valor"].std().reindex(MESES)
    area_std = area_long.groupby("Mes")["valor"].std().reindex(MESES)

    fig_std_obj = px.bar(
        obj_std,
        y="valor",
        title="Desviación Mensual – Objetivos",
        template="plotly_white",
        color_discrete_sequence=["#34495e"]
    )

    fig_std_area = px.bar(
        area_std,
        y="valor",
        title="Desviación Mensual – Áreas",
        template="plotly_white",
        color_discrete_sequence=["#7f8c8d"]
    )

    c1.plotly_chart(fig_std_obj, use_container_width=True)
    c2.plotly_chart(fig_std_area, use_container_width=True)

# =====================================================
# RANKING ÁREAS CRÍTICAS
# =====================================================
with st.expander("🔥 Ranking de Áreas Críticas", expanded=True):
    st.markdown("**Áreas con mayor concentración de ROJO y MORADO.**")

    area_risk = (
        area_long
        .groupby("AREA")
        .agg(
            tareas=("TAREA","count"),
            cumplimiento=("valor","mean"),
            rojos=("Estado", lambda x:(x=="ROJO").sum()),
            morados=("Estado", lambda x:(x=="MORADO").sum())
        )
        .reset_index()
    )

    area_risk["riesgo_%"] = ((area_risk["rojos"]+area_risk["morados"]) / area_risk["tareas"]) * 100
    area_risk = area_risk.sort_values("riesgo_%", ascending=False)

    fig_rank = px.bar(
        area_risk.head(10),
        x="riesgo_%",
        y="AREA",
        orientation="h",
        title="Top 10 Áreas Críticas",
        template="plotly_white",
        color="riesgo_%",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig_rank, use_container_width=True)
    st.dataframe(area_risk, use_container_width=True)

# =====================================================
# ALERTAS
# =====================================================
with st.expander("🚨 Alertas Ejecutivas", expanded=True):
    for _, r in area_risk.iterrows():
        if r["riesgo_%"] >= 40:
            st.error(f"🔴 Área CRÍTICA: {r['AREA']} ({r['riesgo_%']:.1f}%)")
        elif r["riesgo_%"] >= 25:
            st.warning(f"⚠️ Área en riesgo: {r['AREA']} ({r['riesgo_%']:.1f}%)")

# =====================================================
# EXPORT PDF
# =====================================================
def generar_pdf(area_risk):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Informe Ejecutivo – Dashboard Estratégico 2023</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Resumen Ejecutivo</b>", styles["Heading2"]))
    story.append(Paragraph(f"Cumplimiento Objetivos: {obj_long['valor'].mean()*100:.1f}%", styles["Normal"]))
    story.append(Paragraph(f"Cumplimiento Áreas: {area_long['valor'].mean()*100:.1f}%", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Áreas Críticas</b>", styles["Heading2"]))

    table = [["Área","Riesgo %","Cumplimiento %"]]
    for _, r in area_risk.iterrows():
        table.append([r["AREA"], f"{r['riesgo_%']:.1f}%", f"{r['cumplimiento']*100:.1f}%"])

    story.append(Table(table))
    doc.build(story)
    buffer.seek(0)
    return buffer

st.subheader("📄 Exportar Informe Ejecutivo")
pdf = generar_pdf(area_risk)

st.download_button(
    "📥 Descargar Informe PDF",
    pdf,
    "Informe_Estrategico_2023.pdf",
    "application/pdf"
)

st.caption("Dashboard Ejecutivo · Fondo blanco · Análisis de desviación y riesgo")





