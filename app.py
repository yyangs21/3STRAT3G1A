import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(
    page_title="Dashboard Estratégico 2023",
    layout="wide"
)

# Forzar fondo blanco
st.markdown("""
<style>
    .stApp {
        background-color: black;
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Estratégico y de Ejecución 2023")

# =====================================================
# AUTENTICACIÓN GOOGLE SHEETS
# =====================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

CREDS = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES
)

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
    "MORADO": None  # No subido
}

color_map = {
    "VERDE": "#2ecc71",
    "AMARILLO": "#f1c40f",
    "ROJO": "#e74c3c",
    "MORADO": "#9b59b6"
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
def normalizar(df, id_cols):
    return df.melt(
        id_vars=id_cols,
        value_vars=MESES,
        var_name="Mes",
        value_name="Estado"
    )

obj_long = normalizar(
    df_obj,
    ["Objetivo","Tipo Objetivo","Frecuencia Medición"]
)

area_long = normalizar(
    df_area,
    ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","¿Realizada?"]
)

# =====================================================
# MAPEO
# =====================================================
obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# FILTROS (SIDEBAR)
# =====================================================
st.sidebar.header("🎛️ Filtros")

mes_sel = st.sidebar.multiselect(
    "Mes",
    MESES,
    default=MESES
)

obj_long = obj_long[obj_long["Mes"].isin(mes_sel)]
area_long = area_long[area_long["Mes"].isin(mes_sel)]

# =====================================================
# RESÚMENES
# =====================================================
obj_resumen = obj_long.groupby("Objetivo", as_index=False).agg(
    score=("valor","mean"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

area_resumen = area_long.groupby("AREA", as_index=False).agg(
    score=("valor","mean"),
    rojos=("Estado", lambda x: (x=="ROJO").sum()),
    morados=("Estado", lambda x: (x=="MORADO").sum())
)

# =====================================================
# KPIs
# =====================================================
st.subheader("📌 Indicadores Ejecutivos")

k1,k2,k3,k4 = st.columns(4)

k1.metric("Objetivos Estratégicos", len(obj_resumen))
k2.metric("Áreas Evaluadas", len(area_resumen))
k3.metric("Alertas Críticas", obj_resumen["rojos"].sum())
k4.metric("No Subidos", obj_resumen["morados"].sum())

# =====================================================
# MEDIDORES (GAUGE)
# =====================================================
g1,g2 = st.columns(2)

with g1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=obj_resumen["score"].mean()*100,
        title={'text':"Cumplimiento Estratégico 2023"},
        gauge={'axis': {'range':[0,100]},
               'bar': {'color':"#2c3e50"},
               'steps': [
                   {'range':[0,60],'color':'#e74c3c'},
                   {'range':[60,85],'color':'#f1c40f'},
                   {'range':[85,100],'color':'#2ecc71'}
               ]}
    ))
    st.plotly_chart(fig, use_container_width=True)

with g2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=area_resumen["score"].mean()*100,
        title={'text':"Ejecución Operativa 2023"},
        gauge={'axis': {'range':[0,100]},
               'bar': {'color':"#34495e"},
               'steps': [
                   {'range':[0,60],'color':'#e74c3c'},
                   {'range':[60,85],'color':'#f1c40f'},
                   {'range':[85,100],'color':'#2ecc71'}
               ]}
    ))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TABS DE GRÁFICAS
# =====================================================
tab1, tab2 = st.tabs(["📊 Estratégico 2023", "🏭 Áreas 2023"])

with tab1:
    st.markdown("**Distribución del estado de los objetivos estratégicos**")
    fig = px.histogram(
        obj_long,
        x="Estado",
        color="Estado",
        color_discrete_map=color_map
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("**Ranking de áreas críticas**")
    fig = px.bar(
        area_resumen.sort_values("score"),
        x="score",
        y="AREA",
        orientation="h",
        color="score",
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ALERTAS
# =====================================================
st.subheader("🚨 Alertas Automáticas")

alertas = obj_resumen[obj_resumen["rojos"] > 0]

st.dataframe(alertas, use_container_width=True)

# =====================================================
# TABLAS DETALLE
# =====================================================
with st.expander("📋 Datos Estratégicos"):
    st.dataframe(obj_long, use_container_width=True)

with st.expander("📋 Datos Áreas"):
    st.dataframe(area_long, use_container_width=True)

st.caption("Dashboard Ejecutivo")

# =====================================================
# EXPORTAR INFORME EJECUTIVO A PDF
# =====================================================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from datetime import datetime
import os

def exportar_pdf():
    file_name = "Informe_Ejecutivo_2023.pdf"
    doc = SimpleDocTemplate(
        file_name,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    elements = []

    # ---------- PORTADA ----------
    elements.append(Paragraph("<b>INFORME EJECUTIVO DE CUMPLIMIENTO 2023</b>", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Dashboard Estratégico y de Ejecución", styles["h2"]))
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(PageBreak())

    # ---------- KPIs ----------
    elements.append(Paragraph("<b>Indicadores Clave</b>", styles["h1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Objetivos Estratégicos: {len(obj_resumen)}", styles["Normal"]))
    elements.append(Paragraph(f"Áreas Evaluadas: {len(area_resumen)}", styles["Normal"]))
    elements.append(Paragraph(f"Alertas Críticas: {obj_resumen['rojos'].sum()}", styles["Normal"]))
    elements.append(Paragraph(f"No Subidos: {obj_resumen['morados'].sum()}", styles["Normal"]))
    elements.append(PageBreak())

    # ---------- EXPORTAR GRÁFICAS ----------
    os.makedirs("tmp", exist_ok=True)

    fig_g1.write_image("tmp/gauge_estrategico.png", scale=2)
    fig_g2.write_image("tmp/gauge_operativo.png", scale=2)

    elements.append(Paragraph("<b>Cumplimiento Estratégico</b>", styles["h1"]))
    elements.append(Image("tmp/gauge_estrategico.png", width=14*cm, height=8*cm))
    elements.append(PageBreak())

    elements.append(Paragraph("<b>Ejecución Operativa</b>", styles["h1"]))
    elements.append(Image("tmp/gauge_operativo.png", width=14*cm, height=8*cm))
    elements.append(PageBreak())

    # ---------- RANKING ÁREAS ----------
    fig_rank = px.bar(
        area_resumen.sort_values("score"),
        x="score",
        y="AREA",
        orientation="h",
        color="score",
        color_continuous_scale="RdYlGn"
    )
    fig_rank.write_image("tmp/ranking_areas.png", scale=2)

    elements.append(Paragraph("<b>Ranking de Áreas Críticas</b>", styles["h1"]))
    elements.append(Image("tmp/ranking_areas.png", width=15*cm, height=10*cm))
    elements.append(PageBreak())

    # ---------- ALERTAS ----------
    elements.append(Paragraph("<b>Alertas Detectadas</b>", styles["h1"]))
    for _, row in alertas.iterrows():
        elements.append(Paragraph(f"- {row['Objetivo']} presenta desviaciones críticas.", styles["Normal"]))

    doc.build(elements)

    return file_name

# =====================================================
# BOTÓN STREAMLIT
# =====================================================
st.subheader("📄 Informe Ejecutivo")

if st.button("📥 Exportar Informe Ejecutivo PDF"):
    pdf = exportar_pdf()
    with open(pdf, "rb") as f:
        st.download_button(
            label="⬇️ Descargar PDF",
            data=f,
            file_name=pdf,
            mime="application/pdf"
        )

