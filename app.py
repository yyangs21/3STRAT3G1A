# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
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
st.markdown("**Seguimiento ejecutivo de objetivos estratégicos y ejecución operativa por áreas.**")

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

# =====================================================
# NORMALIZACIÓN
# =====================================================
def normalizar(df, id_cols):
    meses_validos = [m for m in MESES if m in df.columns]
    return (
        df.melt(
            id_vars=id_cols,
            value_vars=meses_validos,
            var_name="Mes",
            value_name="Estado"
        )
        .dropna(subset=["Estado"])
    )

obj_long = normalizar(
    df_obj,
    ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
)

area_long = normalizar(
    df_area,
    ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
)

obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# KPIs EJECUTIVOS
# =====================================================
st.subheader("📌 Indicadores Clave")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Objetivos Estratégicos", obj_long["Objetivo"].nunique())
k2.metric("Áreas Ejecutoras", area_long["AREA"].nunique())
k3.metric("Tareas Totales", area_long["TAREA"].nunique())
k4.metric("Cumplimiento Global", f"{obj_long['valor'].mean()*100:.1f}%")

# =====================================================
# ESTADO GENERAL
# =====================================================
with st.expander("📊 Estado General de Cumplimiento", expanded=True):
    st.markdown("**Distribución del estado de avance durante el año 2023.**")

    c1, c2 = st.columns(2)

    fig_obj_estado = px.bar(
        obj_long.groupby("Estado").size().reset_index(name="Total"),
        x="Estado", y="Total",
        color="Estado",
        title="Objetivos Estratégicos 2023",
        template="plotly_white",
        color_discrete_map={
            "VERDE":"green","AMARILLO":"gold","ROJO":"red","MORADO":"purple"
        }
    )

    fig_area_estado = px.bar(
        area_long.groupby("Estado").size().reset_index(name="Total"),
        x="Estado", y="Total",
        color="Estado",
        title="Áreas Operativas 2023",
        template="plotly_white",
        color_discrete_map={
            "VERDE":"green","AMARILLO":"gold","ROJO":"red","MORADO":"purple"
        }
    )

    c1.plotly_chart(fig_obj_estado, use_container_width=True)
    c2.plotly_chart(fig_area_estado, use_container_width=True)

# =====================================================
# TENDENCIA MENSUAL
# =====================================================
with st.expander("📈 Tendencia Mensual de Cumplimiento"):
    st.markdown("**Evolución mensual del cumplimiento promedio.**")

    c1, c2 = st.columns(2)

    obj_mes = obj_long.groupby("Mes")["valor"].mean().reindex(MESES)
    area_mes = area_long.groupby("Mes")["valor"].mean().reindex(MESES)

    fig_obj_line = px.line(
        obj_mes, y="valor", markers=True,
        title="Objetivos 2023",
        template="plotly_white"
    )

    fig_area_line = px.line(
        area_mes, y="valor", markers=True,
        title="Áreas 2023",
        template="plotly_white"
    )

    c1.plotly_chart(fig_obj_line, use_container_width=True)
    c2.plotly_chart(fig_area_line, use_container_width=True)

# =====================================================
# RANKING DE ÁREAS CRÍTICAS
# =====================================================
with st.expander("🔥 Ranking de Áreas Críticas", expanded=True):
    st.markdown("**Áreas con mayor concentración de estados ROJO y MORADO.**")

    area_risk = (
        area_long
        .groupby("AREA")
        .agg(
            tareas=("TAREA","count"),
            cumplimiento_promedio=("valor","mean"),
            rojos=("Estado", lambda x: (x=="ROJO").sum()),
            morados=("Estado", lambda x: (x=="MORADO").sum())
        )
        .reset_index()
    )

    area_risk["riesgo_%"] = (
        (area_risk["rojos"] + area_risk["morados"]) / area_risk["tareas"]
    ) * 100

    area_risk = area_risk.sort_values("riesgo_%", ascending=False)

    fig_ranking = px.bar(
        area_risk.head(10),
        x="riesgo_%",
        y="AREA",
        orientation="h",
        title="Top 10 Áreas Críticas",
        template="plotly_white",
        color="riesgo_%",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig_ranking, use_container_width=True)
    st.dataframe(area_risk, use_container_width=True)

# =====================================================
# ALERTAS EJECUTIVAS
# =====================================================
with st.expander("🚨 Alertas Ejecutivas", expanded=True):
    st.markdown("**Alertas generadas automáticamente según riesgo detectado.**")

    for _, r in area_risk.iterrows():
        if r["riesgo_%"] >= 40:
            st.error(f"🔴 Área CRÍTICA: {r['AREA']} ({r['riesgo_%']:.1f}%)")
        elif r["riesgo_%"] >= 25:
            st.warning(f"⚠️ Área en riesgo: {r['AREA']} ({r['riesgo_%']:.1f}%)")
        elif r["cumplimiento_promedio"] < 0.6:
            st.info(f"🟣 Seguimiento requerido: {r['AREA']}")

# =====================================================
# EXPORTAR PDF EJECUTIVO
# =====================================================
def generar_pdf(area_risk):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Informe Ejecutivo – Dashboard Estratégico 2023</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Resumen General</b>", styles["Heading2"]))
    story.append(Paragraph(f"Objetivos Estratégicos: {obj_long['Objetivo'].nunique()}", styles["Normal"]))
    story.append(Paragraph(f"Áreas Ejecutoras: {area_long['AREA'].nunique()}", styles["Normal"]))
    story.append(Paragraph(f"Cumplimiento Global: {obj_long['valor'].mean()*100:.1f}%", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Ranking de Áreas Críticas</b>", styles["Heading2"]))

    table_data = [["Área","Riesgo %","Cumplimiento %"]]
    for _, r in area_risk.iterrows():
        table_data.append([
            r["AREA"],
            f"{r['riesgo_%']:.1f}%",
            f"{r['cumplimiento_promedio']*100:.1f}%"
        ])

    story.append(Table(table_data))
    doc.build(story)
    buffer.seek(0)
    return buffer

st.subheader("📄 Exportar Informe Ejecutivo")

pdf = generar_pdf(area_risk)

st.download_button(
    "📥 Descargar Informe PDF",
    data=pdf,
    file_name="Informe_Estrategico_2023.pdf",
    mime="application/pdf"
)

# =====================================================
# DATOS
# =====================================================
with st.expander("📋 Datos Detallados"):
    st.subheader("Objetivos Estratégicos")
    st.dataframe(obj_long, use_container_width=True)

    st.subheader("Áreas y Tareas")
    st.dataframe(area_long, use_container_width=True)

st.caption("Dashboard Ejecutivo · Fondo blanco · Actualización automática")





