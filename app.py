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
        "Realizada?": "¿Realizada?",
        "Puesto Responsable": "PUESTO RESPONSABLE"
    }, inplace=True)

    return df_obj, df_area

df_obj, df_area = load_data()

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
estado_map = {"VERDE":1,"AMARILLO":0.5,"ROJO":0,"MORADO":0}
frecuencia_map = {
    "Mensual":12,"Bimestral":6,"Trimestral":4,
    "Cuatrimestral":3,"Semestral":2,"Anual":1
}

# =====================================================
# NORMALIZAR MESES
# =====================================================
def normalizar_meses(df, id_cols):
    meses_presentes = [m for m in MESES if m in df.columns]
    return df.melt(
        id_vars=id_cols,
        value_vars=meses_presentes,
        var_name="Mes",
        value_name="Estado"
    ).dropna(subset=["Estado"])

obj_long = normalizar_meses(df_obj, ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"])
area_long = normalizar_meses(df_area, ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"])

obj_long["valor"] = obj_long["Estado"].map(estado_map)
area_long["valor"] = area_long["Estado"].map(estado_map)

# =====================================================
# RESUMEN OBJETIVOS
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
# FILTROS
# =====================================================
st.sidebar.header("🔎 Filtros")
filtro_estado = st.sidebar.multiselect(
    "Estado Ejecutivo",
    obj_resumen["estado_ejecutivo"].unique(),
    obj_resumen["estado_ejecutivo"].unique()
)

filtro_area = st.sidebar.multiselect(
    "Área",
    area_long["AREA"].unique(),
    area_long["AREA"].unique()
)

obj_resumen_f = obj_resumen[obj_resumen["estado_ejecutivo"].isin(filtro_estado)]
area_long_f = area_long[area_long["AREA"].isin(filtro_area)]

# =====================================================
# GAUGES
# =====================================================
st.subheader("🎯 Medidores Estratégicos")

g1, g2 = st.columns(2)

cum_obj = obj_resumen_f["cumplimiento_%"].mean()
cum_area = area_long_f.groupby("AREA")["valor"].mean().mean() * 100

def gauge(valor, titulo):
    return go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=valor,
        delta={"reference":90},
        gauge={
            "axis":{"range":[0,100]},
            "steps":[
                {"range":[0,60],"color":"red"},
                {"range":[60,90],"color":"yellow"},
                {"range":[90,100],"color":"green"}
            ],
            "bar":{"color":"green"}
        },
        title={"text":titulo}
    ))

g1.plotly_chart(gauge(cum_obj,"Cumplimiento Estratégico 2023"), use_container_width=True)
g2.plotly_chart(gauge(cum_area,"Cumplimiento Operativo por Áreas"), use_container_width=True)

# =====================================================
# ALERTAS AUTOMÁTICAS
# =====================================================
st.subheader("🚨 Alertas Automáticas")

alertas = []

for _, r in obj_resumen_f.iterrows():
    if r["estado_ejecutivo"] == "CRÍTICO":
        alertas.append(f"🚨 **ALERTA CRÍTICA:** Objetivo *{r['Objetivo']}*")
    if r["estado_ejecutivo"] == "NO SUBIDO":
        alertas.append(f"🟣 **NO SUBIDO:** Objetivo *{r['Objetivo']}*")

area_alert = area_long_f.groupby("AREA")["valor"].mean()
for area, v in area_alert.items():
    if v < 0.6:
        alertas.append(f"⚠️ **RIESGO OPERATIVO:** Área *{area}* con bajo cumplimiento")

if alertas:
    for a in alertas:
        st.error(a)
else:
    st.success("✅ No se detectan alertas críticas")

# =====================================================
# VISUALIZACIONES
# =====================================================
st.header("📈 Visualización Ejecutiva")

c1, c2 = st.columns(2)

with c1:
    fig_pie = px.pie(
        obj_resumen_f,
        names="estado_ejecutivo",
        title="Distribución Estado Ejecutivo"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    fig_bar = px.bar(
        obj_resumen_f,
        x="Objetivo",
        y="cumplimiento_%",
        color="estado_ejecutivo",
        title="Desviación de Cumplimiento por Objetivo"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =====================================================
# REPORTE EJECUTIVO
# =====================================================
st.header("📝 Reporte Ejecutivo Automático")

st.markdown(f"""
**Resumen General 2023**

- Cumplimiento Estratégico: **{cum_obj:.1f}%**
- Cumplimiento Operativo: **{cum_area:.1f}%**
- Objetivos Críticos: **{(obj_resumen_f['estado_ejecutivo']=='CRÍTICO').sum()}**
- Objetivos No Subidos: **{(obj_resumen_f['estado_ejecutivo']=='NO SUBIDO').sum()}**

**Recomendación:**
Priorizar los objetivos críticos y reforzar las áreas con desempeño inferior al 60%.
""")

st.caption("Fuente: Google Sheets · Actualización automática cada 5 minutos")
