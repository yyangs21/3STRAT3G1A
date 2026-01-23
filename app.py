import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# CONFIG STREAMLIT
# =====================================================
st.set_page_config(page_title="Dashboard Estratégico", layout="wide")
st.title("📊 Dashboard Estratégico y de Control")

# =====================================================
# GOOGLE SHEETS AUTH
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
# CONFIG GENERAL
# =====================================================
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

estado_map = {"VERDE": 1, "AMARILLO": 0.5, "ROJO": 0, "MORADO": 0}
frecuencia_map = {"Mensual": 12, "Bimestral": 6, "Trimestral": 4, "Cuatrimestral": 3, "Semestral": 2, "Anual": 1}

COLOR_ESTADO = {
    "CUMPLIDO": "green",
    "EN SEGUIMIENTO": "gold",
    "RIESGO": "red",
    "CRÍTICO": "darkred",
    "NO SUBIDO": "purple"
}

# =====================================================
# HELPERS
# =====================================================
def safe_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", " ")
    return df

def standardize_area_cols(df_area: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres para soportar:
    - AREA vs DEPARTAMENTO
    - PUESTO RESPONSABLE vs PUESTO
    """
    df = df_area.copy()
    # normalizar may/min
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # equivalencias comunes
    rename_map = {
        "Área": "AREA",
        "AREA": "AREA",
        "Departamento": "DEPARTAMENTO",
        "DEPARTAMENTO": "DEPARTAMENTO",
        "Puesto Responsable": "PUESTO RESPONSABLE",
        "PUESTO RESPONSABLE": "PUESTO RESPONSABLE",
        "PUESTO": "PUESTO RESPONSABLE",
        "Realizada?": "¿Realizada?",
        "¿Realizada?": "¿Realizada?",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Si no hay AREA pero hay DEPARTAMENTO, creamos AREA=DEPARTAMENTO para tu dashboard
    if "AREA" not in df.columns and "DEPARTAMENTO" in df.columns:
        df["AREA"] = df["DEPARTAMENTO"]

    return df

def normalizar_meses(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    meses_presentes = [m for m in MESES if m in df.columns]
    return (
        df.melt(id_vars=id_cols, value_vars=meses_presentes, var_name="Mes", value_name="Estado")
          .dropna(subset=["Estado"])
    )

def clasificar_estado(row) -> str:
    if row.get("morados", 0) > 0:
        return "NO SUBIDO"
    if row.get("rojos", 0) > 0:
        return "RIESGO"
    if row.get("cumplimiento_%", 0) >= 90:
        return "CUMPLIDO"
    if row.get("cumplimiento_%", 0) >= 60:
        return "EN SEGUIMIENTO"
    return "CRÍTICO"

def apply_filter(df: pd.DataFrame, col: str, selected: list) -> pd.DataFrame:
    if not selected or col not in df.columns:
        return df
    return df[df[col].isin(selected)]

# =====================================================
# LOAD: detectar años disponibles
# =====================================================
@st.cache_data(ttl=300)
def get_available_years():
    sh = client.open(SHEET_NAME)
    titles = [ws.title.strip() for ws in sh.worksheets()]
    years = []
    for t in titles:
        if t.isdigit():
            years.append(int(t))
    years = sorted(list(set(years)))
    return years, titles

@st.cache_data(ttl=300)
def load_year_data(year: int):
    sh = client.open(SHEET_NAME)

    # Objetivos
    df_obj = None
    try:
        df_obj = pd.DataFrame(sh.worksheet(str(year)).get_all_records())
    except Exception:
        df_obj = None

    # Areas
    df_area = None
    try:
        df_area = pd.DataFrame(sh.worksheet(f"{year} AREAS").get_all_records())
    except Exception:
        df_area = None

    if df_obj is not None:
        df_obj = safe_strip_cols(df_obj)

    if df_area is not None:
        df_area = safe_strip_cols(df_area)
        df_area = standardize_area_cols(df_area)

    return df_obj, df_area

# =====================================================
# SIDEBAR: selección de años + filtros
# =====================================================
years, sheet_titles = get_available_years()
if not years:
    st.error("No encontré hojas tipo '2023', '2024', '2025' en Google Sheets. Crea tabs con esos nombres.")
    st.stop()

default_year = max(years)
st.sidebar.header("🗓️ Años")
sel_years = st.sidebar.multiselect(
    "Selecciona años para comparar",
    options=years,
    default=[default_year]
)

st.sidebar.divider()
st.sidebar.header("🔎 Filtros (opcionales)")

# cargamos data de los años seleccionados para construir opciones de filtros
data_by_year = {}
for y in sel_years:
    df_obj_y, df_area_y = load_year_data(y)
    if df_obj_y is None and df_area_y is None:
        continue
    data_by_year[y] = (df_obj_y, df_area_y)

if not data_by_year:
    st.error("Los años seleccionados no tienen hojas válidas.")
    st.stop()

# construir data "unificada" (para opciones de filtro)
objs_all = []
areas_all = []
for y, (o, a) in data_by_year.items():
    if o is not None:
        tmp = o.copy()
        tmp["AÑO"] = y
        objs_all.append(tmp)
    if a is not None:
        tmp = a.copy()
        tmp["AÑO"] = y
        areas_all.append(tmp)

df_obj_all = pd.concat(objs_all, ignore_index=True) if objs_all else pd.DataFrame()
df_area_all = pd.concat(areas_all, ignore_index=True) if areas_all else pd.DataFrame()

# Filtros nuevos (desde tu Excel: Tipo, Perspectiva, Departamento)
f_tipo_plan = st.sidebar.multiselect("Tipo (POA / PEC)", sorted(df_obj_all["Tipo"].dropna().unique()) if "Tipo" in df_obj_all.columns else [])
f_persp = st.sidebar.multiselect("Perspectiva", sorted(df_obj_all["Perspectiva"].dropna().unique()) if "Perspectiva" in df_obj_all.columns else [])
f_depto = st.sidebar.multiselect("Departamento", sorted(df_obj_all["Departamento"].dropna().unique()) if "Departamento" in df_obj_all.columns else [])
f_eje = st.sidebar.multiselect("Eje", sorted(df_obj_all["Eje"].dropna().unique()) if "Eje" in df_obj_all.columns else [])

# Filtros clásicos
f_tipo_obj = st.sidebar.multiselect("Tipo Objetivo", sorted(df_obj_all["Tipo Objetivo"].dropna().unique()) if "Tipo Objetivo" in df_obj_all.columns else [])
# Nota: estado_ejecutivo se calcula después, lo hacemos abajo por año y luego unificamos opciones.
f_area = st.sidebar.multiselect("Área (AREAS)", sorted(df_area_all["AREA"].dropna().unique()) if "AREA" in df_area_all.columns else [])
f_puesto = st.sidebar.multiselect("Puesto Responsable", sorted(df_area_all["PUESTO RESPONSABLE"].dropna().unique()) if "PUESTO RESPONSABLE" in df_area_all.columns else [])

st.sidebar.divider()
st.sidebar.caption("Si no seleccionas filtros, verás el dashboard completo.")

# =====================================================
# PROCESAMIENTO por año
# =====================================================
def compute_obj_metrics(df_obj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
    - obj_long
    - obj_resumen (con estado_ejecutivo y cumplimiento_%)
    """
    id_cols = ["Objetivo","Tipo Objetivo","Fecha Inicio","Fecha Fin","Frecuencia Medición"]
    # agregar nuevos si existen
    for c in ["Tipo","Perspectiva","Eje","Departamento"]:
        if c in df_obj.columns and c not in id_cols:
            id_cols.insert(0, c)

    obj_long = normalizar_meses(df_obj, id_cols)
    obj_long["valor"] = obj_long["Estado"].map(estado_map).fillna(0)

    # resumen base
    group_cols = [c for c in ["Tipo","Perspectiva","Eje","Departamento","Objetivo","Tipo Objetivo","Frecuencia Medición"] if c in df_obj.columns]
    if "Objetivo" not in group_cols:
        group_cols.append("Objetivo")
    if "Tipo Objetivo" not in group_cols and "Tipo Objetivo" in df_obj.columns:
        group_cols.append("Tipo Objetivo")
    if "Frecuencia Medición" not in group_cols and "Frecuencia Medición" in df_obj.columns:
        group_cols.append("Frecuencia Medición")

    obj_resumen = obj_long.groupby(group_cols, as_index=False).agg(
        score_total=("valor","sum"),
        verdes=("Estado", lambda x: (x=="VERDE").sum()),
        amarillos=("Estado", lambda x: (x=="AMARILLO").sum()),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum()),
        meses_reportados=("Mes","count")
    )

    if "Frecuencia Medición" in obj_resumen.columns:
        obj_resumen["meses_esperados"] = obj_resumen["Frecuencia Medición"].map(frecuencia_map).fillna(12)
    else:
        obj_resumen["meses_esperados"] = 12

    obj_resumen["cumplimiento_%"] = (obj_resumen["score_total"] / obj_resumen["meses_esperados"]).clip(0,1)*100
    obj_resumen["estado_ejecutivo"] = obj_resumen.apply(clasificar_estado, axis=1)
    return obj_long, obj_resumen

def compute_area_metrics(df_area: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
    - area_long
    - area_resumen (por AREA, y por PUESTO RESPONSABLE)
    """
    id_cols = ["OBJETIVO","AREA","PUESTO RESPONSABLE","TAREA","Fecha Inicio","Fecha Fin","¿Realizada?"]
    # agregar nuevos si existen
    for c in ["TIPO","PERSPECTIVA","EJE","DEPARTAMENTO"]:
        if c in df_area.columns and c not in id_cols:
            id_cols.insert(0, c)

    # Garantizar columnas mínimas
    for c in ["AREA","PUESTO RESPONSABLE"]:
        if c not in df_area.columns:
            df_area[c] = "SIN_DATO"

    area_long = normalizar_meses(df_area, id_cols)
    area_long["valor"] = area_long["Estado"].map(estado_map).fillna(0)

    area_resumen_area = area_long.groupby(["AREA"], as_index=False).agg(
        cumplimiento=("valor","mean"),
        tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
        rojos=("Estado", lambda x: (x=="ROJO").sum()),
        morados=("Estado", lambda x: (x=="MORADO").sum())
    )
    area_resumen_area["cumplimiento_%"] = area_resumen_area["cumplimiento"]*100

    area_resumen_puesto = area_long.groupby(["AREA","PUESTO RESPONSABLE"], as_index=False).agg(
        cumplimiento=("valor","mean"),
        tareas=("TAREA","nunique") if "TAREA" in area_long.columns else ("Estado","count"),
    )
    area_resumen_puesto["cumplimiento_%"] = area_resumen_puesto["cumplimiento"]*100

    return area_long, area_resumen_area, area_resumen_puesto

# almacenar métricas por año
metrics = {}
for y, (df_obj_y, df_area_y) in data_by_year.items():
    obj_long = obj_res = None
    area_long = area_res_area = area_res_puesto = None

    if df_obj_y is not None and not df_obj_y.empty:
        obj_long, obj_res = compute_obj_metrics(df_obj_y)
        obj_long["AÑO"] = y
        obj_res["AÑO"] = y

    if df_area_y is not None and not df_area_y.empty:
        area_long, area_res_area, area_res_puesto = compute_area_metrics(df_area_y)
        area_long["AÑO"] = y
        area_res_area["AÑO"] = y
        area_res_puesto["AÑO"] = y

    metrics[y] = {
        "obj_long": obj_long, "obj_res": obj_res,
        "area_long": area_long, "area_res_area": area_res_area, "area_res_puesto": area_res_puesto
    }

# concatenar para análisis multi-año
obj_res_all = pd.concat([m["obj_res"] for m in metrics.values() if m["obj_res"] is not None], ignore_index=True) if any(m["obj_res"] is not None for m in metrics.values()) else pd.DataFrame()
obj_long_all = pd.concat([m["obj_long"] for m in metrics.values() if m["obj_long"] is not None], ignore_index=True) if any(m["obj_long"] is not None for m in metrics.values()) else pd.DataFrame()
area_long_all = pd.concat([m["area_long"] for m in metrics.values() if m["area_long"] is not None], ignore_index=True) if any(m["area_long"] is not None for m in metrics.values()) else pd.DataFrame()
area_res_area_all = pd.concat([m["area_res_area"] for m in metrics.values() if m["area_res_area"] is not None], ignore_index=True) if any(m["area_res_area"] is not None for m in metrics.values()) else pd.DataFrame()
area_res_puesto_all = pd.concat([m["area_res_puesto"] for m in metrics.values() if m["area_res_puesto"] is not None], ignore_index=True) if any(m["area_res_puesto"] is not None for m in metrics.values()) else pd.DataFrame()

# aplicar filtros a objetivos (si están)
for col, sel in [("Tipo", f_tipo_plan), ("Perspectiva", f_persp), ("Departamento", f_depto), ("Eje", f_eje), ("Tipo Objetivo", f_tipo_obj)]:
    obj_res_all = apply_filter(obj_res_all, col, sel)
    obj_long_all = apply_filter(obj_long_all, col, sel)

# filtros a areas (si están)
area_long_all = apply_filter(area_long_all, "AREA", f_area)
area_res_area_all = apply_filter(area_res_area_all, "AREA", f_area)
area_res_puesto_all = apply_filter(area_res_puesto_all, "AREA", f_area)
area_long_all = apply_filter(area_long_all, "PUESTO RESPONSABLE", f_puesto)
area_res_puesto_all = apply_filter(area_res_puesto_all, "PUESTO RESPONSABLE", f_puesto)

# =====================================================
# UI: Tabs (más orden)
# =====================================================
tab_res, tab_obj, tab_area, tab_comp, tab_export, tab_data = st.tabs([
    "📌 Resumen", "🎯 Objetivos", "🏢 Áreas", "📊 Comparativos", "📄 Exportar", "📋 Datos"
])

# =====================================================
# TAB: RESUMEN
# =====================================================
with tab_res:
    st.subheader("📌 Indicadores Clave (según filtros)")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_obj = int(obj_res_all.shape[0]) if not obj_res_all.empty else 0
    cumplidos = int((obj_res_all.get("estado_ejecutivo") == "CUMPLIDO").sum()) if "estado_ejecutivo" in obj_res_all.columns else 0
    riesgo = int((obj_res_all.get("estado_ejecutivo") == "RIESGO").sum()) if "estado_ejecutivo" in obj_res_all.columns else 0
    no_sub = int((obj_res_all.get("estado_ejecutivo") == "NO SUBIDO").sum()) if "estado_ejecutivo" in obj_res_all.columns else 0
    avg_cump = float(obj_res_all["cumplimiento_%"].mean()) if "cumplimiento_%" in obj_res_all.columns and not obj_res_all.empty else 0

    col1.metric("Objetivos", total_obj)
    col2.metric("Cumplidos", cumplidos)
    col3.metric("En Riesgo", riesgo)
    col4.metric("No Subidos", no_sub)
    col5.metric("Cumplimiento Promedio", f"{avg_cump:.1f}%")

    # Gauges por año (2 gauges: objetivos y áreas)
    st.markdown("### 🎛️ Medidores por año")
    for y in sel_years:
        m = metrics.get(y, {})
        obj_res = m.get("obj_res")
        area_long = m.get("area_long")
        if obj_res is None and area_long is None:
            continue

        # aplicar filtros a cada año también (para coherencia visual)
        obj_res_y = obj_res.copy() if obj_res is not None else None
        if obj_res_y is not None:
            for col, sel in [("Tipo", f_tipo_plan), ("Perspectiva", f_persp), ("Departamento", f_depto), ("Eje", f_eje), ("Tipo Objetivo", f_tipo_obj)]:
                obj_res_y = apply_filter(obj_res_y, col, sel)

        area_long_y = area_long.copy() if area_long is not None else None
        if area_long_y is not None:
            area_long_y = apply_filter(area_long_y, "AREA", f_area)
            area_long_y = apply_filter(area_long_y, "PUESTO RESPONSABLE", f_puesto)

        g1, g2 = st.columns(2)

        v_obj = float(obj_res_y["cumplimiento_%"].mean()) if obj_res_y is not None and not obj_res_y.empty else 0
        v_area = float(area_long_y["valor"].mean()*100) if area_long_y is not None and not area_long_y.empty else 0

        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v_obj,
            gauge={"axis": {"range": [0, 100]},
                   "steps": [{"range":[0,60],"color":"red"},
                             {"range":[60,90],"color":"yellow"},
                             {"range":[90,100],"color":"green"}]},
            title={"text": f"{y} – Cumplimiento Estratégico (Objetivos)"}
        ))
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v_area,
            gauge={"axis": {"range": [0, 100]},
                   "steps": [{"range":[0,60],"color":"red"},
                             {"range":[60,90],"color":"yellow"},
                             {"range":[90,100],"color":"green"}]},
            title={"text": f"{y} – Cumplimiento Operativo (Áreas)"}
        ))
        g1.plotly_chart(fig_g1, use_container_width=True)
        g2.plotly_chart(fig_g2, use_container_width=True)

    # Alertas automáticas (multi-año)
    st.markdown("### 🚨 Alertas Automáticas")
    alertas = []

    if "estado_ejecutivo" in obj_res_all.columns and "Objetivo" in obj_res_all.columns:
        crit = obj_res_all[obj_res_all["estado_ejecutivo"].isin(["CRÍTICO","RIESGO","NO SUBIDO"])]
        for _, r in crit.head(30).iterrows():
            alertas.append(f"⚠️ ({int(r['AÑO'])}) Objetivo **{r['Objetivo']}** → **{r['estado_ejecutivo']}**")

    if not area_res_area_all.empty:
        rr = area_res_area_all[area_res_area_all["cumplimiento_%"] < 60].sort_values("cumplimiento_%").head(30)
        for _, r in rr.iterrows():
            alertas.append(f"🔴 ({int(r['AÑO'])}) Área **{r['AREA']}** → **{r['cumplimiento_%']:.1f}%**")

    if alertas:
        for a in alertas:
            st.warning(a)
    else:
        st.success("✅ No se detectan alertas críticas con los filtros actuales.")

# =====================================================
# TAB: OBJETIVOS
# =====================================================
with tab_obj:
    st.subheader("🎯 Objetivos – Visualización Ejecutiva")

    if obj_res_all.empty:
        st.info("No hay datos de objetivos para los años/filtros seleccionados.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Distribución de estados (por año)**")
            fig = px.histogram(
                obj_res_all,
                x="estado_ejecutivo",
                color="AÑO",
                barmode="group",
                category_orders={"estado_ejecutivo": ["CUMPLIDO","EN SEGUIMIENTO","RIESGO","CRÍTICO","NO SUBIDO"]},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Cumplimiento promedio por año**")
            byy = obj_res_all.groupby("AÑO")["cumplimiento_%"].mean().reset_index()
            fig = px.bar(byy, x="AÑO", y="cumplimiento_%", text="cumplimiento_%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top objetivos críticos (peor cumplimiento)**")
        top_bad = obj_res_all.sort_values("cumplimiento_%").head(15)
        fig = px.bar(top_bad, x="cumplimiento_%", y="Objetivo", orientation="h", color="AÑO")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Tendencia mensual (promedio ponderado)**")
        if not obj_long_all.empty and "valor" in obj_long_all.columns:
            trend = obj_long_all.groupby(["AÑO","Mes"])["valor"].mean().reset_index()
            trend["cumplimiento_%"] = trend["valor"]*100
            fig = px.line(trend, x="Mes", y="cumplimiento_%", color="AÑO", markers=True,
                          category_orders={"Mes": MESES})
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB: AREAS
# =====================================================
with tab_area:
    st.subheader("🏢 Áreas – Control Operativo")

    if area_res_area_all.empty:
        st.info("No hay datos de áreas para los años/filtros seleccionados.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Ranking de áreas (peor → mejor) por año**")
            # mostramos por año en facets para orden visual
            fig = px.bar(
                area_res_area_all.sort_values(["AÑO","cumplimiento_%"]),
                x="cumplimiento_%", y="AREA",
                orientation="h",
                color="AÑO",
                facet_col="AÑO",
                facet_col_wrap=2
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.replace("AÑO=", "Año ")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Responsables con más tareas (carga operativa)**")
            if not area_res_puesto_all.empty:
                carga = area_res_puesto_all.groupby(["AÑO","PUESTO RESPONSABLE"])["tareas"].sum().reset_index()
                carga = carga.sort_values("tareas", ascending=False).head(15)
                fig = px.bar(carga, x="tareas", y="PUESTO RESPONSABLE", orientation="h", color="AÑO")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay data de responsables para los filtros actuales.")

        st.markdown("**Heatmap Área vs Mes (por año)**")
        if not area_long_all.empty:
            # heatmap por año: mostramos selector interno para no saturar
            year_hm = st.selectbox("Año para Heatmap", sel_years, index=len(sel_years)-1)
            al = area_long_all[area_long_all["AÑO"] == year_hm].copy()
            if al.empty:
                st.info("Sin datos para ese año.")
            else:
                heat = al.pivot_table(index="AREA", columns="Mes", values="valor", fill_value=0)
                fig = px.imshow(heat, color_continuous_scale=["red","yellow","green"])
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB: COMPARATIVOS (2024 vs 2025 y AREAS)
# =====================================================
with tab_comp:
    st.subheader("📊 Comparativos (multi-año)")

    if len(sel_years) < 2:
        st.info("Selecciona al menos 2 años en el sidebar para ver comparativos.")
    else:
        # Comparativo 1: Cumplimiento promedio objetivos (YoY)
        if not obj_res_all.empty:
            comp = obj_res_all.groupby("AÑO")["cumplimiento_%"].mean().reset_index().sort_values("AÑO")
            comp["delta_vs_prev"] = comp["cumplimiento_%"].diff()
            st.markdown("**Comparativo anual: Objetivos (promedio y delta vs año anterior)**")
            fig = px.bar(comp, x="AÑO", y="cumplimiento_%", text="cumplimiento_%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comp, use_container_width=True)

        # Comparativo 2: Áreas críticas por año
        if not area_res_area_all.empty:
            st.markdown("**Comparativo anual: % de Áreas bajo 60%**")
            tmp = area_res_area_all.copy()
            tmp["critica"] = tmp["cumplimiento_%"] < 60
            pct = tmp.groupby("AÑO")["critica"].mean().reset_index()
            pct["critica_%"] = pct["critica"]*100
            fig = px.line(pct, x="AÑO", y="critica_%", markers=True)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB: EXPORTAR (HTML con gráficas incluidas)
# =====================================================
with tab_export:
    st.subheader("📄 Exportar Reporte (HTML descargable – imprimible a PDF)")

    st.caption("En Streamlit Cloud, PDF directo suele fallar por librerías del sistema. "
               "El HTML mantiene gráficas y lo puedes imprimir a PDF desde el navegador (Ctrl+P).")

    # Gráficas clave para export
    figs = []

    if not obj_res_all.empty:
        fig_estado = px.pie(obj_res_all, names="estado_ejecutivo", title="Distribución de Estados (objetivos)")
        figs.append(fig_estado)

        byy = obj_res_all.groupby("AÑO")["cumplimiento_%"].mean().reset_index()
        fig_byy = px.bar(byy, x="AÑO", y="cumplimiento_%", title="Cumplimiento promedio por año (objetivos)")
        figs.append(fig_byy)

    if not area_res_area_all.empty:
        fig_rank = px.bar(
            area_res_area_all.sort_values(["AÑO","cumplimiento_%"]).head(50),
            x="cumplimiento_%", y="AREA", orientation="h", color="AÑO",
            title="Ranking de áreas (muestra)"
        )
        figs.append(fig_rank)

    def build_report_html():
        html_parts = []
        html_parts.append(f"""
        <html><head>
          <meta charset="utf-8"/>
          <title>Reporte Estratégico</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            h1 {{ margin-bottom: 0; }}
            .muted {{ color: #666; margin-top: 4px; }}
            .kpis {{ display:flex; gap:16px; margin: 16px 0; flex-wrap:wrap; }}
            .kpi {{ border:1px solid #eee; padding:12px 14px; border-radius:10px; min-width:180px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
            th {{ background: #f5f5f5; }}
          </style>
        </head><body>
        <h1>Reporte Estratégico y de Control</h1>
        <div class="muted">Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

        <div class="kpis">
          <div class="kpi"><b>Objetivos</b><br/>{total_obj}</div>
          <div class="kpi"><b>Cumplidos</b><br/>{cumplidos}</div>
          <div class="kpi"><b>En Riesgo</b><br/>{riesgo}</div>
          <div class="kpi"><b>No Subidos</b><br/>{no_sub}</div>
          <div class="kpi"><b>Cumplimiento Promedio</b><br/>{avg_cump:.1f}%</div>
        </div>

        <h2>Alertas</h2>
        <ul>
        """)
        if alertas:
            for a in alertas[:40]:
                html_parts.append(f"<li>{a}</li>")
        else:
            html_parts.append("<li>Sin alertas críticas.</li>")
        html_parts.append("</ul>")

        # Insertar gráficos plotly (CDN)
        for f in figs:
            html_parts.append(f.to_html(full_html=False, include_plotlyjs="cdn"))

        # Tablas resumen
        if not obj_res_all.empty:
            html_parts.append("<h2>Tabla: Resumen Objetivos</h2>")
            html_parts.append(obj_res_all.head(200).to_html(index=False))

        if not area_res_area_all.empty:
            html_parts.append("<h2>Tabla: Resumen Áreas</h2>")
            html_parts.append(area_res_area_all.head(200).to_html(index=False))

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    html_report = build_report_html()

    st.download_button(
        "⬇️ Descargar Reporte HTML",
        data=html_report,
        file_name="Reporte_Estrategico.html",
        mime="text/html"
    )

    st.info("Tip: abre el HTML en Chrome/Edge → Ctrl+P → Guardar como PDF.")

# =====================================================
# TAB: DATA (para auditoría / transparencia)
# =====================================================
with tab_data:
    st.subheader("📋 Datos (para auditoría y detalle)")

    with st.expander("Objetivos – Resumen"):
        st.dataframe(obj_res_all, use_container_width=True)

    with st.expander("Objetivos – Long (mensual)"):
        st.dataframe(obj_long_all, use_container_width=True)

    with st.expander("Áreas – Resumen por Área"):
        st.dataframe(area_res_area_all, use_container_width=True)

    with st.expander("Áreas – Resumen por Puesto"):
        st.dataframe(area_res_puesto_all, use_container_width=True)

    with st.expander("Áreas – Long (mensual)"):
        st.dataframe(area_long_all, use_container_width=True)

st.caption("Fuente: Google Sheets · Dashboard Estratégico")




