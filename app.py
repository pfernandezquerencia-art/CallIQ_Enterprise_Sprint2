# ============================================================
# DASHBOARD EJECUTIVO CALLIQ ENTERPRISE
# ============================================================

import streamlit as st
import pandas as pd
import json
import glob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

st.set_page_config(
    page_title="CallIQ Quality Analytics",
    page_icon="🎧",
    layout="wide"
)

# ------------------------------------------------------------
# ESTILOS
# ------------------------------------------------------------

st.markdown("""
<style>

.metric-container {
background-color:#f8f9fa;
padding:15px;
border-radius:8px;
border-left:5px solid #ff0000;
}

.kpi-title{
font-size:16px;
color:#666;
}

.kpi-value{
font-size:28px;
font-weight:bold;
}

.insight-box{
background:#eef2ff;
padding:15px;
border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------

@st.cache_data
def load_data():

    rutas_json = glob.glob("./data/outputs/*.json")

    if not rutas_json:
        return pd.DataFrame()

    registros = []

    for ruta in rutas_json:

        with open(ruta, "r", encoding="utf-8") as f:

            data = json.load(f)

            evaluacion = data.get("quality_evaluation", {})
            bloques = evaluacion.get("details", {}).get("block_scores", {})
            finops = data.get("finops", {})

            fecha_str = finops.get(
                "processed_at",
                datetime.now().isoformat()
            )

            registro = {

                "Fecha": pd.to_datetime(fecha_str),

                "Agente": data.get(
                    "agent_id",
                    "AG-Generico"
                ),

                "Llamada_ID": data.get(
                    "conversation_id",
                    "N/A"
                ),

                "Nota_Final": evaluacion.get(
                    "final_score",
                    0
                ),

                "KO": evaluacion.get(
                    "bi_export",
                    {}
                ).get(
                    "ko",
                    False
                ),

                "Identificacion": bloques.get(
                    "Identificación y Cumplimiento",
                    0
                ),

                "Diagnostico": bloques.get(
                    "Diagnóstico y Eficacia Técnica",
                    0
                ),

                "Empatia": bloques.get(
                    "Empatía y Gestión Emocional",
                    0
                ),

                "Cierre": bloques.get(
                    "Cierre y Seguimiento",
                    0
                )
            }

            registros.append(registro)

    df = pd.DataFrame(registros)

    df["Fecha"] = pd.to_datetime(df["Fecha"])

    df = df.sort_values(by="Fecha")

    return df


df = load_data()

if df.empty:
    st.warning("⚠️ No hay evaluaciones todavía.")
    st.stop()

# ------------------------------------------------------------
# FILTROS
# ------------------------------------------------------------

st.sidebar.title("🎧 CallIQ Analytics")

agentes = ["Todos"] + list(df["Agente"].unique())

agente_seleccionado = st.sidebar.selectbox(
    "Seleccionar Gestor",
    agentes
)

rango_fecha = st.sidebar.date_input(
    "Filtro temporal",
    []
)

df_filtrado = df.copy()

if agente_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Agente"] == agente_seleccionado]

if len(rango_fecha) == 2:
    df_filtrado = df_filtrado[
        (df_filtrado["Fecha"] >= pd.to_datetime(rango_fecha[0])) &
        (df_filtrado["Fecha"] <= pd.to_datetime(rango_fecha[1]))
    ]

# ------------------------------------------------------------
# CABECERA
# ------------------------------------------------------------

st.title("Ficha Calidad Gestor")
st.markdown("### CallIQ Enterprise Analytics")

st.markdown("---")

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

nota_media = df_filtrado["Nota_Final"].mean()
kos = df_filtrado["KO"].sum()

with col1:

    st.markdown(f"""
    <div class="metric-container">
    <div class="kpi-title">Índice Calidad</div>
    <div class="kpi-value">{nota_media:.2f}/10</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.metric(
        "Alertas KO",
        kos
    )

with col3:

    st.metric(
        "Empatía Media",
        f"{df_filtrado['Empatia'].mean():.2f}"
    )

with col4:

    st.metric(
        "Llamadas Auditadas",
        len(df_filtrado)
    )

# ------------------------------------------------------------
# VELOCÍMETRO
# ------------------------------------------------------------

fig_gauge = go.Figure(go.Indicator(

    mode="gauge+number",

    value=nota_media,

    title={"text": "Calidad Global"},

    gauge={

        "axis": {"range": [0, 10]},

        "steps": [

            {"range": [0, 5], "color": "#ff4d4d"},
            {"range": [5, 7], "color": "#ffa500"},
            {"range": [7, 9], "color": "#ffd700"},
            {"range": [9, 10], "color": "#28a745"}

        ]

    }

))

st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------
# EVOLUCIÓN TEMPORAL
# ------------------------------------------------------------

st.subheader("📈 Evolución de Calidad")

df_evo = df_filtrado.groupby("Fecha")["Nota_Final"].mean().reset_index()

fig_line = px.line(
    df_evo,
    x="Fecha",
    y="Nota_Final",
    markers=True
)

fig_line.update_yaxes(range=[0, 10])

st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------------------------
# RADAR EPÍGRAFES
# ------------------------------------------------------------

st.subheader("📊 Desglose por Epígrafes")

epigrafes = ["Identificacion", "Diagnostico", "Empatia", "Cierre"]

medias = [df_filtrado[e].mean() for e in epigrafes]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(

    r=medias + [medias[0]],

    theta=epigrafes + [epigrafes[0]],

    fill="toself"

))

fig_radar.update_layout(

    polar=dict(

        radialaxis=dict(

            visible=True,
            range=[0, 10]

        )

    )

)

st.plotly_chart(fig_radar, use_container_width=True)

# ------------------------------------------------------------
# DISTRIBUCIÓN NOTAS
# ------------------------------------------------------------

st.subheader("Distribución de Calidad")

fig_hist = px.histogram(
    df_filtrado,
    x="Nota_Final",
    nbins=10
)

st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

st.subheader("Heatmap de desempeño")

heat = df_filtrado[
    ["Identificacion", "Diagnostico", "Empatia", "Cierre"]
]

fig_heat = px.imshow(
    heat,
    aspect="auto",
    color_continuous_scale="RdYlGn"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------------------
# RANKING GESTORES
# ------------------------------------------------------------

st.subheader("Ranking Gestores")

ranking = df.groupby("Agente")["Nota_Final"].mean().sort_values(ascending=False)

ranking_df = ranking.reset_index()

ranking_df.columns = ["Agente", "Nota Media"]

st.dataframe(ranking_df)

# ------------------------------------------------------------
# INSIGHTS AUTOMÁTICOS
# ------------------------------------------------------------

st.subheader("🔎 Insights Automáticos")

bloque_debil = heat.mean().idxmin()

insights = []

if kos > 0:
    insights.append(f"Se detectan {kos} alertas críticas (KO).")

insights.append(f"El bloque más débil es **{bloque_debil}**.")

if nota_media > 8:
    insights.append("La calidad global del gestor es alta.")

if nota_media < 6:
    insights.append("Existe riesgo de calidad baja.")

st.markdown(
    "<div class='insight-box'>" +
    "<br>".join(insights) +
    "</div>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# TABLA DETALLE
# ------------------------------------------------------------

st.markdown("---")

st.subheader("Detalle Llamadas")

st.dataframe(

    df_filtrado[
        [
            "Fecha",
            "Llamada_ID",
            "Nota_Final",
            "Identificacion",
            "Diagnostico",
            "Empatia",
            "Cierre",
            "KO"
        ]
    ],

    use_container_width=True
)