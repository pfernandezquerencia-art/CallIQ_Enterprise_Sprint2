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
.dept-card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
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
            contexto = data.get("context", {})

            fecha_str = finops.get("processed_at", datetime.now().isoformat())

            identificacion = next((v for k, v in bloques.items() if "identificaci" in k.lower()), 0)
            diagnostico = next((v for k, v in bloques.items() if "diagn" in k.lower() or "necesidad" in k.lower()), 0)
            empatia = next((v for k, v in bloques.items() if "empat" in k.lower() or "argumentaci" in k.lower() or "retenci" in k.lower()), 0)
            cierre = next((v for k, v in bloques.items() if "cierr" in k.lower()), 0)

            registro = {
                "Fecha": pd.to_datetime(fecha_str),
                "Agente": contexto.get("agent_id", data.get("agent_id", "AG-Generico")),
                "Departamento": contexto.get("queue", "General"),
                "Llamada_ID": data.get("conversation_id", "N/A"),
                "Nota_Final": evaluacion.get("final_score", 0),
                "KO": evaluacion.get("bi_export", {}).get("ko", False),
                "Identificacion": identificacion,
                "Diagnostico": diagnostico,
                "Empatia": empatia,
                "Cierre": cierre
            }

            registros.append(registro)

    df = pd.DataFrame(registros)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(by="Fecha")
    
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ No hay evaluaciones todavía. Procesa algunos audios primero.")
    st.stop()

# ------------------------------------------------------------
# FILTROS
# ------------------------------------------------------------

st.sidebar.title("🎧 CallIQ Analytics")

agentes = ["Todos"] + list(df["Agente"].unique())
agente_seleccionado = st.sidebar.selectbox("Seleccionar Gestor", agentes)

departamentos = ["Todos"] + list(df["Departamento"].unique())
departamento_seleccionado = st.sidebar.selectbox("Seleccionar Departamento", departamentos)

rango_fecha = st.sidebar.date_input("Filtro temporal", [])

df_filtrado = df.copy()

if agente_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Agente"] == agente_seleccionado]

if departamento_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Departamento"] == departamento_seleccionado]

if len(rango_fecha) == 2:
    df_filtrado = df_filtrado[
        (df_filtrado["Fecha"].dt.date >= rango_fecha[0]) &
        (df_filtrado["Fecha"].dt.date <= rango_fecha[1])
    ]

if df_filtrado.empty:
    st.warning("⚠️ No hay datos para los filtros seleccionados.")
    st.stop()

# ------------------------------------------------------------
# CABECERA
# ------------------------------------------------------------

st.title("Dashboard de Calidad Multi-Dominio")
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
    <div class="kpi-title">Índice Calidad Global</div>
    <div class="kpi-value">{nota_media:.2f}/10</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Alertas KO (Muerte Súbita)", int(kos))

with col3:
    st.metric("Empatía Media", f"{df_filtrado['Empatia'].mean():.2f}")

with col4:
    st.metric("Llamadas Auditadas", len(df_filtrado))

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
st.plotly_chart(fig_gauge, width='stretch')

# ------------------------------------------------------------
# EVOLUCIÓN TEMPORAL
# ------------------------------------------------------------

st.subheader("📈 Evolución de Calidad")
df_evo = df_filtrado.groupby("Fecha")["Nota_Final"].mean().reset_index()

fig_line = px.line(
    df_evo, x="Fecha", y="Nota_Final", markers=True
)
fig_line.update_yaxes(range=[0, 10])
st.plotly_chart(fig_line, width='stretch')

# ------------------------------------------------------------
# RADAR Y HEATMAP (En dos columnas para ahorrar espacio)
# ------------------------------------------------------------
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("📊 Desglose por Epígrafes")
    epigrafes = ["Identificacion", "Diagnostico", "Empatia", "Cierre"]
    medias = [df_filtrado[e].mean() for e in epigrafes]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=medias + [medias[0]],
        theta=epigrafes + [epigrafes[0]],
        fill="toself",
        line_color="#1f77b4"
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
    st.plotly_chart(fig_radar, width='stretch')

with col_viz2:
    st.subheader("🔥 Heatmap de Desempeño")
    heat = df_filtrado[["Identificacion", "Diagnostico", "Empatia", "Cierre"]]
    
    # 1. Usamos una escala analítica ("Blues" o "YlGnBu") en lugar de semáforo
    # 2. text_auto=".1f" escribe la nota exacta dentro de cada recuadro (¡A los jefes les encanta esto!)
    # 3. Ajustamos el contraste dinámicamente al min y max real de los datos
    fig_heat = px.imshow(
        heat, 
        aspect="auto", 
        color_continuous_scale="Blues", 
        text_auto=".1f",
        range_color=[heat.min().min(), heat.max().max()] # Contraste dinámico
    )
    
    # Quitamos la barra lateral de color para que el gráfico sea más grande y limpio
    fig_heat.update_layout(coloraxis_showscale=False)
    
    st.plotly_chart(fig_heat, width='stretch')
# ------------------------------------------------------------
# RANKING GESTORES (Gráfico + Tabla)
# ------------------------------------------------------------

st.markdown("---")
st.subheader("🏆 Ranking de Gestores")

# Agrupamos por agente (aplicamos filtro global)
ranking = df_filtrado.groupby("Agente")["Nota_Final"].mean().sort_values(ascending=True) # Ascendente para el gráfico horizontal
ranking_df = ranking.reset_index()
ranking_df.columns = ["Agente", "Nota Media"]

col_rank1, col_rank2 = st.columns([1, 2])

with col_rank1:
    # Mostramos la tabla ordenada de mayor a menor
    st.dataframe(ranking_df.sort_values(by="Nota Media", ascending=False).style.format({"Nota Media": "{:.2f}"}), use_container_width=True)

with col_rank2:
    # Gráfico de barras horizontal
    fig_ranking = px.bar(
        ranking_df,
        x="Nota Media",
        y="Agente",
        orientation="h",
        text="Nota Media",
        color="Nota Media",
        color_continuous_scale="Blues"
    )
    fig_ranking.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_ranking.update_layout(xaxis=dict(range=[0, 10]), coloraxis_showscale=False)
    st.plotly_chart(fig_ranking, width='stretch')

# ------------------------------------------------------------
# ANÁLISIS POR DEPARTAMENTO (FORTALEZAS Y DEBILIDADES)
# ------------------------------------------------------------

st.markdown("---")
st.subheader("🏢 Fortalezas y Áreas de Mejora por Departamento")

# Agrupar datos por departamento
df_dept = df_filtrado.groupby("Departamento")[["Identificacion", "Diagnostico", "Empatia", "Cierre"]].mean()

if not df_dept.empty:
    # Imprimir tarjetas de fortalezas/debilidades
    cols_dept = st.columns(len(df_dept))
    
    for idx, (dept, row) in enumerate(df_dept.iterrows()):
        with cols_dept[idx]:
            fortaleza = row.idxmax()
            nota_max = row.max()
            debilidad = row.idxmin()
            nota_min = row.min()
            
            st.markdown(f"""
            <div class="dept-card">
                <h4 style="text-align:center; color:#333;">{dept.replace('_', ' ')}</h4>
                <p style="color:green;"><b>⭐ Fuerte:</b> {fortaleza} ({nota_max:.1f}/10)</p>
                <p style="color:red;"><b>⚠️ Mejora:</b> {debilidad} ({nota_min:.1f}/10)</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("") # Espaciador
    
    # Gráfico comparativo de departamentos
    df_dept_melted = df_dept.reset_index().melt(id_vars="Departamento", var_name="Bloque", value_name="Puntuación")
    fig_dept = px.bar(
        df_dept_melted, 
        x="Departamento", 
        y="Puntuación", 
        color="Bloque", 
        barmode="group",
        title="Comparativa de Habilidades entre Departamentos"
    )
    fig_dept.update_yaxes(range=[0, 10])
    st.plotly_chart(fig_dept, width='stretch')
else:
    st.info("No hay datos suficientes para analizar por departamento.")

# ------------------------------------------------------------
# INSIGHTS AUTOMÁTICOS
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔎 Insights Automáticos (IA)")

if not heat.empty:
    bloque_debil = heat.mean().idxmin()
    insights = []

    if kos > 0:
        insights.append(f"🚨 Se han detectado **{int(kos)} interacciones con alertas críticas** (Muerte Súbita/KO). Revisar procedimientos urgentes.")
    
    insights.append(f"📉 A nivel global, el área operativa más débil es **{bloque_debil}** con una media de {heat.mean().min():.2f}/10.")

    if nota_media > 8:
        insights.append("⭐ La calidad global operativa se mantiene en niveles de **excelencia** (> 8.0).")
    if nota_media < 6:
        insights.append("⚠️ Existe **riesgo crítico de calidad** en el conjunto de interacciones auditadas.")

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
st.subheader("Detalle de Auditorías (Interacciones)")

st.dataframe(
    df_filtrado[
        [
            "Fecha",
            "Agente",
            "Departamento",
            "Llamada_ID",
            "Nota_Final",
            "Identificacion",
            "Diagnostico",
            "Empatia",
            "Cierre",
            "KO"
        ]
    ].sort_values(by="Fecha", ascending=False).style.format({
        "Nota_Final": "{:.2f}",
        "Identificacion": "{:.1f}",
        "Diagnostico": "{:.1f}",
        "Empatia": "{:.1f}",
        "Cierre": "{:.1f}"
    }),
    use_container_width=True
)