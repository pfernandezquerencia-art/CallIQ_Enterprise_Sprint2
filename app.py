# ============================================================\r
# DASHBOARD EJECUTIVO CALLIQ (INSPIRADO EN CRE MORA)
# ============================================================\r

import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
import plotly.graph_objects as go

# 1. CONFIGURACIÓN DE PÁGINA (Debe ser la primera línea)
st.set_page_config(page_title="Ficha Calidad Gestor", page_icon="🎧", layout="wide")

# Estilos CSS para imitar el informe PDF
st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #ff0000; }
    .kpi-title { font-size: 16px; color: #666; margin-bottom: 5px; }
    .kpi-value { font-size: 28px; font-weight: bold; color: #1e1e1e; }
    </style>
""", unsafe_allow_html=True)

# 2. CARGA DE DATOS (Simulando la lectura de tus JSON y CSV)
@st.cache_data
def load_data():
    # En un caso real, aquí leerías tu CSV calliq_export_BI_...csv
    # Para el esqueleto, creamos un DataFrame con la estructura de tus logs:
    datos = {
        'Fecha': pd.to_datetime(['2026-02-24', '2026-02-25', '2026-02-26', '2026-02-27', '2026-02-27']),
        'Agente': ['AG-102', 'AG-102', 'AG-102', 'AG-102', 'AG-105'],
        'Llamada_ID': ['10249222001', '10249696001', '10249787001', '10249954001', '10250000001'],
        'Nota_Final': [6.2, 6.8, 8.3, 8.05, 9.0],
        'KO': [False, False, False, False, False],
        'Identificacion': [8.0, 10.0, 7.5, 9.5, 10.0],
        'Diagnostico': [4.0, 4.0, 8.0, 9.5, 10.0],
        'Empatia': [9.0, 9.0, 9.5, 9.0, 9.5],
        'Cierre': [6.0, 6.0, 8.5, 5.0, 7.0]
    }
    return pd.DataFrame(datos)

df = load_data()

# 3. BARRA LATERAL (Filtros)
st.sidebar.markdown("# 🎧 CallIQ Analytics")
st.sidebar.header("Filtros de Búsqueda")

agentes_disponibles = ["Todos"] + list(df['Agente'].unique())
agente_seleccionado = st.sidebar.selectbox("👤 Seleccionar Gestor", agentes_disponibles)

# Filtrar el DataFrame
if agente_seleccionado != "Todos":
    df_filtrado = df[df['Agente'] == agente_seleccionado]
else:
    df_filtrado = df

# 4. CABECERA
st.title(f"Ficha Calidad Gestor: {agente_seleccionado}")
st.markdown("CallIQ Enterprise")
st.markdown("---")

# 5. KPIs PRINCIPALES (Como en la Ficha CRE Mora)
col1, col2, col3, col4 = st.columns(4)

nota_media = df_filtrado['Nota_Final'].mean()
kos_totales = df_filtrado['KO'].sum()

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="kpi-title">Índice de Calidad Objetiva</div>
        <div class="kpi-value">{nota_media:.2f} / 10</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container" style="border-left-color: {'#ff0000' if kos_totales > 0 else '#28a745'}">
        <div class="kpi-title">Alertas Críticas (KO)</div>
        <div class="kpi-value">{kos_totales}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.metric(label="Experiencia Cliente (Empatía)", value=f"{df_filtrado['Empatia'].mean():.2f}")

with col4:
    st.metric(label="Llamadas Auditadas", value=len(df_filtrado))

st.markdown("<br>", unsafe_allow_html=True)

# 6. GRÁFICAS: EVOLUCIÓN Y EPÍGRAFES
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("📈 Evolución Temporal de la Calidad")
    # Agrupamos por fecha
    df_evolucion = df_filtrado.groupby('Fecha')['Nota_Final'].mean().reset_index()
    
    fig_line = px.line(df_evolucion, x='Fecha', y='Nota_Final', markers=True, 
                       title="Tendencia de Calidad", 
                       line_shape='spline', # Curvas suaves
                       color_discrete_sequence=['#ff0000'])
    fig_line.update_yaxes(range=[0, 10])
    st.plotly_chart(fig_line, use_container_width=True)

with col_der:
    st.subheader("📊 Desglose por Epígrafes")
    # Calculamos la media de cada bloque de evaluación
    epigrafes = ['Identificacion', 'Diagnostico', 'Empatia', 'Cierre']
    medias_epigrafes = [df_filtrado[ep].mean() for ep in epigrafes]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=medias_epigrafes + [medias_epigrafes[0]], # Cerrar el radar
        theta=epigrafes + [epigrafes[0]],
        fill='toself',
        line_color='#ff0000'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False)
    st.plotly_chart(fig_radar, use_container_width=True)

# 7. TABLA DE DETALLE
st.markdown("---")
st.subheader("📋 Detalle de Llamadas")
st.dataframe(df_filtrado[['Fecha', 'Llamada_ID', 'Nota_Final', 'Identificacion', 'Diagnostico', 'Empatia', 'Cierre', 'KO']], use_container_width=True)