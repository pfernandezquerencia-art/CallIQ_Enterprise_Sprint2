# ============================================================
# DASHBOARD EJECUTIVO CALLIQ (ENTERPRISE EDITION v1.5.3)
# ============================================================
# Visualiza métricas de Calidad, Riesgo, FinOps y Auditoría
# Conecta con: calliq_registry.db (SQLite)
# ============================================================

import streamlit as st
import sqlite3
import pandas as pd
import time
import os

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="CallIQ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para dar look corporativo
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; }
    .metric-card { border: 1px solid #e6e6e6; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 CallIQ: Centro de Control de Calidad y FinOps")
st.markdown("---")

# 2. FUNCIÓN DE CARGA DE DATOS
def load_data():
    db_path = "calliq_registry.db"
    
    # Verificar si existe la BD
    if not os.path.exists(db_path):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        # Cargamos datos de BI incluyendo el nuevo desglose de costes v1.5.3
        query = "SELECT * FROM evaluations_bi ORDER BY processed_at DESC"
        df = pd.read_sql(query, conn)
        conn.close()

        if not df.empty:
            df['processed_at'] = pd.to_datetime(df['processed_at'])
            
            # Retrocompatibilidad con v1.5.2 (si existe cost_usd antiguo)
            if 'total_cost' not in df.columns:
                df['total_cost'] = df.get('cost_usd', 0.0)
            if 'stt_cost' not in df.columns:
                df['stt_cost'] = df['total_cost'] * 0.7  # Estimación si no existe
            if 'llm_cost' not in df.columns:
                df['llm_cost'] = df['total_cost'] * 0.3  # Estimación si no existe
                
        return df
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return pd.DataFrame()

# Cargar el DataFrame
df = load_data()

# 3. LÓGICA DEL DASHBOARD
if df.empty:
    st.warning("⚠️ No hay datos disponibles todavía.")
    st.info("👉 Ejecuta el motor 'calliq_pipeline_enterprise_v1.5.3.py' para procesar llamadas y generar datos.")
    
    if st.button("🔄 Reintentar conexión"):
        st.rerun()

else:
    # --- BARRA LATERAL (FILTROS) ---
    st.sidebar.header("🔍 Filtros de Visualización")
    
    # Filtro por Cliente (Tenant)
    tenants_list = ["Todos"] + list(df['tenant_id'].unique())
    selected_tenant = st.sidebar.selectbox("Cliente / Tenant", tenants_list)

    # Filtro por Versión de Modelo
    versions_list = ["Todas"] + list(df['model_ver'].unique())
    selected_version = st.sidebar.selectbox("Versión del Modelo", versions_list)

    # Aplicar filtros
    df_view = df.copy()
    if selected_tenant != "Todos":
        df_view = df_view[df_view['tenant_id'] == selected_tenant]
    
    if selected_version != "Todas":
        df_view = df_view[df_view['model_ver'] == selected_version]

    # --- FILA 1: KPIs PRINCIPALES (Métricas) ---
    st.subheader("Resumen Ejecutivo")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Cálculo de métricas
    total_calls = len(df_view)
    avg_score = df_view['final_score'].mean()
    total_kos = df_view['ko_triggered'].sum()
    ko_rate = (total_kos / total_calls * 100) if total_calls > 0 else 0
    total_cost = df_view['total_cost'].sum()

    kpi1.metric(
        label="📞 Llamadas Auditadas",
        value=total_calls
    )
    
    kpi2.metric(
        label="⭐ Calidad Media (0-10)",
        value=f"{avg_score:.2f}",
        delta=f"{avg_score - 8.0:.2f} vs Objetivo" # Simulación de un objetivo de 8.0
    )
    
    kpi3.metric(
        label="🚨 Eliminatorias (KO)",
        value=int(total_kos),
        delta=f"{ko_rate:.1f}% Tasa Fallo",
        delta_color="inverse" # Rojo si sube es malo
    )

    kpi4.metric(
        label="💰 Coste Operativo Total",
        value=f"${total_cost:.4f}",
        help="Suma exacta de STT + Inferencia LLM"
    )

    st.markdown("---")

    # --- FILA 2: GRÁFICOS ANALÍTICOS ---
    chart1, chart2, chart3 = st.columns([2, 1, 1])

    with chart1:
        st.subheader("📈 Evolución Temporal de Calidad")
        if total_calls > 0:
            # Gráfico de línea temporal
            chart_data = df_view.set_index('processed_at')[['final_score']]
            st.line_chart(chart_data, color="#0068c9")
        else:
            st.write("Sin datos para gráfico.")

    with chart2:
        st.subheader("⚖️ Distribución Notas")
        if total_calls > 0:
            # Histograma simple de scores redondeados
            scores_dist = df_view['final_score'].round().value_counts().sort_index()
            st.bar_chart(scores_dist, color="#29b5e8")

    with chart3:
        st.subheader("💸 Desglose FinOps")
        if total_calls > 0:
            # Gráfico de desglose de costes (STT vs LLM)
            cost_data = pd.DataFrame({
                "Concepto": ["Transcripción (STT)", "Inferencia IA (LLM)"],
                "Coste ($)": [df_view['stt_cost'].sum(), df_view['llm_cost'].sum()]
            }).set_index("Concepto")
            st.bar_chart(cost_data, color="#ff2b2b")

    # --- FILA 3: DETALLE DE DATOS ---
    st.subheader("📋 Auditoría Detallada")
    
    # Preparar tabla para visualización limpia
    table_df = df_view[['processed_at', 'tenant_id', 'call_id', 'model_ver', 'final_score', 'ko_triggered', 'total_cost']].copy()
    
    # Formato visual
    st.dataframe(
        table_df,
        column_config={
            "processed_at": st.column_config.DatetimeColumn("Fecha/Hora", format="DD/MM/YYYY HH:mm"),
            "tenant_id": "Cliente",
            "call_id": "ID Interacción",
            "model_ver": "Modelo",
            "final_score": st.column_config.ProgressColumn(
                "Nota Final",
                format="%.2f",
                min_value=0,
                max_value=10,
            ),
            "ko_triggered": st.column_config.CheckboxColumn("KO (Fallo)", disabled=True),
            "total_cost": st.column_config.NumberColumn("Coste Total ($)", format="$%.4f")
        },
        use_container_width=True,
        hide_index=True
    )

    # --- PIE DE PÁGINA ---
    st.markdown("---")
    col_l, col_r = st.columns([8, 2])
    with col_l:
        st.caption(f"🛡️ Ecosistema CallIQ Enterprise v1.5.3 (Gold Master) | Base de datos validada: {len(df)} registros.")
    with col_r:
        if st.button("🔄 Actualizar Datos"):
            st.rerun()