
![Python](https://img.shields.io/badge/python-3.10-blue)
![Status](https://img.shields.io/badge/status-research_project-orange)
![License](https://img.shields.io/badge/license-MIT-green)


# 🚀 CallIQ Enterprise v1.5.4 (Multi-Domain)

**Motor de Evaluación Automática de Calidad Conversacional con Inteligencia Artificial**

CallIQ Enterprise es un sistema de análisis automático de interacciones de contact center que combina **modelos de lenguaje (LLM)** con **algoritmos deterministas** para evaluar la calidad de conversaciones entre agentes y clientes.

El diseño del sistema separa explícitamente:

* **Comprensión semántica mediante LLM**
* **Cálculo determinista del score en Python**

Esta arquitectura híbrida permite:

* trazabilidad completa del proceso
* reproducibilidad del scoring
* control de alucinaciones del modelo
* auditabilidad en entornos regulados

---

# 🎯 Objetivo del Proyecto

Demostrar la viabilidad técnica de un sistema capaz de:

* Analizar automáticamente interacciones de voz
* Clasificar dinámicamente el tipo de llamada
* Evaluar la calidad del agente mediante manuales operativos
* Reducir el coste de auditoría manual
* Generar métricas estructuradas para Business Intelligence

---

# 🧩 Características Principales

* Evaluación automática de calidad conversacional
* Arquitectura modular desacoplada
* Clasificación semántica del tipo de interacción
* Evaluación híbrida **LLM + reglas deterministas**
* Anonimización de datos sensibles (GDPR)
* Dashboard analítico en Streamlit
* Exportación de resultados a CSV / BI

---

# 🏗️ Arquitectura del Sistema

El sistema sigue una arquitectura modular basada en un pipeline de procesamiento:

```
Audio
 ↓
Speech-to-Text (AssemblyAI)
 ↓
Anonimización GDPR (Regex + SpaCy)
 ↓
Router semántico (clasificación del tipo de llamada)
 ↓
Carga dinámica del manual de calidad (RAG)
 ↓
Evaluación cognitiva mediante LLM
 ↓
Cálculo determinista del score y reglas KO
 ↓
Cálculo de costes (FinOps)
 ↓
Exportación a CSV / BI
```

---

# ⚙️ Componentes del Sistema

| Componente                   | Función                              |
| ---------------------------- | ------------------------------------ |
| IngestionModule              | Ingesta de audios y transcripción    |
| AnonymizationModule          | Anonimización de datos sensibles     |
| Semantic Router              | Clasificación del tipo de llamada    |
| Evaluation Model Builder     | Interpretación del manual de calidad |
| Dynamic Evaluation Engine    | Evaluación semántica mediante LLM    |
| Deterministic Scoring Engine | Cálculo matemático del score         |
| FinOps Engine                | Cálculo de coste por interacción     |

---
## ⚙️ Ejecución del Pipeline

El sistema procesa las interacciones mediante un pipeline de evaluación automático.

Etapas principales:

- Speech-to-Text de la llamada
- Anonimización GDPR
- Clasificación semántica del tipo de interacción
- Inyección dinámica de manual de calidad (RAG)
- Evaluación cognitiva mediante LLM
- Cálculo determinista del score
- Exportación de resultados para BI

<p align="center">
<img src="docs/images/pipeline_processing.jpg" width="900">
</p>


# 📊 Dashboard Analítico

El sistema incluye un **dashboard interactivo desarrollado en Streamlit** para analizar las métricas generadas.

Características principales:

* Índice de calidad global con velocímetro
* Evolución temporal de la calidad
* Radar de epígrafes de evaluación
* Heatmap de desempeño
* Ranking de gestores
* Insights automáticos

<p align="center">
<img src="docs/images/dashboard_calliq.jpg" width="900" alt="Dashboard CallIQ">
</p>

---

# 🛠️ Instalación

## 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/calliq-enterprise.git
cd calliq-enterprise
```

---

## 2. Crear entorno virtual

```bash
python -m venv venv
```

Activar entorno:

Windows

```bash
.\venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Instalar modelo de SpaCy:

```bash
python -m spacy download es_core_news_md
```

---

# 🔑 Configuración de Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto.

```
ASSEMBLYAI_API_KEY=tu_clave_de_assemblyai
GEMINI_API_KEY=tu_clave_de_google_gemini
```

---

# 📁 Estructura de Datos

```
data/
 ├── audios/
 ├── outputs/
 ├── manual_calidad.txt
 ├── manual_calidad_ventas.txt
 ├── manual_calidad_soporte.txt
 └── manual_calidad_retencion.txt
```

* **audios/** → grabaciones de entrada
* **outputs/** → evaluaciones generadas en JSON

---

# ▶️ Ejecución del Sistema

## Procesar audios

```bash
python calliq_pipeline_enterprise_v1.5.3.py
```

El sistema:

* procesa los audios (Speech-to text transcription)
* NLP-based anonimizador
* LLM evaluación semántica
* genera un JSON por interacción
* exporta métricas a CSV para BI



## Lanzar dashboard

```bash
streamlit run app.py
```

---

# 📊 Exportación para BI

El sistema genera automáticamente un archivo CSV con métricas estructuradas.

Ejemplo:

| conversation_id | agent_id | score | KO    |
| --------------- | -------- | ----- | ----- |
| 10249222001     | AG-102   | 8.45  | False |
| 10249696001     | AG-102   | 9.12  | False |

Este dataset puede integrarse con:

* Power BI
* Tableau
* herramientas de reporting

---

# 🔭 Roadmap

Próximas mejoras:

* procesamiento asíncrono para grandes volúmenes
* integración con plataformas CCaaS
* despliegue en arquitectura cloud

---

# 👨‍💻 Autores

EOI - Grupo 4 - (abril 2025)


Proyecto desarrollado como investigación sobre **inteligencia conversacional aplicada a contact centers**.


Este sistema es un prototipo académico y presenta algunas limitaciones:

* Procesamiento secuencial de llamadas

* Dependencia de APIs externas

* Clasificación semántica basada en LLM

* Sin despliegue en infraestructura cloud

## 🎥 Demo del sistema
El sistema incluye una interfaz de demostración basada en Streamlit.

Permite:

- subir grabaciones de llamadas
- ejecutar el pipeline de evaluación
- visualizar la puntuación de calidad
- inspeccionar el detalle de la auditoría

Para lanzar la demo:

streamlit run app.py

## 🚀 Elementos clave

- Automatic speech transcription using AssemblyAI
- GDPR-compliant anonymization pipeline
- LLM-based semantic routing of conversations
- Dynamic evaluation model generation from quality manuals (RAG)
- Hybrid evaluation engine (LLM + deterministic scoring)
- Prompt injection protection
- Model registry with integrity verification
- Cost estimation with FinOps metrics
- Export-ready dataset for Business Intelligence tools



---
### Instalación automática (Windows)

El proyecto incluye un script de instalación que prepara automáticamente el entorno.

Ejecutar:

```bash
setup.bat