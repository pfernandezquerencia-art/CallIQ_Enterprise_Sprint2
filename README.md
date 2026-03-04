README.md

![Python](https://img.shields.io/badge/python-3.10-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License](https://img.shields.io/badge/license-MIT-green)


CallIQ Enterprise
Motor de Evaluación Automática de Calidad Conversacional con Inteligencia Artificial
Descripción

CallIQ Enterprise es un sistema de análisis automático de interacciones de contact center que combina modelos de lenguaje (LLM) con algoritmos deterministas para evaluar la calidad de las conversaciones entre agentes y clientes.

El sistema procesa grabaciones de llamadas, transcribe el audio, anonimiza los datos sensibles, clasifica el tipo de interacción y aplica automáticamente un modelo de evaluación de calidad basado en manuales operativos.

A diferencia de soluciones tradicionales, el cálculo final de la puntuación es matemático y determinista, garantizando trazabilidad, consistencia y auditabilidad en entornos regulados.

Objetivo del proyecto

El objetivo del proyecto es demostrar la viabilidad técnica y económica de un sistema capaz de:

Analizar automáticamente interacciones de voz en contact centers

Evaluar la calidad del agente en base a manuales operativos

Reducir el coste de auditoría manual

Aumentar la cobertura de evaluaciones

Generar métricas estructuradas para herramientas de Business Intelligence

Arquitectura del sistema

El sistema está diseñado como un pipeline de procesamiento modular, compuesto por varias etapas consecutivas.



Componentes principales

Audio Calls
     │
     ▼
Speech-to-Text (AssemblyAI)
     │
     ▼
Anonymization Engine
     │
     ▼
Semantic Router (LLM)
     │
     ▼
Dynamic Quality Model (RAG)
     │
     ▼
Evaluation Engine (LLM)
     │
     ▼
Deterministic Scoring Engine
     │
     ▼
FinOps Cost Engine
     │
     ▼
BI Export (CSV / SQLite)


1. IngestionModule

Se encarga de:

Subir el audio al servicio de transcripción

Obtener la transcripción con diarización de hablantes

Extraer metadatos de duración

Tecnología utilizada:

AssemblyAI Speech-to-Text API

2. AnonymizationModule

Aplica anonimización automática de información sensible para cumplir con principios de protección de datos.

Métodos utilizados:

Expresiones regulares

Reconocimiento de entidades mediante SpaCy (NER)

Tipos de datos anonimizados:

Teléfonos

Emails

Identificadores

Nombres de persona

Localizaciones

El sistema elimina el texto original en memoria y conserva únicamente un hash criptográfico para trazabilidad.

3. Semantic Router

Clasifica automáticamente el tipo de interacción utilizando un modelo LLM.

Categorías soportadas:

Ventas

Recobro

Soporte técnico

Retención

Información general

Esta clasificación permite seleccionar dinámicamente el manual de calidad específico que se utilizará para la evaluación.

4. Evaluation Model Builder

Este módulo interpreta manuales de calidad (documentos operativos de auditoría) y los convierte en un modelo estructurado en formato JSON.

El modelo contiene:

Bloques de evaluación

Criterios

Pesos de cada bloque

Reglas eliminatorias (KO)

Esto permite adaptar el sistema a diferentes organizaciones o campañas.

5. Dynamic Evaluation Engine

Este componente realiza la evaluación cognitiva de la llamada utilizando un modelo de lenguaje.

El proceso consta de dos fases:

Extracción de señales

Identificación de:

Sentimiento del agente

Riesgo de abandono (churn)

Emociones detectadas

Evaluación formal

Aplicación estricta del modelo de calidad:

Clasificación del tipo de llamada

Puntuación por bloques

Detección de eliminatorias

Para garantizar consistencia, el resultado del LLM es sanitizado y validado estructuralmente.

6. Motor de cálculo determinista

El cálculo final de la puntuación se realiza mediante un algoritmo matemático.

Características:

Aplicación de pesos definidos en el modelo

Normalización automática

Aplicación de reglas eliminatorias

Puntuación final entre 0 y 10

Esto evita dependencia directa del LLM para el resultado final.

7. Model Registry

El sistema incorpora un registro de modelos que permite:

Versionado de modelos de evaluación

Control de integridad mediante hashes SHA256

Persistencia de modelos activos

Trazabilidad histórica

La persistencia se implementa mediante SQLite.

8. FinOps Engine

El sistema calcula automáticamente el coste de cada evaluación considerando:

Coste de transcripción

Coste de tokens del modelo LLM

Esto permite estimar el coste operativo del sistema a escala.

9. Exportación de datos para BI

Los resultados de las evaluaciones se almacenan en una base de datos SQLite y pueden exportarse automáticamente a CSV para su análisis en herramientas de BI como:

Power BI

Tableau

Excel

Tecnologías utilizadas

Lenguaje principal:

Python

Librerías principales:

requests

spacy

pypdf

sqlite3

pandas

Servicios externos:

AssemblyAI (Speech to Text)

Google Gemini API (LLM)

Estructura del proyecto
calliq/
│
├── calliq_pipeline_enterprise.py
│
├── data/
│   ├── audios/
│   ├── outputs/
│   ├── manual_calidad.txt
│   ├── manual_calidad_ventas.txt
│   ├── manual_calidad_soporte.txt
│   └── manual_calidad_retencion.txt
│
├── calliq_registry.db
└── pipeline.log
Ejecución del sistema

Configurar variables de entorno en un archivo .env:

ASSEMBLYAI_API_KEY=your_key
GEMINI_API_KEY=your_key

Colocar audios en la carpeta:

data/audios

Ejecutar el pipeline:

python calliq_pipeline_enterprise.py

El sistema procesará automáticamente todos los audios disponibles.

Salidas generadas

Por cada interacción procesada se generan:

JSON detallado de evaluación

Registro en base de datos SQLite

Exportación CSV para BI

Logs del pipeline

Limitaciones del prototipo

Este sistema es un prototipo académico y presenta algunas limitaciones:

Procesamiento secuencial de llamadas

Dependencia de APIs externas

Clasificación semántica basada en LLM

Sin despliegue en infraestructura cloud

Trabajo futuro

Líneas de evolución del sistema:

procesamiento paralelo de llamadas

integración con plataformas de contact center

router semántico basado en embeddings

dashboard analítico en tiempo real

despliegue en arquitectura cloud

Autor

Proyecto desarrollado como Trabajo Final de Máster en Inteligencia Artificial aplicada a Contact Centers.

## 🎥 Demo del sistema
El sistema incluye una interfaz de demostración basada en Streamlit.

Permite:

- subir grabaciones de llamadas
- ejecutar el pipeline de evaluación
- visualizar la puntuación de calidad
- inspeccionar el detalle de la auditoría

Para lanzar la demo:

streamlit run app.py

## 🚀 Key Features

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

## ⚙️ Requisitos e Instalación

### Requisitos

- Python 3.10 o superior
- Conexión a internet (APIs de transcripción y LLM)

### Instalación automática (Windows)

El proyecto incluye un script de instalación que prepara automáticamente el entorno.

Ejecutar:

```bash
setup.bat