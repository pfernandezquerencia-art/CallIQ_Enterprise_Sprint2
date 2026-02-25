# CallIQ Enterprise v1.5.3 🚀
**Sistema de Análisis Conversacional Cognitivo-Determinista para Contact Centers**

CallIQ es una arquitectura modular orientada a la evaluación automática de calidad en entornos regulados. 
El diseño separa explícitamente la comprensión semántica (LLM) del cálculo determinista de la nota (Reglas de Negocio en Python), garantizando trazabilidad y reproducibilidad.

---

## 🏗️ Arquitectura Modular (Sprint 2)

El pipeline se compone de cuatro módulos lógicos desacoplados:

* **`ETL-01`** → Ingesta de audio, diarización y extracción determinista de metadatos técnicos.
* **`GOB-01`** → Gobernanza criptográfica, anonimización híbrida (RegEx + SpaCy NER) y control Anti-Prompt Injection.
* **`CAL-01`** → Evaluación híbrida (Zero-Shot + RAG) con cálculo matemático determinista externo al LLM.
* **`DLM-01`** (`DLM_Google_Drive.py`) → Microservicio REST (FastAPI) para gestión del ciclo de vida del dato. No es necesaria su ejecución, se establece como módulo opcional *Se adjunta el código fuente como evidencia arquitectónica de la conexión Cloud/FinOps.*

---

## ⚙️ Requisitos e Instalación

* Python 3.10+
* Conexión a internet (API STT y LLM)

Instalación automática (Windows):
```bash
setup.bat