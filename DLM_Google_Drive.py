# ==============================================================================
# CALLIQ – DATA LIFECYCLE MANAGEMENT (DLM) MICROSERVICE v1.0
# ==============================================================================
#
# DESCRIPCIÓN GLOBAL:
# Microservicio desacoplado orientado a la optimización del ciclo de vida del dato
# en entornos de Contact Center (CCaaS).
#
# Este módulo complementa al Core Engine de CallIQ (v1.5.3+) proporcionando una
# capa opcional de enrutamiento inteligente de grabaciones de audio hacia
# almacenamiento optimizado, en función de criterios regulatorios y financieros.
#
# PRINCIPIO ARQUITECTÓNICO:
# CallIQ no compite con plataformas CCaaS ni asume por defecto la custodia del audio.
# Sin embargo, en escenarios donde el cliente incurre en sobrecostes de almacenamiento
# (Fair Use Policies, exceso de cuota, requisitos regulatorios específicos),
# este microservicio permite orquestar la exportación selectiva hacia capas de
# almacenamiento de menor coste.
#
# EJES ESTRATÉGICOS:
# 1. FinOps:
#    - Optimización de costes en plataformas CCaaS.
#    - Clasificación de datos "Dark Data" y purga de interacciones sin valor.
#
# 2. Regulación:
#    - Identificación de llamadas con valor probatorio (ej. MiFID).
#    - Enrutamiento a almacenamiento legal diferenciado.
#
# 3. Modularidad:
#    - Arquitectura REST desacoplada (FastAPI).
#    - Integración vía API con el Core Engine o sistemas externos.
#    - Despliegue independiente y escalable.
#
# NIVELES DE CLASIFICACIÓN:
#   Nivel 0 → Purgado (duración irrelevante / sin valor)
#   Nivel 1 → Almacenamiento operativo
#   Nivel 3 → Almacenamiento regulatorio (legal)
#
# NOTA:
# Este módulo es opcional dentro del ecosistema CallIQ y no altera el núcleo
# cognitivo-determinista del sistema. Su activación depende del modelo de cliente
# (capa analítica sobre CCaaS existente vs. despliegue SaaS independiente).
#
# AUTORÍA:
# Diseño e implementación del microservicio DLM realizado como extensión
# arquitectónica para demostrar capacidad de gestión distribuida y FinOps.
#
# ESTADO: MVP Funcional – Arquitectura preparada para producción escalable.
# ==============================================================================


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import logging
import re
from datetime import datetime, timezone

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

APP_VERSION = "1.6"  # Actualizar la versión cada vez que se hagan cambios significativos 

# Variables de entorno
FOLDER_LEGAL = os.getenv("DRIVE_FOLDER_LEGAL", "id_de_carpeta_legal")
FOLDER_OPERATIVO = os.getenv("DRIVE_FOLDER_OPERATIVO", "id_de_carpeta_operativa")
CREDENTIALS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("calliq-dlm")

# Inicializa el cliente Google Drive para MVP, con manejo de errores para evitar caídas si las credenciales no están configuradas correctamente
try:
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive']
    )
    drive_service = build('drive', 'v3', credentials=creds)
    logger.info("Cliente de Google Drive inicializado con éxito.")
except Exception as e:
    logger.warning(f"No se pudo inicializar Google Drive. Comprueba tu credentials.json: {e}")
    drive_service = None


app = FastAPI(
    title="CallIQ DLM Microservice - Edición Google Drive",
    version=APP_VERSION,
    description="Motor de Enrutamiento de Almacenamiento (MVP)"
)


class CallMetadata(BaseModel):
    call_id: str = Field(..., min_length=1)
    duration_seconds: int = Field(..., ge=0)
    is_mifid_regulated: bool
    audio_file_path: str


def sanitize_call_id(call_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
    return clean or "unknown_call"


def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Error al eliminar archivo local {path}: {e}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "drive_connected": drive_service is not None,
        "timestamp": utc_now_iso()
    }


@app.post("/dlm/enrutar")
def enrutar_llamada(meta: CallMetadata):

    safe_id = sanitize_call_id(meta.call_id)
    audio_path = meta.audio_file_path
    now_ts = utc_now_iso()
    should_delete_local = False

    logger.info(f"Procesando call_id={safe_id}")

    # Validación básica del archivo de audio
    if not audio_path.lower().endswith(".mp3"):
        raise HTTPException(400, "Solo se aceptan archivos MP3")

    if not os.path.exists(audio_path):
        raise HTTPException(400, "Archivo de audio no encontrado")

    try:
        # Nivel 0 (FinOps / Datos Basura)
        if meta.duration_seconds < 3:
            logger.info(f"Duración <3s → purgado call_id={safe_id}")
            should_delete_local = True
            return {
                "status": "purged",
                "level": 0,
                "version": APP_VERSION,
                "timestamp": now_ts
            }

        # Clasificación
        nivel = 3 if meta.is_mifid_regulated else 1
        folder_id = FOLDER_LEGAL if nivel == 3 else FOLDER_OPERATIVO
        file_name = f"{safe_id}.mp3"

        if not drive_service:
            logger.error("Servicio de Google Drive no disponible")
            raise HTTPException(500, "Servicio de Google Drive no disponible")

        # Idempotencia
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, spaces='drive', fields='files(id, name)'
        ).execute()

        items = results.get('files', [])

        if items:
            logger.info(f"Archivo ya existente en Drive: {file_name} → {items[0]['id']}")
            should_delete_local = True
            return {
                "status": "already_processed",
                "level": nivel,
                "drive_file_id": items[0]['id'],
                "version": APP_VERSION,
                "timestamp": now_ts
            }

        # Subida a Drive
        file_metadata = {
            'name': file_name,
            'parents': [folder_id],
            # Metadata adicional para futuras funcionalidades (p.ej. búsqueda avanzada, auditoría)
            'description': f"stored_at={now_ts};version={APP_VERSION}"
        }
        media = MediaFileUpload(audio_path, mimetype='audio/mpeg', resumable=False)

        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        logger.info(
            f"Almacenado call_id={safe_id} nivel={nivel} "
            f"folder={folder_id} drive_id={uploaded_file.get('id')} ts={now_ts}"
        )

        should_delete_local = True

        return {
            "status": "success",
            "level": nivel,
            "folder_id": folder_id,
            "drive_file_id": uploaded_file.get('id'),
            "version": APP_VERSION,
            "timestamp": now_ts
        }

    except HTTPException:
        # No toca should_delet_local: si falla Drive, conserva el archivo
        raise
    except Exception as e:
        logger.exception(f"Error inesperado procesando call_id={safe_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # Este bloque se ejecuta SIEMPRE al salir de la función, y garantiza la limpieza local solo si se definió should_delete_local=True
        if should_delete_local:
            safe_remove(audio_path)
            logger.info(f"Limpieza local ejecutada para call_id={safe_id}")