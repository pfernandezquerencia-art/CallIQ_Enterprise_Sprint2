# ============================================================
# CALLIQ DATA CONTRACT: METADATA_PACK v3 (Data Classes)
# ============================================================
# Este archivo define la estructura canónica del dato para todo el pipeline.
# Garantiza que ETL, Anonimización, Evaluación y FinOps hablen el mismo idioma.
# ESTADO: Alineado con v1.5.3 Gold Master.
# ============================================================

import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

# ------------------------------------------------------------
# 1. SUB-ESTRUCTURAS (BLOQUES LÓGICOS)
# ------------------------------------------------------------

@dataclass
class SourceInfo:
    """Datos técnicos del archivo de origen."""
    filename: str
    file_path: str
    file_size_bytes: int
    format: str
    provider: str = "filesystem"  # o 'genesys', 'google_drive'
    ingestion_source: str = "batch_folder"

@dataclass
class TechnicalMetadata:
    """Mediciones objetivas del audio (Base para FinOps)."""
    duration_ms: int = 0
    duration_sec: float = 0.0
    bitrate: int = 0
    sample_rate: int = 0
    channels: int = 0
    audio_hash_sha256: str = ""  # SHA256 para no-repudio

@dataclass
class Utterance:
    """Una intervención individual en la conversación (Diarización)."""
    speaker_id: str  # 'A', 'B', 'Agent', 'Customer'
    start_ms: int
    end_ms: int
    text: str
    confidence: float = 1.0

@dataclass
class TranscriptionData:
    """Resultado del proceso STT."""
    status: str = "pending"  # pending, completed, failed
    provider: str = "assemblyai"
    language_code: str = "es"
    full_text_clean: str = "" # Solo guardamos el texto tras anonimizar
    utterances: List[Utterance] = field(default_factory=list)
    word_count: int = 0
    processing_latency_ms: int = 0

@dataclass
class GovernanceData:
    """Control de cumplimiento, privacidad y seguridad (GOB-01)."""
    pii_anonymized: bool = False
    anonymization_method: str = "Hybrid (SpaCy NER + RegEx)"
    raw_text_hash: str = ""
    prompt_injection_detected: bool = False
    model_integrity_hash: str = "" # El Kill Switch del manual de calidad
    retention_policy_days: int = 30

@dataclass
class FinOpsData:
    """Traza económica exacta de la transacción."""
    currency: str = "USD"
    stt_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    total_transaction_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class ProcessingControl:
    """Trazabilidad de ejecución del orquestador."""
    pipeline_version: str = "1.5.3"
    conversation_id: str = ""
    ingested_at: str = ""
    processed_at: str = ""
    status: str = "initialized"
    error_log: Optional[str] = None

# ------------------------------------------------------------
# 2. ESTRUCTURA MAESTRA (EL CONTRATO METADATA PACK V3)
# ------------------------------------------------------------

@dataclass
class MetadataPack:
    """
    METADATA_PACK v3: La única fuente de verdad.
    Este objeto viaja desde ETL-01 hasta la capa de persistencia.
    """
    # Identificación
    id: str  # UUID interno
    metadata_pack_version: str = "3.0"
    
    # Bloques de Datos Técnicos
    source: SourceInfo
    technical: TechnicalMetadata
    transcription: TranscriptionData = field(default_factory=TranscriptionData)
    governance: GovernanceData = field(default_factory=GovernanceData)
    finops: FinOpsData = field(default_factory=FinOpsData)
    control: ProcessingControl = field(default_factory=ProcessingControl)
    
    # Bloque de Evaluación Cognitiva-Determinista (CAL-01)
    quality_evaluation: Dict = field(default_factory=dict)  # Score final, KOs, Bloques, Señales

    @classmethod
    def create_new(cls, file_path: str, provider: str = "local"):
        """Factory Method: Crea un paquete nuevo interceptando el archivo."""
        # Evita caídas si el archivo es simulado (como en los test batch)
        if os.path.exists(file_path):
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
        else:
            file_size = 0
            
        filename = os.path.basename(file_path)
        now = datetime.now(timezone.utc).isoformat()
        unique_id = str(uuid.uuid4())

        return cls(
            id=unique_id,
            source=SourceInfo(
                filename=filename,
                file_path=file_path,
                file_size_bytes=file_size,
                format=filename.split('.')[-1] if '.' in filename else 'unknown',
                provider=provider
            ),
            technical=TechnicalMetadata(),
            control=ProcessingControl(
                conversation_id=unique_id,
                ingested_at=now
            )
        )

    def to_json(self) -> str:
        """Serializa el paquete a JSON estricto para la Base de Datos o API."""
        return json.dumps(asdict(self), default=str, indent=2, ensure_ascii=False)

    def calculate_totals(self):
        """Método helper para consolidar la factura final (FinOps)."""
        self.finops.total_transaction_cost_usd = round(
            self.finops.stt_cost_usd + self.finops.llm_cost_usd, 6
        )