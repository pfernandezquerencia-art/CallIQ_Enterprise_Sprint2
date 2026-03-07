import os
import json
import random
import uuid
import glob
from datetime import datetime, timedelta, timezone

def generar_metadatos_audios(directorio_audios="data/audios"):
    print("📞 Iniciando simulador de PBX/CTI (Genesys Cloud Mock)...")
    
    # Buscar todos los audios en la carpeta
    audios = glob.glob(os.path.join(directorio_audios, "*.wav")) + glob.glob(os.path.join(directorio_audios, "*.mp3"))
    
    if not audios:
        print(f"❌ No se encontraron audios en {directorio_audios}")
        return

    # --- NUESTRO MAPEO REAL DE AGENTES BASADO EN LAS TRANSCRIPCIONES ---
    mapeo_real = {
        # --- LOTE 5 ---
        "10232984001": {"agente": "AG-BELEN",     "cola": "Soporte_Tecnico"},  # Jorge Peralta, corte en Granadero Baigorria.
        "10231565001": {"agente": "AG-GONZALO",   "cola": "Soporte_Tecnico"},  # Julián, confirmación visita técnica (Edificio Pasco).
        "10246500001": {"agente": "AG-CAMILA",    "cola": "Atencion_Cliente"}, # Gabriela, traslado domicilio, sin stock de módems.
        "10232682001": {"agente": "AG-CINTIA",    "cola": "Soporte_Tecnico"},  # Llamada abandonada / fantasma.
        "10233574001": {"agente": "AG-FLORENCIA", "cola": "Retencion"},        # Claudia pide baja por mudanza (Deriva Cintia).

        # --- LOTE 4 ---
        "10231875001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},
        "10233161001": {"agente": "AG-CARLA",   "cola": "Atencion_Cliente"},
        "10231958001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},
        "10231722001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},
        "10232888001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},

        # --- LOTE 3 ---
        "10246432001": {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"},  
        "10232697001": {"agente": "AG-CINTIA",  "cola": "Soporte_Tecnico"},  
        "10233439001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},  
        "10248132001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"},  
        "10233414001": {"agente": "AG-INGE",    "cola": "Soporte_Tecnico"},  
        
        # --- LOTE 2 ---
        "10249954001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"}, 
        "16140901":    {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"}, 
        "10232530001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},  
        "10232705001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},  
        "10245406001": {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"},  
        "10233540001": {"agente": "AG-CINTIA",  "cola": "Soporte_Tecnico"},  
        "10245608001": {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"},  
        "10248994001": {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"},  
        "10249696001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"},  
        "10248092001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"},  

        # --- LOTE 1 ---
        "10246870001": {"agente": "AG-NATALIA", "cola": "Retencion"},        
        "10232684001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},  
        "10231909001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},  
        "10232448001": {"agente": "AG-CINTIA",  "cola": "Soporte_Tecnico"},  
        "10232926001": {"agente": "AG-ALICIA",  "cola": "Retencion"},        
        "10249864001": {"agente": "AG-GABRIEL", "cola": "Atencion_Cliente"}, 
        "10231524001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},  
        "10248768001": {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"},  
        "10231711001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},  
        "10248971001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"},
        "10249787001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"},
        "10248770001": {"agente": "AG-GABRIEL", "cola": "Soporte_Tecnico"},
        "10232186001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},
        "10231966001": {"agente": "AG-BELEN",   "cola": "Soporte_Tecnico"},
        "10231491001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},
        "10231468001": {"agente": "AG-MARCELO", "cola": "Soporte_Tecnico"},
        "10249028001": {"agente": "AG-CAMILA",  "cola": "Soporte_Tecnico"},
        "10247965001": {"agente": "AG-CAMILA",  "cola": "Atencion_Cliente"},
        "10233199001": {"agente": "AG-CINTIA",  "cola": "Soporte_Tecnico"}
    }
    
    agentes_fallback = ["AG-GABRIEL", "AG-BELEN", "AG-MARCELO", "AG-CAMILA", "AG-CINTIA"]
    colas_fallback = ["Soporte_Tecnico", "Atencion_Cliente", "Ventas", "Retencion"]

    for audio_path in audios:
        # Sacar el nombre sin la extensión (ej: de "10248971001.mp3" saca "10248971001")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(directorio_audios, f"{base_name}.json")
        
        if os.path.exists(json_path):
            print(f"⏩ Saltando (ya existe): {os.path.basename(json_path)}")
            continue
            
        # Generar fechas aleatorias de los últimos 5 días
        dias_atras = random.randint(0, 5)
        horas_atras = random.randint(0, 23)
        fecha_llamada = datetime.now(timezone.utc) - timedelta(days=dias_atras, hours=horas_atras)
        
        # Buscar si el audio está en nuestro mapeo real. Si no, usar aleatorio.
        datos_agente = mapeo_real.get(base_name, {
            "agente": random.choice(agentes_fallback),
            "cola": random.choice(colas_fallback)
        })
        
        # Crear los metadatos dummy
        metadata_pbx = {
            "ucid": str(uuid.uuid4()).replace("-", "").upper()[:20],
            "agent_id": datos_agente["agente"],
            "extension": f"7{random.randint(100, 999)}",
            "queue": datos_agente["cola"],
            "call_timestamp": fecha_llamada.isoformat(),
            "ani": f"+346{random.randint(10000000, 99999999)}",
            "dnis": "900123456"
        }
        
        # Guardar el JSON junto al audio
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_pbx, f, indent=4)
            
        print(f"✅ Creado Sidecar CTI: {os.path.basename(json_path)} -> {datos_agente['agente']} ({datos_agente['cola']})")

if __name__ == "__main__":
    # Asegúrate de que la ruta apunte a tu carpeta de audios
    generar_metadatos_audios(directorio_audios="data/audios")