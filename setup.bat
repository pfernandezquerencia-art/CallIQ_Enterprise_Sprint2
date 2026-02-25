@echo off
echo ===================================================
echo Iniciando instalacion de CallIQ Enterprise v1.5.3
echo ===================================================

echo [1/2] Instalando librerias base...
pip install -r requirements.txt

echo [2/2] Descargando modelo cognitivo (SpaCy NER)...
python -m spacy download es_core_news_sm

echo ===================================================
echo Instalacion completada con exito.
echo Ya puedes ejecutar el pipeline o app.py
echo ===================================================
pause