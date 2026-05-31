from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64, os, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import logging

# .env dosyasını yükle
load_dotenv()

app = FastAPI(title="PaleographAI", version="1.0.0")

# API key kontrolü
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmamış!")

client = OpenAI(api_key=api_key)

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- AI VISION ----------------
def vision(image_b64: str) -> dict:
    """
    Manuscript görüntüsünü analiz et ve metadata döndür.
    
    Args:
        image_b64: Base64 encoded görüntü
        
    Returns:
        dict: script, period, transcription, confidence, notes
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON with keys: script, period, transcription, confidence, notes"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this manuscript image carefully"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            temperature=0.3
        )
        
        content = res.choices[0].message.content
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw content: {content[:100]}")
            return {
                "script": "unknown",
                "period": "unknown",
                "transcription": content,
                "confidence": 0.5,
                "notes": "JSON parse failed, raw output returned"
            }
        except Exception as e:
            logger.error(f"Unexpected error while parsing response: {e}")
            return {
                "script": "error",
                "period": "error",
                "transcription": "",
                "confidence": 0.0,
                "notes": f"Unexpected error: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        raise

# ---------------- API ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Manuscript görüntüsünü yükle ve analiz et.
    """
    # Dosya türü kontrolü
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece görüntü dosyaları kabul edilir")
    
    # Dosya boyutu kontrolü (10MB)
    MAX_SIZE = 10 * 1024 * 1024
    content = await file.read()
    
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Dosya çok büyük (max 10MB)")
    
    try:
        b64 = base64.b64encode(content).decode()
        result = vision(b64)
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Analyze endpoint error: {e}")
        raise HTTPException(status_code=500, detail="İşlem sırasında hata oluştu")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}
