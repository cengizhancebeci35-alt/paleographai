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
    try:
        img = await file.read()

        if not img:
            return {"error": "empty_file"}

        b64 = base64.b64encode(img).decode()

        result = vision(b64)

        return result

    except Exception as e:
        return {
            "error": "analyze_failed",
            "message": str(e)
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}
