from fastapi import FastAPI, UploadFile, File
import base64, os, json, re
from openai import OpenAI
from dotenv import load_dotenv
import logging

# .env dosyasını yükle
load_dotenv()

app = FastAPI(title="PaleographAI", version="1.0.0")

# API key kontrolü
api_key = os.getenv("OPENAI_API_KEY", "senin_api_key")
if not api_key or api_key == "senin_api_key":
    raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmamış!")

client = OpenAI(api_key=api_key)

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- HELPERS ----------------
def safe_json_extract(text):
    try:
        return json.loads(text)
    except:
        # JSON yakala (GPT bozulursa bile)
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {
            "raw": text,
            "warning": "invalid_json"
        }

# ---------------- AI VISION ----------------
def vision(image_b64):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON with keys: script, period, transcription, confidence, notes"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this manuscript image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )

        content = response.choices[0].message.content
        return safe_json_extract(content)

    except Exception as e:
        logger.error(f"Vision API error: {str(e)}")
        return {
            "error": "vision_failed",
            "message": str(e)
        }

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
        logger.error(f"Analyze endpoint error: {str(e)}")
        return {
            "error": "analyze_failed",
            "message": str(e)
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run("main:app", host="0.0.0.0", port=port)
