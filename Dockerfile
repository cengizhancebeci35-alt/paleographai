FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle (gerekirse)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Requirements yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod kopyala
COPY . .

# Environment değişkenini ayarla
ENV OPENAI_API_KEY=""

# Health check ekle
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Uygulamayı çalıştır
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
