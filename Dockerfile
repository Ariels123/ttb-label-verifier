# TTB Label Verifier — container image.
# Self-contained: Tesseract (local OCR) is installed in the image, so there are
# NO outbound cloud calls at runtime (matches TTB's firewall constraint).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        fonts-dejavu-core \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py verifier.py ocr.py ./
COPY examples/ ./examples/

# Hugging Face Spaces expects the app on 7860; $PORT lets other hosts override.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} -w 2 --threads 4 -t 120 app:app"]
