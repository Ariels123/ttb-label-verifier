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

# The application code lives in code/; examples/ stays at the image root (app.py resolves
# it as ../examples relative to itself). Dev helpers in code/ are excluded via .dockerignore.
COPY code/ ./code/
COPY examples/ ./examples/

# Hugging Face Spaces expects the app on 7860; $PORT lets other hosts override.
# --chdir code: load app:app from the code/ folder and put it on the import path, so app.py's
# `import ocr` / `import verifier` resolve. examples/ is then found at ../examples (= /app/examples).
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn --chdir code -b 0.0.0.0:${PORT:-7860} -w 2 --threads 4 -t 120 app:app"]
