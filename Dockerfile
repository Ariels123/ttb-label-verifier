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

# The application code lives in code/; sample_images/ stays at the image root (app.py resolves
# it as ../sample_images relative to itself). Dev helpers and the real-world test photos
# (sample_images/wm*) are excluded from the image via .dockerignore — only the synthetic demo
# labels the "Try an example" links need are shipped.
COPY code/ ./code/
COPY sample_images/ ./sample_images/

# Hugging Face Spaces expects the app on 7860; $PORT lets other hosts override.
# --chdir code: load app:app from the code/ folder and put it on the import path, so app.py's
# `import ocr` / `import verifier` resolve. sample_images/ is then found at ../sample_images.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn --chdir code -b 0.0.0.0:${PORT:-7860} -w 2 --threads 4 -t 120 app:app"]
