# Serving image. Runs on Render, Hugging Face Spaces or any container host.
# The host can set PORT (Render does); 7860 is the local default.
FROM python:3.13-slim

# libgomp1 is required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/home/user/app/src:/home/user/app \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user models ./models

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
