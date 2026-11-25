# Serving image. Runs on Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.13-slim

# libgomp1 is required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/home/user/app/src:/home/user/app \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user models ./models

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
