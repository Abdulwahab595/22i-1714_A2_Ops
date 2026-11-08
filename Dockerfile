FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 MALLOC_ARENA_MAX=2 HOST=0.0.0.0 PORT=8000
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p artifacts models && chown -R appuser:appuser artifacts models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health',timeout=8)"
CMD ["sh", "-c", "python -m scripts.fetch_champion && exec python serve.py"]
