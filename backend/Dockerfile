FROM python:3.11-slim

# UID pengguna container. Samakan dengan UID host kalau ./eprints_vector_db
# di-bind-mount: kalau berbeda, SQLite menolak menulis dan startup gagal
# dengan "attempt to write a readonly database".
ARG APP_UID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend.py .

RUN mkdir -p /app/eprints_vector_db /model-cache \
    && useradd --create-home --uid ${APP_UID} app \
    && chown -R app:app /app /model-cache

USER app

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
