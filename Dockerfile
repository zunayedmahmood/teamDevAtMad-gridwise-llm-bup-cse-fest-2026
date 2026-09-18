FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import os, urllib.request; port=os.environ.get('PORT') or os.environ.get('APP_PORT','8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host \"${APP_HOST:-0.0.0.0}\" --port \"${PORT:-${APP_PORT:-8000}}\" --workers 1"]
