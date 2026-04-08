ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY . /app/env

WORKDIR /app/env

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "clinical_trial_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
