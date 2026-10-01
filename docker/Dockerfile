# Production image for the prediction API.
# Build from the PROJECT ROOT (the build context must contain src/ and artifacts/):
#   docker build -f docker/Dockerfile -t cancer-api:1.0.0 .

# ---------- Stage 1: builder -- install dependencies into a virtualenv ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirement files first. Docker caches each layer: as long as these
# two files don't change, the slow `pip install` layer is reused on every rebuild,
# even when the source code changes.
COPY requirements-serve.txt requirements.lock.txt /tmp/
# -c = constraints: install only what requirements-serve.txt asks for, but at the
# EXACT versions recorded in the lock file (same scikit-learn that trained the model).
RUN pip install -r /tmp/requirements-serve.txt -c /tmp/requirements.lock.txt


# ---------- Stage 2: runtime -- only what is needed to run ----------
# Starts from a clean image: pip caches and build leftovers from stage 1 are left behind.
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/artifacts/model

# A dedicated unprivileged user: if someone breaks into the app, they are not root.
# Fixed numeric UID so Kubernetes (runAsNonRoot) can verify it is not 0.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ src/
# The model is baked into the image: image tag <-> model version, fully reproducible.
COPY artifacts/model/ artifacts/model/

USER 10001:10001
EXPOSE 8000

# Docker itself polls this; `docker ps` shows (healthy)/(unhealthy).
# Uses only the Python standard library -- no curl or requests needed in the image.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# Exec form (JSON array): uvicorn runs as PID 1 and receives SIGTERM directly,
# so `docker stop` / Kubernetes shut it down gracefully instead of killing it.
# One process per container: scale by running MORE containers, not more workers.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
