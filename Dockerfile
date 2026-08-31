# Agent Ledger chat-only Cloud Run image.
# Author: Miguel Medina Cantos

FROM python:3.12-slim

ARG EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

WORKDIR /app

COPY requirements-cloud-run.txt /tmp/requirements-cloud-run.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-cloud-run.txt

# Bake MiniLM into the image so a Cloud Run cold start never downloads it.
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY config/ /app/config/
COPY src/ /app/src/
COPY adk_invoice_chat/ /app/agents/adk_invoice_chat/

RUN useradd --system --create-home --uid 10001 agentledger \
    && chown -R agentledger:agentledger /app /opt/huggingface

USER agentledger

EXPOSE 8080

# ADK Web is intentionally enabled for the public competition demo.
CMD ["sh", "-c", "exec adk api_server --with_ui --host=0.0.0.0 --port=${PORT:-8080} --session_service_uri=memory:// --artifact_service_uri=memory:// --log_level=info /app/agents"]
