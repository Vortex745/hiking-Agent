FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ai-service/requirements.txt ai-service/requirements-minimal.txt ./
ARG PUBLIC_PYPI_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --upgrade pip --index-url "${PUBLIC_PYPI_INDEX_URL}" \
    && python -m pip install --index-url "${PUBLIC_PYPI_INDEX_URL}" -r requirements.txt

# Restore the pinned fork offline, including Git metadata and prompt assets
# absent from a plain outer checkout. Use the same validator as CI.
COPY scripts/restore-uniharness-fork.py /opt/fork-restore/scripts/restore-uniharness-fork.py
COPY third_party/uniharness-fork.bundle third_party/uniharness-provenance.md /opt/fork-restore/third_party/
RUN python /opt/fork-restore/scripts/restore-uniharness-fork.py --target /opt/uniharness \
    && rm -rf /opt/fork-restore
# The service runs as a non-root user; allow its runtime identity check to
# inspect the root-owned vendored Git checkout without triggering dubious-
# ownership protection.
RUN git config --system --add safe.directory /opt/uniharness
RUN python -m pip install -e /opt/uniharness/libs/uniharness

RUN addgroup --system app && adduser --system --ingroup app app

COPY --chown=app:app ai-service/ .
RUN mkdir -p memory_store rag_docs workspace agent_state \
    && chown -R app:app memory_store rag_docs workspace agent_state

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
