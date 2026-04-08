FROM python:3.11-slim

# ── System ───────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# ── Run ──────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
