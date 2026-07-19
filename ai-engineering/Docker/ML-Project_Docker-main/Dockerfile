# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Titanic ML Pipeline – RF, XGBoost, SVM + Streamlit Dashboard"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Create runtime dirs ───────────────────────────────────────────────────────
RUN mkdir -p /app/models /app/data

# ── Entrypoint script ─────────────────────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
