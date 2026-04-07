# ============================================================
# Dockerfile multi-stage — Enterprise Agent Platform
# Stage 1: installa dipendenze
# Stage 2: immagine finale leggera con utente non-root
# ============================================================

# ── STAGE 1 — Builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Installa dipendenze di sistema necessarie per compilare pacchetti
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia solo requirements per sfruttare la cache Docker
# Se requirements.txt non cambia, questo layer viene cachato
COPY requirements.txt .

# Installa dipendenze in una cartella isolata
# --no-cache-dir: non salva cache pip nell't immagine
# --prefix: installa in /install invece di sistema
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── STAGE 2 — Runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Installa solo libpq runtime (non gcc) — immagine più leggera
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia le dipendenze installate nello stage 1
COPY --from=builder /install /usr/local

# Crea utente non-root per sicurezza
# -r: utente di sistema (no home, no login)
# -g: crea anche il gruppo appuser
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copia il codice dell'applicazione
COPY app/       ./app/
COPY scripts/   ./scripts/
COPY ui.py      .

# Crea cartelle necessarie con permessi corretti
RUN mkdir -p credentials evaluation && \
    chown -R appuser:appuser /app

# Passa all'utente non-root
# Da qui in poi nessun comando gira come root
USER appuser

# Porta esposta dall'applicazione
EXPOSE 8000

# HEALTHCHECK — verifica che /health risponda ogni 30s
# --interval: quanto spesso controllare
# --timeout:  quanto aspettare prima di considerare fallito
# --retries:  quanti fallimenti consecutivi = unhealthy
# --start-period: grace period all'avvio (lascia tempo al lifespan)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando di avvio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]