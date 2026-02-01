FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------
# System Dependencies
# -----------------------------
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    openssl \
    wget \
    curl \
    git \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy Application Files
# -----------------------------
COPY scripts scripts
COPY resources resources

COPY app.sh .
RUN chmod +x /app/app.sh

# -----------------------------
# Python Dependencies
# -----------------------------
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Install yt-dlp (stable PyPI version)
# -----------------------------
RUN pip install --no-cache-dir -U yt-dlp

# -----------------------------
# Start Container
# -----------------------------
CMD ["./app.sh"]
