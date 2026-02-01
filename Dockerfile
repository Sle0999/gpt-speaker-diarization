FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# (Optional) You don't need USER root:root; default is root
# USER root

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    bash \
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

WORKDIR /app

COPY scripts scripts
COPY resources resources

COPY app.sh .
RUN chmod +x /app/app.sh

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yt-dlp from master (ok, but consider pinning to a release for reproducibility)
RUN pip install --no-cache-dir --force-reinstall \
    https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz

CMD ["./app.sh"]
