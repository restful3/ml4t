FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Jupyter Lab 설치
RUN pip3 install jupyterlab 