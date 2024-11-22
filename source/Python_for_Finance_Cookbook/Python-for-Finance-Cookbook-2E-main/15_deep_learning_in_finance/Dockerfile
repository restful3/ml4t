FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Jupyter Lab 확장 기능 설치
RUN pip3 install \
    jupyter \
    notebook==6.5.5 \
    jupyterlab \
    jupyter_contrib_nbextensions \
    jupyter_nbextensions_configurator

# Jupyter Notebook 확장 기능 설정
RUN jupyter contrib nbextension install --system && \
    jupyter nbextensions_configurator enable --system

# Jupyter 커널 설정
RUN python3 -m ipykernel install --user --name python3 --display-name "Python 3 (CUDA)"

# Jupyter Lab 설정
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/workspace'" >> /root/.jupyter/jupyter_notebook_config.py

# 작업 디렉토리 권한 설정
RUN chmod -R 777 /workspace

# 포트 설정
EXPOSE 8888

# 시작 명령어
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]