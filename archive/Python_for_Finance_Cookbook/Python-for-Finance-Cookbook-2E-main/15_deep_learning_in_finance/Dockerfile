FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 두 개의 가상환경 생성
RUN python3 -m pip install virtualenv && \
    python3 -m virtualenv /opt/venv-main && \
    python3 -m virtualenv /opt/venv-prophet

# 메인 환경 설정
COPY requirements-main.txt requirements-main.txt
RUN . /opt/venv-main/bin/activate && \
    pip install --no-cache-dir -r requirements-main.txt && \
    python3 -m ipykernel install --user --name venv-main --display-name "Python (Main)"

# Prophet 환경 설정
COPY requirements-prophet.txt requirements-prophet.txt
RUN . /opt/venv-prophet/bin/activate && \
    pip install --no-cache-dir numpy==1.25.2 && \
    pip install --no-cache-dir -r requirements-prophet.txt && \
    python3 -m ipykernel install --user --name venv-prophet --display-name "Python (Prophet)"

# Jupyter 설정
RUN pip3 install \
    jupyter \
    notebook==6.5.5 \
    jupyterlab \
    jupyter_contrib_nbextensions \
    jupyter_nbextensions_configurator

RUN jupyter contrib nbextension install --system && \
    jupyter nbextensions_configurator enable --system

RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/workspace'" >> /root/.jupyter/jupyter_notebook_config.py

RUN chmod -R 777 /workspace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]