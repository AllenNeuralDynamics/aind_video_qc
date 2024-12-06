# hash:sha256:652eb8b0dc0e7747e0b9ca56d3eb359162557ce8ade08a51619fc998ec728ee7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=21.2.6-0ubuntu0.1~20.04.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    harp-python==0.2.0 \
    numpy==2.0.2 \
    opencv-python-headless==4.10.0.84 \
    pandas==2.2.3 \
    seaborn==0.13.2 \
    pillow==11.0.0 \
    -e git+https://github.com/AllenNeuralDynamics/aind-video-utils@main#egg=aind-video-utils

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
