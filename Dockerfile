# CUDA가 포함된 Ubuntu 20.04 기반 베이스 이미지 사용
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    git \
    build-essential \
    python3-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# 환경 변수 설정
ENV PATH=/opt/conda/bin:$PATH

# 작업 디렉토리 설정
WORKDIR /app

# 환경 설정 파일 복사
COPY requirements_aws_poly.txt .

# 아나콘다 환경 생성 및 패키지 설치
RUN conda create -n polyformer python=3.7 -y && \
    /bin/bash -c "source /opt/conda/bin/activate polyformer && \
    python -m pip install pip==21.2.4 && \
    pip install -r requirements_aws_poly.txt"

# 환경 변수 업데이트
ENV PATH=/opt/conda/envs/polyformer/bin:$PATH
RUN conda init bash


# 애플리케이션 코드 복사
COPY . .

# 기본 쉘 설정
SHELL ["/bin/bash", "-c"]

# 필요한 경우 포트 열기
# EXPOSE 8080

# 컨테이너 시작 시 실행할 명령어 (필요한 경우)
# CMD ["python", "your_script.py"]