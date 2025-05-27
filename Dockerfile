FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Установка системных зависимостей и Python
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git curl unzip build-essential \
    && ln -sf python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Установка pip и Torch (GPU)
RUN python -m pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app
COPY . .

# Установка зависимостей, кроме diso
RUN grep -v "SarahWeiii/diso" requirements.txt > temp-req.txt && \
    pip install -r temp-req.txt && \
    rm temp-req.txt

# Установка diso (требует torch и компиляции)
RUN pip install "git+https://github.com/SarahWeiii/diso.git"

CMD ["python", "inference_triposg.py"]
