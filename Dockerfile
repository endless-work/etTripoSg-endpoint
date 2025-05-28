FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Установка Python, системных зависимостей и CUDA SDK
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip \
    git curl unzip build-essential ninja-build \
    cuda-toolkit-12-1 \
    && ln -sf python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка pip и Torch (GPU)
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    rm -rf ~/.cache /root/.cache

# Копирование исходников
WORKDIR /app
COPY . .

# Установка зависимостей без diso
RUN grep -v "SarahWeiii/diso" requirements.txt > temp-req.txt && \
    pip install --no-cache-dir -r temp-req.txt && \
    rm temp-req.txt && \
    rm -rf ~/.cache /root/.cache


# Установка diso с патчем setup.py
COPY patched_setup.py /tmp/setup.py
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"

RUN git clone https://github.com/SarahWeiii/diso.git /tmp/diso && \
    cd /tmp/diso && pip install . && \
    rm -rf /tmp/diso

# Запуск inference-скрипта
CMD ["python", "inference_triposg.py"]
