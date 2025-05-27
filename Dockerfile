FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Установка системных зависимостей, Python и CUDA toolkit
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip \
    git curl unzip build-essential ninja-build \
    cuda-toolkit-12-1 \
    && ln -sf python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка pip и Torch (GPU) + удаление кэша
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    rm -rf ~/.cache /root/.cache

# Создание рабочей директории
WORKDIR /app

# Копирование всех файлов проекта
COPY . .

# Установка зависимостей без diso
RUN grep -v "SarahWeiii/diso" requirements.txt > temp-req.txt && \
    pip install --no-cache-dir -r temp-req.txt && \
    rm temp-req.txt && \
    rm -rf ~/.cache /root/.cache

# Установка diso (последним шагом)
RUN pip install --no-cache-dir "git+https://github.com/SarahWeiii/diso.git" && \
    rm -rf ~/.cache /root/.cache

# Запуск inference-скрипта
CMD ["python", "inference_triposg.py"]
