FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git curl unzip \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip и torch (до requirements.txt)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio

# Рабочая директория
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Установка зависимостей
RUN pip install -r requirements.txt

# Установка отдельной зависимости diso (после torch)
RUN pip install "git+https://github.com/SarahWeiii/diso.git"

# Запуск скрипта
CMD ["python", "inference_triposg.py"]
