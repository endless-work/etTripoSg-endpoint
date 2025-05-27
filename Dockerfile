FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git curl unzip build-essential \
    && rm -rf /var/lib/apt/lists/*

    
# Создаём директорию приложения
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем pip и torch (до остальных)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio

# Устанавливаем все зависимости, кроме diso
RUN grep -v "SarahWeiii/diso" requirements.txt > temp-req.txt && \
    pip install -r temp-req.txt && \
    rm temp-req.txt

# Устанавливаем diso отдельно
RUN pip install "git+https://github.com/SarahWeiii/diso.git"

# Команда по умолчанию (если нужна)
CMD ["python", "inference_triposg.py"]
