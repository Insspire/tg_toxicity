# Используем официальный образ Python 3.12
FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml ./

# Устанавливаем pip и зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit nest-asyncio telethon torch transformers datasets accelerate scikit-learn

# Копируем весь проект
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Команда для запуска Streamlit приложения
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
