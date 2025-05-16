FROM python:3.12-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libpoppler-cpp-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Uygulama dizini
WORKDIR /app

# Gereksinimler
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod
COPY . .

EXPOSE 8501

# Başlatıcı
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
