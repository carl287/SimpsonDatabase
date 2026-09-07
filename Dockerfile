FROM python:3.12-slim

WORKDIR /app

# Copiar e instalar dependencias primero (aprovecha la caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar los archivos del proyecto
COPY . .

# Puerto para servidor Jupyter (opcional)
EXPOSE 8888