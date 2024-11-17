# Imagen base de Python
FROM python:3.12-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# correr las dependencias y las librerias
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

# Copiado de los archivos del proyecto al contenedor
COPY . /app

# Instalar requirements
RUN pip install --no-cache-dir -r requirements.txt

# ejecuciónn
EXPOSE 8000

# Define el comando por defecto al iniciar el contenedor
CMD ["python", "main.py"]
