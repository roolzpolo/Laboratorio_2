Laboratorio 2: Predicción Automatizada con Machine Learning

Este proyecto consiste en la creación de un sistema automatizado para realizar predicciones mediante modelos de Machine Learning. Incluye tanto un servicio de procesamiento por lotes como una API desarrollada con FastAPI para realizar predicciones en tiempo real.

Objetivos

Implementar un pipeline automatizado de Machine Learning.
Diseñar un servicio de predicciones por lotes que procese datos continuamente.
Desarrollar una API para predicciones en tiempo real basada en FastAPI.
Construir y desplegar el sistema dentro de un contenedor Docker para garantizar la portabilidad.
Estructura del Proyecto

El proyecto está organizado en los siguientes componentes:

- main.py: Script principal que selecciona entre el modo de ejecución por lotes (Batch) o API, dependiendo de la variable de entorno DEPLOYMENT_TYPE.
- batch_prediction.py: Servicio que procesa archivos de datos en lotes ubicados en una carpeta específica.
- api_prediction.py: Servicio API basado en FastAPI que permite enviar datos y obtener predicciones en tiempo real.
- model.py: Contiene la lógica para entrenar y cargar el modelo de Machine Learning.
- Dockerfile: Configuración para crear una imagen Docker que encapsula todo el proyecto.
- requirements.txt: Archivo que lista las dependencias necesarias para el proyecto.

Carpeta data:
input/: Contiene los datos de entrada en formato .parquet para el servicio de predicción por lotes.
output/: Almacena los resultados generados por el procesamiento por lotes.

Requisitos

Software
- Python 3.12 
- Docker instalado
- Librerías de Python
Las dependencias están listadas en el archivo requirements.txt. Algunas de las principales son:

  - pandas
  - fastapi
  - uvicorn
  - joblib
  - dotenv

