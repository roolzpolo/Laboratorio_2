Laboratorio 2: Predicción Automatizada con Machine Learning

Este proyecto consiste en la creación de un sistema automatizado para realizar predicciones mediante modelos de Machine Learning. Incluye tanto un servicio de procesamiento por lotes como una API desarrollada con FastAPI para realizar predicciones en tiempo real.

Objetivos

Implementar un pipeline automatizado de Machine Learning.
Diseñar un servicio de predicciones por lotes que procese datos continuamente.
Desarrollar una API para predicciones en tiempo real basada en FastAPI.
Construir y desplegar el sistema dentro de un contenedor Docker para garantizar la portabilidad.
Estructura del Proyecto

El proyecto está organizado en los siguientes componentes:

main.py: Script principal que selecciona entre el modo de ejecución por lotes (Batch) o API, dependiendo de la variable de entorno DEPLOYMENT_TYPE.
batch_prediction.py: Servicio que procesa archivos de datos en lotes ubicados en una carpeta específica.
api_prediction.py: Servicio API basado en FastAPI que permite enviar datos y obtener predicciones en tiempo real.
model.py: Contiene la lógica para entrenar y cargar el modelo de Machine Learning.
Dockerfile: Configuración para crear una imagen Docker que encapsula todo el proyecto.
requirements.txt: Archivo que lista las dependencias necesarias para el proyecto.
Carpeta data:
input/: Contiene los datos de entrada en formato .parquet para el servicio de predicción por lotes.
output/: Almacena los resultados generados por el procesamiento por lotes.
Requisitos

Software
Python 3.8 o superior
Docker
Librerías de Python
Las dependencias están listadas en el archivo requirements.txt. Algunas de las principales son:

pandas
fastapi
uvicorn
joblib
dotenv
Para instalarlas, ejecuta:

pip install -r requirements.txt
Uso

Ejecución Local
1. Configuración del entorno

Crea los archivos .env en la raíz del proyecto:

batch_prediction.env:
DATASET="/data/input/dataset.parquet"
TARGET="nombre_columna_objetivo"
MODEL="RandomForest"
TRIALS=10
DEPLOYMENT_TYPE="Batch"
INPUT_FOLDER="/data/input"
OUTPUT_FOLDER="/data/output"
api_prediction.env:
DATASET="/data/input/dataset.parquet"
TARGET="nombre_columna_objetivo"
MODEL="GradientBoosting"
TRIALS=15
DEPLOYMENT_TYPE="API"
PORT=8000
2. Modo por lotes

Ejecuta:

python main.py
Configura DEPLOYMENT_TYPE="Batch" en el archivo .env.

3. Modo API

Ejecuta:

python main.py
Configura DEPLOYMENT_TYPE="API" en el archivo .env. Luego, prueba la API visitando http://localhost:8000/docs.

Ejecución con Docker
Construye la imagen:
docker build -t automl-dockerizer:latest .
Ejecuta en modo API:
docker run -e DEPLOYMENT_TYPE="API" -p 8000:8000 -v $(pwd)/data:/data automl-dockerizer:latest
Ejecuta en modo por lotes:
docker run -e DEPLOYMENT_TYPE="Batch" -v $(pwd)/data:/data automl-dockerizer:latest
Endpoints de la API

GET /health: Comprueba el estado del servicio.
POST /predict: Enviar datos para obtener predicciones.
Input:
{
  "data": [
    {"feature1": value1, "feature2": value2, ...}
  ]
}
Output:
{
  "predictions": [
    {"Clase1": probabilidad1, "Clase2": probabilidad2, ...}
  ]
}
Notas Técnicas

Formato de datos de entrada:
Los datos deben estar en formato .parquet para el modo por lotes.
Para la API, los datos deben enviarse como JSON.
Archivo preprocessor.pkl:
El preprocesador debe generarse previamente y estar disponible en la raíz del contenedor o directorio de trabajo.
Entrenamiento del modelo:
El modelo se entrena automáticamente al iniciar el sistema si no se encuentra uno previamente generado.
Autores

Este laboratorio fue diseñado para Desarrollo de Modelos Automatizados como parte del curso de Machine Learning.

Contribuciones realizadas por: [Tu Nombre].