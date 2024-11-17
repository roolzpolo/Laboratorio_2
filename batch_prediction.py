#manejo de system y tiempos de espera
import os
import time
import shutil

import pandas as pd
from model import train_model

#proprocesador	
import joblib

# registro de eventos
import logging

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

# función principal
def run_batch():
    logging.info('Start Modeling')
    model = train_model()

    INPUT_FOLDER = os.getenv("INPUT_FOLDER")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
    PROCESSED_FOLDER = os.path.join(INPUT_FOLDER, 'processed')
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    logging.info(f"Loading new input data: {INPUT_FOLDER}")

    inactivity_counter = 0
    MAX_INACTIVITY_CYCLES = 6

    while True:
        archivos_nuevos = [file for file in os.listdir(INPUT_FOLDER) if file.endswith('.parquet')]

        if not archivos_nuevos:
            logging.warning("Files not found. waiting...")
            inactivity_counter += 1
            if inactivity_counter >= MAX_INACTIVITY_CYCLES:
                logging.warning("Files not found in 1 minute. Closing process..")
                break
            time.sleep(10)
            continue
        
        # Define el número máximo de ciclos de inactividad antes de finalizar el proceso (en este caso, 1 minuto).
        inactivity_counter = 0 

        # Busqueda de archivos nuevos
        
        for file_name in archivos_nuevos:
            logging.info(f"Procesesing file: {file_name}")
            input_file = os.path.join(INPUT_FOLDER, file_name)
            input_df = pd.read_parquet(input_file)

            if os.path.exists('preprocessor.pkl'):
                preprocessor = joblib.load('preprocessor.pkl')
                X_input = preprocessor.transform(input_df)
            else:
                raise FileNotFoundError("Processor not found.")

            probabilities = model.predict_proba(X_input)

            predictions = [{f"Clase{i+1}": prob for i, prob in enumerate(prob_row)} for prob_row in probabilities]

            output_df = pd.DataFrame(predictions)
            output_file = os.path.join(OUTPUT_FOLDER, f"predictions_{file_name}")
            output_df.to_parquet(output_file)

            logging.info(f"predictions saved in: {output_file}")

            shutil.move(input_file, os.path.join(PROCESSED_FOLDER, file_name))
            logging.info(f"processed file move to: {PROCESSED_FOLDER}")
