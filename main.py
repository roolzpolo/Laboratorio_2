import os
from batch_prediction import run_batch
from dotenv import load_dotenv

from api_prediction import run_api
import logging

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    deployment_type = os.getenv("DEPLOYMENT_TYPE")
    
    if deployment_type == "Batch":
        logging.info('Ejecución predict de Batch')
        run_batch()
    elif deployment_type == "API":
        logging.info('Ejecución predict APII')
        run_api()
    else:
        raise ValueError("DEPLOYMENT_TYPE no válido (Batch/API)")

if __name__ == "__main__":
    main()
