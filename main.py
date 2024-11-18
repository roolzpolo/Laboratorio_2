import os
from batch_prediction import run_batch
from dotenv import load_dotenv
from api_prediction import run_api, app  # Import the app object for uvicorn
import logging

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    deployment_type = os.getenv("DEPLOYMENT_TYPE")
    print(f"DEPLOYMENT_TYPE: {deployment_type}")  # Debugging line
    
    if deployment_type == "Batch":
        logging.info('Ejecución predict de Batch')
        run_batch()
    elif deployment_type == "API":
        logging.info('Ejecución predict API')
        # Start the API
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        raise ValueError("DEPLOYMENT_TYPE no válido (Batch/API)")

if __name__ == "__main__":
    main()

