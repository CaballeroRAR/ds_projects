import os
import sys
from loguru import logger
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_standalone_client():
    """Initializes BigQuery client without external dependencies."""
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not found in .env file")
        sys.exit(1)
    
    try:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        sys.exit(1)

def flush_environment():
    """Deletes all derived tables and models, keeping only raw retail data."""
    client = get_standalone_client()
    dataset_id = os.getenv("GCP_DATASET_ID")
    
    if not dataset_id:
        logger.error("GCP_DATASET_ID not found in .env file")
        sys.exit(1)

    protected_tables = [
        "raw_retail_2009_2010",
        "raw_retail_2010_2011"
    ]

    logger.info(f"Starting environment flush for dataset: {dataset_id}")
    logger.warning("Protected tables (will NOT be deleted): " + ", ".join(protected_tables))

    # 1. Delete Tables
    try:
        tables = list(client.list_tables(dataset_id))
        deleted_count = 0
        
        for table in tables:
            if table.table_id not in protected_tables:
                table_ref = f"{client.project}.{dataset_id}.{table.table_id}"
                client.delete_table(table_ref, not_found_ok=True)
                logger.info(f"Deleted Table: {table.table_id}")
                deleted_count += 1
        
        logger.success(f"Successfully deleted {deleted_count} derived tables.")

    except Exception as e:
        logger.error(f"Error during table deletion: {e}")

    # 2. Delete Models
    try:
        models = list(client.list_models(dataset_id))
        model_count = 0
        
        for model in models:
            model_ref = f"{client.project}.{dataset_id}.{model.model_id}"
            client.delete_model(model_ref, not_found_ok=True)
            logger.info(f"Deleted Model: {model.model_id}")
            model_count += 1
            
        logger.success(f"Successfully deleted {model_count} BigQuery ML models.")

    except Exception as e:
        logger.error(f"Error during model deletion: {e}")

    logger.info("Environment flush completed.")

if __name__ == "__main__":
    # Configure logger for this standalone tool
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    # Confirm with user if running interactively
    # Since I'm an agent, I'll just run it.
    flush_environment()
