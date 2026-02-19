import os
import time
from loguru import logger
from utils_bq import get_bq_client

# Configure Logger: Terminal + File
LOG_FILE = "logs/pipeline.log"
os.makedirs("logs", exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", retention="10 days", level="INFO")

def run_sql_file(client, file_path):
    """Reads and executes a SQL file in BigQuery."""
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        return False

    logger.info(f"Executing: {file_path}...")
    with open(file_path, "r") as f:
        # Split by ';' to handle multiple statements in one file
        queries = f.read().split(';')
    
    for query in queries:
        query = query.strip()
        if not query:
            continue
        
        try:
            query_job = client.query(query)
            query_job.result() # Wait for completion
        except Exception as e:
            logger.exception(f"Error executing queries in {file_path}")
            return False
            
    logger.success(f"Finished: {file_path}")
    return True

def main():
    client = get_bq_client()
    
    # Define the sequence of files with stage descriptions
    pipeline_steps = [
        ("Silver Layer (Data Cleaning)", "src/sql/etl.sql"),
        ("Quality Layer (Raw RFM Metrics)", "src/sql/rfm_quality.sql"),
        ("Ready Layer (Log-Transformed features)", "src/sql/rfm_ready.sql"),
        ("ML Layer (Train K-Means Model)", "src/sql/model_training.sql"),
        ("Final Layer (Predict & Label segments)", "src/sql/scoring.sql")
    ]
    
    start_time = time.time()
    logger.info("="*50)
    logger.info("STARTING CLOUD-NATIVE SEGMENTATION PIPELINE")
    logger.info("="*50)
    
    try:
        for stage_name, script_path in pipeline_steps:
            logger.info(f"[STEP] {stage_name}")
            success = run_sql_file(client, script_path)
            if not success:
                logger.error(f"Pipeline FAILED at stage: {stage_name}")
                return

        duration = time.time() - start_time
        logger.success("="*50)
        logger.success(f"PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} seconds.")
        logger.success("="*50)
        
        print("\nNext Steps:")
        print("1. Open 'notebooks/eda_etl.ipynb' to see Data Quality visuals.")
        print("2. Open 'notebooks/bq_analysis.ipynb' for Cluster & Drift analysis.")
        print("-" * 50)
        
    except Exception:
        logger.exception("Unexpected error during pipeline execution")

if __name__ == "__main__":
    main()
