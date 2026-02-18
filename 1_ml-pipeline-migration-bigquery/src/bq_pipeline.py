import os
import time
from utils_bq import get_bq_client

def run_sql_file(client, file_path):
    """Reads and executes a SQL file in BigQuery."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    print(f"Executing: {file_path}...")
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
            print(f"Error in {file_path}:\n{e}")
            return False
            
    print(f"Finished: {file_path}")
    return True

def main():
    client = get_bq_client()
    
    # Define the sequence of files in dependency order
    pipeline_steps = [
        "src/sql/etl.sql",               # Silver Layer
        "src/sql/rfm.sql",               # Gold Layer
        "src/sql/model_training.sql",    # ML Training
        "src/sql/scoring.sql"            # Final Predictions & Labels
    ]
    
    start_time = time.time()
    print("--- Starting BigQuery ML Pipeline ---")
    
    for step in pipeline_steps:
        success = run_sql_file(client, step)
        if not success:
            print(f"Pipeline FAILED at step: {step}")
            return

    duration = time.time() - start_time
    print(f"\n Pipeline COMPLETED SUCCESSFULLY in {duration:.2f} seconds.")
    print("--- Check your BigQuery console for the 'final_scored' tables ---")

if __name__ == "__main__":
    main()
