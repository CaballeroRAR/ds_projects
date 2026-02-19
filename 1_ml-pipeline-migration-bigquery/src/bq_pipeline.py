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
    
    # Define the sequence of files with stage descriptions
    pipeline_steps = [
        ("Silver Layer (Data Cleaning)", "src/sql/etl.sql"),
        ("Quality Layer (Raw RFM Metrics)", "src/sql/rfm_quality.sql"),
        ("Ready Layer (Log-Transformed features)", "src/sql/rfm_ready.sql"),
        ("ML Layer (Train K-Means Model)", "src/sql/model_training.sql"),
        ("Final Layer (Predict & Label segments)", "src/sql/scoring.sql")
    ]
    
    start_time = time.time()
    print("--------------------------------------------------")
    print("STARTING CLOUD-NATIVE SEGMENTATION PIPELINE")
    print("--------------------------------------------------")
    
    for stage_name, script_path in pipeline_steps:
        print(f"\n[STEP] {stage_name}")
        success = run_sql_file(client, script_path)
        if not success:
            print(f"Pipeline FAILED at stage: {stage_name}")
            return

    duration = time.time() - start_time
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} seconds.")
    print("="*50)
    print("\nNext Steps:")
    print("1. Open 'notebooks/eda_etl.ipynb' to see Data Quality visuals.")
    print("2. Open 'notebooks/bq_analysis.ipynb' for Cluster & Drift analysis.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
