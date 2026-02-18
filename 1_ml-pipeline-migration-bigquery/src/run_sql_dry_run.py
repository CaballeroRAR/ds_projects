import sys
import os
from utils_bq import get_bq_client
from google.cloud import bigquery as bq

def dry_run_sql(file_path):
    client = get_bq_client()
    
    # Read the SQL file
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, "r") as f:
        sql_query = f.read()

    # Configure the Dry Run
    job_config = bq.QueryJobConfig(dry_run=True, use_query_cache=False)

    try:
        # Request the dry run
        query_job = client.query(sql_query, job_config=job_config)
        
        print(f"SQL Syntax Valid: {file_path}")
        print(f"This query will process {query_job.total_bytes_processed / (1024**2):.2f} MB.")
    except Exception as e:
        print(f"SQL Syntax Error in {file_path}:")
        print(e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/run_sql_dry_run.py <path_to_sql_file>")
    else:
        dry_run_sql(sys.argv[1])