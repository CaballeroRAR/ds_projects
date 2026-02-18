import pandas as pd
from utils_bq import get_bq_client
from google.cloud import bigquery as bq
import os

def ingest_data(file_path, sheet_name, table_name):
    """
    Reads a specific sheet from Excel and uploads it to BigQuery.
    """
    client = get_bq_client()
    
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    DATASET_ID = os.getenv("GCP_DATASET_ID")
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    print(f"--- Processing Sheet: {sheet_name} ---")
    print(f"Reading data...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # ----- Pyarrow cant parse correclty to objetc (str) so, it is forced here -----
    df['Invoice'] = df['Invoice'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)
    # ----- END -----
    print(f"Uploading {len(df)} rows to {table_id}...")
    
    job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE") # Resets the table (change in next step)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    print(f"Successfully uploaded {sheet_name} to {table_name}")

if __name__ == "__main__":
    # 1. Use a raw string (r"") for Windows paths to fix the SyntaxWarning
    DATA_PATH = r"D:\datasets_complete\datasets-projects\online+retail+ii\online_retail_II.xlsx" 
    
    # 2. Define the mapping: {Sheet Name : BigQuery Table Name}
    # Adjust the sheet names if they are different in your file
    sheets_to_upload = {
        "Year 2009-2010": "raw_retail_2009_2010",
        "Year 2010-2011": "raw_retail_2010_2011"
    }
    
    try:
        for sheet, table in sheets_to_upload.items():
            ingest_data(DATA_PATH, sheet, table)
            
    except Exception as e:
        print(f"Error during ingestion: {e}")