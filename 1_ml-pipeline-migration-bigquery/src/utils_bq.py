import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

def get_bq_client():
    """
    Initializes and returns a BigQuery client using service account credentials.
    Requires GOOGLE_APPLICATION_CREDENTIALS path in .env file.
    """
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in .env file")
    
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    return client

if __name__ == "__main__":
    # Connectivity Test
    try:
        client = get_bq_client()
        print(f"Successfully connected to BigQuery project: {client.project}")
        
        # List datasets as a simple test
        datasets = list(client.list_datasets())
        if datasets:
            print("Accessible datasets:")
            for ds in datasets:
                print(f" - {ds.dataset_id}")
        else:
            print("No datasets found in this project.")
            
    except Exception as e:
        print(f"Error connecting to BigQuery: {e}")
