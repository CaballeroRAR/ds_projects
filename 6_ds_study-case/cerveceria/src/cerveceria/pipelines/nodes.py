import pandas as pd
import gspread
import logging

def fetch_sheet_data() -> pd.DataFrame:
    """Fetches data from the Google Sheet, bypassing the empty header rows."""
    gc = gspread.service_account(filename='sa-key.json')
    sh = gc.open_by_key('12HKmitm6TO9hhfXqZ0NijE7728N5tji_CuvwsRNQhz8') # unique document ID
    worksheet = sh.worksheet('test-db')
    data = worksheet.get_all_values()
    
    # Locate the header row dynamically
    header_row_idx = 0
    for i, row in enumerate(data):
        if 'id' in row and 'nombre' in row:
            header_row_idx = i
            break
            
    # Load into DataFrame
    df = pd.DataFrame(data[header_row_idx+1:], columns=data[header_row_idx])
    
    # Drop completely empty columns
    df = df.loc[:, df.columns != '']
    
    # Drop rows that are completely empty
    df.replace('', pd.NA, inplace=True)
    df.dropna(how='all', inplace=True)
    
    # Ensure types for BigQuery (ID as int)
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype(int)
    
    return df

def validate_data_entry(df: pd.DataFrame) -> pd.DataFrame:
    """Validates the extracted Google Sheet data before pushing to GCP."""
    logger = logging.getLogger(__name__)
    
    initial_rows = len(df)
    
    # 1. Deduplicate by ID
    df = df.drop_duplicates(subset=['id'])
    
    # 2. Check for missing IDs (should be handled by extraction, but validating)
    if df['id'].isnull().any():
        logger.warning("Found null 'id' values. Dropping them.")
        df = df.dropna(subset=['id'])
        
    # 3. Basic email validation
    if 'email' in df.columns:
        invalid_emails = df[~df['email'].astype(str).str.contains('@', na=False) & df['email'].notnull()]
        if not invalid_emails.empty:
            logger.warning(f"Found {len(invalid_emails)} rows with invalid emails. Nullifying invalid emails.")
            df.loc[invalid_emails.index, 'email'] = pd.NA

    final_rows = len(df)
    logger.info(f"Data validation complete. Rows before: {initial_rows}, Rows after: {final_rows}")
    
    return df
