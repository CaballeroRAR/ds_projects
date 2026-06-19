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

def db_compare(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Compares the newest extract with the existing DB records.
    Logs updates, insertions, and deletions.
    """
    logger = logging.getLogger(__name__)

    # Handle cases where existing_df is None or empty
    if existing_df is None or existing_df.empty:
        logger.info("Existing database table is empty or could not be loaded. All records are new inserts.")
        logger.info(f"New inserts (IDs): {new_df['id'].tolist()}")
        return new_df

    # Align data types for comparison
    new_df = new_df.copy()
    existing_df = existing_df.copy()
    new_df['id'] = new_df['id'].astype(int)
    existing_df['id'] = existing_df['id'].astype(int)

    # Perform outer merge to compare
    common_cols = list(set(new_df.columns).intersection(set(existing_df.columns)))
    if 'id' not in common_cols:
        common_cols.append('id')

    merged = pd.merge(
        new_df[common_cols],
        existing_df[common_cols],
        on='id',
        how='outer',
        suffixes=('_new', '_existing'),
        indicator=True
    )

    # 1. New Inserts
    inserts = merged[merged['_merge'] == 'left_only']
    if not inserts.empty:
        logger.info(f"New inserts detected: {len(inserts)} rows. IDs: {inserts['id'].tolist()}")
    else:
        logger.info("No new inserts detected.")

    # 2. Deletions
    deletions = merged[merged['_merge'] == 'right_only']
    if not deletions.empty:
        logger.info(f"Deletions detected: {len(deletions)} rows. IDs: {deletions['id'].tolist()}")
    else:
        logger.info("No deletions detected.")

    # 3. Updates
    both = merged[merged['_merge'] == 'both']
    updated_ids = []
    
    compare_cols = [c for c in common_cols if c != 'id']
    for idx, row in both.iterrows():
        is_updated = False
        for col in compare_cols:
            val_new = row[f"{col}_new"]
            val_existing = row[f"{col}_existing"]
            if pd.isna(val_new) and pd.isna(val_existing):
                continue
            if val_new != val_existing:
                is_updated = True
                break
        if is_updated:
            updated_ids.append(int(row['id']))

    if updated_ids:
        logger.info(f"Updates detected: {len(updated_ids)} rows. IDs: {updated_ids}")
    else:
        logger.info("No updates detected.")

    return new_df
