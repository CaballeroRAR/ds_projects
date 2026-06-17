import pandas as pd
import gspread

def fetch_sheet_data() -> pd.DataFrame:
    """Fetches data from the Google Sheet, bypassing the empty header rows."""
    gc = gspread.service_account(filename='sa-key.json')
    sh = gc.open_by_key('12HKmitm6TO9hhfXqZ0NijE7728N5tji_CuvwsRNQhz8')
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
