"""
AEMO DispatchIS Data Downloader
Downloads, extracts, and processes NEM 5-minute dispatch price data.
"""

import os
import re
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/DispatchIS_Reports/"
OUTPUT_DIR = Path(__file__).parent / "data"
COMBINED_CSV = OUTPUT_DIR / "combined_dispatch_prices.csv"

def get_zip_links(base_url: str) -> list[str]:
    """Scrape the directory listing for all .zip file links."""
    print(f"Fetching file list from {base_url}")
    response = requests.get(base_url)
    response.raise_for_status()
    
    # Parse HTML to find all .zip links
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.zip') and 'DISPATCHIS' in href:
            # Handle relative vs absolute URLs
            if href.startswith('http'):
                links.append(href)
            else:
                links.append(base_url + href.split('/')[-1])
    
    print(f"Found {len(links)} ZIP files")
    return links


def download_and_extract_zip(url: str) -> pd.DataFrame | None:
    """Download a ZIP file and extract the price data from the CSV inside."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract ZIP in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Get the CSV filename inside the ZIP
            csv_files = [f for f in zf.namelist() if f.endswith('.CSV')]
            if not csv_files:
                return None
            
            # Read the CSV
            with zf.open(csv_files[0]) as csv_file:
                # Read raw lines and filter for DISPATCH,PRICE rows
                lines = csv_file.read().decode('utf-8').splitlines()
                
                # Find the header row for DISPATCH,PRICE
                header_line = None
                data_lines = []
                
                for line in lines:
                    if line.startswith('I,DISPATCH,PRICE,'):
                        # This is the header definition line
                        header_line = line
                    elif line.startswith('D,DISPATCH,PRICE,'):
                        # This is a data row
                        data_lines.append(line)
                
                if not header_line or not data_lines:
                    return None
                
                # Parse header: I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,...
                header_parts = header_line.split(',')
                columns = header_parts[4:]  # Skip I,DISPATCH,PRICE,5
                
                # Parse data rows
                records = []
                for line in data_lines:
                    parts = line.split(',')
                    values = parts[4:]  # Skip D,DISPATCH,PRICE,5
                    if len(values) >= len(columns):
                        records.append(values[:len(columns)])
                
                if records:
                    df = pd.DataFrame(records, columns=columns)
                    return df
                    
        return None
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None


def download_all_data(max_files: int = None) -> pd.DataFrame:
    """Download all available dispatch files and combine into one DataFrame."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of all ZIP files
    zip_links = get_zip_links(BASE_URL)
    
    if max_files:
        zip_links = zip_links[:max_files]
        print(f"Limiting to {max_files} files for testing")
    
    # Download and process files in parallel
    all_dfs = []
    
    print(f"Downloading and extracting {len(zip_links)} files...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(download_and_extract_zip, url): url for url in zip_links}
        
        for i, future in enumerate(as_completed(future_to_url)):
            df = future.result()
            if df is not None:
                all_dfs.append(df)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(zip_links)} files...")
    
    if not all_dfs:
        print("No data was extracted!")
        return pd.DataFrame()
    
    # Combine all DataFrames
    print(f"Combining {len(all_dfs)} DataFrames...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean up the data
    # Remove quotes from SETTLEMENTDATE
    combined_df['SETTLEMENTDATE'] = combined_df['SETTLEMENTDATE'].str.strip('"')
    
    # Convert to proper types
    combined_df['SETTLEMENTDATE'] = pd.to_datetime(combined_df['SETTLEMENTDATE'])
    combined_df['RRP'] = pd.to_numeric(combined_df['RRP'], errors='coerce')
    
    # Sort by date and region
    combined_df = combined_df.sort_values(['SETTLEMENTDATE', 'REGIONID']).reset_index(drop=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'])
    
    print(f"Combined dataset: {len(combined_df)} rows")
    print(f"Date range: {combined_df['SETTLEMENTDATE'].min()} to {combined_df['SETTLEMENTDATE'].max()}")
    print(f"Regions: {combined_df['REGIONID'].unique().tolist()}")
    
    # Save to CSV
    combined_df.to_csv(COMBINED_CSV, index=False)
    print(f"Saved to: {COMBINED_CSV}")
    
    return combined_df


def load_existing_data() -> pd.DataFrame:
    """Load previously downloaded data if it exists."""
    if COMBINED_CSV.exists():
        print(f"Loading existing data from {COMBINED_CSV}")
        df = pd.read_csv(COMBINED_CSV)
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
        return df
    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("  AEMO NEM Dispatch Price Data Downloader")
    print("=" * 60)
    
    # Download all available data (use max_files=10 for testing)
    df = download_all_data(max_files=None)
    
    if not df.empty:
        print("Sample data:")
        print(df[['SETTLEMENTDATE', 'REGIONID', 'RRP']].head(10))
        
        print("Price statistics by region:")
        stats = df.groupby('REGIONID')['RRP'].agg(['min', 'mean', 'max', 'count'])
        print(stats)
