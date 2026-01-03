"""
AEMO DispatchIS Data Downloader
Downloads, extracts, and processes NEM 5-minute dispatch price data.
Supports incremental downloads to minimize bandwidth and runtime.
"""

import os
import re
import io
import json
import zipfile
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Set
from zoneinfo import ZoneInfo
from datetime import datetime

# AEMO uses Australian Eastern Standard Time (AEST/AEDT)
AEMO_TIMEZONE = ZoneInfo('Australia/Sydney')

# TradingIS_Reports has ~5 min delay vs 2-3 hours for DispatchIS_Reports
BASE_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/TradingIS_Reports/"
OUTPUT_DIR = Path(__file__).parent / "data"
COMBINED_CSV = OUTPUT_DIR / "combined_dispatch_prices.csv"
DOWNLOAD_STATE_FILE = OUTPUT_DIR / "download_state.json"

def get_zip_links(base_url: str) -> List[str]:
    """Scrape the directory listing for all .zip file links."""
    print(f"Fetching file list from {base_url}")
    response = requests.get(base_url, timeout=30)
    response.raise_for_status()
    
    # Parse HTML to find all .zip links
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.zip') and 'TRADINGIS' in href:
            # Handle relative vs absolute URLs
            if href.startswith('http'):
                links.append(href)
            else:
                links.append(base_url + href.split('/')[-1])
    
    print(f"Found {len(links)} ZIP files")
    return links


def load_download_state() -> dict:
    """Load the download state tracking which files have been processed."""
    if DOWNLOAD_STATE_FILE.exists():
        try:
            with open(DOWNLOAD_STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        'downloaded_files': [],
        'last_update': None,
        'total_files_processed': 0
    }


def save_download_state(state: dict):
    """Save the download state."""
    state['last_update'] = datetime.now().isoformat()
    with open(DOWNLOAD_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_new_files(all_links: List[str], state: dict) -> List[str]:
    """Filter to only files we haven't downloaded yet."""
    downloaded = set(state.get('downloaded_files', []))
    # Extract just the filename for comparison
    downloaded_names = {url.split('/')[-1] for url in downloaded}
    
    new_links = []
    for link in all_links:
        filename = link.split('/')[-1]
        if filename not in downloaded_names:
            new_links.append(link)
    
    return new_links


def download_and_extract_zip(url: str) -> Optional[pd.DataFrame]:
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
                    if line.startswith('I,TRADING,PRICE,'):
                        # This is the header definition line
                        header_line = line
                    elif line.startswith('D,TRADING,PRICE,'):
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


def download_all_data(max_files: int = None, force_full: bool = False) -> pd.DataFrame:
    """
    Download dispatch files and combine into one DataFrame.
    
    Supports incremental downloads - only fetches new files since last run.
    
    Args:
        max_files: Limit number of files to download (for testing)
        force_full: If True, ignore download state and re-download everything
        
    Returns:
        Combined DataFrame with all price data
    """
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing data and download state
    existing_df = load_existing_data()
    state = load_download_state() if not force_full else {'downloaded_files': [], 'last_update': None, 'total_files_processed': 0}
    
    # Get list of all ZIP files
    all_links = get_zip_links(BASE_URL)
    
    # Filter to only new files (incremental download)
    if not force_full:
        new_links = get_new_files(all_links, state)
        print(f"Incremental mode: {len(new_links)} new files to download (skipping {len(all_links) - len(new_links)} already processed)")
    else:
        new_links = all_links
        print(f"Full download mode: downloading all {len(new_links)} files")
    
    if max_files:
        new_links = new_links[:max_files]
        print(f"Limiting to {max_files} files for testing")
    
    # If no new files, return existing data
    if not new_links:
        print("No new files to download. Data is up to date.")
        return existing_df
    
    # Download and process files in parallel
    all_dfs = []
    successfully_downloaded = []
    
    print(f"Downloading and extracting {len(new_links)} files...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(download_and_extract_zip, url): url for url in new_links}
        
        for i, future in enumerate(as_completed(future_to_url)):
            url = future_to_url[future]
            df = future.result()
            if df is not None:
                all_dfs.append(df)
                successfully_downloaded.append(url)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(new_links)} files...")
    
    # Update download state with successfully processed files
    state['downloaded_files'] = list(set(state.get('downloaded_files', []) + successfully_downloaded))
    # Keep only the last 1000 files in state to prevent unbounded growth
    if len(state['downloaded_files']) > 1000:
        state['downloaded_files'] = state['downloaded_files'][-1000:]
    state['total_files_processed'] = state.get('total_files_processed', 0) + len(successfully_downloaded)
    save_download_state(state)
    
    if not all_dfs:
        print("No new data was extracted!")
        return existing_df
    
    # Combine new DataFrames
    print(f"Combining {len(all_dfs)} new DataFrames...")
    new_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean up the data
    # Remove quotes from SETTLEMENTDATE
    new_df['SETTLEMENTDATE'] = new_df['SETTLEMENTDATE'].str.strip('"')
    
    # Convert to proper types with timezone awareness
    new_df['SETTLEMENTDATE'] = pd.to_datetime(new_df['SETTLEMENTDATE'])
    # Localize to AEMO timezone (AEST/AEDT) if naive
    if new_df['SETTLEMENTDATE'].dt.tz is None:
        new_df['SETTLEMENTDATE'] = new_df['SETTLEMENTDATE'].dt.tz_localize(
            AEMO_TIMEZONE, ambiguous='infer', nonexistent='shift_forward'
        )
    new_df['RRP'] = pd.to_numeric(new_df['RRP'], errors='coerce')
    
    # Merge with existing data
    if not existing_df.empty:
        print(f"Merging with existing data ({len(existing_df)} rows)...")
        # Ensure existing data has timezone info
        if existing_df['SETTLEMENTDATE'].dt.tz is None:
            existing_df['SETTLEMENTDATE'] = pd.to_datetime(existing_df['SETTLEMENTDATE'])
            existing_df['SETTLEMENTDATE'] = existing_df['SETTLEMENTDATE'].dt.tz_localize(
                AEMO_TIMEZONE, ambiguous='infer', nonexistent='shift_forward'
            )
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="AEMO NEM Dispatch Price Data Downloader")
    parser.add_argument('--force', '-f', action='store_true', help='Force full download (ignore incremental state)')
    parser.add_argument('--max-files', '-m', type=int, default=None, help='Limit number of files to download')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  AEMO NEM Dispatch Price Data Downloader")
    print("=" * 60)
    
    # Download data (incremental by default)
    df = download_all_data(max_files=args.max_files, force_full=args.force)
    
    if not df.empty:
        print("\nSample data:")
        print(df[['SETTLEMENTDATE', 'REGIONID', 'RRP']].head(10))
        
        print("\nPrice statistics by region:")
        stats = df.groupby('REGIONID')['RRP'].agg(['min', 'mean', 'max', 'count'])
        print(stats)
