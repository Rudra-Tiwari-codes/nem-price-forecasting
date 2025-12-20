"""
Data loader module for AEMO dispatch price data.

Handles loading, cleaning, and preprocessing of AEMO NEMWEB data
for use in arbitrage simulations.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


def load_dispatch_data(
    filepath: str,
    regions: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and preprocess AEMO dispatch price data.
    
    Args:
        filepath: Path to the CSV file
        regions: Optional list of regions to filter (e.g., ['NSW1', 'VIC1'])
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: SETTLEMENTDATE, REGIONID, RRP, etc.
    """
    df = pd.read_csv(filepath)
    
    # Standardize column names (handle variations)
    df.columns = df.columns.str.upper().str.strip()
    
    # Parse datetime
    date_col = None
    for col in ['SETTLEMENTDATE', 'SETTLEMENT_DATE', 'DATETIME', 'TIMESTAMP']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df['SETTLEMENTDATE'] = pd.to_datetime(df[date_col])
        if date_col != 'SETTLEMENTDATE':
            df = df.drop(columns=[date_col])
    
    # Filter by region if specified
    if regions and 'REGIONID' in df.columns:
        df = df[df['REGIONID'].isin(regions)]
    
    # Filter by date range
    if start_date:
        df = df[df['SETTLEMENTDATE'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['SETTLEMENTDATE'] <= pd.to_datetime(end_date)]
    
    # Sort by datetime
    df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)
    
    return df


def get_price_series(df: pd.DataFrame, region: Optional[str] = None) -> pd.Series:
    """
    Extract price series from dispatch data.
    
    Args:
        df: DataFrame from load_dispatch_data
        region: Optional region to filter
        
    Returns:
        Series of RRP values indexed by SETTLEMENTDATE
    """
    data = df.copy()
    
    if region and 'REGIONID' in data.columns:
        data = data[data['REGIONID'] == region]
    
    return data.set_index('SETTLEMENTDATE')['RRP']


def get_price_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for price data.
    
    Args:
        df: DataFrame with RRP column
        
    Returns:
        Dictionary of statistics
    """
    rrp = df['RRP']
    
    return {
        'count': len(rrp),
        'mean': rrp.mean(),
        'median': rrp.median(),
        'std': rrp.std(),
        'min': rrp.min(),
        'max': rrp.max(),
        'negative_count': (rrp < 0).sum(),
        'negative_percent': (rrp < 0).mean() * 100,
        'spike_count': (rrp > 300).sum(),  # Prices > $300/MWh
        'spike_percent': (rrp > 300).mean() * 100,
    }


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return a report.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check for required columns
    required_cols = ['SETTLEMENTDATE', 'RRP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    if 'RRP' in df.columns:
        null_count = df['RRP'].isna().sum()
        if null_count > 0:
            issues.append(f"Missing RRP values: {null_count}")
    
    # Check for duplicate timestamps
    if 'SETTLEMENTDATE' in df.columns and 'REGIONID' in df.columns:
        dupes = df.duplicated(subset=['SETTLEMENTDATE', 'REGIONID']).sum()
        if dupes > 0:
            issues.append(f"Duplicate timestamp-region pairs: {dupes}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'row_count': len(df),
        'date_range': (
            df['SETTLEMENTDATE'].min() if 'SETTLEMENTDATE' in df.columns else None,
            df['SETTLEMENTDATE'].max() if 'SETTLEMENTDATE' in df.columns else None
        )
    }


if __name__ == "__main__":
    # Test with actual data
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        print(f"Loading data from: {data_path}")
        df = load_dispatch_data(str(data_path))
        
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        
        print("\nValidation:")
        validation = validate_data(df)
        print(f"  Valid: {validation['valid']}")
        print(f"  Rows: {validation['row_count']}")
        print(f"  Date range: {validation['date_range']}")
        
        if 'REGIONID' in df.columns:
            print(f"\nRegions: {df['REGIONID'].unique()}")
        
        print("\nPrice Statistics:")
        stats = get_price_statistics(df)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Data file not found: {data_path}")
