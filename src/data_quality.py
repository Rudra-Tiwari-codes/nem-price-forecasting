"""
Data Quality Analysis Module.

Provides comprehensive data quality metrics and monitoring for price data:
- Gap detection and analysis
- Missing interval identification
- Outlier statistics
- Data freshness monitoring
- Quality score calculation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    timestamp: str
    region: str
    total_rows: int
    date_range_start: str
    date_range_end: str
    expected_intervals: int
    actual_intervals: int
    completeness_pct: float
    gap_count: int
    gaps: List[Dict]
    outlier_count: int
    outliers: Dict
    duplicate_count: int
    quality_score: float
    issues: List[str]
    

def detect_gaps(
    df: pd.DataFrame,
    expected_interval_minutes: int = 5,
    max_gap_to_report: int = 50
) -> Tuple[List[Dict], int]:
    """
    Detect gaps in time series data.
    
    Args:
        df: DataFrame with SETTLEMENTDATE column
        expected_interval_minutes: Expected interval between data points
        max_gap_to_report: Maximum number of gaps to include in report
        
    Returns:
        Tuple of (list of gap details, total gap count)
    """
    if 'SETTLEMENTDATE' not in df.columns or len(df) < 2:
        return [], 0
    
    df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)
    
    # Handle timezone-aware datetimes
    if df['SETTLEMENTDATE'].dt.tz is not None:
        timestamps = df['SETTLEMENTDATE'].dt.tz_localize(None)
    else:
        timestamps = df['SETTLEMENTDATE']
    
    # Calculate time differences
    time_diffs = timestamps.diff().dropna()
    expected_delta = timedelta(minutes=expected_interval_minutes)
    
    # Find gaps (where diff > expected + tolerance)
    tolerance = timedelta(minutes=1)
    gap_mask = time_diffs > (expected_delta + tolerance)
    gap_indices = gap_mask[gap_mask].index.tolist()
    
    gaps = []
    for idx in gap_indices[:max_gap_to_report]:
        gap_start = timestamps.iloc[idx - 1]
        gap_end = timestamps.iloc[idx]
        gap_duration = time_diffs.iloc[idx]
        missing_intervals = int(gap_duration.total_seconds() / (expected_interval_minutes * 60)) - 1
        
        gaps.append({
            'start': str(gap_start),
            'end': str(gap_end),
            'duration_minutes': gap_duration.total_seconds() / 60,
            'missing_intervals': missing_intervals
        })
    
    return gaps, len(gap_indices)


def detect_outliers(
    df: pd.DataFrame,
    column: str = 'RRP',
    spike_threshold: float = 300.0,
    negative_threshold: float = -100.0,
    iqr_multiplier: float = 3.0
) -> Dict:
    """
    Detect outliers in price data.
    
    Args:
        df: DataFrame with price column
        column: Column to analyze
        spike_threshold: Price above this is a spike
        negative_threshold: Price below this is extreme negative
        iqr_multiplier: Multiplier for IQR-based outlier detection
        
    Returns:
        Dictionary with outlier statistics
    """
    if column not in df.columns:
        return {'error': f'Column {column} not found'}
    
    prices = df[column].dropna()
    
    if len(prices) == 0:
        return {'error': 'No valid prices'}
    
    # IQR-based outliers
    q1, q3 = prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    iqr_outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
    
    # Threshold-based
    spikes = prices[prices > spike_threshold]
    extreme_negatives = prices[prices < negative_threshold]
    
    return {
        'total_observations': len(prices),
        'spike_count': len(spikes),
        'spike_percent': len(spikes) / len(prices) * 100,
        'spike_max': float(spikes.max()) if len(spikes) > 0 else None,
        'extreme_negative_count': len(extreme_negatives),
        'extreme_negative_percent': len(extreme_negatives) / len(prices) * 100,
        'extreme_negative_min': float(extreme_negatives.min()) if len(extreme_negatives) > 0 else None,
        'iqr_outlier_count': len(iqr_outliers),
        'iqr_outlier_percent': len(iqr_outliers) / len(prices) * 100,
        'iqr_bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
        'price_stats': {
            'min': float(prices.min()),
            'max': float(prices.max()),
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std())
        }
    }


def check_duplicates(df: pd.DataFrame) -> Dict:
    """Check for duplicate records."""
    if 'SETTLEMENTDATE' not in df.columns:
        return {'error': 'SETTLEMENTDATE column not found'}
    
    # Check duplicates with and without region
    if 'REGIONID' in df.columns:
        dupes = df.duplicated(subset=['SETTLEMENTDATE', 'REGIONID'])
        dupe_count = dupes.sum()
        
        # Get sample of duplicates
        if dupe_count > 0:
            dupe_sample = df[dupes].head(5)[['SETTLEMENTDATE', 'REGIONID']].to_dict('records')
        else:
            dupe_sample = []
            
        return {
            'duplicate_count': int(dupe_count),
            'duplicate_percent': dupe_count / len(df) * 100,
            'sample': dupe_sample
        }
    else:
        dupes = df.duplicated(subset=['SETTLEMENTDATE'])
        return {
            'duplicate_count': int(dupes.sum()),
            'duplicate_percent': dupes.sum() / len(df) * 100
        }


def calculate_completeness(
    df: pd.DataFrame,
    expected_interval_minutes: int = 5
) -> Dict:
    """
    Calculate data completeness metrics.
    
    Args:
        df: DataFrame with SETTLEMENTDATE column
        expected_interval_minutes: Expected interval between points
        
    Returns:
        Dictionary with completeness metrics
    """
    if 'SETTLEMENTDATE' not in df.columns or len(df) == 0:
        return {'error': 'Invalid data'}
    
    min_date = df['SETTLEMENTDATE'].min()
    max_date = df['SETTLEMENTDATE'].max()
    
    # Calculate expected number of intervals
    if hasattr(min_date, 'tz') and min_date.tz is not None:
        time_span = (max_date.tz_localize(None) - min_date.tz_localize(None))
    else:
        time_span = (max_date - min_date)
    
    expected_intervals = int(time_span.total_seconds() / (expected_interval_minutes * 60)) + 1
    
    # Count unique intervals per region
    if 'REGIONID' in df.columns:
        actual_per_region = df.groupby('REGIONID')['SETTLEMENTDATE'].nunique().to_dict()
        regions = list(actual_per_region.keys())
    else:
        actual_per_region = {'ALL': df['SETTLEMENTDATE'].nunique()}
        regions = ['ALL']
    
    actual_intervals = df['SETTLEMENTDATE'].nunique()
    
    return {
        'date_range': {
            'start': str(min_date),
            'end': str(max_date),
            'days': time_span.days
        },
        'expected_intervals': expected_intervals,
        'actual_intervals': actual_intervals,
        'completeness_percent': (actual_intervals / expected_intervals * 100) if expected_intervals > 0 else 0,
        'regions': regions,
        'intervals_per_region': actual_per_region
    }


def calculate_quality_score(
    completeness_pct: float,
    gap_count: int,
    outlier_pct: float,
    duplicate_count: int
) -> float:
    """
    Calculate overall data quality score (0-100).
    
    Weights:
    - Completeness: 40%
    - No gaps: 25%
    - No excessive outliers: 20%
    - No duplicates: 15%
    """
    # Completeness score (0-40)
    completeness_score = min(40, completeness_pct * 0.4)
    
    # Gap score (0-25) - penalize for gaps
    gap_penalty = min(25, gap_count * 2)
    gap_score = 25 - gap_penalty
    
    # Outlier score (0-20) - allow some outliers (market spikes are normal)
    # Penalize only if > 5% are outliers
    if outlier_pct <= 5:
        outlier_score = 20
    else:
        outlier_score = max(0, 20 - (outlier_pct - 5) * 2)
    
    # Duplicate score (0-15)
    if duplicate_count == 0:
        duplicate_score = 15
    else:
        duplicate_score = max(0, 15 - duplicate_count * 0.5)
    
    return round(completeness_score + gap_score + outlier_score + duplicate_score, 1)


def generate_quality_report(
    df: pd.DataFrame,
    region: str = 'ALL'
) -> DataQualityReport:
    """
    Generate comprehensive data quality report.
    
    Args:
        df: DataFrame with price data
        region: Region identifier
        
    Returns:
        DataQualityReport object
    """
    issues = []
    
    # Basic validation
    if df.empty:
        return DataQualityReport(
            timestamp=datetime.now().isoformat(),
            region=region,
            total_rows=0,
            date_range_start='N/A',
            date_range_end='N/A',
            expected_intervals=0,
            actual_intervals=0,
            completeness_pct=0,
            gap_count=0,
            gaps=[],
            outlier_count=0,
            outliers={},
            duplicate_count=0,
            quality_score=0,
            issues=['No data available']
        )
    
    # Completeness
    completeness = calculate_completeness(df)
    if completeness.get('completeness_percent', 0) < 95:
        issues.append(f"Data completeness below 95%: {completeness.get('completeness_percent', 0):.1f}%")
    
    # Gaps
    gaps, gap_count = detect_gaps(df)
    if gap_count > 0:
        issues.append(f"Found {gap_count} gaps in time series")
    
    # Outliers
    outliers = detect_outliers(df)
    outlier_count = outliers.get('iqr_outlier_count', 0)
    outlier_pct = outliers.get('iqr_outlier_percent', 0)
    if outlier_pct > 10:
        issues.append(f"High outlier percentage: {outlier_pct:.1f}%")
    
    # Duplicates
    duplicates = check_duplicates(df)
    duplicate_count = duplicates.get('duplicate_count', 0)
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate records")
    
    # Quality score
    quality_score = calculate_quality_score(
        completeness.get('completeness_percent', 0),
        gap_count,
        outlier_pct,
        duplicate_count
    )
    
    return DataQualityReport(
        timestamp=datetime.now().isoformat(),
        region=region,
        total_rows=len(df),
        date_range_start=completeness.get('date_range', {}).get('start', 'N/A'),
        date_range_end=completeness.get('date_range', {}).get('end', 'N/A'),
        expected_intervals=completeness.get('expected_intervals', 0),
        actual_intervals=completeness.get('actual_intervals', 0),
        completeness_pct=round(completeness.get('completeness_percent', 0), 2),
        gap_count=gap_count,
        gaps=gaps,
        outlier_count=outlier_count,
        outliers=outliers,
        duplicate_count=duplicate_count,
        quality_score=quality_score,
        issues=issues
    )


def generate_quality_dashboard_data(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Generate quality dashboard data for all regions.
    
    Args:
        df: Full DataFrame with all regions
        output_path: Optional path to save JSON report
        
    Returns:
        Dictionary with quality data for dashboard
    """
    reports = {}
    
    # Overall report
    overall_report = generate_quality_report(df, 'ALL')
    reports['ALL'] = {
        'quality_score': overall_report.quality_score,
        'total_rows': overall_report.total_rows,
        'completeness_pct': overall_report.completeness_pct,
        'gap_count': overall_report.gap_count,
        'outlier_count': overall_report.outlier_count,
        'issues': overall_report.issues
    }
    
    # Per-region reports
    if 'REGIONID' in df.columns:
        for region in df['REGIONID'].unique():
            region_df = df[df['REGIONID'] == region]
            report = generate_quality_report(region_df, region)
            reports[region] = {
                'quality_score': report.quality_score,
                'total_rows': report.total_rows,
                'completeness_pct': report.completeness_pct,
                'gap_count': report.gap_count,
                'outlier_count': report.outlier_count,
                'issues': report.issues
            }
    
    dashboard_data = {
        'generated_at': datetime.now().isoformat(),
        'overall_score': overall_report.quality_score,
        'date_range': {
            'start': overall_report.date_range_start,
            'end': overall_report.date_range_end
        },
        'regions': reports,
        'summary': {
            'total_issues': len(overall_report.issues),
            'status': 'good' if overall_report.quality_score >= 80 else ('warning' if overall_report.quality_score >= 60 else 'critical')
        }
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        logger.info(f"Quality report saved to {output_path}")
    
    return dashboard_data


def print_quality_report(df: pd.DataFrame, region: str = 'ALL'):
    """Print a formatted quality report to console."""
    report = generate_quality_report(df, region)
    
    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Generated: {report.timestamp}")
    print(f"  Region: {report.region}")
    print(f"  Quality Score: {report.quality_score}/100")
    print("-" * 60)
    
    print(f"\nData Overview:")
    print(f"  Total rows: {report.total_rows:,}")
    print(f"  Date range: {report.date_range_start} to {report.date_range_end}")
    print(f"  Expected intervals: {report.expected_intervals:,}")
    print(f"  Actual intervals: {report.actual_intervals:,}")
    print(f"  Completeness: {report.completeness_pct:.1f}%")
    
    print(f"\nData Issues:")
    print(f"  Gaps found: {report.gap_count}")
    print(f"  Outliers: {report.outlier_count}")
    print(f"  Duplicates: {report.duplicate_count}")
    
    if report.issues:
        print(f"\nIssues Detected:")
        for issue in report.issues:
            print(f"  ⚠ {issue}")
    else:
        print(f"\n✓ No significant issues detected")
    
    if report.gaps:
        print(f"\nGap Details (showing first {len(report.gaps)}):")
        for gap in report.gaps[:5]:
            print(f"  {gap['start']} → {gap['end']} ({gap['missing_intervals']} missing intervals)")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    
    if data_path.exists():
        df = load_dispatch_data(str(data_path))
        print(f"Loaded {len(df)} rows")
        
        # Print overall report
        print_quality_report(df, 'ALL')
        
        # Generate dashboard JSON
        output_path = Path(__file__).parent.parent / "dashboard" / "public" / "data_quality.json"
        dashboard_data = generate_quality_dashboard_data(df, output_path)
        print(f"\nDashboard data saved to {output_path}")
    else:
        print(f"Data not found: {data_path}")
