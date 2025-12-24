"""
Exploratory Data Analysis for electricity price data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def price_distribution_analysis(df):
    """Compute distribution statistics for RRP."""
    rrp = df['RRP']
    
    return {
        'mean': rrp.mean(),
        'median': rrp.median(),
        'std': rrp.std(),
        'skewness': stats.skew(rrp.dropna()),
        'kurtosis': stats.kurtosis(rrp.dropna()),
        'iqr': rrp.quantile(0.75) - rrp.quantile(0.25),
        'percentile_5': rrp.quantile(0.05),
        'percentile_95': rrp.quantile(0.95),
    }


def volatility_analysis(df, window=288):
    """
    Calculate rolling volatility. Default window is 288 intervals (1 day at 5min).
    """
    df = df.copy()
    df['rolling_std'] = df['RRP'].rolling(window=window, min_periods=1).std()
    df['rolling_mean'] = df['RRP'].rolling(window=window, min_periods=1).mean()
    df['volatility_ratio'] = df['rolling_std'] / df['rolling_mean'].abs().replace(0, np.nan)
    
    return df[['SETTLEMENTDATE', 'RRP', 'rolling_std', 'rolling_mean', 'volatility_ratio']]


def temporal_patterns(df):
    """Analyze price patterns by hour and day of week."""
    df = df.copy()
    df['hour'] = df['SETTLEMENTDATE'].dt.hour
    df['dayofweek'] = df['SETTLEMENTDATE'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    
    hourly = df.groupby('hour')['RRP'].agg(['mean', 'std', 'median'])
    daily = df.groupby('dayofweek')['RRP'].agg(['mean', 'std', 'median'])
    
    weekend_avg = df[df['is_weekend']]['RRP'].mean()
    weekday_avg = df[~df['is_weekend']]['RRP'].mean()
    
    peak_hours = [7, 8, 9, 17, 18, 19, 20]
    peak_avg = df[df['hour'].isin(peak_hours)]['RRP'].mean()
    offpeak_avg = df[~df['hour'].isin(peak_hours)]['RRP'].mean()
    
    return {
        'hourly_stats': hourly,
        'daily_stats': daily,
        'weekend_avg': weekend_avg,
        'weekday_avg': weekday_avg,
        'peak_avg': peak_avg,
        'offpeak_avg': offpeak_avg,
        'peak_premium': (peak_avg - offpeak_avg) / offpeak_avg * 100 if offpeak_avg else 0
    }


def outlier_analysis(df, spike_threshold=300, negative_threshold=0):
    """Identify and analyze price spikes and negative prices."""
    spikes = df[df['RRP'] > spike_threshold].copy()
    negatives = df[df['RRP'] < negative_threshold].copy()
    
    q1 = df['RRP'].quantile(0.25)
    q3 = df['RRP'].quantile(0.75)
    iqr = q3 - q1
    outlier_low = q1 - 1.5 * iqr
    outlier_high = q3 + 1.5 * iqr
    
    iqr_outliers = df[(df['RRP'] < outlier_low) | (df['RRP'] > outlier_high)]
    
    return {
        'spike_count': len(spikes),
        'spike_percent': len(spikes) / len(df) * 100,
        'spike_max': spikes['RRP'].max() if len(spikes) > 0 else None,
        'negative_count': len(negatives),
        'negative_percent': len(negatives) / len(df) * 100,
        'negative_min': negatives['RRP'].min() if len(negatives) > 0 else None,
        'iqr_outlier_count': len(iqr_outliers),
        'outlier_bounds': (outlier_low, outlier_high),
        'spike_events': spikes,
        'negative_events': negatives
    }


def plot_price_histogram(df, save_path=None):
    """Distribution histogram with statistics overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rrp = df['RRP']
    
    # Full distribution
    axes[0].hist(rrp, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(rrp.mean(), color='red', linestyle='--', label=f'Mean: ${rrp.mean():.1f}')
    axes[0].axvline(rrp.median(), color='orange', linestyle='--', label=f'Median: ${rrp.median():.1f}')
    axes[0].set_xlabel('Price ($/MWh)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution (Full Range)')
    axes[0].legend()
    
    # Clipped for detail
    clipped = rrp[(rrp > -50) & (rrp < 300)]
    axes[1].hist(clipped, bins=80, edgecolor='black', alpha=0.7, color='teal')
    axes[1].axvline(clipped.mean(), color='red', linestyle='--', label=f'Mean: ${clipped.mean():.1f}')
    axes[1].set_xlabel('Price ($/MWh)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Price Distribution (-$50 to $300)')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_volatility(df, save_path=None):
    """Rolling volatility over time."""
    vol_df = volatility_analysis(df)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    axes[0].plot(vol_df['SETTLEMENTDATE'], vol_df['RRP'], alpha=0.5, linewidth=0.5, color='gray')
    axes[0].plot(vol_df['SETTLEMENTDATE'], vol_df['rolling_mean'], color='blue', linewidth=1, label='24h Rolling Mean')
    axes[0].set_ylabel('Price ($/MWh)')
    axes[0].set_title('Price and Rolling Mean')
    axes[0].legend()
    
    axes[1].fill_between(vol_df['SETTLEMENTDATE'], vol_df['rolling_std'], alpha=0.5, color='coral')
    axes[1].set_ylabel('Volatility (Std Dev)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Rolling Volatility (24h Window)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_temporal_patterns(df, save_path=None):
    """Hourly and daily price patterns."""
    patterns = temporal_patterns(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hourly = patterns['hourly_stats']
    axes[0].bar(hourly.index, hourly['mean'], color='steelblue', alpha=0.8)
    axes[0].errorbar(hourly.index, hourly['mean'], yerr=hourly['std']/2, fmt='none', color='black', capsize=2)
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average Price ($/MWh)')
    axes[0].set_title('Hourly Price Pattern')
    axes[0].set_xticks(range(0, 24, 2))
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = patterns['daily_stats']
    daily_indices = daily.index.tolist()
    daily_labels = [day_names[i] for i in daily_indices]
    colors = ['coral' if i >= 5 else 'steelblue' for i in daily_indices]
    axes[1].bar(range(len(daily)), daily['mean'], color=colors, alpha=0.8)
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Average Price ($/MWh)')
    axes[1].set_title('Daily Price Pattern')
    axes[1].set_xticks(range(len(daily)))
    axes[1].set_xticklabels(daily_labels)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_outliers(df, save_path=None):
    """Timeline of extreme price events."""
    outliers = outlier_analysis(df)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    if len(outliers['spike_events']) > 0:
        spikes = outliers['spike_events']
        axes[0].scatter(spikes['SETTLEMENTDATE'], spikes['RRP'], c='red', s=20, alpha=0.6)
        axes[0].set_ylabel('Price ($/MWh)')
        axes[0].set_title(f'Price Spikes (>{300} $/MWh) - {len(spikes)} events')
    else:
        axes[0].text(0.5, 0.5, 'No spikes detected', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Price Spikes')
    
    if len(outliers['negative_events']) > 0:
        negs = outliers['negative_events']
        axes[1].scatter(negs['SETTLEMENTDATE'], negs['RRP'], c='blue', s=20, alpha=0.6)
        axes[1].set_ylabel('Price ($/MWh)')
        axes[1].set_title(f'Negative Prices - {len(negs)} events')
    else:
        axes[1].text(0.5, 0.5, 'No negative prices detected', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Negative Prices')
    
    axes[1].set_xlabel('Date')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def generate_eda_report(data_path, output_dir='charts'):
    """Run full EDA and save all charts."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dispatch_data
    
    df = load_dispatch_data(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Running Exploratory Data Analysis...")
    print("-" * 40)
    
    dist = price_distribution_analysis(df)
    print("\nDistribution Statistics:")
    print(f"  Mean: ${dist['mean']:.2f}")
    print(f"  Median: ${dist['median']:.2f}")
    print(f"  Std Dev: ${dist['std']:.2f}")
    print(f"  Skewness: {dist['skewness']:.2f}")
    print(f"  Kurtosis: {dist['kurtosis']:.2f}")
    
    patterns = temporal_patterns(df)
    print("\nTemporal Patterns:")
    print(f"  Weekday avg: ${patterns['weekday_avg']:.2f}")
    print(f"  Weekend avg: ${patterns['weekend_avg']:.2f}")
    print(f"  Peak avg: ${patterns['peak_avg']:.2f}")
    print(f"  Off-peak avg: ${patterns['offpeak_avg']:.2f}")
    print(f"  Peak premium: {patterns['peak_premium']:.1f}%")
    
    outliers = outlier_analysis(df)
    print("\nOutlier Analysis:")
    print(f"  Spike events (>$300): {outliers['spike_count']} ({outliers['spike_percent']:.2f}%)")
    print(f"  Negative prices: {outliers['negative_count']} ({outliers['negative_percent']:.2f}%)")
    
    print("\nGenerating charts...")
    saved = []
    
    path = output_dir / 'eda_price_distribution.png'
    plot_price_histogram(df, str(path))
    saved.append(str(path))
    
    path = output_dir / 'eda_volatility.png'
    plot_volatility(df, str(path))
    saved.append(str(path))
    
    path = output_dir / 'eda_temporal_patterns.png'
    plot_temporal_patterns(df, str(path))
    saved.append(str(path))
    
    path = output_dir / 'eda_outliers.png'
    plot_outliers(df, str(path))
    saved.append(str(path))
    
    print(f"Saved {len(saved)} charts to {output_dir}/")
    return saved


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "combined_dispatch_prices.csv"
    if data_path.exists():
        generate_eda_report(str(data_path))
    else:
        print(f"Data not found: {data_path}")
