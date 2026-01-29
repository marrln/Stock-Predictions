"""
Add Cumulative Features to Existing Sentiment Data
===================================================

If you already have daily_sentiment.csv, this script adds
cumulative features WITHOUT re-running FinBERT.

Usage:
    python add_cumulative_features.py --input data_stats/daily_sentiment.csv
    
This will create: data_stats/daily_sentiment_v2.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def add_cumulative_features(
    df: pd.DataFrame,
    windows: list = [3, 5, 10, 20]
) -> pd.DataFrame:
    """Add cumulative sentiment features to existing daily sentiment data."""
    
    df = df.copy()
    
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure required columns exist
    if 'daily_sentiment' not in df.columns:
        raise ValueError("Missing 'daily_sentiment' column")
    if 'ticker' not in df.columns:
        raise ValueError("Missing 'ticker' column")
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Process each ticker
    result_dfs = []
    
    for ticker in tqdm(df['ticker'].unique(), desc="Adding cumulative features"):
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Ensure n_articles exists
        if 'n_articles' not in ticker_df.columns:
            ticker_df['n_articles'] = (ticker_df['daily_sentiment'] != 0).astype(int)
        
        # Sentiment strength
        ticker_df['sentiment_strength'] = (
            ticker_df['daily_sentiment'].abs() * 
            np.log1p(ticker_df['n_articles'])
        )
        
        for window in windows:
            # Cumulative sum
            ticker_df[f'sentiment_cum_{window}d'] = (
                ticker_df['daily_sentiment']
                .rolling(window, min_periods=1)
                .sum()
            )
            
            # Moving average
            ticker_df[f'sentiment_ma_{window}d'] = (
                ticker_df['daily_sentiment']
                .rolling(window, min_periods=1)
                .mean()
            )
            
            # Exponential weighted mean
            ticker_df[f'sentiment_ewm_{window}d'] = (
                ticker_df['daily_sentiment']
                .ewm(span=window, min_periods=1)
                .mean()
            )
            
            # News intensity
            ticker_df[f'news_intensity_{window}d'] = (
                ticker_df['n_articles']
                .rolling(window, min_periods=1)
                .sum()
            )
            
            # Sentiment volatility
            ticker_df[f'sentiment_vol_{window}d'] = (
                ticker_df['daily_sentiment']
                .rolling(window, min_periods=1)
                .std()
                .fillna(0)
            )
        
        # Momentum
        ticker_df['sentiment_momentum'] = ticker_df['sentiment_cum_5d'].diff()
        ticker_df['sentiment_acceleration'] = ticker_df['sentiment_momentum'].diff()
        
        # Surprise
        ticker_df['sentiment_surprise'] = (
            ticker_df['daily_sentiment'] - ticker_df['sentiment_ma_20d']
        )
        
        # News spike
        news_ma = ticker_df['n_articles'].rolling(20, min_periods=1).mean()
        news_std = ticker_df['n_articles'].rolling(20, min_periods=1).std().fillna(1).replace(0, 1)
        ticker_df['news_spike'] = ((ticker_df['n_articles'] - news_ma) / news_std).clip(-3, 3)
        
        # Binary flags
        ticker_df['has_news'] = (ticker_df['n_articles'] > 0).astype(int)
        ticker_df['strong_sentiment'] = (ticker_df['daily_sentiment'].abs() > 0.3).astype(int)
        
        result_dfs.append(ticker_df)
    
    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df = final_df.fillna(0)
    
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Add cumulative features to sentiment data")
    
    # Paths
    parser.add_argument('--input', default='data_stats/daily_sentiment.csv', help='Input sentiment CSV')
    parser.add_argument('--output', default=None, help='Output CSV (default: replace v2 with v2 in input name)')
    # Windows
    parser.add_argument('--windows', default='3,5,10,20', help='Comma-separated window sizes for cumulative features')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / input_path.name.replace('.csv', '_v2.csv')
        if str(output_path) == str(input_path):
            output_path = input_path.parent / (input_path.stem + '_v2.csv')
    else:
        output_path = Path(args.output)
    
    print("\n" + "="*60)
    print("ADD CUMULATIVE FEATURES TO SENTIMENT DATA (V2)")
    print("="*60)
    print(f"\nInput:  {args.input}")
    print(f"Output: {output_path}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records, {df['Ticker' if 'Ticker' in df.columns else 'ticker'].nunique()} tickers")
    
    # Add features
    windows = [int(w) for w in args.windows.split(',')]
    print(f"\nAdding cumulative features with windows: {windows}")
    
    df_enhanced = add_cumulative_features(df, windows)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enhanced.to_csv(output_path, index=False)
    
    print(f"\nSaved to {output_path}")
    print(f"New columns added:")
    
    original_cols = set(df.columns.str.lower())
    for col in df_enhanced.columns:
        if col.lower() not in original_cols:
            print(f"  - {col}")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Run: python3 tf_1_combine_data.py --tickers AAPL,MSFT,NVDA --sentiment-csv {output_path}")
    print(f"  2. Run: python3 tf_2_train.py --data-dir processed_data/csv_v2 --ablation")


if __name__ == "__main__":
    main()
