"""
TF 1 Combine Data - With Cumulative Sentiment Features
==========================================================

- Uses daily_sentiment_v2.csv which has cumulative features
- Includes sentiment_cum_5d, sentiment_cum_10d, sentiment_momentum, etc.
- Better handling of missing data (forward fill for cumulative features)

Usage:
    python3 tf_1_combine_data.py --tickers AAPL,MSFT,NVDA --verbose
    
    python3 tf_1_combine_data.py --all --verbose
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
import argparse
import json
import pandas as pd


# Paths
DATA_STATS_DIR = "data_stats"
NEWS_STATS_JSON = os.path.join(DATA_STATS_DIR, "news_stats.json")
DAILY_SENTIMENT_CSV_V2 = os.path.join(DATA_STATS_DIR, "daily_sentiment_v2.csv") 
DAILY_SENTIMENT_CSV = os.path.join(DATA_STATS_DIR, "daily_sentiment.csv")  # Fallback
FULL_HISTORY_DIR = "Stock_price/full_history"
PROCESSED_DATA_DIR = "processed_data/csv_v2"  # New output dir


def load_top_tickers(n: int) -> List[str]:
    """Load top N tickers by article count."""
    if not os.path.exists(NEWS_STATS_JSON):
        return []
    
    with open(NEWS_STATS_JSON, 'r') as f:
        stats = json.load(f)
    
    ticker_counts = []
    for ticker, data in stats.items():
        if isinstance(data, dict) and 'total_articles' in data:
            ticker_counts.append((ticker, data['total_articles']))
        elif isinstance(data, (int, float)):
            ticker_counts.append((ticker, data))
    
    ticker_counts.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in ticker_counts[:n]]


def combine_data(
    sentiment_csv: str = DAILY_SENTIMENT_CSV_V2,
    full_history_dir: str = FULL_HISTORY_DIR,
    output_dir: str = PROCESSED_DATA_DIR,
    tickers: Optional[List[str]] = None,
    verbose: bool = False
) -> List[Path]:
    """
    Combine price history with cumulative sentiment features.
    
    - Includes cumulative features (sentiment_cum_5d, etc.)
    - Forward-fills cumulative features (they should persist)
    - Fills raw daily_sentiment with 0 (no news = neutral)
    """
    
    sentiment_csv = Path(sentiment_csv)
    full_history_dir = Path(full_history_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sentiment data
    print(f"[INFO] Loading sentiment from {sentiment_csv}")
    sent_df = pd.read_csv(sentiment_csv)
    
    # Standardize column names
    sent_df.columns = [c.lower() for c in sent_df.columns]
    sent_df['date'] = pd.to_datetime(sent_df['date'])
    sent_df['ticker'] = sent_df['ticker'].str.upper()
    
    print(f"[INFO] Sentiment data: {len(sent_df)} records, {sent_df['ticker'].nunique()} tickers")
    print(f"[INFO] Sentiment columns: {list(sent_df.columns)}")
    
    # Get price files
    price_files = {p.stem.upper(): p for p in full_history_dir.glob("*.csv")}
    
    if tickers is None:
        tickers = sorted(price_files.keys())
    else:
        tickers = [t.upper() for t in tickers]
    
    written = []
    
    for ticker in tickers:
        if ticker not in price_files:
            print(f"[WARN] No price file for {ticker}")
            continue
        
        # Load price data
        price_df = pd.read_csv(price_files[ticker])
        price_df.columns = [c.lower().replace(' ', '_') for c in price_df.columns]
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date').reset_index(drop=True)
        
        # Filter sentiment for this ticker
        sent_ticker = sent_df[sent_df['ticker'] == ticker].copy()

        # Avoid duplicate 'ticker' column after merge by removing from inputs
        if 'ticker' in price_df.columns:
            price_df = price_df.drop(columns=['ticker'])
        if 'ticker' in sent_ticker.columns:
            sent_ticker = sent_ticker.drop(columns=['ticker'])
        
        if verbose:
            print(f"\n{ticker}: {len(price_df)} price days, {len(sent_ticker)} sentiment days")
        
        # Merge
        merged = price_df.merge(sent_ticker, on='date', how='left')
        
        # Identify column types for filling
        cumulative_cols = [c for c in merged.columns if 'cum_' in c or 'ma_' in c or 
                          'ewm_' in c or 'intensity' in c or 'vol_' in c]
        raw_cols = ['daily_sentiment', 'n_articles', 'sentiment_momentum', 
                   'sentiment_acceleration', 'sentiment_surprise', 'news_spike',
                   'has_news', 'strong_sentiment', 'sentiment_std', 'sentiment_strength']
        
        # Fill cumulative features: forward fill (they persist)
        for col in cumulative_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().fillna(0)
        
        # Fill raw features: 0 (no news = neutral)
        for col in raw_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Fill confidence
        if 'daily_confidence' in merged.columns:
            merged['daily_confidence'] = merged['daily_confidence'].fillna(0)
        
        # Add ticker column (safe: inputs no longer include 'ticker')
        merged.insert(0, 'ticker', ticker)
        
        # Remove duplicate ticker column if exists
        if 'ticker_y' in merged.columns:
            merged = merged.drop('ticker_y', axis=1)
        if 'ticker_x' in merged.columns:
            merged = merged.rename(columns={'ticker_x': 'ticker'})
        
        # Sort and save
        merged = merged.sort_values('date').reset_index(drop=True)
        
        out_path = output_dir / f"{ticker}.csv"
        merged.to_csv(out_path, index=False)
        
        if verbose:
            coverage = (merged['n_articles'] > 0).mean() * 100 if 'n_articles' in merged.columns else 0
            print(f"  → Saved {out_path} ({len(merged)} rows, {coverage:.1f}% news coverage)")
        
        written.append(out_path)
    
    return written


def main():
    parser = argparse.ArgumentParser(description="Combine price with cumulative sentiment")
    
    parser.add_argument('--sentiment-csv', default=DAILY_SENTIMENT_CSV_V2)
    parser.add_argument('--full-history-dir', default=FULL_HISTORY_DIR)
    parser.add_argument('--output-dir', default=PROCESSED_DATA_DIR)
    
    parser.add_argument('--tickers', type=str, default='', help='Comma-separated tickers')
    parser.add_argument('--top', type=int, default=0, help='Process top N tickers')
    parser.add_argument('--all', action='store_true', help='Process all tickers')
    
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TF 1: Combine Data (with Cumulative Sentiment)")
    print("="*60)
    
    # Determine tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    elif args.top > 0:
        tickers = load_top_tickers(args.top)
    elif args.all:
        tickers = None
    else:
        print("[ERROR] Specify --tickers, --top, or --all")
        return
    
    print(f"\nSettings:")
    print(f"  Sentiment CSV: {args.sentiment_csv}")
    print(f"  Price dir: {args.full_history_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Tickers: {'ALL' if tickers is None else tickers}")
    
    written = combine_data(
        sentiment_csv=args.sentiment_csv,
        full_history_dir=args.full_history_dir,
        output_dir=args.output_dir,
        tickers=tickers,
        verbose=args.verbose
    )
    
    print(f"\n[DONE] Wrote {len(written)} files to {args.output_dir}")
    
    print(f"\nNext steps:")
    print(f"  1. Run: python3 tf_2_train.py --data-dir processed_data/csv_v2 --ablation")


if __name__ == "__main__":
    main()
