"""Download SPY (S&P 500 ETF) price data for market benchmark features."""
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_spy_data(output_path="data_stats/SPY.csv", start_date="2015-01-01"):
    """
    Download SPY price data from Yahoo Finance.
    
    Args:
        output_path: Where to save the CSV file
        start_date: Start date for historical data (format: YYYY-MM-DD)
    """
    print(f"Downloading SPY data from {start_date} to present...")
    
    # Download SPY data
    spy = yf.Ticker("SPY")
    df = spy.history(start=start_date, auto_adjust=False)
    
    # Rename columns to match our stock price format
    df = df.reset_index()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'date': 'Date'
    })
    
    # Select only the columns we need
    df = df[['Date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Downloaded {len(df)} days of SPY data")
    print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"✓ Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Download SPY data starting from 2015 (to cover our stock data range)
    df = download_spy_data()
    
    # Show summary statistics
    print("\nSPY Data Summary:")
    print(df.describe())
    print(f"\nSPY data is now available at: data_stats/SPY.csv")
    print("Market features will be automatically enabled in your models!")
