import os
import pandas as pd
import json

STOCKS_CSV = "data_stats/sp500.csv"
PRICE_STATS = "data_stats/price_stats.json"
STOCKS_PRICE_DIR = "Stock_price/full_history"

YEAR_START = 2018
YEAR_END = 2023


def select_stocks_in_range(stocks_csv, stocks_price_dir, year_start, year_end):
    """Select stocks that have complete price data from year_start to year_end."""
    stock_tickers = set(pd.read_csv(stocks_csv)['Symbol'])
    for fname in os.listdir(stocks_price_dir):
        if not fname.endswith('.csv'):
            continue
        ticker = fname[:-4]
        file_path = os.path.join(stocks_price_dir, fname)
        # Remove file if ticker not in stock list
        if ticker not in stock_tickers:
            try:
                os.remove(file_path)
                print(f"[INFO] Deleted file for unavailable ticker: {fname}")
            except Exception as e:
                print(f"[ERROR] Could not delete {fname}: {e}")
            continue
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            mask = (df['date'] >= f"{year_start}-01-01") & (df['date'] <= f"{year_end}-12-31")
            df = df.loc[mask, ['date','volume','open','high','low','close','adj close']]
            # Delete file if empty after filtering
            if df.empty:
                os.remove(file_path)
                print(f"[INFO] Deleted empty file after date filtering: {fname}")
                continue
            # Check for complete year coverage
            years_present = set(df['date'].dt.year.unique())
            if not all(year in years_present for year in range(year_start, year_end + 1)):
                os.remove(file_path)
                print(f"[INFO] Deleted incomplete data file (missing years): {fname}")
                continue
            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"[ERROR] Could not process {fname}: {e}")


def calc_csv_stats(price_stats, stocks_price_dir, year_start, year_end):
    """Calculate and save statistics for each stock CSV."""
    stats = {}
    for fname in os.listdir(stocks_price_dir):
        if not fname.endswith('.csv'):
            continue
        ticker = fname[:-4]
        file_path = os.path.join(stocks_price_dir, fname)
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            df['year'] = df['date'].dt.year
            ticker_stats = {}
            for year in range(year_start, year_end + 1):
                year_df = df[df['year'] == year]
                if year_df.empty:
                    continue
                n_rows = int(year_df['close'].notna().sum())
                median_close = year_df['close'].median()
                median_volume = year_df['volume'].median()
                min_close = year_df['close'].min()
                max_close = year_df['close'].max()
                volatility = year_df['close'].std()
                # Return: pct change from first to last valid close, robust to NaN/zero
                year_df = year_df.sort_values('date')
                valid_closes = year_df['close'].dropna()
                if len(valid_closes) > 1 and valid_closes.iloc[0] != 0:
                    annual_return = (valid_closes.iloc[-1] - valid_closes.iloc[0]) / valid_closes.iloc[0]
                else:
                    annual_return = None
                # Correlation between open and close
                if year_df['open'].std() > 0 and year_df['close'].std() > 0:
                    corr_open_close = year_df['open'].corr(year_df['close'])
                else:
                    corr_open_close = None
                ticker_stats[str(year)] = {
                    'rows': n_rows,
                    'median_close': median_close,
                    'median_volume': median_volume,
                    'min_close': min_close,
                    'max_close': max_close,
                    'volatility': volatility,
                    'return': annual_return,
                    'corr_open_close': corr_open_close
                }
            stats[ticker] = ticker_stats
        except Exception as e:
            print(f"[ERROR] Could not calculate stats for {fname}: {e}")
    with open(price_stats, 'w') as f:
        json.dump(stats, f, indent=4)
        
        
if __name__ == "__main__":
    select_stocks_in_range(STOCKS_CSV, STOCKS_PRICE_DIR, YEAR_START, YEAR_END)
    calc_csv_stats(PRICE_STATS, STOCKS_PRICE_DIR, YEAR_START, YEAR_END)