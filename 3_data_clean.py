import os
import json
import pandas as pd

NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
STOCK_PRICE_DIR = "Stock_price/full_history"
SP500_CSV = os.path.join("data_stats", "sp500.csv")

YEAR_START = 2018
YEAR_END = 2023

def find_tickers_without_news_for_all_years(stock_price_dir, year_start, year_end):
    # Get all tickers from stock price directory
    all_tickers = set()
    for fname in os.listdir(stock_price_dir):
        if fname.endswith('.csv'):
            all_tickers.add(fname[:-4])
    print(f"[INFO] Found {len(all_tickers)} tickers in stock price directory.", flush=True)
    # Read news_stats data
    with open(NEWS_STATS_JSON, 'r') as f:
        news_stats = json.load(f)
    tickers_missing_news = []
    tickers_no_news = []
    for ticker in all_tickers:
        if ticker not in news_stats:
            tickers_no_news.append(ticker)
            print(f"[MISSING] Ticker {ticker} has no news data at all.", flush=True)
            continue
        ticker_years = news_stats[ticker]
        required_years = set(str(y) for y in range(year_start, year_end + 1))
        # Find years with zero news
        zero_years = [str(y) for y in range(year_start, year_end + 1) if ticker_years.get(str(y), 0) == 0]
        # If all years are zero, it's no news at all
        if len(zero_years) == len(required_years):
            tickers_no_news.append(ticker)
            print(f"[MISSING] Ticker {ticker} has no news for any year.", flush=True)
        # If any year is zero, but not all, it's missing for some years
        elif zero_years:
            tickers_missing_news.append(ticker)
            print(f"[MISSING] Ticker {ticker} is missing news for years: {sorted(zero_years)}", flush=True)
    return tickers_missing_news, tickers_no_news

if __name__ == "__main__":
    
    # Step 1: Find tickers missing news for some years or all years.
    tickers_missing_news, tickers_no_news = find_tickers_without_news_for_all_years(
        STOCK_PRICE_DIR,
        YEAR_START,
        YEAR_END
    )
    print(f"\nTotal tickers missing news for some years: {len(tickers_missing_news)}", flush=True)
    print(f"Total tickers with no news at all: {len(tickers_no_news)}\n", flush=True)

    # Step 2: Delete CSV files for tickers with no news at all
    for ticker in tickers_no_news:
        csv_path = os.path.join(STOCK_PRICE_DIR, f"{ticker}.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"[DELETE] Removed {csv_path} (no news for any year)", flush=True)
            
    sp500_df = pd.read_csv(SP500_CSV)
    for ticker in tickers_missing_news:
        row = sp500_df[sp500_df['Symbol'] == ticker]
        if not row.empty:
            date_added = row.iloc[0].get('Date added', None)
            if date_added:
                try:
                    year_added = int(str(date_added)[:4])
                    if year_added >= YEAR_START:
                        print(f"[INFO] Ticker {ticker} was added to S&P500 in {year_added} (>= {YEAR_START})", flush=True)
                except Exception as e:
                    print(f"[WARN] Could not parse 'Date added' for {ticker}: {date_added}", flush=True)
        else:
            print(f"[WARN] Ticker {ticker} not found in SP500 list.", flush=True) # Should not happen
   