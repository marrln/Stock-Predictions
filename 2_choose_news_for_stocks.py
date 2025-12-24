import os
import pandas as pd
import json

NEWS_CSV = "Stock_news/nasdaq_exteral_data.csv"
OUTPUT_DIR = "Stock_news"
FILTERED_NEWS_CSV = os.path.join(OUTPUT_DIR, "sp500_news.csv")

NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
PRICE_DIR = "Stock_price/full_history"

YEAR_START = 2018
YEAR_END = 2023

# Get valid tickers from price directory
valid_tickers = set()
for fname in os.listdir(PRICE_DIR):
    if fname.endswith('.csv'):
        valid_tickers.add(fname[:-4])

def filter_and_stats_news(news_csv, output_dir, filtered_news_csv, news_stats_json, year_start, year_end, tickers):
    os.makedirs(output_dir, exist_ok=True)
    # Remove output file if it exists to avoid appending to old data
    if os.path.exists(filtered_news_csv):
        os.remove(filtered_news_csv)
    news_stats = {}
    first_chunk = True
    chunk_idx = 0
    # Determine columns to read (all except drop_cols)
    drop_cols = ['Unnamed: 0', 'Url', 'Author', 'Publisher', 'Lsa_summary', 'Textrank_summary', 'Lexrank_summary', 'Lunh_summary'] 
    all_cols = pd.read_csv(news_csv, nrows=0).columns.tolist()
    for chunk in pd.read_csv(news_csv, usecols=[c for c in all_cols if c not in drop_cols], chunksize=50_000):
        try:
            chunk_idx += 1
            print(f"[INFO] Processing chunk {chunk_idx}/?", flush=True)

            # Drop rows with missing Article
            before = len(chunk)
            chunk = chunk.dropna(subset=['Article'])
            after = len(chunk)
            if before != after:
                print(f"[CLEAN] Dropped {before - after} rows with missing Article in chunk {chunk_idx}", flush=True)

            if chunk.empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after Article drop.", flush=True)
                continue
            
            chunk = chunk[chunk['Stock_symbol'].isin(tickers)]
            if chunk.empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after ticker filter.", flush=True)
                continue
            
            chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
            chunk = chunk[(chunk['Date'].dt.year >= year_start) & (chunk['Date'].dt.year <= year_end)]
            
            if chunk.empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after date filter.", flush=True)
                continue
            
            chunk.to_csv(filtered_news_csv, index=False, mode='a', header=first_chunk)
            first_chunk = False
            
            # Efficiently update stats using value_counts
            chunk['year'] = chunk['Date'].dt.year
            counts = chunk.value_counts(['Stock_symbol', 'year'])
            for (symbol, year), count in counts.items():
                if symbol not in news_stats:
                    news_stats[symbol] = {}
                news_stats[symbol][str(year)] = news_stats[symbol].get(str(year), 0) + int(count)
                
        except Exception as e:
            print(f"[ERROR] Exception in chunk {chunk_idx}: {e}", flush=True)
            
    # Ensure all stocks and all years are present in stats
    all_years = [str(y) for y in range(year_start, year_end + 1)]
    for ticker in tickers:
        if ticker not in news_stats:
            news_stats[ticker] = {}
        for year in all_years:
            if year not in news_stats[ticker]:
                news_stats[ticker][year] = 0
                
    with open(news_stats_json, 'w') as f:
        json.dump(news_stats, f, indent=4)
        
    print(f"[INFO] Filtered news CSV saved to {filtered_news_csv}")
    print(f"[INFO] News stats saved to {news_stats_json}")

if __name__ == "__main__":
    filter_and_stats_news(NEWS_CSV, OUTPUT_DIR, FILTERED_NEWS_CSV, NEWS_STATS_JSON, YEAR_START, YEAR_END, valid_tickers)
