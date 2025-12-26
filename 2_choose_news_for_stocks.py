"""
Filters financial news articles to match valid S&P 500 tickers and date range (2018-2023).
Drops unnecessary columns, removes articles with missing text, and counts news per ticker/year.
Outputs sp500_news.csv and news_stats.json for training data preparation.
"""

import os
import pandas as pd
import json
from collections import defaultdict

NEWS_CSV = "Stock_news/nasdaq_exteral_data.csv"
OUTPUT_DIR = "Stock_news"
FILTERED_NEWS_CSV = os.path.join(OUTPUT_DIR, "sp500_news.csv")
TEMP_DEDUP_TRACKER = os.path.join(OUTPUT_DIR, "temp_dedup_tracker.json")

NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
PRICE_DIR = "Stock_price/full_history"

YEAR_START = 2018
YEAR_END = 2023

# Get valid tickers from price directory
valid_tickers = set()
for fname in sorted(os.listdir(PRICE_DIR)):
    if fname.endswith('.csv'):
        valid_tickers.add(fname[:-4])


def load_dedup_tracker():
    """Load deduplication tracker from JSON file."""
    if os.path.exists(TEMP_DEDUP_TRACKER):
        with open(TEMP_DEDUP_TRACKER, 'r') as f:
            return json.load(f)
    return {}


def save_dedup_tracker(tracker):
    """Save deduplication tracker to JSON file."""
    with open(TEMP_DEDUP_TRACKER, 'w') as f:
        json.dump(tracker, f)


def filter_by_date(chunk, year_start, year_end, chunk_idx):
    """Filter chunk by date range and drop invalid dates."""
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    
    # Drop rows with invalid dates (NaT values)
    before_date_drop = len(chunk)
    chunk = chunk.dropna(subset=['Date'])
    if len(chunk) < before_date_drop:
        print(f"[CLEAN] Dropped {before_date_drop - len(chunk)} rows with invalid dates in chunk {chunk_idx}", flush=True)
    
    if chunk.empty:
        return chunk, True
    
    # Filter to only our date range of interest
    chunk = chunk[(chunk['Date'].dt.year >= year_start) & (chunk['Date'].dt.year <= year_end)]
    
    return chunk, chunk.empty


def filter_by_tickers(chunk, tickers):
    """Filter chunk to only include valid tickers."""
    chunk = chunk[chunk['Stock_symbol'].isin(tickers)]
    return chunk


def deduplicate_and_aggregate(chunk, dedup_tracker, chunk_idx, index_offset):
    """Deduplicate articles and aggregate stock symbols, tracking indices for later merging."""
    before_dedup = len(chunk)
    
    # Add index column for tracking
    chunk = chunk.reset_index(drop=True)
    chunk['row_index'] = chunk.index + index_offset
    
    # Track which articles are duplicates and need merging
    for _, row in chunk.iterrows():
        title = row['Article_title']
        symbol = row['Stock_symbol']
        row_idx = row['row_index']
        
        if title in dedup_tracker:
            # This is a duplicate - add stock symbol to existing entry
            if symbol not in dedup_tracker[title]['symbols']:
                dedup_tracker[title]['symbols'].append(symbol)
            dedup_tracker[title]['duplicate_indices'].append(row_idx)
        else:
            # First occurrence of this article
            dedup_tracker[title] = {
                'first_index': row_idx,
                'symbols': [symbol],
                'duplicate_indices': []
            }
    
    # Keep only first occurrence of each article title
    chunk = chunk.drop_duplicates(subset=['Article_title'], keep='first')
    
    if len(chunk) < before_dedup:
        print(f"[DEDUP] Removed {before_dedup - len(chunk)} duplicate article titles in chunk {chunk_idx}", flush=True)
    
    return chunk, dedup_tracker


def check_missing_articles(chunk, total_missing_articles, year_start, year_end, chunk_idx):
    """Check for and remove rows with missing article text."""
    before = len(chunk)
    chunk = chunk.dropna(subset=['Article'])
    after = len(chunk)
    
    if before != after:
        missing_count = before - after
        total_missing_articles += missing_count
        print(f"[WARN] Dropped {missing_count} rows with missing Article in date range {year_start}-{year_end} (chunk {chunk_idx})", flush=True)
    
    return chunk, total_missing_articles


def update_news_stats(chunk, news_stats):
    """Update news statistics for each ticker mentioned in articles."""
    chunk['year'] = chunk['Date'].dt.year
    chunk = chunk.dropna(subset=['year'])
    
    for _, row in chunk.iterrows():
        year = str(int(row['year']))
        stock_symbols = row['Stock_symbol']
        
        # Handle both list and single symbol cases
        if isinstance(stock_symbols, list):
            symbols = stock_symbols
        else:
            symbols = [stock_symbols]
        
        # Count article once for each ticker it mentions
        for symbol in symbols:
            if symbol not in news_stats:
                news_stats[symbol] = {}
            news_stats[symbol][year] = news_stats[symbol].get(year, 0) + 1
    
    return news_stats


def merge_duplicate_symbols(filtered_csv, dedup_tracker, chunk_size=50_000):
    """Merge stock symbols for duplicate articles in the final CSV."""
    print("\n[INFO] Merging duplicate stock symbols...", flush=True)
    
    # Build index to symbols mapping
    index_to_symbols = {}
    for title, data in dedup_tracker.items():
        first_idx = data['first_index']
        all_symbols = data['symbols']
        index_to_symbols[first_idx] = all_symbols
    
    # Read, update, and write back in chunks
    temp_output = filtered_csv + ".temp"
    first_write = True
    
    for chunk in pd.read_csv(filtered_csv, chunksize=chunk_size):
        # Update Stock_symbol column for rows that need it
        if 'row_index' in chunk.columns:
            for idx, row in chunk.iterrows():
                row_idx = row['row_index']
                if row_idx in index_to_symbols:
                    chunk.at[idx, 'Stock_symbol'] = index_to_symbols[row_idx]
            
            # Remove temporary row_index column
            chunk = chunk.drop(columns=['row_index'])
        
        chunk.to_csv(temp_output, index=False, mode='a', header=first_write)
        first_write = False
    
    # Replace original with updated file
    os.replace(temp_output, filtered_csv)
    print(f"[INFO] Successfully merged duplicate stock symbols", flush=True)

def filter_and_stats_news(news_csv, output_dir, filtered_news_csv, news_stats_json, year_start, year_end, tickers):
    """Main function to filter and collect statistics on news data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove output file if it exists to avoid appending to old data
    if os.path.exists(filtered_news_csv):
        os.remove(filtered_news_csv)
    
    # Remove old dedup tracker if exists
    if os.path.exists(TEMP_DEDUP_TRACKER):
        os.remove(TEMP_DEDUP_TRACKER)
    
    news_stats = {}
    first_chunk = True
    chunk_idx = 0
    index_offset = 0
    
    # Load or initialize deduplication tracker
    dedup_tracker = load_dedup_tracker()
    
    # Counter for missing articles
    total_missing_articles = 0
    
    # Determine columns to read (all except drop_cols)
    drop_cols = ['Unnamed: 0', 'Url', 'Author', 'Publisher', 'Lsa_summary', 'Textrank_summary', 'Lexrank_summary', 'Lunh_summary']
    try:
        all_cols = pd.read_csv(news_csv, nrows=0).columns.tolist()
    except FileNotFoundError:
        print(f"[ERROR] News CSV file not found: {news_csv}", flush=True)
        return
    except Exception as e:
        print(f"[ERROR] Could not read news CSV columns: {e}", flush=True)
        return
    
    # Process chunks
    for chunk in pd.read_csv(news_csv, usecols=[c for c in all_cols if c not in drop_cols], chunksize=50_000):
        try:
            chunk_idx += 1
            print(f"[INFO] Processing chunk {chunk_idx}/?", flush=True)

            # STEP 1: Filter by date
            chunk, is_empty = filter_by_date(chunk, year_start, year_end, chunk_idx)
            if is_empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after date filter (outside {year_start}-{year_end}).", flush=True)
                continue

            # STEP 2: Filter by valid tickers
            chunk = filter_by_tickers(chunk, tickers)
            if chunk.empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after ticker filter.", flush=True)
                continue

            # STEP 3: Deduplicate and track for later symbol aggregation
            chunk, dedup_tracker = deduplicate_and_aggregate(chunk, dedup_tracker, chunk_idx, index_offset)
            if chunk.empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after deduplication.", flush=True)
                continue

            # STEP 4: Check for missing articles
            chunk, total_missing_articles = check_missing_articles(chunk, total_missing_articles, year_start, year_end, chunk_idx)
            if chunk.empty:
                print(f"[INFO] Chunk {chunk_idx} is empty after Article validation.", flush=True)
                continue
            
            # Save chunk to CSV
            chunk.to_csv(filtered_news_csv, index=False, mode='a', header=first_chunk)
            first_chunk = False
            
            # Update index offset for next chunk
            index_offset += 50_000
            
            # Periodically save dedup tracker
            if chunk_idx % 10 == 0:
                save_dedup_tracker(dedup_tracker)
                
        except Exception as e:
            print(f"[ERROR] Exception in chunk {chunk_idx}: {e}", flush=True)
    
    # Final save of dedup tracker
    save_dedup_tracker(dedup_tracker)
    
    # STEP 5: Merge duplicate stock symbols in final CSV
    merge_duplicate_symbols(filtered_news_csv, dedup_tracker)
    
    # Clean up temporary tracker file
    if os.path.exists(TEMP_DEDUP_TRACKER):
        os.remove(TEMP_DEDUP_TRACKER)
    
    # STEP 6: Calculate news statistics AFTER merging (now Stock_symbol lists are complete)
    print("\n[INFO] Calculating news statistics from final merged data...", flush=True)
    for chunk in pd.read_csv(filtered_news_csv, chunksize=50_000):
        news_stats = update_news_stats(chunk, news_stats)
    
    # Ensure all stocks and all years are present in stats
    all_years = [str(y) for y in range(year_start, year_end + 1)]
    for ticker in sorted(tickers):
        if ticker not in news_stats:
            news_stats[ticker] = {}
        for year in all_years:
            if year not in news_stats[ticker]:
                news_stats[ticker][year] = 0
    
    # Sort the dictionary by ticker for consistent output
    news_stats = dict(sorted(news_stats.items()))
    
    # Calculate and print summary statistics
    total_articles = sum(sum(years.values()) for ticker, years in news_stats.items() if ticker != '_metadata')
    tickers_with_news = sum(1 for ticker, years in news_stats.items() if sum(years.values()) > 0)
    print(f"\n[SUMMARY] Total articles retained: {total_articles:,}")
    print(f"[SUMMARY] Total missing articles found: {total_missing_articles:,}")
    print(f"[SUMMARY] Tickers with news: {tickers_with_news}/{len(tickers)}")
    print(f"[SUMMARY] Average articles per ticker: {total_articles / len(tickers):.1f}")
                
    with open(news_stats_json, 'w') as f:
        json.dump(news_stats, f, indent=4)
        
    print(f"[INFO] Filtered news CSV saved to {filtered_news_csv}")
    print(f"[INFO] News stats saved to {news_stats_json}")


if __name__ == "__main__":
    filter_and_stats_news(NEWS_CSV, OUTPUT_DIR, FILTERED_NEWS_CSV, NEWS_STATS_JSON, YEAR_START, YEAR_END, valid_tickers)
