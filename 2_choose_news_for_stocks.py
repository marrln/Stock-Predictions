"""
Filters financial news articles to match valid S&P 500 tickers and date range (2018-2023).
Uses metadata separation: sp500_news_meta.csv for metadata and sp500_news_content.csv for articles.
Drops unnecessary columns, removes articles with missing text, and counts news per ticker/year.
Outputs separate metadata and content files plus news_stats.json.
"""

import os
import pandas as pd
import json
import time
import ast
import hashlib


NEWS_CSV = "Stock_news/nasdaq_exteral_data.csv"
OUTPUT_DIR = "Stock_news"
METADATA_CSV = os.path.join(OUTPUT_DIR, "sp500_news_meta.csv")
CONTENT_CSV = os.path.join(OUTPUT_DIR, "sp500_news_content.csv")

# New paths for filtered and dedup-by-url artifacts
FILTERED_CSV = os.path.join(OUTPUT_DIR, "filtered_news.csv")
DEDUP_JSON = os.path.join(OUTPUT_DIR, "dedup_by_url.json")

NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
PRICE_DIR = "Stock_price/full_history"

YEAR_START = 2018
YEAR_END = 2023

# Get valid tickers from price directory
valid_tickers = set()
for fname in sorted(os.listdir(PRICE_DIR)):
    if fname.endswith('.csv'):
        valid_tickers.add(fname[:-4])


def load_dedup_json(path=DEDUP_JSON):
    """Load dedup-by-url JSON mapping (urlhash -> {url, article_index, stocks})."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read dedup json: {e}", flush=True)
    return {}


def save_dedup_json(d, path=DEDUP_JSON):
    """Save dedup mapping to JSON file."""
    try:
        with open(path, 'w') as f:
            json.dump(d, f, indent=2, sort_keys=True)
        size = os.path.getsize(path)
        print(f"[INFO] Saved dedup json ({path}) size={size/1024/1024:.1f} MB", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to save dedup json: {e}", flush=True)


def filter_chunk(chunk, year_start, year_end, valid_tickers, chunk_idx):
    """Apply date and ticker filters in one pass."""
    original_size = len(chunk)
    
    # Convert date and filter invalid dates
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    
    # Create filter mask (excluding Article check for now)
    # Ensure case-insensitive matching by coercing to string and normalizing; handle NaN/non-strings
    num_non_string = chunk['Stock_symbol'].apply(lambda x: not isinstance(x, str) and pd.notna(x)).sum()
    if num_non_string > 0:
        print(f"[WARN] Chunk {chunk_idx} has {num_non_string} non-string Stock_symbol entries; coercing to string", flush=True)
    chunk['Stock_symbol'] = chunk['Stock_symbol'].fillna('').astype(str).str.strip().str.lower()
    valid_tickers_lower = set(t.lower() for t in valid_tickers)

    mask = (
        chunk['Date'].notna() &
        (chunk['Date'].dt.year >= year_start) &
        (chunk['Date'].dt.year <= year_end) &
        chunk['Stock_symbol'].isin(valid_tickers_lower)
    )
    
    filtered_chunk = chunk[mask].copy()
    
    dropped_count = original_size - len(filtered_chunk)
    if dropped_count > 0:
        print(f"[FILTER] Dropped {dropped_count} rows in chunk {chunk_idx} (invalid date/ticker)", flush=True)
    
    return filtered_chunk


def deduplicate_and_aggregate(chunk, chunk_idx):
    """Aggregate rows within a chunk by Url (one row per Url)."""
    if chunk.empty:
        return chunk

    chunk['Url'] = chunk['Url'].astype(str).str.strip()

    def _agg(g):
        row = g.iloc[0].to_dict()
        row['Stock_symbol'] = sorted({str(s).strip().lower() for s in g['Stock_symbol'] if pd.notna(s) and str(s).strip()})
        row['Article'] = next((a for a in g['Article'] if pd.notna(a) and str(a).strip()), None)
        return row

    deduped = pd.DataFrame([_agg(g) for _, g in chunk.groupby('Url')]).reset_index(drop=True)

    merged = len(chunk) - len(deduped)
    if merged > 0:
        print(f"[DEDUP] Aggregated {merged} rows by Url in chunk {chunk_idx}", flush=True)

    return deduped


def save_news_stats(news_stats, news_stats_json):
    """Helper to save news_stats to JSON file."""
    with open(news_stats_json, 'w') as f:
        json.dump(news_stats, f, indent=4)


def update_news_stats(chunk, news_stats):
    """Update news statistics for each ticker mentioned in articles."""
    if chunk.empty:
        return news_stats

    # Extract year from date
    chunk['year'] = chunk['Date'].dt.year
    chunk = chunk.dropna(subset=['year'])

    for _, row in chunk.iterrows():
        year = str(int(row['year']))
        stock_symbols = row.get('Stock_symbol', [])

        # Normalize and parse possible string representations
        if isinstance(stock_symbols, str):
            try:
                if stock_symbols.strip().startswith('['):
                    symbols = list(ast.literal_eval(stock_symbols))
                else:
                    symbols = [s.strip() for s in stock_symbols.split(',') if s.strip()]
            except Exception:
                symbols = [stock_symbols.strip()]
        elif isinstance(stock_symbols, list):
            symbols = stock_symbols
        else:
            symbols = [stock_symbols]

        for symbol in symbols:
            if not isinstance(symbol, str):
                symbol = str(symbol)
            symbol = symbol.strip().lower()
            if not symbol:
                continue
            if symbol not in news_stats:
                news_stats[symbol] = {}
            news_stats[symbol][year] = news_stats[symbol].get(year, 0) + 1

    return news_stats


def filter_and_stats_news(news_csv, metadata_csv, content_csv, 
                         news_stats_json, year_start, year_end, tickers):
    """Main function to filter and collect statistics on news data."""
    os.makedirs(os.path.dirname(news_stats_json), exist_ok=True)
    
    # Normalize tickers to lowercase for consistent processing
    tickers = set(t.lower() for t in tickers)

    # Remove output files if they exist (start fresh)
    for f in [metadata_csv, content_csv, FILTERED_CSV, DEDUP_JSON]:
        if os.path.exists(f):
            os.remove(f)
    
    news_stats = {}
    first_chunk_content = True
    chunk_idx = 0
    
    # Load or initialize dedup-by-url mapping
    dedup_map = load_dedup_json()

    # Determine starting content index (if content CSV exists)
    content_index = 0
    if os.path.exists(content_csv):
        try:
            existing = pd.read_csv(content_csv)
            if 'content_id' in existing.columns and len(existing) > 0:
                content_index = int(existing['content_id'].max()) + 1
            else:
                content_index = len(existing)
        except Exception:
            content_index = 0

    # Track whether we've written filtered CSV and content CSV headers
    first_chunk_filtered = True
    first_chunk_content = not os.path.exists(content_csv)
    
    # Determine columns to read
    drop_cols = ['Unnamed: 0', 'Author', 'Publisher', 
                 'Lsa_summary', 'Textrank_summary', 
                 'Lexrank_summary', 'Lunh_summary']
    
    try:
        all_cols = pd.read_csv(news_csv, nrows=0).columns.tolist()
        columns_to_use = [c for c in all_cols if c not in drop_cols]
    except FileNotFoundError:
        print(f"[ERROR] News CSV file not found: {news_csv}", flush=True)
        return
    except Exception as e:
        print(f"[ERROR] Could not read news CSV columns: {e}", flush=True)
        return
    
    start_time = time.time()

    # PHASE 1: Filter-only pass - write filtered CSV without Article to reduce size
    print("[INFO] Phase 1: Filtering and writing reduced CSV...", flush=True)
    filtered_rows_total = 0
    p_chunk = 0
    for chunk in pd.read_csv(news_csv, usecols=columns_to_use, chunksize=80_000):
        p_chunk += 1
        chunk = filter_chunk(chunk, year_start, year_end, tickers, p_chunk)
        if chunk.empty:
            continue
        filtered_no_article = chunk.drop(columns=['Article'], errors='ignore')

        # Serialize Stock_symbol consistently
        def _serialize_symbols(x):
            try:
                if isinstance(x, (list, dict)):
                    return json.dumps(x)
                return '' if pd.isna(x) else str(x)
            except Exception:
                return str(x)

        filtered_no_article['Stock_symbol'] = filtered_no_article['Stock_symbol'].apply(_serialize_symbols)
        filtered_no_article.to_csv(FILTERED_CSV, index=False, mode='a', header=(p_chunk == 1))
        filtered_rows_total += len(filtered_no_article)

    print(f"[INFO] Phase 1 complete: wrote {filtered_rows_total} filtered rows to {FILTERED_CSV}", flush=True)

    # PHASE 2: Deduplicate and materialize content by Url using the original CSV (re-applying filters)
    print("[INFO] Phase 2: Building dedup map and content CSV from filtered data...", flush=True)
    chunk_idx = 0

    for chunk in pd.read_csv(news_csv, usecols=columns_to_use, chunksize=80_000):
        chunk_idx += 1
        print(f"[INFO] Processing chunk {chunk_idx}/195 for dedup...", flush=True)
        chunk = filter_chunk(chunk, year_start, year_end, tickers, chunk_idx)
        if chunk.empty:
            continue

        # Aggregate within-chunk by Url
        chunk_deduped = deduplicate_and_aggregate(chunk, chunk_idx)

        # Update global dedup_map and materialize first article when found
        for _, row in chunk_deduped.iterrows():
            url = str(row.get('Url', '')).strip()
            if not url:
                continue
            url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
            entry = dedup_map.get(url_hash, {'Url': url, 'article_index': -1, 'stocks': []})

            # merge stocks
            stocks = row.get('Stock_symbol', [])
            if isinstance(stocks, list):
                merged = sorted(set(entry.get('stocks', [])) | set(stocks))
            else:
                merged = sorted(set(entry.get('stocks', [])) | set([str(stocks).strip().lower()]))
            entry['stocks'] = merged

            # if we don't have content yet but this row has it, materialize
            article_text = row.get('Article')
            if entry.get('article_index', -1) == -1 and pd.notna(article_text) and str(article_text).strip():
                with open(CONTENT_CSV, 'a' if not first_chunk_content else 'w', encoding='utf-8') as f:
                    if first_chunk_content:
                        f.write('content_id,Article\n')
                        first_chunk_content = False
                    safe_text = str(article_text).replace('"', '""').replace('\n', ' ')
                    f.write(f"{content_index},\"{safe_text}\"\n")
                entry['article_index'] = content_index
                content_index += 1

            dedup_map[url_hash] = entry

        # Periodically save dedup map
        if chunk_idx % 5 == 0:
            save_dedup_json({k: {'Url': v['Url'], 'article_index': v['article_index'], 'stocks': v['stocks']} for k, v in dedup_map.items()})

    # Save final dedup mapping
    save_dedup_json(dedup_map)

    # Build final metadata.csv from dedup map
    metadata_rows = []
    for k, v in dedup_map.items():
        metadata_rows.append({'Url': v.get('Url', v.get('url', '')), 'content_id': v['article_index'], 'Stock_symbol': v['stocks']})
    metadata_df = pd.DataFrame(metadata_rows)

    # Serialize stocks as JSON strings for disk
    if not metadata_df.empty:
        metadata_df['Stock_symbol'] = metadata_df['Stock_symbol'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else ('' if pd.isna(x) else str(x)))
        metadata_df.to_csv(metadata_csv, index=False)
    else:
        metadata_df = pd.DataFrame()

    # Compute news_stats by grouping filtered CSV by Url and counting unique articles per ticker per year
    news_stats = {}
    if os.path.exists(FILTERED_CSV):
        filtered_df = pd.read_csv(FILTERED_CSV, parse_dates=['Date'], low_memory=False)
        # group by Url and pick earliest Date per Url
        for url, grp in filtered_df.groupby('Url'):
            try:
                url_hash = hashlib.sha256(str(url).encode('utf-8')).hexdigest()
            except Exception:
                continue
            entry = dedup_map.get(url_hash, None)
            if not entry:
                continue
            stocks = entry.get('stocks', [])
            date_vals = pd.to_datetime(grp['Date'], errors='coerce')
            date_vals = date_vals.dropna()
            if date_vals.empty:
                continue
            year = str(date_vals.min().year)
            for symbol in stocks:
                sym = symbol.strip().lower()
                if not sym:
                    continue
                if sym not in news_stats:
                    news_stats[sym] = {}
                news_stats[sym][year] = news_stats[sym].get(year, 0) + 1

    # Sort the dictionary by ticker for consistent output
    # Ensure all stocks and all years are present in stats
    news_stats = dict(sorted(news_stats.items()))
    all_years = [str(y) for y in range(year_start, year_end + 1)]
    for ticker in sorted(tickers):
        if ticker not in news_stats:
            news_stats[ticker] = {}
        for year in all_years:
            if year not in news_stats[ticker]:
                news_stats[ticker][year] = 0
    
    # Calculate summary statistics
    total_articles = sum(sum(years.values()) for years in news_stats.values())
    tickers_with_news = sum(1 for years in news_stats.values() if sum(years.values()) > 0)
    
    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"[SUMMARY] Processing completed in {elapsed_time:.2f} seconds")
    print(f"[SUMMARY] Total articles retained: {total_articles:,}")
    print(f"[SUMMARY] Tickers with news: {tickers_with_news}/{len(tickers)}")
    avg_articles = total_articles / len(tickers) if len(tickers) else 0
    print(f"[SUMMARY] Average articles per ticker: {avg_articles:.1f}")
    print(f"{'='*60}")
    
    save_news_stats(news_stats, news_stats_json)
    print(f"[INFO] News stats saved to {news_stats_json}")

if __name__ == "__main__":
    filter_and_stats_news(NEWS_CSV, METADATA_CSV, CONTENT_CSV,
                         NEWS_STATS_JSON, YEAR_START, YEAR_END, valid_tickers)