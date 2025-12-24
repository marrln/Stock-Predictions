import os
import json
import pandas as pd
import ast

NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
STOCK_PRICE_DIR = "Stock_price/full_history"
SP500_CSV = os.path.join("data_stats", "sp500.csv")

STOCK_NEWS_DIR = "Stock_news"
SP500_NEWS_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news.csv")
SP500_NEWS_NO_ARTICLES_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news_no_articles.csv")
SP500_NEWS_ARTICLES_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news_articles.csv")
SP500_NEWS_NO_ARTICLES_NEW_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news_no_articles_new.csv")
SP500_NEWS_DEDUP_NO_ARTICLES_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news_dedup_no_articles.csv")
SP500_NEWS_DEDUP_ARTICLES_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news_dedup_articles.csv")
SP500_NEWS_DEDUP_FINAL_NO_ARTICLES_CSV = os.path.join(STOCK_NEWS_DIR, "sp500_news_dedup_final_no_articles.csv")

YEAR_START = 2018
YEAR_END = 2023

def find_tickers_without_news_for_all_years(stock_price_dir, year_start, year_end):
    """Find tickers missing news for some or all years."""
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


def delete_tickers_without_news(tickers_no_news, stock_price_dir):
    """Delete CSV files for tickers with no news at all."""
    for ticker in tickers_no_news:
        csv_path = os.path.join(stock_price_dir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"[DELETE] Removed {csv_path} (no news for any year)", flush=True)


def validate_tickers_with_partial_news(tickers_missing_news, year_start):
    """Validate tickers with partial news based on S&P 500 date added."""
    sp500_df = pd.read_csv(SP500_CSV)
    for ticker in tickers_missing_news:
        row = sp500_df[sp500_df['Symbol'] == ticker]
        if not row.empty:
            date_added = row.iloc[0].get('Date added', None)
            if date_added:
                try:
                    year_added = int(str(date_added)[:4])
                    if year_added >= year_start:
                        print(f"[INFO] Ticker {ticker} was added to S&P500 in {year_added} (>= {year_start})", flush=True)
                except Exception as e:
                    print(f"[WARN] Could not parse 'Date added' for {ticker}: {date_added}", flush=True)
        else:
            print(f"[WARN] Ticker {ticker} not found in SP500 list.", flush=True)


def separate_articles_from_news(input_csv, output_no_articles_csv, output_articles_csv, chunk_size=50_000):
    """Separate articles into a separate CSV file, keeping other columns together."""
    # Remove output files if they exist
    for out_path in [output_no_articles_csv, output_articles_csv]:
        if os.path.exists(out_path):
            os.remove(out_path)

    index_offset = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        # Add index column
        chunk = chunk.reset_index(drop=True)
        chunk['index'] = chunk.index + index_offset
        index_offset += len(chunk)
        # Save all columns except 'Article'
        cols_no_article = [c for c in chunk.columns if c != 'Article']
        chunk[cols_no_article].to_csv(output_no_articles_csv, index=False, mode='a', header=not os.path.exists(output_no_articles_csv))
        # Save only 'index' and 'Article'
        chunk[['index', 'Article']].to_csv(output_articles_csv, index=False, mode='a', header=not os.path.exists(output_articles_csv))
    print(f"[INFO] Saved {output_no_articles_csv} (all columns except Article)", flush=True)
    print(f"[INFO] Saved {output_articles_csv} (index and Article only)", flush=True)


def deduplicate_articles(
    input_no_articles_csv,
    output_dedup_no_articles_csv,
    input_articles_csv,
    output_dedup_articles_csv,
    output_final_no_articles_csv,
    chunk_size=50_000
):
    """Deduplicate articles based on Article_title and update indices."""
    # Sort df by date
    df = pd.read_csv(input_no_articles_csv)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df.to_csv(input_no_articles_csv, index=False)
    print(f"[INFO] Sorted data by Date and saved to {input_no_articles_csv}", flush=True)

    # Deduplicate df based on Article_title
    dedup_df = df.groupby('Article_title').agg({
        'Date': 'first',
        'index': lambda x: list(x),
        'Stock_symbol': lambda x: list(x)
    }).reset_index()
    dedup_df.to_csv(output_dedup_no_articles_csv, index=False)
    print(f"[INFO] Saved deduplicated articles to {output_dedup_no_articles_csv}", flush=True)

    # Get indices to keep
    indices_to_keep = set()
    for idx_list in dedup_df['index']:
        if isinstance(idx_list, str):
            idxs = ast.literal_eval(idx_list)
        else:
            idxs = idx_list
        if idxs:
            indices_to_keep.add(idxs[0])
    print(f"[INFO] Total unique articles after deduplication: {len(indices_to_keep)}", flush=True)

    # Chunked filtering and writing of articles
    first = True
    for chunk in pd.read_csv(input_articles_csv, chunksize=chunk_size):
        filtered = chunk[chunk['index'].isin(indices_to_keep)]
        filtered.to_csv(output_dedup_articles_csv, mode='a', index=False, header=first)
        first = False
    print(f"[INFO] Saved deduplicated articles (only) to {output_dedup_articles_csv}", flush=True)

    # Update index column to keep only the first index
    final_df = pd.read_csv(output_dedup_no_articles_csv)
    final_df['index'] = final_df['index'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0])
    final_df.to_csv(output_final_no_articles_csv, index=False)
    print(f"[INFO] Updated 'index' column in {output_final_no_articles_csv} to keep only first index", flush=True)

if __name__ == "__main__":
    # Step 1: Find tickers missing news for some years or all years
    tickers_missing_news, tickers_no_news = find_tickers_without_news_for_all_years(
        STOCK_PRICE_DIR,
        YEAR_START,
        YEAR_END
    )
    print(f"\nTotal tickers missing news for some years: {len(tickers_missing_news)}", flush=True)
    print(f"Total tickers with no news at all: {len(tickers_no_news)}\n", flush=True)

    # Step 2: Delete CSV files for tickers with no news at all
    delete_tickers_without_news(tickers_no_news, STOCK_PRICE_DIR)

    # Step 2.5: Validate tickers with partial news
    validate_tickers_with_partial_news(tickers_missing_news, YEAR_START)

    # Step 3: Separate articles from sp500 news and save 2 different CSVs in chunks
    separate_articles_from_news(
        SP500_NEWS_CSV,
        SP500_NEWS_NO_ARTICLES_CSV,
        SP500_NEWS_ARTICLES_CSV
    )

    # Step 4: Deduplicate articles based on Article_title
    deduplicate_articles(
        SP500_NEWS_NO_ARTICLES_NEW_CSV,
        SP500_NEWS_DEDUP_NO_ARTICLES_CSV,
        SP500_NEWS_ARTICLES_CSV,
        SP500_NEWS_DEDUP_ARTICLES_CSV,
        SP500_NEWS_DEDUP_FINAL_NO_ARTICLES_CSV
    )
    
    # Delete intermediate files
    os.remove(SP500_NEWS_NO_ARTICLES_CSV)
    os.remove(SP500_NEWS_ARTICLES_CSV)
    os.remove(SP500_NEWS_NO_ARTICLES_NEW_CSV)
    os.remove(SP500_NEWS_DEDUP_NO_ARTICLES_CSV)
    print(f"[CLEANUP] Removed intermediate files.", flush=True)