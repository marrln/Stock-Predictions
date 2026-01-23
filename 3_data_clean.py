"""
Clean and sync news metadata and articles.

Operations performed:
1. Remove the prefix "https://www.nasdaq.com/articles/" from URL fields to save space.
2. For each metadata entry, remove tickers that are not present in `data_stats/news_stats.json`.
    - If a metadata entry's ticker list becomes empty, the metadata entry is deleted and the
    corresponding article (by `Article_Index`) is removed from `Stock_news/articles.csv`.
    - Track counts of deleted metadata entries and removed articles.
3. Parse the `Date` column (assumed UTC) and add a `Posted_After_Close` boolean column
indicating whether the article was posted at or after US market close (4:00 PM US/Eastern).
    - If the article's date is not a valid market day (based on `data_stats/valid_market_days.csv`),
    it is considered "before close" automatically (weekends/holidays).
    - If the article date is after the last valid market day in the file, the entry is removed.

Usage: run the script from the repo root: python3 3_data_clean.py
"""

import os
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import Set

import pandas as pd

# Paths
NEWS_DIR = "Stock_news"
METADATA_CSV = os.path.join(NEWS_DIR, "metadata.csv")
ARTICLES_CSV = os.path.join(NEWS_DIR, "articles.csv")
METADATA_JSON = os.path.join(NEWS_DIR, "url_metadata.json")
NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
VALID_MARKET_DAYS = os.path.join("data_stats", "valid_market_days.csv")
STOCKS_PRICE_DIR = os.path.join("Stock_price", "full_history")

# Constants
NASDAQ_PREFIX = "https://www.nasdaq.com/articles/"
US_EAST = ZoneInfo("America/New_York")
MARKET_CLOSE_TIME = time(16, 0)  # 16:00 (4:00 PM) US/Eastern


def load_news_stats() -> Set[str]:
    if not os.path.exists(NEWS_STATS_JSON):
        print(f"[WARN] News stats not found: {NEWS_STATS_JSON} - ticker filtering will be skipped")
        return set()
    with open(NEWS_STATS_JSON, 'r') as f:
        data = json.load(f)
    return {k.strip().upper() for k in data.keys()}


def load_valid_days() -> Set[str]:
    if not os.path.exists(VALID_MARKET_DAYS):
        print(f"[WARN] Valid market days file not found: {VALID_MARKET_DAYS} - weekends/holidays logic disabled")
        return set()
    df = pd.read_csv(VALID_MARKET_DAYS, header=0)
    # Normalize to ISO date strings
    return {str(d) for d in df['date'].astype('string').str.slice(0, 10)}


def parse_date_to_utc(dt_str: str):
    if not isinstance(dt_str, str) or not dt_str.strip():
        return None
    try:
        # Let pandas parse with timezone awareness; assume UTC if tz missing
        ts = pd.to_datetime(dt_str, utc=True, errors='coerce')
        return ts
    except Exception:
        return None


def remove_price_files_without_news(news_tickers: Set[str]) -> int:
    """Delete stock CSVs from `Stock_price/full_history` if ticker not present in news tickers.
    Returns number of files removed."""
    if not news_tickers:
        print("[INFO] No news tickers available; skipping removal of stock price files.")
        return 0
    if not os.path.exists(STOCKS_PRICE_DIR):
        print(f"[WARN] Stocks price directory not found: {STOCKS_PRICE_DIR}")
        return 0

    removed = 0
    for fname in os.listdir(STOCKS_PRICE_DIR):
        if not fname.lower().endswith('.csv'):
            continue
        ticker = os.path.splitext(fname)[0].upper()
        if ticker not in news_tickers:
            path = os.path.join(STOCKS_PRICE_DIR, fname)
            try:
                os.remove(path)
                removed += 1
                print(f"[REMOVE] Deleted price file: {fname}")
            except Exception as e:
                print(f"[ERROR] Failed to delete {fname}: {e}")

    print(f"[INFO] Removed {removed} stock price files not present in news stats")
    return removed


def main():
    print("[INFO] Starting data clean: metadata & articles")

    news_tickers = load_news_stats()
    valid_days = load_valid_days()
    last_valid_day = None
    if valid_days:
        parsed_days = sorted(valid_days)
        last_valid_day = datetime.fromisoformat(parsed_days[-1]).date()

    # Load metadata JSON
    if not os.path.exists(METADATA_JSON):
        print(f"[ERROR] Metadata JSON not found: {METADATA_JSON}")
        return

    with open(METADATA_JSON, 'r') as f:
        metadata = json.load(f)

    removed_metadata_keys = []
    removed_articles_indices = set()
    removed_due_future_date = []
    updated_count = 0

    for key, v in list(metadata.items()):
        # 1) Trim URL prefix
        url = v.get('Url', '') or ''
        if url.startswith(NASDAQ_PREFIX):
            v['Url'] = url[len(NASDAQ_PREFIX):]

        # 2) Filter tickers against news_stats (if available)
        tickers = v.get('Ticker_List') or sorted(v.get('Ticker_Set', []))
        # normalize
        tickers_norm = [t.strip().upper() for t in (tickers or []) if isinstance(t, str) and t.strip()]
        if news_tickers:
            tickers_filtered = [t for t in tickers_norm if t in news_tickers]
        else:
            # If news_stats not present, keep existing tickers
            tickers_filtered = tickers_norm

        # 3) Date parsing and posted-before/after logic
        date_str = v.get('Date')
        ts_utc = parse_date_to_utc(date_str)

        # If date missing/unparsable, keep but mark as before-close
        posted_after = False
        remove_for_future = False

        if ts_utc is not None:
            dt_date = ts_utc.date()

            # Remove if after last valid day
            if last_valid_day is not None and dt_date > last_valid_day:
                remove_for_future = True

            iso_date = dt_date.isoformat()
            if not remove_for_future:
                if valid_days and iso_date not in valid_days:
                    # Weekend/holiday => consider before close
                    posted_after = False
                else:
                    # Market day: convert to US/Eastern and compare time
                    try:
                        eastern = ts_utc.tz_convert(US_EAST)
                        posted_after = eastern.time() >= MARKET_CLOSE_TIME
                    except Exception:
                        posted_after = False
        else:
            # Unable to parse date -> treat as before close
            posted_after = False

        # If date beyond last valid day -> remove
        if remove_for_future:
            removed_metadata_keys.append(key)
            # also mark article for removal
            idx = v.get('Article_Index')
            if idx is not None:
                removed_articles_indices.add(int(idx))
            removed_due_future_date.append(key)
            continue

        # Update tickers and Posted_After_Close flag
        if tickers_filtered:
            # Ensure canonical Ticker_List and remove any in-memory sets
            v['Ticker_List'] = sorted({t.strip().upper() for t in tickers_filtered})
            v.pop('Ticker_Set', None)
            v['Posted_After_Close'] = bool(posted_after)
            updated_count += 1
        else:
            # No valid tickers -> remove metadata and article
            removed_metadata_keys.append(key)
            idx = v.get('Article_Index')
            if idx is not None:
                removed_articles_indices.add(int(idx))

    # Remove keys from metadata
    for k in removed_metadata_keys:
        metadata.pop(k, None)

    # Save updated metadata JSON atomically
    tmp = METADATA_JSON + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(metadata, f, separators=(',', ':'), indent=None)
    os.replace(tmp, METADATA_JSON)

    print(f"[INFO] Metadata updated. Entries updated: {updated_count:,}. Entries removed: {len(removed_metadata_keys):,} (future-date: {len(removed_due_future_date):,}).")

    # Update metadata.csv
    # Recreate metadata CSV from remaining metadata entries
    rows = []
    for v in metadata.values():
        if v.get('Article_Index') is None:
            continue
        tickers = v.get('Ticker_List') or []
        rows.append({
            'Ticker_List': ','.join(tickers) if tickers else '',
            'Date': v.get('Date'),
            'Url': v.get('Url'),
            'Title': v.get('Article_Title'),
            'Article_Index': v.get('Article_Index'),
            'Posted_After_Close': v.get('Posted_After_Close', False)
        })

    if rows:
        df_meta = pd.DataFrame(rows)
        df_meta.sort_values('Article_Index', inplace=True)
        df_meta.to_csv(METADATA_CSV, index=False)
        print(f"[INFO] Wrote {len(df_meta):,} rows to {METADATA_CSV}")
    else:
        # Create empty metadata file
        pd.DataFrame(columns=['Ticker_List', 'Date', 'Url', 'Title', 'Article_Index', 'Posted_After_Close']).to_csv(METADATA_CSV, index=False)
        print(f"[INFO] No metadata rows remaining; created empty {METADATA_CSV}")

    # Update articles.csv: remove articles with indices in removed_articles_indices and also update Urls trimmed
    if os.path.exists(ARTICLES_CSV):
        df_articles = pd.read_csv(ARTICLES_CSV)
        initial_articles = len(df_articles)

        # Trim URLs in articles
        if 'Url' in df_articles.columns:
            df_articles['Url'] = df_articles['Url'].fillna('').apply(lambda u: u[len(NASDAQ_PREFIX):] if isinstance(u, str) and u.startswith(NASDAQ_PREFIX) else u)

        if removed_articles_indices:
            df_articles = df_articles[~df_articles['Index'].isin(removed_articles_indices)].copy()
            removed_articles = initial_articles - len(df_articles)
        else:
            removed_articles = 0

        df_articles.to_csv(ARTICLES_CSV, index=False)
        print(f"[INFO] Articles updated. Removed articles: {removed_articles:,}. Remaining: {len(df_articles):,}.")
    else:
        print(f"[WARN] Articles file missing: {ARTICLES_CSV}")

    # Remove stock price files not present in news stats
    removed_price_files = remove_price_files_without_news(news_tickers)
    print(f"[INFO] Removed {removed_price_files} stock price files not present in {NEWS_STATS_JSON}")

    # Compute percentage of articles posted before/after (based on metadata CSV)
    if os.path.exists(METADATA_CSV):
        dfm = pd.read_csv(METADATA_CSV, usecols=['Article_Index', 'Posted_After_Close'])
        total = len(dfm)
        if total > 0:
            after_count = int(dfm['Posted_After_Close'].sum())
            before_count = total - after_count
            print(f"[INFO] Posted before market close: {before_count}/{total} ({before_count/total:.2%})")
            print(f"[INFO] Posted at/after market close: {after_count}/{total} ({after_count/total:.2%})")
        else:
            print("[INFO] No metadata rows to compute before/after statistics.")

    print("[INFO] Data clean complete.")


if __name__ == '__main__':
    main()
