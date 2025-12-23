import os
import pandas as pd
import json

PRICE_DIR = "Stock_price/full_history"
NEWS_CSV = "Stock_news/nasdaq_exteral_data.csv"
OUTPUT_DIR = "data_stats"
STOCK_NEWS_DIR = "Stock_news"
START_YEAR = 2018
END_YEAR = 2023


def filter_csv_years(price_dir, start_year, end_year):
    """Filter each CSV to only have data within the given years and delete empty CSVs."""
    csv_files = [f for f in os.listdir(price_dir) if f.endswith('.csv')]
    valid_csvs = []

    for csv_file in csv_files:
        path = os.path.join(price_dir, csv_file)
        chunks = []
        for chunk in pd.read_csv(path, chunksize=100_000, parse_dates=['date'], low_memory=False):
            chunk = chunk[(chunk['date'].dt.year >= start_year) & (chunk['date'].dt.year <= end_year)]
            if not chunk.empty:
                chunks.append(chunk)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            df.to_csv(path, index=False)
            valid_csvs.append(csv_file)
        else:
            os.remove(path)  # Step 2: Delete empty CSVs
            print(f"[INFO] Deleted empty CSV: {csv_file}")
    return valid_csvs


def filter_complete_tickers(price_dir, csv_files, start_year, end_year):
    """Keep only CSVs with samples in all years; delete others."""
    complete_csvs = []

    for csv_file in csv_files:
        path = os.path.join(price_dir, csv_file)
        years_present = set()
        for chunk in pd.read_csv(path, chunksize=100_000, parse_dates=['date'], low_memory=False):
            years_present.update(chunk['date'].dt.year.unique())
        if all(year in years_present for year in range(start_year, end_year + 1)):
            complete_csvs.append(csv_file)
        else:
            os.remove(path)
            print(f"[INFO] Deleted CSV without full-year coverage: {csv_file}")
    return complete_csvs


def generate_yearly_stats(price_dir, csv_files, start_year, end_year):
    """Create a single JSON file with stats per year."""
    year_stats = {year: {'rows': 0, 'date_range': (None, None)} for year in range(start_year, end_year + 1)}

    for csv_file in csv_files:
        ticker = os.path.splitext(csv_file)[0]
        path = os.path.join(price_dir, csv_file)
        for chunk in pd.read_csv(path, chunksize=100_000, parse_dates=['date'], low_memory=False):
            chunk = chunk[(chunk['date'].dt.year >= start_year) & (chunk['date'].dt.year <= end_year)]
            for year in range(start_year, end_year + 1):
                year_mask = chunk['date'].dt.year == year
                if year_mask.any():
                    filtered = chunk[year_mask]
                    ys = year_stats[year]
                    ys['rows'] += len(filtered)
                    min_date = filtered['date'].min()
                    max_date = filtered['date'].max()
                    ys['date_range'] = (
                        min(min_date, ys['date_range'][0]) if ys['date_range'][0] else min_date,
                        max(max_date, ys['date_range'][1]) if ys['date_range'][1] else max_date
                    )
    stats_path = os.path.join(OUTPUT_DIR, "price_stats.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(year_stats, f, indent=2, default=str)
    print(f"[INFO] Yearly stats saved to {stats_path}")
    return year_stats


def filter_news_csv(news_csv, output_dir, start_year, end_year, tickers):
    """
    Filter news CSV by date, tickers, and drop unnecessary columns.
    Writes a single CSV in output_dir as chunks to save memory.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "filtered_news.csv")
    
    first_chunk = True  # Write header only once

    for chunk in pd.read_csv(news_csv, chunksize=50_000, low_memory=False):
        print(f"[INFO] Processing a chunk of news data...")

        # Parse date safely
        chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')

        # Filter by year
        if chunk['Date'].notna().any():
            chunk = chunk[(chunk['Date'].dt.year >= start_year) & (chunk['Date'].dt.year <= end_year)]
        else:
            continue

        # Filter by tickers
        if 'Stock_symbol' in chunk.columns and tickers:
            chunk = chunk[chunk['Stock_symbol'].isin(tickers)]

        # Drop unnecessary columns
        drop_cols = ['Unnamed: 0', 'Url', 'Author', 'Publisher']
        chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])

        # Write chunk immediately
        if not chunk.empty:
            chunk.to_csv(out_path, index=False, mode='a', header=first_chunk)
            first_chunk = False

    print(f"[INFO] Filtered news CSV saved to {out_path}")


if __name__ == "__main__":

    valid_csvs = filter_csv_years(PRICE_DIR, START_YEAR, END_YEAR)
    print(f"[INFO] Filtered CSVs with data in range {START_YEAR}-{END_YEAR}: {len(valid_csvs)} files remain.")

    complete_csvs = filter_complete_tickers(PRICE_DIR, valid_csvs, START_YEAR, END_YEAR)
    csv_files = [f for f in os.listdir(PRICE_DIR) if f.endswith('.csv')]
    remaining_tickers = [os.path.splitext(f)[0] for f in csv_files]
    print(f"[INFO] CSVs with complete year coverage: {len(complete_csvs)} files remain.")
    print(f"[INFO] Found {len(remaining_tickers)} remaining tickers in {PRICE_DIR}.")
    print(f"Sanity Check - complete_csvs vs remaining_tickers match: {set(os.path.splitext(f)[0] for f in complete_csvs) == set(remaining_tickers)}")

    generate_yearly_stats(PRICE_DIR, complete_csvs, START_YEAR, END_YEAR)
    print(f"[INFO] Generated yearly statistics.")

    filter_news_csv(NEWS_CSV, STOCK_NEWS_DIR, START_YEAR, END_YEAR, remaining_tickers)
    print(f"[INFO] Filtered news CSV based on remaining tickers and date range.")