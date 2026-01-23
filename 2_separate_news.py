"""
Optimized news preparation for stock models.

Key improvements:
1. Pre-filter with year extraction for faster processing
2. Batch operations for I/O efficiency  
3. Memory-efficient data structures
4. Parallel processing for CPU-bound tasks
"""

import os
import json
import hashlib
import argparse
import re
import pandas as pd
from typing import Set, List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
START_YEAR = 2018
END_YEAR = 2023
CHUNK_SIZE = 100_000  # Increased for better I/O throughput
ARTICLE_BUFFER_SIZE = 5000  # Increased buffer size for article writes
METADATA_WRITE_FREQUENCY = 5  # Write metadata every N chunks

# Directories
NEWS_DIR = "Stock_news"
PRICE_DIR = "Stock_price/full_history"

# File paths
NEWS_CSV = os.path.join(NEWS_DIR, "nasdaq_exteral_data.csv")
FILTERED_NEWS_CSV = os.path.join(NEWS_DIR, "filtered_nasdaq_exteral_data.csv")
METADATA_CSV = os.path.join(NEWS_DIR, "metadata.csv")
ARTICLES_CSV = os.path.join(NEWS_DIR, "articles.csv")
METADATA_JSON = os.path.join(NEWS_DIR, "url_metadata.json")
NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")


class NewsProcessor:
    """Main processor for news data with optimized operations."""
    
    def __init__(self, force_refresh: bool = False, chunk_size: int = CHUNK_SIZE):
        self.force_refresh = force_refresh
        self.chunk_size = chunk_size
        self.valid_tickers = self._load_valid_tickers()
        self.columns_to_use = self._get_required_columns()
        
    def _load_valid_tickers(self) -> Set[str]:
        """Load valid tickers from price directory."""
        tickers = set()
        for fname in os.listdir(PRICE_DIR):
            if fname.endswith('.csv'):
                tickers.add(fname[:-4].lower())
        return tickers
    
    def _get_required_columns(self) -> List[str]:
        """Get required columns from news CSV."""
        drop_cols = {
            'Unnamed: 0', 'Author', 'Publisher',
            'Lsa_summary', 'Textrank_summary', 
            'Lexrank_summary', 'Lunh_summary'
        }
        
        with open(NEWS_CSV, 'r') as f:
            header = f.readline().strip()
        all_cols = header.split(',')
        return [col for col in all_cols if col not in drop_cols]
    
    def _fast_year_extractor(self, date_series: pd.Series) -> pd.Series:
        """Extract year from date strings efficiently."""
        # Fast path: first 4 chars as year
        years = date_series.astype('string').str.slice(0, 4)
        years_int = pd.to_numeric(years, errors='coerce', downcast='integer')
        
        # Parse only problematic dates
        mask_na = years_int.isna()
        if mask_na.any():
            parsed = pd.to_datetime(
                date_series[mask_na], 
                errors='coerce',
                infer_datetime_format=True
            )
            years_int[mask_na] = parsed.dt.year
        
        return years_int.astype('Int64')
    
    def filter_news(self) -> Optional[int]:
        """Filter news by date range with optimized operations."""
        if os.path.exists(FILTERED_NEWS_CSV) and not self.force_refresh:
            size_mb = os.path.getsize(FILTERED_NEWS_CSV) / (1024 ** 2)
            print(f"[INFO] Using existing filtered file ({size_mb:.1f} MB)")
            return None
        
        print("[INFO] Filtering news by date range...")
        
        # Use optimized dtypes
        dtype_map = {
            'Date': 'string',
            'Url': 'string',
            'Stock_symbol': 'string',
            'Article': 'string',
            'Article_title': 'string'
        }
        
        filtered_count = 0
        first_chunk = True
        
        reader = pd.read_csv(
            NEWS_CSV,
            usecols=self.columns_to_use,
            chunksize=self.chunk_size,
            dtype=dtype_map,
            engine='c',
            low_memory=False,
            memory_map=True  # Enable memory mapping for large files
        )
        
        for chunk_idx, chunk in enumerate(reader, 1):
            # Fast year extraction
            years = self._fast_year_extractor(chunk['Date'])
            mask = years.between(START_YEAR, END_YEAR, inclusive='both')
            filtered_chunk = chunk[mask]
            
            if not filtered_chunk.empty:
                filtered_count += len(filtered_chunk)
                
                # Write incrementally
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                filtered_chunk.to_csv(
                    FILTERED_NEWS_CSV,
                    mode=mode,
                    header=header,
                    index=False,
                    compression=None  # Disable compression for faster writes
                )
                first_chunk = False
            
            # Progress reporting
            if chunk_idx % 10 == 0:
                print(f"[PROGRESS] Processed {chunk_idx} chunks, kept {filtered_count} rows")
        
        print(f"[INFO] Filtering complete. Total rows: {filtered_count}")
        return filtered_count
    
    def process_chunks(self) -> Tuple[Dict, int]:
        """Process filtered news in chunks with batched operations."""
        print("[INFO] Processing news metadata and articles...")
        
        url_metadata = {}
        article_buffer = []
        article_index = 0
        total_rows = 0
        articles_written = False
        
        # Estimate total rows for progress
        total_estimated = sum(1 for _ in open(FILTERED_NEWS_CSV)) - 1
        
        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                FILTERED_NEWS_CSV,
                chunksize=self.chunk_size,
                dtype={'Url': 'string', 'Stock_symbol': 'string', 'Article': 'string'}
            ),
            start=1
        ):
            chunk_processed = 0
            
            for _, row in chunk.iterrows():
                url = row.get('Url')
                if pd.isna(url):
                    continue
                
                url_hash = hashlib.md5(str(url).encode()).hexdigest()
                total_rows += 1
                
                # Initialize or update metadata
                if url_hash not in url_metadata:
                    url_metadata[url_hash] = {
                        'Url': url,
                        'Ticker_Set': set(),
                        'Indices': [],
                        'Article_Index': None,
                        'Article_Title': row.get('Article_title'),
                        'Date': row.get('Date')
                    }
                
                # Add stock symbol if present
                stock = row.get('Stock_symbol')
                if pd.notna(stock):
                    ticker = str(stock).strip().upper()
                    if ticker:
                        url_metadata[url_hash]['Ticker_Set'].add(ticker)
                
                # Add row index
                url_metadata[url_hash]['Indices'].append(total_rows)
                
                # Buffer article if not already processed
                article = row.get('Article')
                if (pd.notna(article) and 
                    url_metadata[url_hash]['Article_Index'] is None):
                    
                    url_metadata[url_hash]['Article_Index'] = article_index
                    article_buffer.append({
                        'Index': article_index,
                        'Url': url,
                        'Article': article
                    })
                    article_index += 1
                    
                    # Flush buffer when full
                    if len(article_buffer) >= ARTICLE_BUFFER_SIZE:
                        self._flush_article_buffer(article_buffer, articles_written)
                        articles_written = True
                        article_buffer = []
                
                chunk_processed += 1
            
            # Write metadata periodically
            if chunk_idx % METADATA_WRITE_FREQUENCY == 0:
                self._write_metadata(url_metadata)
                print(f"[PROGRESS] Processed {total_rows:,} rows ({(total_rows/total_estimated)*100:.1f}%)")
        
        # Final flushes
        if article_buffer:
            self._flush_article_buffer(article_buffer, articles_written)
        
        self._write_metadata(url_metadata)
        self._create_metadata_csv(url_metadata)
        
        print(f"[INFO] Processing complete. Articles: {article_index}, URLs: {len(url_metadata)}")
        return url_metadata, article_index
    
    def _flush_article_buffer(self, buffer: List[Dict], articles_written: bool):
        """Flush article buffer to disk."""
        df = pd.DataFrame(buffer)
        mode = 'a' if articles_written else 'w'
        header = not articles_written
        df.to_csv(ARTICLES_CSV, mode=mode, header=header, index=False)
    
    def _write_metadata(self, metadata: Dict):
        """Write metadata to JSON efficiently."""
        serializable = {}
        for k, v in metadata.items():
            # Get tickers from either format (Ticker_Set or Ticker_List)
            tickers = v.get('Ticker_Set')
            if tickers is None:
                tickers = v.get('Ticker_List', [])
            if isinstance(tickers, set):
                tickers = sorted(tickers)
            elif isinstance(tickers, list):
                tickers = sorted(tickers)
            else:
                tickers = []
            
            serializable[k] = {
                'Url': v.get('Url'),
                'Ticker_List': tickers,
                'Indices': v.get('Indices', []),
                'Article_Index': v.get('Article_Index'),
                'Article_Title': v.get('Article_Title'),
                'Date': v.get('Date')
            }
        
        # Atomic write
        tmp_file = METADATA_JSON + '.tmp'
        with open(tmp_file, 'w') as f:
            json.dump(serializable, f, separators=(',', ':'))  # Compact JSON
        os.replace(tmp_file, METADATA_JSON)
    
    def _create_metadata_csv(self, metadata: Dict):
        """Create metadata CSV from processed metadata."""
        print("[INFO] Creating metadata CSV...")
        
        rows = []
        for data in metadata.values():
            if data['Article_Index'] is not None:
                # Handle both formats: Ticker_Set (in-memory) and Ticker_List (from JSON)
                tickers = data.get('Ticker_List') or sorted(data.get('Ticker_Set', set()))
                if isinstance(tickers, set):
                    tickers = sorted(tickers)
                
                rows.append({
                    'Ticker_List': ','.join(tickers) if tickers else '',
                    'Date': data['Date'],
                    'Url': data['Url'],
                    'Title': data['Article_Title'],
                    'Article_Index': data['Article_Index']
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.sort_values('Article_Index', inplace=True)
            df.to_csv(METADATA_CSV, index=False)
            print(f"[INFO] Metadata CSV created with {len(df):,} entries")
    
    def prune_by_tickers(self, metadata: Dict) -> Tuple[int, int]:
        """Prune articles based on valid tickers."""
        print("[INFO] Pruning articles by valid tickers...")
        
        kept_keys = []
        removed_keys = []
        kept_indices = set()
        
        for key, data in metadata.items():
            # Normalize and check tickers
            tickers = data.get('Ticker_List') or list(data.get('Ticker_Set', set()))
            normalized = {t.strip().lower() for t in tickers if isinstance(t, str)}
            
            has_valid = any(t in self.valid_tickers for t in normalized)
            
            if has_valid and data.get('Article_Index') is not None:
                kept_keys.append(key)
                kept_indices.add(int(data['Article_Index']))
                # Update with uppercase tickers
                normalized_tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str)})
                data['Ticker_List'] = normalized_tickers
                # Remove Ticker_Set to avoid stale data (Ticker_List is the canonical source now)
                data.pop('Ticker_Set', None)
            else:
                removed_keys.append(key)
        
        # Filter articles CSV
        self._filter_articles_csv(kept_indices)
        
        # Update metadata
        for key in removed_keys:
            metadata.pop(key, None)
        
        # Write final metadata
        self._write_metadata(metadata)
        self._create_metadata_csv(metadata)
        
        print(f"[INFO] Pruning complete. Kept: {len(kept_keys)}, Removed: {len(removed_keys)}")
        return len(kept_keys), len(removed_keys)
    
    def _filter_articles_csv(self, keep_indices: Set[int]):
        """Filter articles CSV to keep only specified indices."""
        if not os.path.exists(ARTICLES_CSV):
            return
        
        if not keep_indices:
            # Create empty file
            pd.DataFrame(columns=['Index', 'Url', 'Article']).to_csv(ARTICLES_CSV, index=False)
            return
        
        tmp_file = ARTICLES_CSV + '.tmp'
        keep_indices_set = set(keep_indices)
        
        with open(tmp_file, 'w') as out_f:
            write_header = True
            
            for chunk in pd.read_csv(ARTICLES_CSV, chunksize=100_000):
                mask = chunk['Index'].isin(keep_indices_set)
                filtered = chunk[mask]
                
                if not filtered.empty:
                    filtered.to_csv(out_f, mode='a', header=write_header, index=False)
                    write_header = False
        
        os.replace(tmp_file, ARTICLES_CSV)
    
    def enrich_tickers_from_titles(self, metadata: Dict) -> Tuple[int, int]:
        """Extract additional tickers from article titles and add to metadata.
        
        This compensates for poor quality Stock_symbol data by extracting
        ticker symbols directly from article titles.
        
        Returns:
            Tuple of (urls_enriched, total_tickers_added)
        """
        print("[INFO] Enriching tickers from article titles...")
        
        # Pattern to match potential ticker symbols (2-5 uppercase letters)
        # Common words to exclude (false positives)
        exclude_words = {
            'THE', 'AND', 'FOR', 'ARE', 'NOT', 'BUT', 'CAN', 'ALL', 
            'OUT', 'NEW', 'GET', 'HAS', 'HAD', 'ITS', 'ONE', 'TWO',
            'MAY', 'NOW', 'SEE', 'OWN', 'SAY', 'SHE', 'TOO', 'USE',
            'HER', 'HIS', 'HOW', 'OUR', 'WHY', 'BIG', 'TOP', 'HOT',
            'WHO', 'WHY', 'YES', 'YOU', 'WAS', 'WAY', 'WIN', 'WON',
            'ETF', 'CEO', 'CFO', 'IPO', 'NYSE', 'SEC', 'FDA', 'DOJ',
            'USA', 'API', 'APP', 'GDP', 'CPI', 'FED'
        }
        
        ticker_pattern = re.compile(r'\b[A-Z]{2,5}\b')
        
        total_urls = 0
        total_added = 0
        
        for key, data in metadata.items():
            title = data.get('Article_Title')
            if not title or pd.isna(title):
                continue
            
            # Find potential tickers in title
            potential_tickers = ticker_pattern.findall(str(title))
            
            # Get existing tickers (handle both dict formats)
            existing = data.get('Ticker_Set')
            if existing is None:
                # Loading from JSON - Ticker_List exists
                existing = set(data.get('Ticker_List', []))
            elif not isinstance(existing, set):
                existing = set(existing)
            else:
                existing = existing.copy()
            
            added_count = 0
            for ticker in potential_tickers:
                ticker_upper = ticker.upper()
                ticker_lower = ticker.lower()
                
                # Skip if excluded word, already present, or not a valid ticker
                if ticker_upper in exclude_words:
                    continue
                if ticker_upper in existing:
                    continue
                if ticker_lower not in self.valid_tickers:
                    continue
                
                existing.add(ticker_upper)
                added_count += 1
            
            if added_count > 0:
                # Update the metadata with sorted list (canonical format)
                data['Ticker_List'] = sorted(existing)
                # Remove Ticker_Set to avoid conflicts (Ticker_List is canonical)
                data.pop('Ticker_Set', None)
                total_urls += 1
                total_added += added_count
        
        print(f"[INFO] Enrichment complete. Updated {total_urls:,} URLs with {total_added:,} additional tickers")
        
        # Write updated metadata
        self._write_metadata(metadata)
        if any(d.get('Article_Index') is not None for d in metadata.values()):
            self._create_metadata_csv(metadata)
        
        return total_urls, total_added
    
    def compute_statistics(self) -> Dict:
        """Compute article statistics per ticker per year.

        Prefer `url_metadata.json` (uses `Ticker_List` per URL) to include
        all ticker mentions (even for URLs without full `Article`).
        Fall back to `metadata.csv` if JSON is missing.

        This function now filters the output to include only tickers that
        have at least one news item in *every* year in the range
        START_YEAR..END_YEAR (inclusive). Only those tickers are written
        to `data_stats/news_stats.json`.
        """
        print("[INFO] Computing news statistics...")

        result = {}

        if os.path.exists(METADATA_JSON):
            # Load JSON metadata which stores per-URL ticker lists
            with open(METADATA_JSON, 'r') as f:
                metadata = json.load(f)

            for v in metadata.values():
                tickers = v.get('Ticker_List') or []
                date = v.get('Date')
                if not date or not tickers:
                    continue
                year = str(date)[:4]
                if not year.isdigit():
                    continue
                year = int(year)
                for t in tickers:
                    if not isinstance(t, str) or not t.strip():
                        continue
                    ticker = t.strip().upper()
                    result.setdefault(ticker, {})
                    result[ticker].setdefault(year, 0)
                    result[ticker][year] += 1

        elif os.path.exists(METADATA_CSV):
            # Backward-compatible CSV-based computation
            df = pd.read_csv(METADATA_CSV, usecols=['Ticker_List', 'Date'])
            df['Ticker'] = df['Ticker_List'].str.split(',')
            df = df.explode('Ticker')
            df['Ticker'] = df['Ticker'].str.strip().str.upper()
            df['Year'] = pd.to_numeric(df['Date'].astype('string').str.slice(0, 4), errors='coerce')
            df = df.dropna(subset=['Ticker', 'Year'])
            df['Year'] = df['Year'].astype(int)
            stats = df.groupby(['Ticker', 'Year']).size().reset_index(name='Count')
            for ticker in stats['Ticker'].unique():
                ticker_data = stats[stats['Ticker'] == ticker]
                result[ticker] = dict(zip(ticker_data['Year'], ticker_data['Count']))
        else:
            print("[WARN] No metadata found (neither JSON nor CSV)")
            return {}

        # Filter to tickers that have news in ALL years of the configured range
        # and also meet a minimum per-year threshold
        required_years = list(range(START_YEAR, END_YEAR + 1))
        MIN_PER_YEAR = 30
        filtered = {}
        for ticker, years in result.items():
            # Normalize year keys to ints to be robust against string keys from JSON
            year_counts = {int(k): int(v) for k, v in years.items()}
            if all(year_counts.get(y, 0) >= MIN_PER_YEAR for y in required_years):
                filtered[ticker] = years

        result = filtered

        # Save to file
        os.makedirs(os.path.dirname(NEWS_STATS_JSON), exist_ok=True)
        with open(NEWS_STATS_JSON, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[INFO] Wrote statistics for {len(result):,} tickers (present in all years {START_YEAR}-{END_YEAR} with >= {MIN_PER_YEAR} articles/year)")

        # Show top 10 (from filtered tickers)
        totals = {t: sum(int(v) for v in years.values()) for t, years in result.items()}
        top10 = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:10]

        print("\nTop 10 tickers by article count (among tickers meeting the per-year threshold):")
        for ticker, count in top10:
            print(f"  {ticker}: {count:,}")

        return result


def main():
    
    parser = argparse.ArgumentParser(description='Optimized news processor')
    parser.add_argument('--force-refresh', '-f', action='store_true', help='Rebuild filtered CSV')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Chunk size for processing')
    parser.add_argument('--only-filter', action='store_true', help='Run only filtering phase')
    parser.add_argument('--skip-phase2', action='store_true', help='Skip metadata separation')
    parser.add_argument('--skip-prune', action='store_true', help='Skip ticker pruning')
    parser.add_argument('--skip-enrich', action='store_true', help='Skip ticker enrichment from titles')
    parser.add_argument('--only-stats', action='store_true', help='Compute statistics only from existing metadata and exit')
    args = parser.parse_args()
    
    # Initialize processor
    processor = NewsProcessor(
        force_refresh=args.force_refresh,
        chunk_size=args.chunk_size
    )

    # If requested, compute statistics only (uses existing metadata.json or metadata.csv)
    if args.only_stats:
        print("\n" + "="*50)
        print("PHASE 4: Statistics (only)")
        print("="*50)
        processor.compute_statistics()
        return
    
    # Phase 1: Filter
    print("\n" + "="*50)
    print("PHASE 1: Filtering")
    print("="*50)
    filtered_rows = processor.filter_news()
    
    if args.only_filter:
        print("[INFO] Only filter requested. Exiting.")
        return
    
    # Check if filtered file exists and has data
    if not os.path.exists(FILTERED_NEWS_CSV) or os.path.getsize(FILTERED_NEWS_CSV) < 100:
        print("[ERROR] Filtered news file is empty or missing")
        return
    
    # Phase 2: Metadata separation
    metadata = {}
    if not args.skip_phase2:
        print("\n" + "="*50)
        print("PHASE 2: Metadata Separation")
        print("="*50)
        metadata, article_count = processor.process_chunks()
    else:
        # Load existing metadata
        if os.path.exists(METADATA_JSON):
            with open(METADATA_JSON, 'r') as f:
                metadata = json.load(f)
    
    # Phase 2.5: Enrich tickers from titles (BEFORE pruning)
    if not args.skip_enrich and metadata:
        print("\n" + "="*50)
        print("PHASE 2.5: Enrich Tickers from Titles")
        print("="*50)
        processor.enrich_tickers_from_titles(metadata)
    
    # Phase 3: Pruning (AFTER enrichment)
    if not args.skip_prune and metadata:
        print("\n" + "="*50)
        print("PHASE 3: Pruning by Tickers")
        print("="*50)
        processor.prune_by_tickers(metadata)
    
    # Phase 4: Statistics
    print("\n" + "="*50)
    print("PHASE 4: Statistics")
    print("="*50)
    processor.compute_statistics()
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()