"""
S&P 500 stock price data processor.
Filters to retain only tickers with complete data from 2018-2023.
Removes NaN values, ensures full year coverage, and generates statistics.
Outputs cleaned CSVs and price_stats.json for downstream analysis.
"""

import os
import shutil
import json
import argparse
import pandas as pd
from typing import Set, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants
YEAR_START = 2018
YEAR_END = 2023
PRICE_COLS = ['volume', 'open', 'high', 'low', 'close', 'adj close']
PRICE_COLS_ROUND = ['open', 'high', 'low', 'close', 'adj close']
EPSILON = 1e-6  # Threshold for near-zero values

# Directories and files
STOCKS_CSV = "data_stats/sp500.csv"
PRICE_STATS = "data_stats/price_stats.json"
STOCKS_PRICE_DIR = "Stock_price/full_history"
INVALID_STOCKS_PRICE_DIR = "Stock_price/invalid"


class StockPriceProcessor:
    """Main processor for stock price data with optimized operations."""
    
    def __init__(self):
        self.valid_tickers = self._load_sp500_tickers()
        self.processed_count = 0
        self.invalid_count = 0
        
    def _load_sp500_tickers(self) -> Set[str]:
        """Load S&P 500 tickers from CSV file."""
        if not os.path.exists(STOCKS_CSV):
            raise FileNotFoundError(f"S&P 500 ticker file not found: {STOCKS_CSV}")
        
        df = pd.read_csv(STOCKS_CSV)
        return set(df['Symbol'].str.strip().str.lower())
    
    def _fast_date_parser(self, date_series: pd.Series) -> pd.Series:
        """Fast date parsing optimized for YYYY-MM-DD format."""
        # Try fast conversion for standard format first
        try:
            return pd.to_datetime(date_series, format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError):
            # Fall back to slower parsing for non-standard formats
            return pd.to_datetime(date_series, errors='coerce')
    
    def filter_and_clean_stocks(self):
        """Filter stocks with complete price data from YEAR_START to YEAR_END."""
        print("[INFO] Filtering stocks by date range and completeness...")
        
        os.makedirs(INVALID_STOCKS_PRICE_DIR, exist_ok=True)
        
        # Get list of files to process
        files = [f for f in os.listdir(STOCKS_PRICE_DIR) if f.endswith('.csv')]
        print(f"[INFO] Found {len(files)} stock price files to process")
        
        valid_files = []
        self.processed_count = 0
        self.invalid_count = 0
        
        for i, fname in enumerate(sorted(files), 1):
            ticker = fname[:-4]
            ticker_lower = ticker.lower()
            file_path = os.path.join(STOCKS_PRICE_DIR, fname)
            
            # Check if ticker is in S&P 500
            if ticker_lower not in self.valid_tickers:
                self._move_to_invalid(fname, file_path, "ticker not in S&P 500")
                continue
            
            try:
                # Read with optimized settings
                df = pd.read_csv(
                    file_path, 
                    usecols=['date'] + PRICE_COLS,
                    parse_dates=['date'],
                    dtype={col: 'float32' for col in PRICE_COLS if col != 'volume'},
                    engine='c'
                )
                
                # Early exit if empty
                if df.empty:
                    self._move_to_invalid(fname, file_path, "empty file")
                    continue
                
                # Filter by date range
                start_date = f"{YEAR_START}-01-01"
                end_date = f"{YEAR_END}-12-31"
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df = df.loc[mask].copy()
                
                if df.empty:
                    self._move_to_invalid(fname, file_path, "no data in date range")
                    continue
                
                # Clean data
                initial_rows = len(df)
                df_clean = self._clean_price_data(df, fname)
                
                if df_clean is None or df_clean.empty:
                    self._move_to_invalid(fname, file_path, "insufficient data after cleaning")
                    continue
                
                # Check year coverage
                if not self._has_complete_year_coverage(df_clean):
                    self._move_to_invalid(fname, file_path, "incomplete year coverage")
                    continue
                
                # Round and format
                df_final = self._format_price_data(df_clean)
                
                # Save cleaned file
                df_final.to_csv(file_path, index=False)
                valid_files.append(fname)
                self.processed_count += 1
                
                # Progress update
                if i % 50 == 0:
                    print(f"[PROGRESS] Processed {i}/{len(files)} files, " f"valid: {self.processed_count}, invalid: {self.invalid_count}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {fname}: {str(e)[:100]}")
                self._move_to_invalid(fname, file_path, f"processing error: {str(e)[:50]}")
        
        print(f"[INFO] Filtering complete. Valid: {self.processed_count}, " f"Invalid: {self.invalid_count}")
        return valid_files
    
    def _clean_price_data(self, df: pd.DataFrame, fname: str) -> Optional[pd.DataFrame]:
        """Clean price data by removing NaNs and invalid values."""
        # Check for required columns
        required_cols = ['date', 'volume'] + [c for c in PRICE_COLS if c != 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"[WARN] Missing columns in {fname}: {missing_cols}")
            return None
        
        # Drop rows with NaNs in key columns
        before_drop = len(df)
        df_clean = df.dropna(subset=['volume'] + [c for c in PRICE_COLS if c != 'volume'])
        dropped = before_drop - len(df_clean)
        
        if dropped > 0:
            print(f"[CLEAN] {fname}: Dropped {dropped} rows with NaNs")
        
        # Remove negative or zero volumes and prices
        mask = df_clean['volume'] > 0
        for col in [c for c in PRICE_COLS if c != 'volume']:
            mask &= df_clean[col] > 0
        
        df_clean = df_clean[mask].copy()
        
        if len(df_clean) < 10:  # Arbitrary minimum rows threshold
            print(f"[WARN] {fname}: Insufficient valid rows after cleaning ({len(df_clean)})")
            return None
        
        return df_clean
    
    def _has_complete_year_coverage(self, df: pd.DataFrame) -> bool:
        """Check if data covers all years from YEAR_START to YEAR_END."""
        years_present = set(df['date'].dt.year.unique())
        required_years = set(range(YEAR_START, YEAR_END + 1))
        return required_years.issubset(years_present)
    
    def _format_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format price data with rounding and proper types."""
        # Round price columns
        for col in PRICE_COLS_ROUND:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        # Convert volume to integer
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype('int64')
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _move_to_invalid(self, fname: str, source_path: str, reason: str):
        """Move file to invalid directory."""
        try:
            os.makedirs(INVALID_STOCKS_PRICE_DIR, exist_ok=True)
            dest_path = os.path.join(INVALID_STOCKS_PRICE_DIR, fname)
            shutil.move(source_path, dest_path)
            self.invalid_count += 1
            print(f"[INVALID] Moved {fname}: {reason}")
        except Exception as e:
            print(f"[ERROR] Failed to move {fname} to invalid: {str(e)[:100]}")
    
    def calculate_statistics(self) -> Dict:
        """Calculate statistics for each valid stock CSV."""
        print("[INFO] Calculating stock price statistics...")
        
        stats = {}
        files = [f for f in os.listdir(STOCKS_PRICE_DIR) if f.endswith('.csv')]
        
        if not files:
            print("[WARN] No stock files found in full_history directory")
            return stats
        
        for i, fname in enumerate(sorted(files), 1):
            ticker = fname[:-4]
            file_path = os.path.join(STOCKS_PRICE_DIR, fname)
            
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                df['year'] = df['date'].dt.year
                
                ticker_stats = {}
                
                for year in range(YEAR_START, YEAR_END + 1):
                    year_df = df[df['year'] == year]
                    
                    if year_df.empty:
                        continue
                    
                    # Basic statistics
                    n_rows = int(year_df['close'].notna().sum())
                    median_close = float(year_df['close'].median())
                    median_volume = float(year_df['volume'].median())
                    min_close = float(year_df['close'].min())
                    max_close = float(year_df['close'].max())
                    volatility = float(year_df['close'].std())
                    
                    # Calculate annual return
                    annual_return = self._calculate_annual_return(year_df)
                    
                    # Calculate correlation
                    corr_open_close = self._calculate_correlation(year_df)
                    
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
                
                if ticker_stats:  # Only add if we have data for at least one year
                    stats[ticker] = ticker_stats
                
                # Progress update
                if i % 50 == 0:
                    print(f"[PROGRESS] Calculated stats for {i}/{len(files)} files")
                    
            except Exception as e:
                print(f"[ERROR] Failed to calculate stats for {fname}: {str(e)[:100]}")
        
        self._save_statistics(stats)
        self._display_statistics_summary(stats)
        
        return stats
    
    def _calculate_annual_return(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate annual return from first to last valid close."""
        df_sorted = df.sort_values('date')
        valid_closes = df_sorted['close'].dropna()
        
        if len(valid_closes) < 2:
            return None
        
        first_close = valid_closes.iloc[0]
        last_close = valid_closes.iloc[-1]
        
        if abs(first_close) < EPSILON:
            return None
        
        return (last_close - first_close) / first_close
    
    def _calculate_correlation(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate correlation between open and close prices."""
        if df['open'].std() < EPSILON or df['close'].std() < EPSILON:
            return None
        
        corr = df['open'].corr(df['close'])
        return float(corr) if not pd.isna(corr) else None
    
    def _save_statistics(self, stats: Dict):
        """Save statistics to JSON file."""
        os.makedirs(os.path.dirname(PRICE_STATS), exist_ok=True)
        
        with open(PRICE_STATS, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[INFO] Statistics saved to {PRICE_STATS}")
    
    def _display_statistics_summary(self, stats: Dict):
        """Display summary of statistics."""
        if not stats:
            print("[WARN] No statistics calculated")
            return
        
        print("\n" + "="*60)
        print("STOCK PRICE STATISTICS SUMMARY")
        print("="*60)
        
        # Count tickers with complete data
        tickers_complete = []
        for ticker, years_data in stats.items():
            if len(years_data) == (YEAR_END - YEAR_START + 1):
                tickers_complete.append(ticker)
        
        print(f"Total tickers processed: {len(stats)}")
        print(f"Tickers with complete {YEAR_START}-{YEAR_END} data: {len(tickers_complete)}")
        
        # Show top 5 by average volume
        avg_volumes = {}
        for ticker, years_data in stats.items():
            volumes = [data['median_volume'] for data in years_data.values()]
            avg_volumes[ticker] = sum(volumes) / len(volumes)
        
        top_volume = sorted(avg_volumes.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 tickers by average volume:")
        for ticker, volume in top_volume:
            print(f"  {ticker}: {volume:,.0f}")
        
        # Show top 5 by average return
        avg_returns = {}
        for ticker, years_data in stats.items():
            returns = [data['return'] for data in years_data.values() if data['return'] is not None]
            if returns:
                avg_returns[ticker] = sum(returns) / len(returns)
        
        if avg_returns:
            top_return = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 tickers by average annual return:")
            for ticker, ret in top_return:
                print(f"  {ticker}: {ret:.2%}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Process S&P 500 stock price data')
    parser.add_argument('--skip-stats', action='store_true', help='Skip statistics calculation')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("S&P 500 STOCK PRICE PROCESSOR")
    print("="*60)
    
    try:
        # Initialize processor
        processor = StockPriceProcessor()
        
        # Phase 1: Filter and clean stocks
        print("\nPHASE 1: Filtering and Cleaning")
        print("-" * 40)
        valid_files = processor.filter_and_clean_stocks()
        
        if not valid_files:
            print("[ERROR] No valid stock files found after filtering")
            return
        
        # Phase 2: Calculate statistics (unless skipped)
        if not args.skip_stats:
            print("\nPHASE 2: Calculating Statistics")
            print("-" * 40)
            stats = processor.calculate_statistics()
        else:
            print("\n[INFO] Skipping statistics calculation as requested")
            stats = {}
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Final results:")
        print(f"  - Valid tickers: {processor.processed_count}")
        print(f"  - Invalid tickers: {processor.invalid_count}")
        if not args.skip_stats:
            print(f"  - Statistics file: {PRICE_STATS}")
            print(f"  - Total tickers in stats: {len(stats)}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()