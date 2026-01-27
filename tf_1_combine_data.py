''' Combine daily sentiment data with full price history per ticker and write per-ticker CSVs. 

Usage:
Use after computing daily sentiment with `5_compute_daily_sentiment.py`.
'''
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import pandas as pd


DAILY_SENTIMENT_CSV = "data_stats/daily_sentiment.csv"
FULL_HISTORY_DIR = "Stock_price/full_history"
PROCESSED_DATA_DIR = "processed_data/csv"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def _standardize_sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types for the sentiment dataframe.

    Expected input columns (case-sensitive in source):
    - Ticker
    - Date
    - Daily_Sentiment
    - n_articles
    """
    # Normalize column names we care about
    col_map: Dict[str, str] = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # Required columns with flexible case
    for src, tgt in (
        ("ticker", "ticker"),
        ("date", "date"),
        ("daily_sentiment", "daily_sentiment"),
        ("n_articles", "n_articles"),
    ):
        if src in cols_lower:
            col_map[cols_lower[src]] = tgt
        else:
            # Specific fallback for the given dataset's casing
            if src == "daily_sentiment" and "daily_sentiment" not in cols_lower and "daily_sentiment" not in df.columns:
                # Check for Daily_Sentiment exact
                if "Daily_Sentiment" in df.columns:
                    col_map["Daily_Sentiment"] = "daily_sentiment"
            if src == "n_articles" and "n_articles" not in cols_lower and "n_articles" not in df.columns:
                if "n_articles" in df.columns:
                    col_map["n_articles"] = "n_articles"

    # Apply renames where discovered
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure existence of key columns
    missing = [c for c in ("ticker", "date", "daily_sentiment") if c not in df.columns]
    if missing:
        raise ValueError(f"Sentiment data missing required columns: {missing}")

    # Parse date and normalize types
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[index]
    df["ticker"] = df["ticker"].astype(str).str.upper()  # type: ignore[index]

    # Optional column default
    if "n_articles" not in df.columns:
        df["n_articles"] = 0

    # Drop duplicates and sort
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _load_price_file_map(full_history_dir: Path) -> Dict[str, Path]:
    """Map UPPERCASE ticker -> csv path from a directory of price files."""
    file_map: Dict[str, Path] = {}
    for p in full_history_dir.glob("*.csv"):
        # Use filename stem, uppercased, as ticker key
        ticker = p.stem.upper()
        file_map[ticker] = p
    return file_map


def _load_price_history(path: Path) -> pd.DataFrame:
    """Load and standardize a single ticker price history csv."""
    df = pd.read_csv(path)

    # Normalize column names
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}

    # Build rename mapping to canonical names
    rename_map: Dict[str, str] = {}
    for canonical in ("date", "open", "high", "low", "close", "adj_close", "volume"):
        if canonical in cols:
            rename_map[cols[canonical]] = canonical
        else:
            # Special case: sometimes adjusted close is 'adj close'
            if canonical == "adj_close" and "adj close" in {c.lower(): c for c in df.columns}:
                orig = {c.lower(): c for c in df.columns}["adj close"]
                rename_map[orig] = "adj_close"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns
    required = ["date", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Price file {path.name} missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])  # type: ignore[index]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def combine_data(
    sentiment_csv: str | Path = DAILY_SENTIMENT_CSV,
    full_history_dir: str | Path = FULL_HISTORY_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    tickers: Optional[Iterable[str]] = None,
    join_how: str = "left",
    fill_missing_sentiment: Optional[float] = 0.0,
    fill_missing_n_articles: Optional[int] = 0,
    output_filename_pattern: str = "{ticker}.csv",
) -> List[Path]:
    """Combine daily sentiment with full price history per ticker and write per-ticker CSVs.

    Parameters
    ----------
    sentiment_csv : str | Path
        Path to daily sentiment CSV with columns: Ticker, Date, Daily_Sentiment, n_articles.
    full_history_dir : str | Path
        Directory containing per-ticker price history CSVs (one file per ticker).
    output_dir : str | Path
        Directory to write merged per-ticker CSVs.
    tickers : Optional[Iterable[str]]
        Specific tickers to process. If None, process all tickers found in full_history_dir.
    join_how : str
        Merge strategy when combining price (left) and sentiment (right). Default 'left'.
    fill_missing_sentiment : Optional[float]
        If provided, fill missing sentiment values with this number (e.g., 0.0).
    fill_missing_n_articles : Optional[int]
        If provided, fill missing article counts with this integer (e.g., 0).
    output_filename_pattern : str
        Filename pattern for outputs; must include '{ticker}'. Default '{ticker}.csv'.

    Returns
    -------
    List[Path]
        The list of written file paths.
    """
    sentiment_csv = Path(sentiment_csv)
    full_history_dir = Path(full_history_dir)
    output_dir = Path(output_dir)

    if "{ticker}" not in output_filename_pattern:
        raise ValueError("output_filename_pattern must contain '{ticker}' placeholder")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and standardize sentiment
    sent_df_raw = pd.read_csv(sentiment_csv)
    sent_df = _standardize_sentiment_df(sent_df_raw)

    # Build price file map and determine tickers to process
    price_map = _load_price_file_map(full_history_dir)

    if tickers is None:
        tickers_list: List[str] = sorted(price_map.keys())
    else:
        tickers_list = sorted({t.upper().strip() for t in tickers})

    written: List[Path] = []
    for ticker in tickers_list:
        price_path = price_map.get(ticker)
        if price_path is None:
            # Try case-insensitive search as a fallback
            cand = next((p for k, p in price_map.items() if k.upper() == ticker.upper()), None)
            if cand is None:
                # Skip if no price file
                # Could log/print, but keep function silent by default
                continue
            price_path = cand

        # Load and merge
        price_df = _load_price_history(price_path)

        # Filter sentiment for ticker
        sent_t = sent_df[sent_df["ticker"] == ticker]

        merged = price_df.merge(
            sent_t[["date", "daily_sentiment", "n_articles"]],
            on="date",
            how=join_how,
        )

        # Add ticker column for clarity
        merged.insert(0, "ticker", ticker)

        # Fill missing sentiment/article counts if requested
        if fill_missing_sentiment is not None and "daily_sentiment" in merged.columns:
            merged["daily_sentiment"] = merged["daily_sentiment"].fillna(fill_missing_sentiment)
        if fill_missing_n_articles is not None and "n_articles" in merged.columns:
            merged["n_articles"] = merged["n_articles"].fillna(fill_missing_n_articles).astype(int)

        # Sort, reset index, and write
        merged = merged.sort_values("date").reset_index(drop=True)

        out_path = output_dir / output_filename_pattern.format(ticker=ticker)
        merged.to_csv(out_path, index=False)
        written.append(out_path)

    return written


__all__ = ["combine_data"]


if __name__ == "__main__":
    # Example usage
    combine_data()