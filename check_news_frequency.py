"""Plot news frequency by valid market days.
"""

import os
from bisect import bisect_right
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Paths
NEWS_DIR = "Stock_news"
METADATA_CSV = os.path.join(NEWS_DIR, "metadata.csv")
METADATA_JSON = os.path.join(NEWS_DIR, "url_metadata.json")
VALID_MARKET_DAYS = os.path.join("data_stats", "valid_market_days.csv")
FIGURES_DIR = "figures"
OUT_PNG = os.path.join(FIGURES_DIR, "news_per_market_day.png")


def load_valid_days():
    if not os.path.exists(VALID_MARKET_DAYS):
        raise FileNotFoundError(f"Valid market days not found: {VALID_MARKET_DAYS}")
    df = pd.read_csv(VALID_MARKET_DAYS, header=0)
    days = pd.to_datetime(df['date'].astype('string').str.slice(0, 10), format='%Y-%m-%d', errors='coerce')
    days = days.dropna().sort_values().dt.date.tolist()
    return days


def load_dates_from_metadata():
    # Prefer CSV (already created by pipeline); fallback to JSON
    if os.path.exists(METADATA_CSV):
        df = pd.read_csv(METADATA_CSV, usecols=['Article_Index', 'Date'])
        # Keep date strings and attempt parsing
        df['parsed'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        return df['parsed'].dropna().dt.date.tolist()

    if os.path.exists(METADATA_JSON):
        import json
        with open(METADATA_JSON, 'r') as f:
            data = json.load(f)
        dates = []
        for v in data.values():
            d = v.get('Date')
            try:
                ts = pd.to_datetime(d, utc=True, errors='coerce')
                if not pd.isna(ts):
                    dates.append(ts.date())
            except Exception:
                continue
        return dates

    raise FileNotFoundError('No metadata source found (metadata.csv or url_metadata.json)')


def assign_to_valid_day(dates, valid_days):
    # valid_days is sorted list of date objects
    assigned = []
    skipped_after_last = 0
    first_valid = valid_days[0]
    last_valid = valid_days[-1]

    for d in dates:
        if d > last_valid:
            skipped_after_last += 1
            continue
        if d < first_valid:
            # assign to first valid day
            assigned.append(first_valid)
            continue

        # Find rightmost index where valid_days[idx] <= d
        idx = bisect_right(valid_days, d) - 1
        if idx < 0:
            assigned.append(first_valid)
        else:
            assigned.append(valid_days[idx])

    return assigned, skipped_after_last


def plot_counts(assigned_dates, valid_days):
    # Count occurrences per valid day (include zeros)
    s = pd.Series(1, index=pd.to_datetime(assigned_dates))
    counts = s.groupby(s.index.date).sum()

    # Ensure all valid days are present with zero if missing
    all_days_index = pd.Index(pd.to_datetime(valid_days).date)
    counts = counts.reindex(all_days_index, fill_value=0)

    # Prepare plotting
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(list(counts.index), counts.values, drawstyle='steps-post')
    plt.xlabel('Market Day')
    plt.ylabel('Number of Articles Assigned')
    plt.title('News Articles per Valid Market Day')
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    return counts


# --- New helpers for per-ticker plots ---

def load_metadata_with_tickers():
    """Load metadata rows with Date and Ticker_List, prefer CSV, fallback to JSON."""
    if os.path.exists(METADATA_CSV):
        df = pd.read_csv(METADATA_CSV, usecols=['Ticker_List', 'Date'])
        # Parse Date
        df['parsed'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        # Normalize tickers into lists
        df['Tickers'] = df['Ticker_List'].fillna('').astype('string').apply(lambda s: [t.strip().upper() for t in s.split(',') if t.strip()])
        df = df.dropna(subset=['parsed'])
        df['date'] = df['parsed'].dt.date
        return df[['Tickers', 'date']]

    if os.path.exists(METADATA_JSON):
        import json
        with open(METADATA_JSON, 'r') as f:
            data = json.load(f)
        rows = []
        for v in data.values():
            d = v.get('Date')
            ts = pd.to_datetime(d, utc=True, errors='coerce')
            if pd.isna(ts):
                continue
            tickers = v.get('Ticker_List') or []
            if isinstance(tickers, list):
                tickers_list = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
            else:
                tickers_list = [t.strip().upper() for t in str(tickers).split(',') if t.strip()]
            rows.append({'Tickers': tickers_list, 'date': ts.date()})
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=['Tickers', 'date'])


def assign_single_to_valid_day(d, valid_days):
    """Assign a single date to the most recent valid market day <= d. Return None if after last valid day."""
    if pd.isna(d):
        return None
    first_valid = valid_days[0]
    last_valid = valid_days[-1]
    if d > last_valid:
        return None
    if d < first_valid:
        return first_valid
    idx = bisect_right(valid_days, d) - 1
    if idx < 0:
        return first_valid
    return valid_days[idx]


def plot_top10_tickers_per_day(valid_days):
    """Aggregate articles per ticker per valid day and plot top 10 tickers."""
    df = load_metadata_with_tickers()
    if df.empty:
        print('[WARN] No metadata available to compute per-ticker counts')
        return

    rows = []
    for _, r in df.iterrows():
        assigned = assign_single_to_valid_day(r['date'], valid_days)
        if assigned is None:
            continue
        for t in r['Tickers']:
            rows.append({'Ticker': t, 'Assigned': assigned})

    if not rows:
        print('[INFO] No ticker assignments found for top-10 plot')
        return

    df2 = pd.DataFrame(rows)
    counts = df2.groupby(['Ticker', 'Assigned']).size().unstack(fill_value=0)

    # Total counts per ticker
    totals = counts.sum(axis=1)
    top10 = totals.nlargest(10).index.tolist()
    if not top10:
        print('[INFO] No tickers to plot')
        return

    # Prepare DataFrame indexed by valid_days
    dates_index = pd.to_datetime(valid_days).date
    per_day = pd.DataFrame(index=dates_index)
    for ticker in top10:
        series = counts.loc[ticker] if ticker in counts.index else pd.Series(dtype=int)
        # series.index are dates; align to dates_index
        ser = pd.Series(0, index=dates_index)
        for d, val in series.items():
            ser[d] = int(val)
        per_day[ticker] = ser

    # Plot top10 with distinct colors
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, 'news_top10_tickers_per_day.png')
    plt.figure(figsize=(16, 7))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(top10))]
    for ticker, color in zip(top10, colors):
        plt.plot(per_day.index, per_day[ticker], label=ticker, color=color, linewidth=1)

    plt.legend(ncol=2, fontsize='small')
    plt.xlabel('Market Day')
    plt.ylabel('Articles assigned')
    plt.title('Top 10 Tickers — Articles per Market Day')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    print(f'[INFO] Saved top-10 tickers plot to {out}')
    print('\nTop 10 tickers by article count:')
    for t in top10:
        print(f'  {t}: {int(totals[t])}')

    # Also print top 50 tickers as a list (by article count)
    top50 = totals.nlargest(50)
    print('\nTop 50 tickers by article count:')
    for t, cnt in top50.items():
        print(f'  {t}: {int(cnt)}')

    # Compact list of top 50 tickers (comma-separated)
    print('\nTop 50 tickers (list):')
    print(', '.join(top50.index.tolist()))


def main():
    print('[INFO] Loading valid market days...')
    valid_days = load_valid_days()
    print(f'[INFO] Loaded {len(valid_days)} valid market days ({valid_days[0]}..{valid_days[-1]})')

    print('[INFO] Loading article dates from metadata...')
    dates = load_dates_from_metadata()
    print(f'[INFO] Loaded {len(dates)} article dates')

    print('[INFO] Assigning article dates to nearest prior valid market day...')
    assigned, skipped = assign_to_valid_day(dates, valid_days)
    print(f'[INFO] Assigned {len(assigned)} articles; skipped {skipped} articles dated after last valid market day')

    print('[INFO] Plotting counts...')
    counts = plot_counts(assigned, valid_days)
    print(f'[INFO] Saved plot to {OUT_PNG}')

    # Print a brief summary of counts for top days
    top = counts.sort_values(ascending=False).head(10)
    print('\nTop 10 market days by assigned article count:')
    for d, c in top.items():
        print(f'  {d}: {int(c)}')

    # New plot: per-ticker for top 10 tickers
    print('\n[INFO] Aggregating and plotting top-10 ticker frequencies...')
    plot_top10_tickers_per_day(valid_days)


if __name__ == '__main__':
    main()
