"""Generate plots for news distribution per year and for tickers with news in all years 2018-2023.

Produces:
 - figures/news_per_year.png (bar chart total news per year)
 - figures/tickers_all_years_hist.png (histogram of total news per ticker for tickers with news in all years)
 - figures/tickers_all_years_top20.png (bar chart of top-20 tickers by total news among those tickers)

Usage: python plot_news_distribution.py
"""
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

DATA_PATH = Path("data_stats/news_stats.json")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

PRICE_DIR = Path("Stock_price/full_history")

YEARS = [str(y) for y in range(2018, 2024)]  # 2018-2023


def load_df(path: Path) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)
    # Convert to DataFrame with tickers as index and years as columns
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(columns=YEARS).fillna(0).astype(int)
    return df


def plot_news_per_year(df: pd.DataFrame, out_path: Path):
    totals = df[YEARS].sum()
    plt.figure(figsize=(9, 5))
    # assign `hue` to use a palette across bars and suppress the legend (future-proof)
    ax = sns.barplot(x=totals.index, y=totals.values, hue=totals.index, palette="Blues_d", dodge=False, legend=False)
    ax.set_title("Total number of news items per year (2018-2023)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total news count")
    for p in ax.patches:
        h = int(p.get_height())
        ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_tickers_all_years(df: pd.DataFrame):
    mask_all_years = (df[YEARS] > 0).all(axis=1)
    tickers = df[mask_all_years].copy()
    tickers['total'] = tickers[YEARS].sum(axis=1)
    return tickers.sort_values('total', ascending=False)


def plot_tickers_hist(tickers_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 5))
    sns.histplot(tickers_df['total'], bins=30, kde=False)
    plt.title(f"Total news per ticker (tickers with news in all years 2018-2023)\nN={len(tickers_df)}")
    plt.xlabel("Total news count (2018-2023)")
    plt.ylabel("Number of tickers")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_top_tickers(tickers_df: pd.DataFrame, out_path: Path, top_n: int = 50):
    top = tickers_df.head(top_n)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=top.index, y=top['total'].values, hue=top.index, palette="viridis", dodge=False, legend=False)
    plt.title(f"Top {top_n} tickers by total news (2018-2023)\n(only tickers with news in all years)")
    plt.xlabel("Ticker")
    plt.ylabel("Total news count (2018-2023)")
    plt.xticks(rotation=90)
    # annotate bars with counts
    for p in ax.patches:
        h = int(p.get_height())
        ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    df = load_df(DATA_PATH)

    # Restrict to tickers that have price files in PRICE_DIR
    valid_tickers = set()
    for fname in os.listdir(PRICE_DIR):
        if fname.endswith('.csv'):
            valid_tickers.add(fname[:-4].lower())
    df = df[df.index.to_series().str.lower().isin(valid_tickers)]
    print(f"Tickers with price files: {len(valid_tickers)}; tickers considered after filtering: {df.shape[0]}")

    # 1) news distribution per year
    plot_news_per_year(df, OUT_DIR / "news_per_year.png")

    # 2) tickers with news in all years
    tickers_all = analyze_tickers_all_years(df)
    tickers_all.to_csv(OUT_DIR / "tickers_all_years.csv")
    plot_tickers_hist(tickers_all, OUT_DIR / "tickers_all_years_hist.png")
    if len(tickers_all) > 0:
        plot_top_tickers(tickers_all, OUT_DIR / "tickers_all_years_top50.png", top_n=50)

    # Print summary
    print("Saved figures in:", OUT_DIR)
    print("Tickers with news in all years (2018-2023):", len(tickers_all))
