"""
Compute FinBERT sentiment for articles and save to metadata.

Features:
- Sentence tokenize articles using NLTK
- Use Hugging Face `ProsusAI/finbert` for sentence-level sentiment
- Map labels: Positive=+1, Negative=-1, Neutral=0; multiply by model confidence
- Per-article sentiment: mean of weighted sentence scores
- Attach `Article_Sentiment` and `Num_Sentences` to `metadata.csv`
- Produce per-ticker, per-day aggregated daily sentiment CSV in `data_stats/daily_sentiment.csv`

Usage examples:

# For GPU (if available)
python3 4_compute_sentiment.py --batch-size 256 --device 0 --chunk-size 10000 --max-sentences 50

# For CPU only
python3 4_compute_sentiment.py --batch-size 64 --device -1 --chunk-size 5000 --max-sentences 30

# Process specific tickers
python3 4_compute_sentiment.py --tickers AAPL,MSFT,GOOGL --batch-size 128 --device 0

# Resume processing (skips already processed)
python3 4_compute_sentiment.py --top 100 --batch-size 256 --device 0

# Force recompute everything
python3 4_compute_sentiment.py --force-refresh --batch-size 256 --device 0

"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Set, Tuple
import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk

# Paths
NEWS_DIR = "Stock_news"
ARTICLES_CSV = os.path.join(NEWS_DIR, "articles.csv")
METADATA_CSV = os.path.join(NEWS_DIR, "metadata.csv")
METADATA_JSON = os.path.join(NEWS_DIR, "url_metadata.json")
NEWS_STATS_JSON = os.path.join("data_stats", "news_stats.json")
DAILY_SENTIMENT_CSV = os.path.join("data_stats", "daily_sentiment.csv")
ARTICLES_SENTIMENT_CSV = os.path.join(NEWS_DIR, "articles_sentiment.csv")

# Model
FINBERT_MODEL = 'ProsusAI/finbert'
LABEL_MAP = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

# Defaults
DEFAULT_BATCH = 64


def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print('[INFO] Downloading NLTK punkt tokenizer...')
        nltk.download('punkt')


def load_top_tickers(n: int) -> List[str]:
    if not os.path.exists(NEWS_STATS_JSON):
        print(f'[WARN] {NEWS_STATS_JSON} missing; --top N not available')
        return []
    with open(NEWS_STATS_JSON, 'r') as f:
        data = json.load(f)
    totals = {t: sum(int(v) for v in years.values()) for t, years in data.items()}
    top = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:n]
    return [t for t, _ in top]


def build_article_sentence_index(metadata: dict, target_tickers: Set[str]) -> Tuple[List[int], List[str]]:
    """Return lists of article indices and corresponding texts to process, filtered by target tickers.
    Only include metadata entries that have an Article_Index and have at least one ticker in target_tickers.
    If target_tickers is empty, include all articles with Article_Index."""
    to_process = []

    if not os.path.exists(ARTICLES_CSV):
        raise FileNotFoundError(f'Articles CSV not found: {ARTICLES_CSV}')

    df_articles = pd.read_csv(ARTICLES_CSV)
    articles_map = {int(r['Index']): r['Article'] for _, r in df_articles.iterrows()}

    for k, v in metadata.items():
        idx = v.get('Article_Index')
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except Exception:
            continue

        # filter by tickers if provided
        tickers = v.get('Ticker_List') or []
        if isinstance(tickers, list):
            normalized = {t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()}
        else:
            normalized = {t.strip().upper() for t in str(tickers).split(',') if t.strip()}

        if target_tickers and normalized.isdisjoint(target_tickers):
            continue

        text = articles_map.get(idx_int)
        if not isinstance(text, str) or not text.strip():
            continue

        to_process.append((idx_int, text))

    return to_process


def _sample_sentences(sentences, max_sentences, method='first'):
    if not max_sentences or len(sentences) <= max_sentences:
        return sentences
    if method == 'first':
        return sentences[:max_sentences]
    if method == 'last':
        return sentences[-max_sentences:]
    if method == 'uniform':
        # pick uniformly spaced sentences
        import math
        step = len(sentences) / max_sentences
        return [sentences[int(i * step)] for i in range(max_sentences)]
    if method == 'headtail':
        h = max_sentences // 2
        t = max_sentences - h
        return sentences[:h] + sentences[-t:]
    return sentences[:max_sentences]


def save_results_incrementally(new_results, force_refresh=False, first_chunk=True):
    """Save results incrementally to avoid memory issues.

    Parameters
    - new_results: dict of idx -> (sent, n)
    - force_refresh: if True, overwrite existing file (but only write header on first chunk)
    - first_chunk: whether this is the first chunk being written in this run
    Returns True if rows were written, False otherwise
    """
    # Load existing if any
    file_exists = os.path.exists(ARTICLES_SENTIMENT_CSV)
    if file_exists and not force_refresh:
        df_old = pd.read_csv(ARTICLES_SENTIMENT_CSV)
        existing_idx = set(int(r) for r in df_old['Index'].tolist())
    else:
        df_old = None
        existing_idx = set()

    rows = []
    for idx, (sent, n) in new_results.items():
        if idx in existing_idx:
            continue
        rows.append({'Index': int(idx), 'Article_Sentiment': float(sent), 'Num_Sentences': int(n)})

    if not rows:
        print('[INFO] No new article sentiment rows to save')
        return False

    df_new = pd.DataFrame(rows)

    # Determine mode and header
    if force_refresh:
        # Overwrite mode: write header only on first chunk
        mode = 'w'
        header = True if first_chunk else False
    else:
        # Append mode if file exists, otherwise write with header
        mode = 'a' if file_exists else 'w'
        header = False if file_exists else True

    df_new.to_csv(ARTICLES_SENTIMENT_CSV, mode=mode, header=header, index=False)

    print(f"[INFO] Appended {len(df_new):,} rows to {ARTICLES_SENTIMENT_CSV}")
    return True


def compute_sentiment_for_articles(to_process: List[Tuple[int, str]], batch_size: int, device: int, force_refresh: bool,
                                chunk_size: int = 5000, max_length: int = 256, max_sentences: int = 0, sample_method: str = 'first'):
    """Optimized streaming processing of articles.

    - Processes articles in chunks to limit memory
    - Samples/caps sentences per article with `max_sentences`
    - Processes sentence batches per article to avoid building a giant sentence list
    - Uses an HF pipeline with truncation/padding and optional GPU/FP16
    """
    if pipeline is None:
        raise RuntimeError('transformers not available; please install transformers and torch')

    # Build classifier with model/tokenizer if available (for FP16)
    use_model_obj = AutoModelForSequenceClassification is not None and AutoTokenizer is not None and torch is not None
    if use_model_obj:
        try:
            torch_device = 'cuda' if (torch.cuda.is_available() and device >= 0) else 'cpu'
            dtype = torch.float16 if (torch.cuda.is_available() and device >= 0) else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, torch_dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
            if torch_device == 'cuda':
                model.to('cuda')

            classifier = pipeline(
                'sentiment-analysis',
                model=model,
                tokenizer=tokenizer,
                device=device,
                return_all_scores=True,
                truncation=True,
                padding=True,
                max_length=max_length,
                batch_size=batch_size
            )
        except Exception as e:
            print(f'[WARN] Failed to load model/tokenizer optimized path: {e}. Falling back to pipeline auto-load')
            classifier = pipeline('sentiment-analysis', model=FINBERT_MODEL, tokenizer=FINBERT_MODEL, device=device, return_all_scores=True)
    else:
        classifier = pipeline('sentiment-analysis', model=FINBERT_MODEL, tokenizer=FINBERT_MODEL, device=device, return_all_scores=True)

    # Load existing sentiment if present unless force_refresh
    existing = {}
    if os.path.exists(ARTICLES_SENTIMENT_CSV) and not force_refresh:
        df_exist = pd.read_csv(ARTICLES_SENTIMENT_CSV, usecols=['Index', 'Article_Sentiment', 'Num_Sentences'])
        existing = {int(r['Index']): (r['Article_Sentiment'], int(r['Num_Sentences'])) for _, r in df_exist.iterrows()}

    # Filter out already processed
    todo = [(idx, text) for idx, text in to_process if idx not in existing]
    if not todo:
        print('[INFO] All articles already processed')
        return existing

    article_scores = defaultdict(list)
    processed_articles = 0
    total_articles = len(todo)

    # Process in chunks of articles to keep memory bounded
    for i in range(0, total_articles, chunk_size):
        chunk = todo[i:i+chunk_size]
        if tqdm is not None:
            iterator = tqdm(chunk, desc=f'Chunk {i//chunk_size + 1}/{(total_articles-1)//chunk_size+1}')
        else:
            iterator = chunk

        for article_idx, text in iterator:
            try:
                sents = nltk.tokenize.sent_tokenize(text)
            except Exception:
                sents = []

            if not sents:
                article_scores[article_idx] = [0.0]
                continue

            # Sample/cap long articles
            sents = _sample_sentences(sents, max_sentences, method=sample_method)

            # Process this article's sentences in batches
            for j in range(0, len(sents), batch_size):
                batch = sents[j:j+batch_size]
                try:
                    outs = classifier(batch)
                except Exception as e:
                    print(f'[WARN] classifier failed for article {article_idx} batch starting {j}: {e}')
                    # fallback: add neutral scores for this batch
                    article_scores[article_idx].extend([0.0] * len(batch))
                    continue

                for out in outs:
                    if not out:
                        article_scores[article_idx].append(0.0)
                        continue
                    best = max(out, key=lambda x: x['score'])
                    label = best['label'].lower()
                    score = float(best['score'])
                    weighted = LABEL_MAP.get(label, 0.0) * score
                    article_scores[article_idx].append(weighted)

            processed_articles += 1

        # After finishing a chunk, compute per-article means and save incrementally
        new_results = {}
        for article_idx, scores in article_scores.items():
            if article_idx in existing:
                continue
            if scores:
                mean_score = float(np.mean(scores)) if np is not None else float(sum(scores)/len(scores))
                new_results[article_idx] = (mean_score, len(scores))
            else:
                new_results[article_idx] = (0.0, 0)

        # Persist results for this chunk
        is_first_chunk = (i == 0)
        save_results_incrementally(new_results, force_refresh, first_chunk=is_first_chunk)
        # Add to existing so next chunks skip them
        existing.update(new_results)
        # clear article_scores for next chunk
        article_scores.clear()

    return existing


def attach_sentiment_to_metadata(results: dict):
    """Attach computed article sentiment to `metadata.csv`

    Updates two columns on `metadata.csv`: `Article_Sentiment` and `Num_Sentences`.
    """
    if not os.path.exists(METADATA_CSV):
        print(f"[WARN] {METADATA_CSV} not found; skipping metadata CSV update")
        return

    df = pd.read_csv(METADATA_CSV)
    if 'Article_Index' not in df.columns:
        print(f"[WARN] {METADATA_CSV} missing Column 'Article_Index'; skipping")
        return

    # Build mapping from article index -> (sentiment, n_sentences)
    mapping_sent = {int(k): v for k, (v, _) in results.items()}
    mapping_n = {int(k): n for k, (_, n) in results.items()}

    def _get_sent(x):
        try:
            return mapping_sent.get(int(x)) if pd.notna(x) else None
        except Exception:
            return None

    def _get_n(x):
        try:
            return mapping_n.get(int(x)) if pd.notna(x) else None
        except Exception:
            return None

    df['Article_Sentiment'] = df['Article_Index'].apply(_get_sent)
    df['Num_Sentences'] = df['Article_Index'].apply(_get_n)

    df.to_csv(METADATA_CSV, index=False)
    updated = df['Article_Sentiment'].notna().sum()
    print(f"[INFO] Updated {updated:,} rows in {METADATA_CSV} with Article_Sentiment and Num_Sentences")


def aggregate_daily_sentiment_to_csv():
    # Load metadata for per-article sentiment
    if not os.path.exists(METADATA_CSV):
        print(f'[WARN] {METADATA_CSV} not found; skipping daily aggregation')
        return
    df = pd.read_csv(METADATA_CSV, usecols=['Ticker_List', 'Date', 'Article_Sentiment'])
    df = df.dropna(subset=['Article_Sentiment'])
    if df.empty:
        print('[INFO] No article sentiments to aggregate')
        return

    rows = []
    for _, r in df.iterrows():
        sents = r['Article_Sentiment']
        try:
            tickers = [t.strip().upper() for t in str(r['Ticker_List']).split(',') if t.strip()]
        except Exception:
            tickers = []
        for t in tickers:
            rows.append({'Ticker': t, 'Date': str(r['Date'])[:10], 'Article_Sentiment': float(sents)})

    if not rows:
        print('[INFO] No rows to write for daily sentiment')
        return

    df2 = pd.DataFrame(rows)
    grouped = df2.groupby(['Ticker', 'Date']).agg(Daily_Sentiment=('Article_Sentiment', 'mean'), n_articles=('Article_Sentiment', 'count')).reset_index()
    os.makedirs(os.path.dirname(DAILY_SENTIMENT_CSV), exist_ok=True)
    grouped.to_csv(DAILY_SENTIMENT_CSV, index=False)
    print(f'[INFO] Wrote daily sentiment to {DAILY_SENTIMENT_CSV} (rows: {len(grouped)})')


def main():
    parser = argparse.ArgumentParser(description='Compute FinBERT sentiment for news articles')
    parser.add_argument('--tickers', type=str, default='', help='Comma-separated list of tickers to process (default: top N or all)')
    parser.add_argument('--top', type=int, default=0, help='If provided, process only the top N tickers from news_stats.json')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH, help='Sentence batch size for the model')
    parser.add_argument('--device', type=int, default=-1, help='Device index for transformers pipeline (-1 CPU, >=0 GPU)')
    parser.add_argument('--force-refresh', action='store_true', help='Recompute sentiments even if existing results exist')
    parser.add_argument('--chunk-size', type=int, default=5000, help='Number of articles to process at once')
    parser.add_argument('--max-length', type=int, default=256, help='Maximum sequence length for tokenizer')
    parser.add_argument('--max-sentences', type=int, default=0, help='Maximum sentences per article (0 = unlimited)')
    parser.add_argument('--sample-method', type=str, default='first', choices=['first', 'last', 'uniform', 'headtail'], help='Method to sample sentences if capping')
    args = parser.parse_args()

    ensure_nltk()

    # Build target tickers set
    target = set()
    if args.top > 0:
        top = load_top_tickers(args.top)
        target.update([t.upper() for t in top])
        print(f'[INFO] Targeting top {args.top} tickers: {top}')
    if args.tickers:
        custom = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
        target.update(custom)
        print(f'[INFO] Added custom tickers: {custom}')

    # Load metadata JSON
    if not os.path.exists(METADATA_JSON):
        print(f'[ERROR] Metadata JSON not found: {METADATA_JSON}')
        return
    with open(METADATA_JSON, 'r') as f:
        metadata = json.load(f)

    to_process = build_article_sentence_index(metadata, target)
    if not to_process:
        print('[INFO] No articles found for processing (check tickers/metadata)')
        return

    results = compute_sentiment_for_articles(
        to_process,
        batch_size=args.batch_size,
        device=args.device,
        force_refresh=args.force_refresh,
        chunk_size=args.chunk_size,
        max_length=args.max_length,
        max_sentences=args.max_sentences,
        sample_method=args.sample_method
    )

    attach_sentiment_to_metadata(results)
    aggregate_daily_sentiment_to_csv()


if __name__ == '__main__':
    main()
