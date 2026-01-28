"""
Article summarization pipeline using Sumy (LexRank).

Key optimizations:
1. Vectorized operations with Pandas
2. Efficient memory usage via chunked reading
3. Parallel summarization for speed and throughput
4. Robust handling of short or problematic articles

Operations:
1. Read articles from Stock_news/articles.csv in chunks (Index, Article)
2. Apply LexRank summarization to every article (no filtering)
3. Sanitize summaries to single-line, CSV-safe strings
4. Save results (Index, summary) to Stock_news/summaries.csv

Usage: python3 4_summarize_articles.py [options]
"""

import os
import argparse
import warnings
from typing import Dict, List, Optional, Set, Tuple, Iterator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Sumy imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lex_rank import LexRankSummarizer

warnings.filterwarnings('ignore')

# Constants
NEWS_DIR = "Stock_news"
ARTICLES_CSV = os.path.join(NEWS_DIR, "articles.csv")
SUMMARIES_CSV = os.path.join(NEWS_DIR, "summaries.csv")

# Default settings
DEFAULT_SENTENCE_COUNT = 5
MIN_ARTICLE_LENGTH = 100
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_MAX_WORKERS = 4  # Adjust based on CPU cores


def sanitize_summary(text: str) -> str:
    """Collapse newlines and normalize whitespace for single-line CSV summaries."""
    if text is None:
        return ""
    s = str(text).replace('\r', ' ').replace('\n', ' ')
    # Collapse consecutive whitespace
    s = ' '.join(s.split())
    return s.strip()


def chunked_article_loader(
    selected_indices: Optional[Set[int]] = None,
    articles_path: str = ARTICLES_CSV,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks (columns: Index, Article).

    Reads the articles CSV in streaming mode (chunks). If ``selected_indices``
    is ``None`` the full chunk is returned; if given as a ``set``, only rows
    with ``Index`` in the set are yielded. An empty set yields nothing.
    """
    # Read CSV in chunks with optimized dtype
    dtypes = {'Index': 'int32', 'Article': 'string'}

    for chunk in pd.read_csv(
        articles_path,
        dtype=dtypes,
        usecols=['Index', 'Article'],
        chunksize=chunk_size,
        encoding='utf-8'
    ):
        if selected_indices is None:
            yield chunk
            continue

        if not selected_indices:
            # empty set -> nothing to yield
            return

        mask = chunk['Index'].isin(selected_indices)
        filtered_chunk = chunk.loc[mask]

        if not filtered_chunk.empty:
            yield filtered_chunk


class SumySummarizer:
    """Optimized wrapper for Sumy LexRank summarizer with caching."""
    
    # Class-level cache for expensive-to-create objects
    _stemmer_cache: Dict[str, Stemmer] = {}
    _stop_words_cache: Dict[str, List[str]] = {}
    
    def __init__(self, language: str = "english", sentence_count: int = DEFAULT_SENTENCE_COUNT):
        self.language = language
        self.sentence_count = sentence_count
        
        # Cache stemmer and stop words
        if language not in self._stemmer_cache:
            self._stemmer_cache[language] = Stemmer(language)
        
        if language not in self._stop_words_cache:
            self._stop_words_cache[language] = get_stop_words(language)
        
        self.stemmer = self._stemmer_cache[language]
        self.stop_words = self._stop_words_cache[language]
        
        # Initialize summarizer once
        self.summarizer = LexRankSummarizer(self.stemmer)
        self.summarizer.stop_words = self.stop_words
    
    def summarize(self, text: str) -> str:
        """Summarize text using LexRank and return combined sentences.

        If the text is shorter than ``MIN_ARTICLE_LENGTH`` the original text
        is returned. On unexpected failures a truncated fallback is returned.
        """
        if not text or len(text.strip()) < MIN_ARTICLE_LENGTH:
            return text or ""
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summary_sentences = self.summarizer(parser.document, self.sentence_count)
            return " ".join(str(s) for s in summary_sentences).strip()
        except Exception:
            # Fallback: return first 500 chars if summarization fails
            return text[:500].strip()


def summarize_batch(
    batch: List[Tuple[int, str]],
    language: str = "english",
    sentence_count: int = DEFAULT_SENTENCE_COUNT
) -> List[Tuple[int, str]]:
    """Summarize a batch of (Index, Article) pairs.

    Returns a list of (Index, sanitized_summary) tuples where summaries are
    collapsed into single-line strings suitable for CSV storage.
    """
    summarizer = SumySummarizer(language, sentence_count)
    results = []
    
    for idx, text in batch:
        summary = summarizer.summarize(text)
        summary = sanitize_summary(summary)
        results.append((idx, summary))
    
    return results


class ArticleSummarizationProcessor:
    """Main processor with parallel processing support."""
    
    def __init__(
        self,
        sentence_count: int = DEFAULT_SENTENCE_COUNT,
        max_workers: int = DEFAULT_MAX_WORKERS,
        batch_size: int = 100
    ):
        self.sentence_count = sentence_count
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    def process_articles(
        self,
        output_path: str = SUMMARIES_CSV
    ) -> pd.DataFrame:
        """Process all articles and write summaries to CSV.

        Reads articles in chunks, summarizes them in parallel, sanitizes
        summaries to single-line strings, and writes the output CSV with
        columns ("Index", "summary").
        """
        print(f"[INFO] Processing ALL articles from {ARTICLES_CSV}")
        selected_indices = None
        
        # Step 2: Prepare batches for parallel processing
        batches = []
        current_batch = []
        
        for chunk in chunked_article_loader(selected_indices):
            for _, row in chunk.iterrows():
                current_batch.append((row['Index'], str(row['Article'])))
                
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        # Step 3: Parallel summarization
        print(f"[INFO] Processing {len(batches)} batches with {self.max_workers} workers...")
        all_results = []
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepare partial function for consistent arguments
            summarize_func = partial(
                summarize_batch,
                language="english",
                sentence_count=self.sentence_count
            )
            
            # Submit all batches
            future_to_batch = {
                executor.submit(summarize_func, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    if processed_count % 1000 == 0:
                        print(f"[PROGRESS] Processed {processed_count:,} articles")
                        
                except Exception as e:
                    print(f"[ERROR] Batch processing failed: {e}")
        
        # Step 4: Create and save results DataFrame
        if not all_results:
            print("[WARNING] No summaries generated.")
            return pd.DataFrame(columns=['Index', 'summary'])
        
        df_results = pd.DataFrame(all_results, columns=['Index', 'summary'])
        df_results = df_results.sort_values('Index').reset_index(drop=True)

        # Ensure summaries are single-line (collapse newlines and normalize whitespace)
        df_results['summary'] = df_results['summary'].astype('string').apply(sanitize_summary)

        # Filter out any summaries that contain Cyrillic characters — treat as non-English
        pre_filter_count = len(df_results)
        non_cyrillic_mask = ~df_results['summary'].str.contains(r'[\u0400-\u04FF]', regex=True, na=False)
        df_results = df_results.loc[non_cyrillic_mask].reset_index(drop=True)
        dropped_non_english = pre_filter_count - len(df_results)
        if dropped_non_english:
            print(f"[INFO] Dropped {dropped_non_english:,} summaries containing non-English (Cyrillic) characters.")
        
        # Save to CSV efficiently
        print(f"[INFO] Saving {len(df_results):,} summaries to {output_path}")
        df_results.to_csv(output_path, index=False, encoding='utf-8')

        return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Article summarization (LexRank) — processes all articles and writes single-line summaries to CSV'
    )
    
    parser.add_argument(
        '--sentences',
        type=int,
        default=DEFAULT_SENTENCE_COUNT,
        help=f'Number of sentences in summary (default: {DEFAULT_SENTENCE_COUNT})'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_MAX_WORKERS})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for parallel processing (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=SUMMARIES_CSV,
        help=f'Output path for single-line summaries CSV (default: {SUMMARIES_CSV})'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ArticleSummarizationProcessor(
        sentence_count=args.sentences,
        max_workers=args.workers,
        batch_size=args.batch_size
    )

    # Process articles (always process all articles)
    df = processor.process_articles(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print(f"Processed: all articles")
    print(f"Articles summarized: {len(df):,}")
    print(f"Sentences per summary: {args.sentences}")
    print(f"Parallel workers used: {args.workers}")


if __name__ == "__main__":
    main()
