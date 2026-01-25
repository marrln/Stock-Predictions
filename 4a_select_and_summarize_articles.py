"""
Article Selection and Summarization Pipeline using Sumy.

This script performs extractive summarization on news articles BEFORE sentiment analysis.

Operations:
1. Load articles from Stock_news/articles.csv
2. Select the top10 tickers according to how many articles they have (and save them in a new csv, and a json)
3. Apply multiple summarization algorithms (LSA, LexRank, Luhn, TextRank) to the articles of this selection
4. Save summarized articles for downstream sentiment analysis

Usage: python3 4a_select_and_summarize_articles.py [options]

Dependencies:
    pip install sumy nltk --break-system-packages
"""

import os
import json
import argparse
import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


# Paths
NEWS_DIR = "Stock_news"
DATA_STATS_DIR = "data_stats"
OUTPUT_DIR = "Stock_news" 

ARTICLES_CSV = os.path.join(NEWS_DIR, "articles.csv")
METADATA_JSON = os.path.join(NEWS_DIR, "url_metadata.json")
NEWS_STATS_JSON = os.path.join(DATA_STATS_DIR, "news_stats.json")

# Output
SELECTED_TICKERS_JSON = os.path.join(OUTPUT_DIR, "selected_tickers.json")
SELECTED_ARTICLES_CSV = os.path.join(OUTPUT_DIR, "selected_articles.csv")
SUMMARIZED_ARTICLES_CSV = os.path.join(OUTPUT_DIR, "articles_summarized.csv")

# Summarization settings
DEFAULT_SENTENCE_COUNT = 5  # Number of sentences in summary
MIN_ARTICLE_LENGTH = 100    # Minimum characters to summarize



# ============================================================================
# PART A: TICKER FILTERING
# ============================================================================
class TickerFilter:
    """Filter tickers based on article frequency."""
    
    def __init__(self, news_stats_path: str = NEWS_STATS_JSON):
        self.news_stats = self._load_news_stats(news_stats_path)
        
    def _load_news_stats(self, path: str) -> Dict:
        """Load news statistics from JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"News stats not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_top_n_tickers(self, n: int = 10) -> List[str]:
        """Get top N tickers by total article count."""
        totals = {}
        for ticker, years in self.news_stats.items():
            totals[ticker] = sum(int(v) for v in years.values())
        
        sorted_tickers = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        top_n = [t[0] for t in sorted_tickers[:n]]
        
        print(f"\n[INFO] Top {n} tickers by article count:")
        for ticker, count in sorted_tickers[:n]:
            print(f"  {ticker}: {count:,} articles")
        
        return top_n
    
    def get_tickers_by_threshold(self, min_per_year: int = 50, start_year: int = 2018, end_year: int = 2023) -> List[str]:
        """Get tickers with at least min_per_year articles each year."""
        required_years = list(range(start_year, end_year + 1))
        qualified = []
        
        for ticker, years in self.news_stats.items():
            year_counts = {int(k): int(v) for k, v in years.items()}
            if all(year_counts.get(y, 0) >= min_per_year for y in required_years):
                qualified.append(ticker)
        
        print(f"\n[INFO] {len(qualified)} tickers with >= {min_per_year} articles/year")
        return qualified
    
    def get_tickers_balanced(self, n: int = 10, min_per_year: int = 30) -> List[str]:
        """Get top N tickers that also meet minimum per-year threshold."""
        
        # First filter by threshold
        qualified = set(self.get_tickers_by_threshold(min_per_year, start_year=2018, end_year=2023))
        
        # Then rank by total and take top N from qualified
        totals = {}
        for ticker in qualified:
            years = self.news_stats[ticker]
            totals[ticker] = sum(int(v) for v in years.values())
        
        sorted_tickers = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        top_n = [t[0] for t in sorted_tickers[:n]]
        
        print(f"\n[INFO] Selected {len(top_n)} tickers (top {n} from qualified):")
        for ticker, count in sorted_tickers[:n]:
            yearly = self.news_stats[ticker]
            print(f"  {ticker}: {count:,} total | per-year: {dict(yearly)}")
        
        return top_n
    
    def save_selected_tickers(self, tickers: List[str], output_path: str = SELECTED_TICKERS_JSON):
        """Save selected tickers to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Include statistics for each ticker
        data = {
            'selected_tickers': tickers,
            'selection_date': datetime.now().isoformat(),
            'ticker_stats': {}
        }
        
        for ticker in tickers:
            if ticker in self.news_stats:
                years = self.news_stats[ticker]
                data['ticker_stats'][ticker] = {
                    'yearly_counts': years,
                    'total': sum(int(v) for v in years.values())
                }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Saved {len(tickers)} selected tickers to {output_path}")
        
        
    def load_articles(self, selected_tickers: List[str] = None, articles_path: str = None, metadata_path: str = METADATA_JSON) -> pd.DataFrame:
        """Load and filter articles for selected tickers."""
        
        # Prefer summarized articles if available
        if articles_path is None:            
            articles_path = ARTICLES_CSV
            print(f"[INFO] Using original articles: {ARTICLES_CSV}")
        
        
        print("[INFO] Loading articles and metadata...")
        
        # Load metadata to get ticker associations
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create mapping: Article_Index -> (tickers, date, title)
        index_to_meta = {}
        for url_hash, data in metadata.items():
            idx = data.get('Article_Index')
            if idx is None:
                continue
            
            tickers = data.get('Ticker_List', [])
            # Filter to selected tickers
            relevant_tickers = [t for t in tickers if t.upper() in selected_tickers]
            
            if relevant_tickers:
                index_to_meta[int(idx)] = {
                    'tickers': relevant_tickers,
                    'date': data.get('Date'),
                    'title': data.get('Article_Title'),
                    'url': data.get('Url')
                }
        
        print(f"[INFO] Found {len(index_to_meta):,} articles for selected tickers")
        
        # Load articles
        df = pd.read_csv(articles_path)
        
        # Filter to relevant articles
        df = df[df['Index'].isin(index_to_meta.keys())].copy()
        
        # Add metadata
        df['Tickers'] = df['Index'].map(lambda x: index_to_meta.get(x, {}).get('tickers', []))
        df['Date'] = df['Index'].map(lambda x: index_to_meta.get(x, {}).get('date'))
        df['Title'] = df['Index'].map(lambda x: index_to_meta.get(x, {}).get('title'))
        
        print(f"[INFO] Loaded {len(df):,} articles")
    
        return df


# ============================================================================
# PART B: SUMY SUMMARIZER
# ============================================================================

class SumySummarizer:
    """
    Extractive summarization using Sumy library.
    
    Algorithms available:
    - LSA (Latent Semantic Analysis): Good for general summarization
    - LexRank: Graph-based, good for news articles
    - Luhn: Classic algorithm, focuses on significant words
    - TextRank: Similar to PageRank, good for key sentences
    - SumBasic: Simple frequency-based
    """
    
    def __init__(self, language: str = "english"):
        self.language = language
        self._setup_nltk()
        self._load_summarizers()
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        import nltk
        
        required = ['punkt', 'stopwords', 'punkt_tab']
        for item in required:
            try:
                nltk.data.find(f'tokenizers/{item}' if 'punkt' in item else f'corpora/{item}')
            except LookupError:
                print(f"[INFO] Downloading NLTK {item}...")
                nltk.download(item, quiet=True)
    
    def _load_summarizers(self):
        """Initialize Sumy summarizers."""
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words
            
            # Summarizer algorithms
            from sumy.summarizers.lsa import LsaSummarizer
            from sumy.summarizers.lex_rank import LexRankSummarizer
            from sumy.summarizers.luhn import LuhnSummarizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            from sumy.summarizers.sum_basic import SumBasicSummarizer
            
            self.PlaintextParser = PlaintextParser
            self.Tokenizer = Tokenizer
            
            stemmer = Stemmer(self.language)
            stop_words = get_stop_words(self.language)
            
            # Initialize all summarizers
            self.summarizers = {
                'lsa': LsaSummarizer(stemmer),
                'lexrank': LexRankSummarizer(stemmer),
                'luhn': LuhnSummarizer(stemmer),
                'textrank': TextRankSummarizer(stemmer),
                'sumbasic': SumBasicSummarizer(stemmer)
            }
            
            # Set stop words for each
            for name, summarizer in self.summarizers.items():
                summarizer.stop_words = stop_words
            
            print(f"[INFO] Loaded {len(self.summarizers)} summarization algorithms")
            
        except ImportError as e:
            print(f"[ERROR] Sumy not installed. Run: pip install sumy --break-system-packages")
            raise e
    
    def summarize(self, text: str, algorithm: str = 'lexrank', sentence_count: int = DEFAULT_SENTENCE_COUNT) -> str:
        """
        Summarize text using specified algorithm.
        
        Args:
            text: Input text to summarize
            algorithm: One of 'lsa', 'lexrank', 'luhn', 'textrank', 'sumbasic'
            sentence_count: Number of sentences in summary
            
        Returns:
            Summarized text
        """
        if not text or len(text.strip()) < MIN_ARTICLE_LENGTH:
            return text if text else ""
        
        try:
            # Parse text
            parser = self.PlaintextParser.from_string(text, self.Tokenizer(self.language))
            
            # Get summarizer
            summarizer = self.summarizers.get(algorithm.lower())
            if summarizer is None:
                print(f"[WARN] Unknown algorithm '{algorithm}', using lexrank")
                summarizer = self.summarizers['lexrank']
            
            # Generate summary
            summary_sentences = summarizer(parser.document, sentence_count)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            
            return summary.strip()
            
        except Exception as e:
            # Return original text on error
            return text[:500] if len(text) > 500 else text
    
    def summarize_all_algorithms(self, text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT) -> Dict[str, str]:
        """Apply all summarization algorithms and return results."""
        results = {}
        
        for algo_name in self.summarizers.keys():
            results[algo_name] = self.summarize(text, algo_name, sentence_count)
        
        return results
    
    def get_best_summary(self, text: str, sentence_count: int = DEFAULT_SENTENCE_COUNT, preferred_algo: str = 'lexrank') -> Tuple[str, str]:
        """
        Get best summary, preferring specified algorithm.
        Falls back to others if preferred fails.
        
        Returns:
            (summary_text, algorithm_used)
        """
        # Try preferred first
        summary = self.summarize(text, preferred_algo, sentence_count)
        if summary and len(summary) > 50:
            return summary, preferred_algo
        
        # Try others in order of preference
        fallback_order = ['lexrank', 'textrank', 'lsa', 'luhn', 'sumbasic']
        for algo in fallback_order:
            if algo == preferred_algo:
                continue
            summary = self.summarize(text, algo, sentence_count)
            if summary and len(summary) > 50:
                return summary, algo
        
        # Return truncated original as last resort
        return text[:500] if len(text) > 500 else text, 'truncated'


# ============================================================================
# ARTICLE PROCESSOR
# ============================================================================
class ArticleSummarizationProcessor:
    """Main processor for article summarization."""
    
    def __init__(self, sentence_count: int = DEFAULT_SENTENCE_COUNT, algorithm: str = 'lexrank'):
        self.sentence_count = sentence_count
        self.algorithm = algorithm
        self.output_path = os.path.join(OUTPUT_DIR, f"articles_summarized_{algorithm}_{sentence_count}sent.csv") 
        
        self.summarizer = SumySummarizer()
        
        # Load metadata for ticker info
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load article metadata."""
        if not os.path.exists(METADATA_JSON):
            print(f"[WARN] Metadata not found: {METADATA_JSON}")
            return {}
        
        with open(METADATA_JSON, 'r') as f:
            return json.load(f)
    
    def _get_article_tickers(self, article_index: int) -> List[str]:
        """Get tickers associated with an article."""
        for url_hash, data in self.metadata.items():
            if data.get('Article_Index') == article_index:
                return data.get('Ticker_List', [])
        return []
    
    def process_articles_csv(self, input_path: str = ARTICLES_CSV, batch_size: int = 1000) -> pd.DataFrame:
        """
        Process all articles and add summaries from the csv file.
        
        Adds columns:
        - summary: General summary using chosen algorithm
        - summary_algorithm: Which algorithm was used
        - ticker_extract: Ticker-specific extracted sentences (if enabled)
        """
        print(f"[INFO] Loading articles from {input_path}...")
        df = pd.read_csv(input_path)
        
        print(f"[INFO] Processing {len(df):,} articles...")
        
        # Initialize new columns
        df['summary'] = ''
        df['summary_algorithm'] = ''
    
        
        # Process in batches
        total = len(df)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                article_text = row.get('Article', '')
                article_index = row.get('Index')
                
                # Generate summary
                if pd.notna(article_text) and len(str(article_text)) >= MIN_ARTICLE_LENGTH:
                    summary, algo_used = self.summarizer.get_best_summary(str(article_text), self.sentence_count, self.algorithm)
                    df.at[df.index[idx], 'summary'] = summary
                    df.at[df.index[idx], 'summary_algorithm'] = algo_used
                else:
                    # Keep original short text
                    df.at[df.index[idx], 'summary'] = str(article_text) if pd.notna(article_text) else ''
                    df.at[df.index[idx], 'summary_algorithm'] = 'original'
            
            # Progress update
            print(f"[PROGRESS] Processed {end_idx:,}/{total:,} articles ({100*end_idx/total:.1f}%)")
        
        # Save results
        print(f"[INFO] Saving summarized articles to {self.output_path}...")
        df.to_csv(self.output_path, index=False)
        
        # Print statistics
        self._print_stats(df)
        
        # Remove the 'Article' column from the DataFrame
        if 'Article' in df.columns:
            df2 = df.drop(columns=['Article'])
        output_path_2 = self.output_path.replace('.csv', '_only.csv')
        print(f"[INFO] Saving summarized articles (only summary) to {output_path_2}...")
        df2.to_csv(output_path_2, index=False)
        
        return df
    
    def process_articles_df(self, input_df: pd.DataFrame = None, output_path: str = None, batch_size: int = 1000) -> pd.DataFrame:
        """
        Process all articles and add summaries from the dataframe.
        
        Adds columns:
        - summary: General summary using chosen algorithm
        - summary_algorithm: Which algorithm was used
        - ticker_extract: Ticker-specific extracted sentences (if enabled)
        """
        print(f"[INFO] Loading articles from input df...")
        df = input_df.copy()
        if output_path == None : 
            output_path = os.path.join(OUTPUT_DIR, f"articles_summarized_{args.algorithm}_{args.sentences}sent_sample.csv")

        
        print(f"[INFO] Processing {len(df):,} articles...")
        
        # Initialize new columns
        df['summary'] = ''
        df['summary_algorithm'] = ''
    
        
        # Process in batches
        total = len(df)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                article_text = row.get('Article', '')
                article_index = row.get('Index')
                
                # Generate summary
                if pd.notna(article_text) and len(str(article_text)) >= MIN_ARTICLE_LENGTH:
                    summary, algo_used = self.summarizer.get_best_summary(str(article_text), self.sentence_count, self.algorithm)
                    df.at[df.index[idx], 'summary'] = summary
                    df.at[df.index[idx], 'summary_algorithm'] = algo_used
                else:
                    # Keep original short text
                    df.at[df.index[idx], 'summary'] = str(article_text) if pd.notna(article_text) else ''
                    df.at[df.index[idx], 'summary_algorithm'] = 'original'
            
            # Progress update
            print(f"[PROGRESS] Processed {end_idx:,}/{total:,} articles ({100*end_idx/total:.1f}%)")
        
        # Save results
        print(f"[INFO] Saving summarized articles (with article) to {output_path}...")
        df.to_csv(output_path, index=False)
         
        # Print statistics
        self._print_stats(df)
        
        # Remove the 'Article' column from the DataFrame
        if 'Article' in df.columns:
            df2 = df.drop(columns=['Article'])
        output_path_2 = output_path.replace('.csv', '_only.csv')
        print(f"[INFO] Saving summarized articles (only summary) to {output_path_2}...")
        df2.to_csv(output_path_2, index=False)      
        
        return df
    
    
    def _print_stats(self, df: pd.DataFrame):
        """Print summarization statistics."""
        print("\n" + "=" * 50)
        print("SUMMARIZATION STATISTICS")
        print("=" * 50)
        
        # Algorithm distribution
        algo_counts = df['summary_algorithm'].value_counts()
        print("\nAlgorithm usage:")
        for algo, count in algo_counts.items():
            print(f"  {algo}: {count:,} ({100*count/len(df):.1f}%)")
        
        # Length comparison
        if 'Article' in df.columns:
            original_lengths = df['Article'].fillna('').str.len()
            summary_lengths = df['summary'].fillna('').str.len()
            
            print(f"\nText length comparison:")
            print(f"  Original avg: {original_lengths.mean():,.0f} chars")
            print(f"  Summary avg:  {summary_lengths.mean():,.0f} chars")
            print(f"  Compression:  {100 * (1 - summary_lengths.mean()/original_lengths.mean()):.1f}%")
        


# ============================================================================
# COMPARE SUMMARIES
# ============================================================================

def compare_algorithms(text: str, sentence_count: int = 3):
    """Compare different summarization algorithms on a sample text."""
    summarizer = SumySummarizer()
    
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    print(f"\nOriginal text ({len(text)} chars):")
    print(text[:500] + "..." if len(text) > 500 else text)
    
    results = summarizer.summarize_all_algorithms(text, sentence_count)
    
    for algo, summary in results.items():
        print(f"\n--- {algo.upper()} ({len(summary)} chars) ---")
        print(summary)
    
    return results




# MAIN
def main():
    parser = argparse.ArgumentParser(description='Article summarization with Sumy')
    
    # Part A: Ticker selection
    parser.add_argument('--top-n', type=int, default=10, help='Select top N tickers by article count')
    parser.add_argument('--min-per-year', type=int, default=100, help='Minimum articles per year threshold')
    
    # Part B: Summarization options
    parser.add_argument('--algorithm', type=str, default='lexrank', choices=['lsa', 'lexrank', 'luhn', 'textrank', 'sumbasic'], help='Summarization algorithm')
    parser.add_argument('--sentences', type=int, default=DEFAULT_SENTENCE_COUNT, help='Number of sentences in summary') 
    
    # Processing options
    parser.add_argument('--only-tickers', action='store_true', help='Only select tickers and exit')
    
    # Testing
    parser.add_argument('--compare', action='store_true', help='Compare algorithms on sample article') 
    parser.add_argument('--sample-size', type=int, default=None, help='Process only N articles (for testing)')
    
    args = parser.parse_args()   
    # print(args)   
    
    
    print("\n" + "-" * 40)
    print("PART A: TOP n TICKER SELECTION by minimum articles per year")
    print("-" * 40)
    
    ticker_filter = TickerFilter()
    selected_tickers = ticker_filter.get_tickers_balanced(n=args.top_n, min_per_year=args.min_per_year )
    ticker_filter.save_selected_tickers(selected_tickers)
    
    if args.only_tickers:
        print("\n[INFO] Only ticker selection requested. Exiting.")
        return
    
    
    # Comparison mode
    if args.compare:
        # Load sample article
        df = pd.read_csv(SELECTED_ARTICLES_CSV, nrows=5)
        row_num = 1
        sample_text = df['Article'].dropna().iloc[row_num]
        compare_algorithms(sample_text, args.sentences)
        print("\n End of comparisons.")
        return
    
    
    print("\n" + "=" * 60)
    print("PART B: SELECTED ARTICLES SUMMARIZATION ")
    print("=" * 60)
    
    # Load articles df from the selected tinker list
    df = ticker_filter.load_articles(selected_tickers, ARTICLES_CSV, METADATA_JSON) 
    df.to_csv(SELECTED_ARTICLES_CSV, index=False)
    
    # Initialize processor
    processor = ArticleSummarizationProcessor(sentence_count=args.sentences, algorithm=args.algorithm)
    
    # Process articles
    if args.sample_size:
        # For testing, only process sample
        df_sample = pd.read_csv(SELECTED_ARTICLES_CSV, nrows=args.sample_size)
        output_path = os.path.join(OUTPUT_DIR, f"articles_summarized_{args.algorithm}_{args.sentences}sent_sample.csv")
        df = processor.process_articles_df(df_sample, output_path)
        
    else:
        df = processor.process_articles_csv(SELECTED_ARTICLES_CSV)
        output_path = os.path.join(OUTPUT_DIR, f"articles_summarized_{args.algorithm}_{args.sentences}sent.csv")
    
    
    print("\n" + "=" * 60)
    print("SUMMARIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Sentences per summary: {args.sentences}")


if __name__ == "__main__":
    main()
