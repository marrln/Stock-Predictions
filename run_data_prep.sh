#!/bin/bash

# Exit on any error
set -e

echo "Starting data preparation pipeline..."

echo "Step 1: Stock selection..."
python3 1_stock_select.py

echo "Step 2: Choosing news for stocks..."
python3 2_separate_news.py

echo "Step 3: Cleaning news data..."
python3 3_data_clean.py

echo "Step 4: Summarizing Articles with LexRank..."
python3 4_summarize_articles.py

echo "Step 5: Computing Sentiment from Summaries with FinBert..."
python3 5_compute_sentiment.py

echo "Data preparation complete!"
