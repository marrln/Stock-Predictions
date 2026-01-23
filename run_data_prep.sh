#!/bin/bash

# Exit on any error
set -e

echo "Starting data preparation pipeline..."

echo "Step 1: Stock selection..."
python3 1_stock_select.py

echo "Step 2: Choosing news for stocks..."
python3 2_separate_news.py

echo "Data preparation complete!"
