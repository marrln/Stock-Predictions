# Stock-Predictions with Financial News Context

This repository contains code and models for predicting stock prices using historical stock data combined with financial news articles using the FSNPID dataset. The project leverages deep learning techniques to analyze the impact of news on stock movements.

## Steps to Mimic our Workflow

1. Download the FSNPID dataset from Hugging Face [FSNPID Dataset](https://huggingface.co/datasets/Zihan1004/FNSPID).
2. Make sure the dataset you downloaded has the following structure:
```
Stock_news/
├── nasdaq_external_data.csv (~20GB)
Stock_price/
└── full_history.zip (~5GB after unzipping)
```
3. Unzip the `full_history.zip` file into the `Stock_price/` directory. Now the structure should look like this:
```
Stock_news/
├── nasdaq_external_data.csv
Stock_price/
├── full_history/
│   ├── A.csv
│   ├── AA.csv
│   └── ...
```

4. Download the S&P 500 tickers CSV:

Download the file from [here](https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv) and rename it to `sp500.csv` and place it in the `data_stats/` directory, so the structure is:
```
data_stats/
├── sp500.csv
Stock_news/
├── nasdaq_external_data.csv (~22GB)
Stock_price/
├── full_history/ (~2GB)
│   ├── A.csv 
│   ├── AA.csv
│   └── ...
```

5. After downloading and unzipping the dataset, run the following scripts to preprocess the data for your experiments:

**Step 1: Filter and clean stock price data**

Run `1_stock_select.py` to:
	- Remove tickers not in the S&P 500 list
	- Filter each stock's CSV to only keep rows from 2018 to 2023
	- Delete any CSVs that are empty after filtering
	- Delete any CSVs that do not have at least one row for every year in the range
	- Generate per-ticker, per-year statistics in `data_stats/price_stats.json`

**Step 2: Filter and clean news data**

Run `2_choose_news_for_stocks.py` (this may take a while) to:
	- Filter the news CSV to only keep articles for the valid tickers and within the date range 2018-2023
	- Drop unnecessary columns
	- Write the filtered news to `Stock_news/sp500_news.csv`
	- Generate per-ticker, per-year news article counts in `data_stats/news_stats.json`

```bash
python 1_stock_select.py
python 2_choose_news_for_stocks.py 
```

Now the structure should look like this:
```
data_stats/ (~1MB)
├── price_stats.json
├── news_stats.json
├── sp500.csv
Stock_news/
├── nasdaq_external_data.csv (delete this to save space)
├── sp500_news.csv (~3GB)
Stock_price/
├── full_history/ (~52MB)
│   ├── A.csv (2018-2023)
│   ├── AAPL.csv
│   └── ...
```

**Step 3: Clean and deduplicate news data**

Run `3_data_clean.py` to:
	- Identify tickers with missing news for some or all years (2018-2023)
	- Delete stock CSV files for tickers with no news data at all
	- Validate tickers with partial news based on S&P 500 "Date added" field
	- Separate article text into a separate CSV file for storage efficiency
	- Deduplicate articles based on Article_title while preserving all stock symbols and dates
	- Update indices to keep only the first occurrence of each unique article

```bash
python 3_data_clean.py
```

Now the structure should look like this:
```
data_stats/ (~1MB)
├── price_stats.json
├── news_stats.json
├── sp500.csv
Stock_news/
├── sp500_news_dedup_final_no_articles.csv (~30MB)
├── sp500_news_dedup_articles.csv (~1.4GB)
Stock_price/
├── full_history/ (~52MB after cleanup)
│   ├── A.csv (2018-2023)
│   ├── AAPL.csv
│   └── ...
```

## Bibliography

- Zihan Dong, Xinyu Fan, Zhiyuan Peng "FNSPID: A Comprehensive Financial News Dataset in Time Series" [Paper Link](https://arxiv.org/abs/2402.06698/)