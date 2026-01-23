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

**Step 2: Filter, separate and clean news data** ✅

Run `2_separate_news.py` (this may take a while) to:
- Filter the raw news CSV to keep articles dated 2018–2023
- Drop unnecessary columns and write the filtered CSV to `Stock_news/filtered_nasdaq_exteral_data.csv`
- Separate metadata and article text into `url_metadata.json`, `Stock_news/articles.csv`, and `Stock_news/metadata.csv` (includes `Article_Index`)
- Compute per-ticker per-year article counts into `data_stats/news_stats.json`

**Step 3: Further clean news data**
Run `3_data_clean.py` to:
- Remove the URL prefix from all URLs in `Stock_news/articles.csv` to save space
- For each metadata entry, remove tickers not present in `data_stats/news_stats.json`
- If a metadata entry's ticker list becomes empty, the metadata entry is deleted and the corresponding article (by `Article_Index`) is removed from `Stock_news/articles.csv`
- Track counts of deleted metadata entries and removed articles
- Parse the `Date` column (assumed UTC) and add a `Posted_After_Close` boolean column indicating whether the article was posted at or after US market close (4:00 PM US/Eastern)
- If the article's date is not a valid market day (based on `data_stats/valid_market_days.csv`), it is considered "before close" automatically (weekends/holidays)
- If the article date is after the last valid market day in the file, the entry is removed
- **Prune stock price files:** delete `.csv` files in `Stock_price/full_history` whose tickers are not present in `data_stats/news_stats.json` 

```bash
python3 1_stock_select.py
python3 2_separate_news.py
python3 3_data_clean.py
```

or run all steps in one go:

```bash
chmod +x run_data_prep.sh
./run_data_prep.sh
```

Expected structure:
```
data_stats/ (under 1MB)
├── price_stats.json
├── news_stats.json
├── sp500.csv
Stock_news/
├── nasdaq_external_data.csv (~22GB) (delete this to save space)
├── filtered_nasdaq_exteral_data.csv (~9GB) (delete this to save space)
├── url_metadata.json (~80MB)
├── metadata.csv (~40MB)
├── articles.csv (~1.2GB)
Stock_price/
├── full_history/ (~20MB)
│   ├── A.csv (2018-2023)
│   ├── AAPL.csv
│   └── ...
```

## Bibliography

- Zihan Dong, Xinyu Fan, Zhiyuan Peng "FNSPID: A Comprehensive Financial News Dataset in Time Series" [Paper Link](https://arxiv.org/abs/2402.06698/)