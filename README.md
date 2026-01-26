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

**Step 4: Summarize articles**

Run `4_summarize_articles.py` to:
- Generate single-line LexRank summaries for every article and write `Stock_news/summaries.csv` (columns: `Index`, `summary`).

**Step 5: Compute sentiment from summaries**

Run `5_compute_sentiment.py` to:
- Compute sentence-level FinBERT sentiment for summaries, save per-article results to `Stock_news/articles_sentiment.csv`, attach sentiment to `Stock_news/metadata.csv`, and aggregate daily sentiment by ticker to `data_stats/daily_sentiment.csv`.
- Note: this step downloads models from Hugging Face on first run and can be accelerated by a GPU.

Run steps individually:

```bash
python3 1_stock_select.py
python3 2_separate_news.py
python3 3_data_clean.py
python3 4_summarize_articles.py
python3 5_compute_sentiment.py
```

Or run all steps (1–5) in one go with `run_data_prep.sh` (recommended after setting up the venv):

```bash
chmod +x run_data_prep.sh
./run_data_prep.sh
```

Before running the pipeline, create and activate the virtual environment to ensure required packages are installed:

```bash
./make_venv.sh
source venv/bin/activate
```

(See `make_venv.sh` — it installs PyTorch plus project libraries such as `sumy`, `transformers`, and `nltk`.)

Expected structure:
```
data_stats/ (~7MB)
├── price_stats.json
├── news_stats.json
├── daily_sentiment.csv        
├── sp500.csv
Stock_news/
├── nasdaq_external_data.csv (~22GB) (delete this to save space)
├── filtered_nasdaq_exteral_data.csv (~9GB) (delete this to save space)
├── url_metadata.json (~80MB)
├── metadata.csv (~40MB)
├── articles.csv (~1.2GB)
├── summaries.csv (~130MB)    
├── articles_sentiment.csv (~6MB)  
Stock_price/
├── full_history/ (~20MB)
│   ├── A.csv (2018-2023)
│   ├── AAPL.csv
│   └── ...
```

## Project file structure

Top-level files and directories:

- `1_stock_select.py` — Filter and prepare stock price CSVs.
- `2_separate_news.py` — Filter and separate news dataset into metadata and articles.
- `3_data_clean.py` — Clean news metadata, normalize dates, and prune stocks.
- `4_summarize_articles.py` — Generate article summaries using Sumy.
- `5_compute_sentiment.py` — Compute sentiment with HuggingFace transformers and NLTK.
- `6_train_models.py` — Train models (uses `core` modules and datasets).
- `7_evaluate_models.py` — Evaluate trained models on test sets.
- `8_compare_models.py` — Compare model performance across experiments.
- `plot_news.py`, `check_news_frequency.py` — Plotting and analysis utilities.
- `make_venv.sh` — Create a virtualenv and install required Python packages (PyTorch + project libraries).
- `run_data_prep.sh` — Run all data prep steps in succession.
- `core/` — Core modules: `Model.py`, `train.py`, `PriceNewsDataset.py`, `plotter.py`, etc.
- `data_stats/` — Generated statistics and auxiliary CSVs (e.g., `news_stats.json`, `price_stats.json`, `sp500.csv`, `valid_market_days.csv`).
- `Stock_news/` — Raw and processed news files (`articles.csv`, `metadata.csv`, `url_metadata.json`).
- `Stock_price/` — Raw stock price CSVs (unzipped `full_history/`).
- `processed_data/` — Serialized datasets for training (`train_ds.pt`, `val_ds.pt`, `test_ds.pt`).
- `experiments/` — Saved model checkpoints and experiment configs.
- `figures/` — Generated plots and figures from analyses.
- `README.md`, `LICENSE`— Documentation and License.


## Bibliography

- Zihan Dong, Xinyu Fan, Zhiyuan Peng "FNSPID: A Comprehensive Financial News Dataset in Time Series" [Paper Link](https://arxiv.org/abs/2402.06698/)