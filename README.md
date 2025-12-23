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
4. After downloading and unzipping the dataset, run the preprocessing script to only keep data from 2018 to 2023 and only keep stocks that have samples for all these years.
```bash
python crop_fnspid.py
```
Now the structure should look like this:
```
data_stats/
├── news_stats.json
├── price_stats.json
Stock_news/
├── nasdaq_external_data.csv (deletable)
├── filtered_news.csv (~8GB)
Stock_price/
├── full_history/ (~600MB)
│   ├── A.csv (2018-2023)
│   ├── AA.csv
│   └── ...
```

## Bibliography

- Zihan Dong, Xinyu Fan, Zhiyuan Peng "FNSPID: A Comprehensive Financial News Dataset in Time Series" [Paper Link](https://arxiv.org/abs/2402.06698/)