# data_collection.py
import yfinance as yf
import pandas as pd

def collect_data(ticker, start_date, end_date, save_path):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(save_path)
    print(f"Data saved to {save_path}")
