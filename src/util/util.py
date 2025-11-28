import csv
import os

import pandas as pd
import requests
from datetime import datetime, timedelta
from bcb import sgs

def merge_keep_first(left, right):
    merged = pd.merge(left, right, on="fiscalDateEnding", how="outer", suffixes=('', '_dup'))

    for col in right.columns:
        if col != "fiscalDateEnding" and col + "_dup" in merged.columns:
            merged.drop(columns=[col + "_dup"], inplace=True)

    return merged

def get_usd_brl_exchange_rate_bcb(date_str):
    """
    Reads the USD/BRL exchange rate from a CSV file for a specific date.
    """

    try:
        current_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print("ERROR: Invalid input date format. Use 'YYYY-MM-DD'.")
        return None

    exchange_rates = {}
    try:
        file_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw"), 'usd_brl_exchange_rate_bcb.csv')
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')

            for row in reader:
                try:
                    date_obj = datetime.strptime(row['date'], '%d/%m/%Y')
                    formatted_date = date_obj.strftime('%Y-%m-%d')

                    exchange_rates[formatted_date] = float(row['value'])
                except (ValueError, KeyError) as e:
                    continue

    except FileNotFoundError:
        print(f"ERROR: CSV file not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading the file: {e}")
        return None

    max_days_back = 5
    for days_back in range(max_days_back):
        date_to_try = current_date - timedelta(days=days_back)
        date_to_try_str = date_to_try.strftime('%Y-%m-%d')

        if date_to_try_str in exchange_rates:
            rate = exchange_rates[date_to_try_str]
            return rate

    print(f"ERROR: Could not find the rate in the CSV after backtracking {max_days_back} days starting from {date_str}.")
    return None