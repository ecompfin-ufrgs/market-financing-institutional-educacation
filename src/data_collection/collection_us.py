import os
from functools import reduce

import requests
import pandas as pd

from src.util import util


class USDataCollection:
    def __init__(self, tickers, api_key, output_dir):
        self.tickers = tickers
        self.api_key = api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def fetch_json(url: str) -> dict:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code} - {url}")
                return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}

    def process_ticker(self, ticker: str) -> pd.DataFrame:
        print(f"Collecting data for {ticker}...")

        functions = {
            "balance_sheet": "BALANCE_SHEET",
            "cash_flow": "CASH_FLOW",
            "income_statement": "INCOME_STATEMENT"
        }

        dfs = []

        for name, func in functions.items():
            url = (
                f"https://www.alphavantage.co/query"
                f"?function={func}&symbol={ticker}&apikey={self.api_key}"
            )
            data = self.fetch_json(url)

            quarterly_reports = data.get("quarterlyReports", [])

            if not quarterly_reports:
                print(f"No data: {func} - {ticker}.")
                print("API Response:", data)
                continue

            df = pd.DataFrame(quarterly_reports)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        df_final = reduce(util.merge_keep_first, dfs)

        df_final = df_final.loc[:, ~df_final.columns.duplicated()]

        df_final["fiscalDateEnding"] = pd.to_datetime(df_final["fiscalDateEnding"])
        df_final = df_final.sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)

        df_final = df_final[df_final["fiscalDateEnding"].dt.year >= 2010]

        return df_final

    def run(self):
        for ticker in self.tickers:
            try:
                df_annual = self.process_ticker(ticker)
                if not df_annual.empty:
                    out_path = os.path.join(self.output_dir, f"{ticker}_balance_annual.csv")
                    df_annual.to_csv(out_path, index=False, encoding="utf-8")
                    print(f"Data saved at: {out_path}")
            except Exception as e:
                print(f"Unexpected error for {ticker}: {e}")

