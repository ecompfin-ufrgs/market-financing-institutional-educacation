import os

import numpy as np
import pandas as pd
from src.data_processing.data_processor_base import DataProcessorBase

class USDataProcessor(DataProcessorBase):
    def __init__(self):
        super().__init__("us")

    def process_file(self, file_path):
        try:
            df = pd.read_csv(file_path)

            filename = os.path.basename(file_path)
            ticker = filename.split("_")[0]
            df["company_name"] = ticker

            df = df.rename(columns={v: k for k, v in self.columns_mapping.items() if v in df.columns})

            df["currency"] = "USD"

            for col in self.header_order:
                if col not in df.columns:
                    df[col] = 0

            numeric_cols = [col for col in self.header_order if col not in ['company_name', 'date', 'currency']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.replace([np.inf, -np.inf], np.nan)

            if "formula_us" in self.config:
                for target_col, formula in self.config["formula_us"].items():
                    if target_col in df.columns:
                        mask = df[target_col].isna() | (df[target_col] == 0)
                        if mask.any():
                            try:
                                result_series = df.eval(formula)
                                df.loc[mask, target_col] = result_series.loc[mask]
                                df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)

                            except Exception as e:
                                print('Formula error:', e)

            df = df.fillna(0)
            return df

        except KeyError as e:
            print(f"Column error while processing {file_path}: {e}.")
            return None
        except Exception as e:
            print(f"Unexpected error while processing {file_path}: {e}")
            return None

if __name__ == "__main__":
    DataProcessorBase('us').run_processing(USDataProcessor(), "us_balance_data.csv")