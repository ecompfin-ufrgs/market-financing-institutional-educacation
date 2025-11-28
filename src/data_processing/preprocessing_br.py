import os
import pandas as pd

from src.data_processing.data_processor_base import DataProcessorBase
from src.util import util


class BRDataProcessor(DataProcessorBase):
    def __init__(self):
        super().__init__("br")

    def process_file(self, file_path):
        try:
            df = pd.read_csv(file_path, sep=";", encoding="utf-8")
            df_result = []

            if "DS_CONTA" not in df.columns or "VL_CONTA" not in df.columns:
                df["DS_CONTA"] = df.get("DS_CONTA", "")
                df["VL_CONTA"] = df.get("VL_CONTA", 0)

            company_col = self.columns_mapping.get("company_name")
            date_col = self.columns_mapping.get("date")
            numeric_cols_to_convert = [c for c in self.columns_mapping.keys() if
                                       c not in ["date", "company_name", "currency"]]

            for company in df[company_col].unique():
                df_company = df[df[company_col] == company]

                for date_val in df_company[date_col].unique():
                    data = dict()
                    data["currency"] = "BRL"
                    data["company_name"] = company
                    data["date"] = date_val

                    df_data_period = df_company[df_company[date_col] == date_val]

                    for target_col, source_account_name in self.columns_mapping.items():
                        if target_col not in ["date", "company_name", "currency"]:
                            matched_rows = df_data_period[df_data_period["DS_CONTA"] == source_account_name]
                            data[target_col] = matched_rows["VL_CONTA"].sum() * 1000 if not matched_rows.empty else 0

                    if "formula_br" in self.config:
                        for target_col, formula in self.config["formula_br"].items():
                            if target_col not in data or data[target_col] in [None, 0, ""]:
                                try:
                                    expression = formula
                                    for var in data.keys():
                                        expression = expression.replace(var, f"data.get('{var}', 0)")
                                    data[target_col] = eval(expression)
                                except Exception as e:
                                    print('Formula error:', e)
                                    data[target_col] = 0

                    exchange_rate = util.get_usd_brl_exchange_rate_bcb(data["date"])
                    conversion_factor = 1 / exchange_rate
                    for col in numeric_cols_to_convert:
                        if col in data and isinstance(data[col], (int, float)):
                            data[col] = data[col] * conversion_factor

                    df_result.append(data)



            return pd.DataFrame(df_result)
        except KeyError as e:
            print(f"Column error while processing {file_path}: {e}.")
            return None
        except Exception as e:
            print(f"Unexpected error while processing {file_path}: {e}")
            return None
