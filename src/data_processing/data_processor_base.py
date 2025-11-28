import os
import pandas as pd
import json
from glob import glob

class DataProcessorBase:
    def __init__(self, country_code, config_file="config.json"):
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_file = os.path.join(self.project_root, "src", config_file)
        self.raw_dir = os.path.join(self.project_root, "data", "raw", country_code)
        self.processed_dir = os.path.join(self.project_root, "data", "processed", country_code)
        self.config = self._load_config()
        self.columns_mapping = self.config[f"columns_{country_code}"]
        self.header_order = self.config["header_order"]

        os.makedirs(self.processed_dir, exist_ok=True)

    def _load_config(self):
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error - FileNotFoundError '{self.config_file}' not found.")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: JSONDecodeError '{self.config_file}'.")
            exit(1)

    def _get_file_paths(self):
        return glob(os.path.join(self.raw_dir, "*.csv"))

    def run_processing(self, processor, output_filename):
        all_dfs = []
        file_paths = self._get_file_paths()
        if not file_paths:
            print(f"Error - No CSV files found in {self.raw_dir}.")
            return

        for f_path in file_paths:
            try:
                processed_df = processor.process_file(f_path)
                if processed_df is not None:
                    all_dfs.append(processed_df)
            except Exception as e:
                print(f"Error - File processing {f_path}: {e}")

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df = final_df[self.header_order]
            output_path = os.path.join(self.processed_dir, output_filename)
            final_df.to_csv(output_path, index=False)
            print(f"Final file saved: {output_path}")
        else:
            print("No data has been processed")
