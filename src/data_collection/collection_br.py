import os
import requests
import zipfile
import io
import pandas as pd
from tqdm import tqdm

class BRDataCollection:
    def __init__(self, companies, required_files, start_year, end_year, output_dir):
        self.companies = companies
        self.required_files = required_files
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def download_zip(url: str) -> zipfile.ZipFile | None:
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                return zipfile.ZipFile(io.BytesIO(r.content))
            return None
        except Exception as e:
            print(f"Download Error {url}: {e}")
            return None

    @staticmethod
    def read_csv_from_zip(z: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
        try:
            with z.open(name) as f:
                try:
                    return pd.read_csv(f, sep=";", dtype=str, encoding="latin1", low_memory=False)
                except Exception:
                    f.seek(0)
                    return pd.read_csv(f, sep=";", dtype=str, encoding="utf-8", low_memory=False)
        except Exception as e:
            print(f"Error in reading {name}: {e}")
            return None

    def process_itr(self, year: int) -> pd.DataFrame:
        url = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_{year}.zip"
        z = self.download_zip(url)
        if not z:
            print(f"{year}: file not found, skipping...")
            return pd.DataFrame()

        df_itr = pd.DataFrame()

        for name in z.namelist():
            if not (any(f in name.lower() for f in self.required_files) and name.lower().endswith(".csv")):
                continue

            df = self.read_csv_from_zip(z, name)
            if df is None or "DENOM_CIA" not in df.columns or "ORDEM_EXERC" not in df.columns or "DT_FIM_EXERC" not in df.columns:
                continue

            df_filtered = df[
                (df["DENOM_CIA"].isin(self.companies)) &
                ((df["ORDEM_EXERC"].str.upper() == "ÚLTIMO") |
                    ((df["ORDEM_EXERC"].str.upper() == "PENÚLTIMO") & (df["DT_FIM_EXERC"].str[5:7] == "12")))
            ]

            if not df_filtered.empty:
                df_itr = pd.concat([df_itr, df_filtered], ignore_index=True)

        return df_itr

    def run(self, consolidate=False):
        all_data = pd.DataFrame()

        for year in tqdm(range(self.start_year, self.end_year)):
            try:
                df_itr = self.process_itr(year)
                if not df_itr.empty:
                    if consolidate:
                        all_data = pd.concat([all_data, df_itr], ignore_index=True)
                    else:
                        out_path = os.path.join(self.output_dir, f"{year}_balance_annual.csv")
                        df_itr.to_csv(out_path, index=False, sep=";", encoding="utf-8")
            except Exception as e:
                print(f"{year}: unexpected error -> {e}")

        if consolidate and not all_data.empty:
            out_path = os.path.join(self.output_dir, "balances_all_years.csv")
            all_data.to_csv(out_path, index=False, sep=";", encoding="utf-8")
            print(f"File saved in {out_path}")