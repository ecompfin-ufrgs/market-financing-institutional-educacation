from src.data_collection.collection_br import BRDataCollection
from src.data_collection.collection_us import USDataCollection
import os

class DataCollector:

    @staticmethod
    def br_data_collection():
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "data", "raw", "br")

        companies_br = ["COGNA EDUCAÇÃO S.A.", "YDUQS PARTICIPACOES S.A.", "SER EDUCACIONAL S.A."]
        required_files = ["_bpa_con_", "_bpp_con_", "_dre_con_", "_dfc_mi_con_"]

        BRDataCollection(
            companies=companies_br,
            required_files=required_files,
            start_year=2011,
            end_year=2025,
            output_dir=output_dir,
        ).run(consolidate=False)


    @staticmethod
    def us_data_collection():
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "data", "raw", "us")

        tickers = ["TWOU", "ATGE", "LOPE", "STRA", "APEI"]
        api_key = "M4V35A5KMSYBMQ32"

        USDataCollection(
            tickers=tickers,
            api_key=api_key,
            output_dir=output_dir,
        ).run()

if __name__ == "__main__":
    DataCollector().us_data_collection()
    DataCollector().br_data_collection()
