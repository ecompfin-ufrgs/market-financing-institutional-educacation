import logging
import time


from src.analysis.analysis import BaseAnalysisRunner
from src.data_collection.data_collector_base import DataCollector
from src.data_processing.data_processor_base import DataProcessorBase
from src.data_processing.preprocessing_br import BRDataProcessor
from src.data_processing.preprocessing_us import USDataProcessor

if __name__ == "__main__":
    total_start = time.perf_counter()

    start = time.perf_counter()
    print("Starting Brazil collection (CVM)...")
    DataCollector().br_data_collection()
    end = time.perf_counter()
    print(f"Brazil collection completed (Duration: {end - start:.2f} seconds)")

    start = time.perf_counter()
    print("Starting Brazil processing...")
    DataProcessorBase('br').run_processing(BRDataProcessor(), "br_balance_data.csv")
    end = time.perf_counter()
    print(f"Brazil processing completed (Duration: {end - start:.2f} seconds)")

    start = time.perf_counter()
    print("Starting US collection (Alpha Vantage)...")
    DataCollector().us_data_collection()
    end = time.perf_counter()
    print(f"US collection completed (Duration: {end - start:.2f} seconds)")

    start = time.perf_counter()
    print("Starting US processing...")
    DataProcessorBase("us").run_processing(USDataProcessor(), "us_balance_data.csv")
    end = time.perf_counter()
    print(f"US processing completed (Duration: {end - start:.2f} seconds)")

    start = time.perf_counter()
    print("Starting Statistical Tests...")
    BaseAnalysisRunner().run_base_analysis()
    end = time.perf_counter()
    print(f"Statistical Tests completed (Duration: {end - start:.2f} seconds)")

    total_end = time.perf_counter()
    print(f"Total duration: {total_end - total_start:.2f} seconds")
