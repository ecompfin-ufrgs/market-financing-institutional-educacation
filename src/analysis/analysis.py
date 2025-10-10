import os
import pandas as pd

from src.analysis.statistical_analysis import StatisticalAnalysis


class BaseAnalysisRunner:
    """
    Class to execute the base statistical analysis steps, such as stationarity tests
    and descriptive statistics calculation, by centralizing function calls
    from existing modules.
    """

    def __init__(self):
        self.analysis_engine = StatisticalAnalysis()
        self.results_dir = self.analysis_engine.results_dir

    def load_and_preprocess_data(self):
        """Loads and applies initial data preprocessing."""
        print("Loading and preprocessing financial data...")
        df = self.analysis_engine.load_data()
        print("Data loaded successfully.")
        return df

    def run_stationarity_tests_by_country(self, df):
        """Runs ADF and KPSS tests and generates the transformation logs."""
        print("Running Stationarity Tests (ADF and KPSS) by Company and Country...")
        stationarity_df, transformation_df = self.analysis_engine.run_stationarity_tests(df)
        print("Creating stationarity summary tables by country...")
        self.analysis_engine.create_stationarity_summary_tables(stationarity_df)
        print("Summary tables saved")
        return stationarity_df, transformation_df

    def calculate_descriptive_stats_by_country(self, df):
        """Calculates and saves descriptive statistics separated by country."""
        print("Calculating Descriptive Statistics...")
        br_stats, us_stats = self.analysis_engine.calculate_descriptive_statistics(df)
        print("Descriptive statistics saved")
        return br_stats, us_stats

    def run_base_analysis(self):
        print("Starting Analysis...")

        df = self.load_and_preprocess_data()

        if df.empty:
            print("No data available for analysis. Exiting.")
            return

        self.run_stationarity_tests_by_country(df)
        self.calculate_descriptive_stats_by_country(df)

        print("Analysis Complete")


if __name__ == "__main__":
    BaseAnalysisRunner().run_base_analysis()