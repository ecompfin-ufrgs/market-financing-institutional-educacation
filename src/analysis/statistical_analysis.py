import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# warnings.filterwarnings('ignore')

class StatisticalAnalysis:
    """
    Class focused on loading and preprocessing data, calculating descriptive
    statistics, and performing stationarity tests (ADF and KPSS).
    """

    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.br_dir = os.path.join(self.project_root, "data", "processed", 'br')
        self.us_dir = os.path.join(self.project_root, "data", "processed", 'us')
        self.results_dir = os.path.join(self.project_root, "data", "final", "analysis")

        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self):
        """Loads and performs essential data preprocessing."""
        try:
            br_data = pd.read_csv(os.path.join(self.br_dir, 'br_balance_data.csv'))
            br_data['country'] = 'Brazil'
        except FileNotFoundError:
            br_data = pd.DataFrame()

        try:
            us_data = pd.read_csv(os.path.join(self.us_dir, 'us_balance_data.csv'))
            us_data['country'] = 'USA'
        except FileNotFoundError:
            us_data = pd.DataFrame()

        if br_data.empty and us_data.empty:
            print("Warning: No processed data files found.")
            return pd.DataFrame()

        df = pd.concat([br_data, us_data], ignore_index=True)

        if 'total_debt' in df.columns and 'total_assets' in df.columns:
            df['debt_to_assets'] = df['total_debt'] / df['total_assets']
        if 'total_debt' in df.columns and 'shareholder_equity' in df.columns:
            df['debt_to_equity'] = df['total_debt'] / df['shareholder_equity']
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa'] = df['net_income'] / df['total_assets']

        if 'total_debt' in df.columns and 'company_name' in df.columns:
            df = df.sort_values(['company_name', 'date'])
            df['debt_change'] = df.groupby('company_name')['total_debt'].diff()

        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def calculate_descriptive_statistics(self, df):
        """
        Calculates comprehensive descriptive statistics and saves them in separate CSVs by country.
        Returns the statistics DataFrames.
        """
        financial_vars = ['total_assets', 'current_assets', 'total_liabilities', 'current_liabilities',
                          'total_debt', 'shareholder_equity', 'dividends_paid', 'net_income',
                          'operation_cash_flow', 'capex', 'revenue',
                          'debt_to_assets', 'debt_to_equity', 'roa',
                          'working_capital']

        valid_vars = [var for var in financial_vars if var in df.columns]

        def get_stats(data):
            return pd.DataFrame({
                'Mean': data.mean(),
                'Median': data.median(),
                'Std_Dev': data.std(),
                'Minimum': data.min(),
                'Maximum': data.max(),
                'Skewness': data.skew(),
                'Observations': data.count()
            }).round(4)

        br_data = df[df['country'] == 'Brazil'][valid_vars]
        br_stats = get_stats(br_data)
        br_stats.to_csv(os.path.join(self.results_dir, 'table1_brazilian_descriptive_stats.csv'))

        us_data = df[df['country'] == 'USA'][valid_vars]
        us_stats = get_stats(us_data)
        us_stats.to_csv(os.path.join(self.results_dir, 'table2_us_descriptive_stats.csv'))

        return br_stats, us_stats

    def _test_stationarity(self, series):
        """Helper function to run both ADF and KPSS tests."""
        try:
            # ADF test
            adf_result = adfuller(series, autolag='AIC')
            adf_dict = {'statistic': adf_result[0], 'pvalue': adf_result[1]}

            # KPSS test
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_dict = {'statistic': kpss_result[0], 'pvalue': kpss_result[1]}

            # Determine stationarity (p <= 0.05 for ADF AND p > 0.05 for KPSS)
            adf_stationary = adf_result[1] <= 0.05
            kpss_stationary = kpss_result[1] > 0.05

            is_stationary = adf_stationary and kpss_stationary

            return adf_dict, kpss_dict, is_stationary

        except Exception:
            return {'statistic': np.nan, 'pvalue': np.nan}, {'statistic': np.nan, 'pvalue': np.nan}, False

    def run_stationarity_tests(self, df):
        """
        Runs ADF and KPSS tests for each variable/company and generates transformation logs.
        Saves detailed results and logs to CSVs. Returns the results DataFrames.
        """
        financial_vars = ['total_assets', 'current_assets', 'total_liabilities', 'current_liabilities',
                          'total_debt', 'shareholder_equity', 'dividends_paid', 'net_income',
                          'operation_cash_flow', 'capex', 'revenue', 'working_capital',
                          'debt_to_assets', 'debt_to_equity', 'roa', 'debt_change']

        stationarity_results = []
        transformation_log = []

        series_diff = None
        is_stationary_diff = False

        for country in ['Brazil', 'USA']:
            country_data = df[df['country'] == country].copy()

            for var in financial_vars:
                if var in country_data.columns:
                    for company in country_data['company_name'].unique():
                        company_data = country_data[country_data['company_name'] == company].sort_values('date')
                        series = company_data[var].dropna()

                        if len(series) < 3:
                            continue

                        adf_result, kpss_result, is_stationary = self._test_stationarity(series)

                        stationarity_results.append({
                            'Country': country, 'Company': company, 'Variable': var,
                            'Level': 'Original', 'ADF_Statistic': adf_result['statistic'],
                            'ADF_pvalue': adf_result['pvalue'], 'KPSS_Statistic': kpss_result['statistic'],
                            'KPSS_pvalue': kpss_result['pvalue'], 'Is_Stationary': is_stationary
                        })

                        transformation_applied = 'None'
                        final_stationary = is_stationary

                        if not is_stationary:
                            series_diff = series.diff().dropna()
                            if len(series_diff) >= 3:
                                adf_result_diff, kpss_result_diff, is_stationary_diff = self._test_stationarity(
                                    series_diff)

                                stationarity_results.append({
                                    'Country': country, 'Company': company, 'Variable': var,
                                    'Level': 'First_Difference', 'ADF_Statistic': adf_result_diff['statistic'],
                                    'ADF_pvalue': adf_result_diff['pvalue'],
                                    'KPSS_Statistic': kpss_result_diff['statistic'],
                                    'KPSS_pvalue': kpss_result_diff['pvalue'], 'Is_Stationary': is_stationary_diff
                                })

                                if is_stationary_diff:
                                    transformation_applied = 'First Difference'
                                    final_stationary = True

                        if not is_stationary_diff:
                            series_diff2 = series_diff.diff().dropna()
                            if len(series_diff2) >= 3:
                                adf_result_diff2, kpss_result_diff2, is_stationary_diff2 = self._test_stationarity(
                                    series_diff2)

                                stationarity_results.append({
                                    'Country': country, 'Company': company, 'Variable': var,
                                    'Level': 'Second_Difference', 'ADF_Statistic': adf_result_diff2['statistic'],
                                    'ADF_pvalue': adf_result_diff2['pvalue'],
                                    'KPSS_Statistic': kpss_result_diff2['statistic'],
                                    'KPSS_pvalue': kpss_result_diff2['pvalue'], 'Is_Stationary': is_stationary_diff2
                                })

                                if is_stationary_diff2:
                                    transformation_applied = 'Second Difference'
                                    final_stationary = True

                        transformation_log.append({
                            'Country': country, 'Company': company, 'Variable': var,
                            'Original_Stationary': is_stationary,
                            'Transformation_Applied': transformation_applied,
                            'Final_Stationary': final_stationary
                        })

        stationarity_df = pd.DataFrame(stationarity_results)
        transformation_df = pd.DataFrame(transformation_log)

        stationarity_df.to_csv(os.path.join(self.results_dir, 'stationarity_tests_detailed.csv'), index=False)
        transformation_df.to_csv(os.path.join(self.results_dir, 'transformation_log.csv'), index=False)

        return stationarity_df, transformation_df

    def create_stationarity_summary_tables(self, stationarity_df):
        """Creates summary tables for stationarity results by country."""

        br_stationarity = stationarity_df[stationarity_df['Country'] == 'Brazil'].copy()
        br_summary = br_stationarity.groupby(['Variable', 'Level']).agg({
            'ADF_pvalue': 'mean',
            'KPSS_pvalue': 'mean',
            'Is_Stationary': 'sum'
        }).round(4)
        br_summary['Companies_Stationary'] = br_summary['Is_Stationary']
        br_summary = br_summary.drop('Is_Stationary', axis=1)
        br_summary.to_csv(os.path.join(self.results_dir, 'table3_brazilian_stationarity.csv'))

        us_stationarity = stationarity_df[stationarity_df['Country'] == 'USA'].copy()
        us_summary = us_stationarity.groupby(['Variable', 'Level']).agg({
            'ADF_pvalue': 'mean',
            'KPSS_pvalue': 'mean',
            'Is_Stationary': 'sum'
        }).round(4)
        us_summary['Companies_Stationary'] = us_summary['Is_Stationary']
        us_summary = us_summary.drop('Is_Stationary', axis=1)
        us_summary.to_csv(os.path.join(self.results_dir, 'table4_us_stationarity.csv'))

        return br_summary, us_summary
