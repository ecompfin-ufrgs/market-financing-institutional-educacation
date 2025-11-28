import os
import pandas as pd
import numpy as np


class DataPreparation:
    """
    A class for loading the final stationary data and engineering the features
    required for the Panel Markov-Switching (PMS-IFE) and PVAR models.

    The main steps include:
    1. Calculating First Differences (Delta) for stationarity.
    2. Simulating the Tariff Impact variable for Brazilian companies.
    3. Creating the interaction term (Profitability * Tariff Impact).
    """

    def __init__(self, file_name='final_stationary_data.csv', tariff_rate=0.5, exposure_rate=0.1):
        """
        Initializes the DataPreparation class.

        Args:
            file_name (str): The name of the final stationary data CSV file.
            tariff_rate (float): The simulated tariff rate (tau).
            exposure_rate (float): The base exposure rate for the TariffImpact calculation.
        """

        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = os.path.join(self.project_root, "data", "final", "analysis", file_name)
        self.tariff_rate = tariff_rate
        self.exposure_rate = exposure_rate
        self.df = None
        self.df_final_model = None

    def load_and_preprocess(self):
        """Loads the data and performs initial cleaning and sorting."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values(by=['company_name', 'date']).reset_index(drop=True)
            print("Data loaded and sorted successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}. Please ensure it is accessible.")
            self.df = pd.DataFrame()

    def calculate_first_differences(self):
        """Calculates the first difference (Delta) for key variables."""
        if self.df is None or self.df.empty:
            return

        # Core variables for the POT test
        panel_vars = ['debt_to_assets', 'roa']

        # Calculate Delta (first difference) grouped by company
        df_diff = self.df.groupby('company_name')[panel_vars].diff().add_prefix('D_')
        self.df = pd.concat([self.df, df_diff], axis=1)
        print(f"First differences calculated for: {list(df_diff.columns)}.")

    def create_exogenous_regime_dummy(self):
        """
        Creates the deterministic exogenous regime dummy variable based on
        the FIES crisis (starting in 2015).
        """
        if self.df is None or self.df.empty:
            return

        regime_break_date = pd.to_datetime('2015-01-01')

        # Stress_t = 1 for the Post-FIES (Stress) period, 0 for the Pre-FIES (Stable) period
        self.df['Stress_t'] = (self.df['date'] >= regime_break_date).astype(int)
        print(f"Regime dummy 'Stress_t' (Break Point: {regime_break_date.date()}) created.")

    def simulate_tariff_impact(self):
        """Creates the simulated TariffImpact variable and the interaction term."""
        if self.df is None or self.df.empty:
            return

        # TariffImpact_it = tau * Exposure_it * TotalAssets_it
        self.df['TariffImpact'] = (
                self.tariff_rate * self.exposure_rate * self.df['total_assets']
        )

        # Apply the shock only to Brazilian companies
        self.df.loc[self.df['country'] == 'Brazil', 'TariffImpact'] = (
                self.tariff_rate * self.exposure_rate * self.df['total_assets']
        )

        # InteractionTerm = D_roa * TariffImpact (using the differenced Profitability)
        self.df['InteractionTerm'] = self.df['D_roa'] * self.df['TariffImpact']
        print(f"TariffImpact (tau={self.tariff_rate}, exposure={self.exposure_rate}) and InteractionTerm created.")

    def final_clean_and_export(self):
        """Removes NaN values resulting from differentiation and prepares the final DataFrame."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.df_final_model = self.df.dropna(subset=['D_debt_to_assets', 'D_roa', 'log_total_assets']).copy()

        self.df_final_model = self.df_final_model.set_index(['company_name', 'date'])

        print(f"\nFinal dataset size: {len(self.df_final_model)} observations.")
        return self.df_final_model

    def run_pipeline(self):
        """Executes the full data preparation pipeline."""
        self.load_and_preprocess()
        if self.df is not None and not self.df.empty:
            self.calculate_first_differences()
            self.create_exogenous_regime_dummy()
            self.simulate_tariff_impact()
            return self.final_clean_and_export()
        return pd.DataFrame()

