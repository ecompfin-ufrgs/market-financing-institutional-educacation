import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.api import VAR  # Used for PVAR IRF simulation


class PanelRegimeEstimator:
    """
    Estimates the POT relationship using Panel OLS with Fixed Effects (PanelOLS)
    and tests two distinct hypotheses:
    1.  Exogenous Regime Hypothesis using an exogenous dummy ('Stress_t' for the FIES crisis).
    2.  Tariff Shock Simulation Hypothesis using the simulated 'TariffImpact' and 'InteractionTerm' variables.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_regime_model(self, country_code: str):
        """
        Runs the panel model with the exogenous regime interaction (FIES crisis).
        """
        df_country = self.df[self.df['country'] == country_code].copy()

        if country_code == 'USA':
            return self.run_base_model(df_country, "USA (Baseline Model)")

        print(f"\n--- Estimating Exogenous Regime Model (PanelOLS w/ Fixed Effects) for: {country_code} ---")

        Y = df_country['D_debt_to_assets']

        X_vars = [
            'D_roa',
            'Stress_t',
            'Interaction_D_roa_Estresse',
            'log_total_assets'
        ]

        df_country['Interaction_D_roa_Estresse'] = df_country['D_roa'] * df_country['Stress_t']
        X = sm.add_constant(df_country[X_vars])
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        Y.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_data = pd.concat([Y, X], axis=1).dropna()
        Y_clean = final_data[Y.name]
        X_clean = final_data[X.columns]

        if Y_clean.empty or X_clean.empty:
            return f"\n{country_code} Error: Insufficient data after cleaning."

        try:
            model = PanelOLS(Y_clean, X_clean, entity_effects=True)
            results = model.fit(cov_type='robust')

            print(results)
            return results

        except Exception as e:
            return f"\n{country_code} Error: PanelOLS estimation failed: {e}"

    def run_shock_simulation_model(self, country_code: str = 'Brazil'):
        """
        Runs the panel model for the simulated TARIFF SHOCK.
        """
        if country_code != 'Brazil':
            print(f"\nSkipping Tariff Shock Simulation for {country_code}.")
            return None

        print(f"\n--- Estimating Tariff Shock Simulation Model (PanelOLS w/ Fixed Effects) for: {country_code} ---")

        df_country = self.df[self.df['country'] == country_code].copy()

        Y = df_country['D_debt_to_assets']

        X_vars = [
            'D_roa',
            'TariffImpact',
            'InteractionTerm'
        ]

        X = sm.add_constant(df_country[X_vars])

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        Y.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_data = pd.concat([Y, X], axis=1).dropna()
        Y_clean = final_data[Y.name]
        X_clean = final_data[X.columns]

        if Y_clean.empty or X_clean.empty:
            return f"\n{country_code} Error: Insufficient data for shock simulation after cleaning."

        try:
            model = PanelOLS(Y_clean, X_clean, entity_effects=True)
            results = model.fit(cov_type='robust')

            print(results)
            return results

        except Exception as e:
            return f"\n{country_code} Error: PanelOLS shock simulation estimation failed: {e}"

    def run_base_model(self, df_country, name: str):
        """Runs the basic POT model (no regimes) for the USA."""
        print(f"\n--- Estimating Base Model (PanelOLS w/ Fixed Effects) for: {name} ---")

        Y = df_country['D_debt_to_assets']
        X_vars = ['D_roa', 'log_total_assets']
        X = sm.add_constant(df_country[X_vars])

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        Y.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_data = pd.concat([Y, X], axis=1).dropna()
        Y_clean = final_data[Y.name]
        X_clean = final_data[X.columns]

        if Y_clean.empty or X_clean.empty:
            return f"\n{name} Error: Insufficient data after cleaning."

        try:
            model = PanelOLS(Y_clean, X_clean, entity_effects=True)
            results = model.fit(cov_type='robust')
            print(results)
            return results
        except Exception as e:
            return f"\n{name} Error: PanelOLS estimation failed: {e}"


class PVAREstimator:
    """
    Estimator for the Panel Vector Autoregression (PVAR) model.

    Simulates PVAR dynamics using Pooled VAR
    for calculating Impulse Response Functions (IRF).
    """

    def __init__(self, df: pd.DataFrame, lag_order: int = 1):
        self.df = df
        self.lag_order = lag_order
        self.pvar_vars = ['D_roa', 'D_debt_to_assets', 'log_total_assets']
        self.model_var = None

    def estimate_var_coefficients(self, country_code: str):
        """Estimates the VAR on the pooled data for coefficient matrix."""

        df_country = self.df[self.df['country'] == country_code].copy()
        df_country = df_country.reset_index()
        df_country = df_country.set_index(['company_name', 'date'])

        data_stacked = df_country[self.pvar_vars].reset_index(level=0, drop=True)

        try:
            print(f"\n--- Estimating Pooled VAR (PVAR Simulation) for {country_code} ---")

            model = VAR(data_stacked).fit(self.lag_order, verbose=False)
            self.model_var = model
            print(self.model_var.summary())

            try:
                pvar_params = self.model_var.params
                pvar_bse = self.model_var.bse
                pvar_tvalues = self.model_var.tvalues
                pvar_pvalues = self.model_var.pvalues

                pvar_results_df = pd.concat([
                    pvar_params.stack(dropna=False).rename('Coefficient'),
                    pvar_bse.stack(dropna=False).rename('Std. Error'),
                    pvar_tvalues.stack(dropna=False).rename('t-stat'),
                    pvar_pvalues.stack(dropna=False).rename('p-value')
                ], axis=1)
                pvar_results_df.index.names = ['Equation', 'Variable']

                csv_file_name = "table_3_pvar_coefficients.csv"
                pvar_results_df.to_csv(csv_file_name)
                print(f"Table 3 (PVAR Coefficients) successfully saved as '{csv_file_name}'")

            except Exception as e:
                print(f"Error saving Table 3 to CSV: {e}")

            return True

        except Exception as e:
            print(f"\nError in VAR estimation: {e}")
            return False

    def calculate_irf(self, periods: int = 8):
        """Calculates the Impulse Response Function (IRF) for the POT test."""
        if not self.model_var:
            print("Error: VAR model was not estimated.")
            return pd.DataFrame()

        shock_var = 'D_roa'
        response_var = 'D_debt_to_assets'

        irf = self.model_var.irf(periods=periods)

        response_index = self.pvar_vars.index(response_var)
        shock_index = self.pvar_vars.index(shock_var)

        irf_pot_response = irf.irfs[:, response_index, shock_index]

        irf_table = pd.DataFrame({
            'Period': range(periods + 1),
            'Response': irf_pot_response
        })

        try:
            csv_file_name = "table_4_irf_data.csv"
            irf_table.to_csv(csv_file_name, index=False)
            print(f"Table 4 (IRF Data) successfully saved as '{csv_file_name}'")
        except Exception as e:
            print(f"Error saving Table 4 to CSV: {e}")

        print("\n--- Impulse Response Function (IRF) Results ---")
        print(f"Response of {response_var} (Leverage Change) to a 1 SD Shock in {shock_var} (Profitability Change):")
        print(irf_table.to_markdown(index=False, floatfmt=".4f"))

        return irf_table