import pandas as pd
import numpy as np
from pre_inferential_stage import DataPreparation
import sys
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.modeling.pvar_models import PanelRegimeEstimator, PVAREstimator

class InferentialAnalysis():
    def inf_analysis_pipeline(self):
        FILE_NAME = 'final_stationary_data.csv'

        print("========================================================")
        print("====== STEP 1: DATA PREPARATION ======")
        print("========================================================")
        data_pipeline = DataPreparation(file_name=FILE_NAME)
        df_prepared = data_pipeline.run_pipeline()

        if df_prepared.empty:
            print("\nAnalysis terminated. Failed to load or prepare data.")
            sys.exit(1)

        print("\n========================================================")
        print("====== STEP 2: STRUCTURAL REGIME ANALYSIS (PanelOLS) ======")
        print("========================================================")
        regime_estimator = PanelRegimeEstimator(df_prepared)

        results_br_regime = regime_estimator.run_regime_model(country_code='Brazil')
        format_panel_regime_results(results_br_regime, "Brazil - Structural Regime Model (Post-2015)")

        results_us_base = regime_estimator.run_regime_model(country_code='USA')
        format_panel_regime_results(results_us_base, "USA - Baseline Model")

        print("\n========================================================")
        print("====== STEP 3: TARIFF SHOCK SIMULATION (PanelOLS) ======")
        print("========================================================")

        TARIFF_RATE = 0.5
        EXPOSURE_SCENARIOS = [0.1, 0.2, 0.3]  # 10%, 20%, 30%

        for exposure in EXPOSURE_SCENARIOS:
            print(f"\n--- Preparing data for Exposure Scenario: {exposure * 100:.0f}% ---")
            temp_data_pipeline = DataPreparation(
                file_name=FILE_NAME,
                tariff_rate=TARIFF_RATE,
                exposure_rate=exposure
            )
            temp_df_prepared = temp_data_pipeline.run_pipeline()

            if temp_df_prepared.empty:
                print(f"Skipping scenario {exposure}, data prep failed.")
                continue

            temp_regime_estimator = PanelRegimeEstimator(temp_df_prepared)
            results_shock_br = temp_regime_estimator.run_shock_simulation_model(country_code='Brazil')
            format_panel_shock_results(results_shock_br, "Brazil - Tariff Shock Simulation", exposure)

        print("\n========================================================")
        print("====== STEP 4: DYNAMIC ANALYSIS (PVAR) ======")
        print("========================================================")

        pvar_estimator = PVAREstimator(df_prepared, lag_order=1)

        if pvar_estimator.estimate_var_coefficients(country_code='Brazil'):
            irf_data = pvar_estimator.calculate_irf(periods=8)

            if not irf_data.empty:
                print("\n--- Generating FIGURE 1 ---")

                try:
                    sns.set_theme(style="ticks")
                    sns.set_context("paper", font_scale=1.2)

                    plt.figure(figsize=(6, 4))

                    plt.plot(
                        irf_data['Period'],
                        irf_data['Response'],
                        color='black',
                        marker='o',
                        markersize=4,
                        linestyle='-',
                        linewidth=1.5
                    )

                    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

                    plt.xlabel('Period')
                    plt.ylabel(r'Response of $\Delta$Debt-to-Assets')

                    plt.title('')
                    plt.suptitle('')

                    sns.despine()

                    plt.tight_layout()
                    file_name_figure = 'irf.png'
                    plt.savefig(file_name_figure, dpi=300)
                    plt.clf()
                    print(f"Saved: {file_name_figure}")

                except Exception as e:
                    print(f"Error generating FIGURE 1: {e}")

        print("\n--- Complete Analysis Pipeline Finished ---")


def format_panel_regime_results(results, model_name: str):
    """
    Formats the PanelOLS results for the STRUCTURAL REGIME model (FIES Crisis).
    """
    print(f"\n\n--- RESULTS TABLE: {model_name} ---")
    print("Model: Panel OLS with Entity Fixed Effects and Robust SEs")

    if isinstance(results, str):
        print(results)
        return

    try:
        nobs = int(results.nobs)
    except:
        nobs = 'N/A'

    if "Brazil" in model_name:
        print(f"\n\n[TABLE 1: Summary of the Exogenous Regime Model (Panel OLS) - Brazil (N={nobs})]")
        csv_file_name = "table_1_brazil_regime.csv"
    elif "USA" in model_name:
        print(f"\n\n[TABLE 2: Summary of the Baseline Model (Panel OLS) - USA (N={nobs})]")
        csv_file_name = "table_2_usa_baseline.csv"
    else:
        print(f"\n\n--- RESULTS TABLE: {model_name} (N={nobs}) ---")
        csv_file_name = f"table_{model_name.lower().replace(' ', '_')}.csv"

    print("Model: Panel OLS with Entity Fixed Effects and Robust SEs")

    summary_df = pd.DataFrame({
        'Coefficient': results.params,
        'Std. Error': results.std_errors,
        't-stat': results.tstats,
        'p-value': results.pvalues
    })

    print(summary_df.to_markdown(floatfmt=".4f"))

    try:
        summary_df.to_csv(csv_file_name)
        print(f"Table successfully saved as '{csv_file_name}'")
    except Exception as e:
        print(f"Error saving table to CSV ({csv_file_name}): {e}")

    if 'D_roa' in results.params and 'Interaction_D_roa_Estresse' in results.params:
        print("\n--- Regime Hypothesis Interpretation (POT) ---")
        try:
            beta1 = results.params['D_roa']
            beta2 = results.params['Interaction_D_roa_Estresse']

            pot_stable_regime = beta1
            pot_stress_regime = beta1 + beta2

            cov = results.cov.loc['D_roa', 'Interaction_D_roa_Estresse']
            var_sum = results.std_errors['D_roa'] ** 2 + results.std_errors['Interaction_D_roa_Estresse'] ** 2 + 2 * cov
            std_err_sum = var_sum ** 0.5
            t_stat_sum = pot_stress_regime / std_err_sum

            df_resid = results.df_resid
            p_val_sum = t.sf(np.abs(t_stat_sum), df_resid) * 2

            print(
                f"   Stable Regime (Pre-2015): Coef. (D_roa) = {pot_stable_regime:.4f} (p={results.pvalues['D_roa']:.4f})")
            print(
                f"   Stress Regime (Post-2015): Coef. (D_roa + Interaction) = {pot_stress_regime:.4f} (p={p_val_sum:.4f})")

            if pot_stress_regime < 0 and p_val_sum < 0.1:
                print("   FINDING: POT adherence is statistically significant in the Stress Regime.")
            elif pot_stable_regime < 0 and results.pvalues['D_roa'] < 0.1:
                print("   FINDING: POT adherence is statistically significant in the Stable Regime.")
            else:
                print("   FINDING: POT adherence is not statistically significant in either regime.")

        except Exception as e:
            print(f"   Error calculating coefficient sum test: {e}")


def format_panel_shock_results(results, model_name: str, exposure_rate: float):
    """
    Formats the PanelOLS results for the TARIFF SHOCK SIMULATION model.
    """
    print(f"\n\n--- RESULTS TABLE: {model_name} (Exposure: {exposure_rate * 100:.0f}%) ---")
    print("Model: Panel OLS with Entity Fixed Effects and Robust SEs")

    if isinstance(results, str):
        print(results)
        return

    summary_df = pd.DataFrame({
        'Coefficient': results.params,
        'Std. Error': results.std_errors,
        't-stat': results.tstats,
        'p-value': results.pvalues
    })

    print(summary_df.to_markdown(floatfmt=".4f"))

    print("\n--- Shock Hypothesis Interpretation (POT) ---")
    try:
        beta = results.params['D_roa']
        p_beta = results.pvalues['D_roa']
        gamma = results.params['TariffImpact']
        p_gamma = results.pvalues['TariffImpact']
        delta = results.params['InteractionTerm']
        p_delta = results.pvalues['InteractionTerm']

        print(f"   Beta (POT): Coef. (D_roa) = {beta:.4f} (p={p_beta:.4f})")
        print(f"   Gamma (Direct Shock): Coef. (TariffImpact) = {gamma:.4e} (p={p_gamma:.4f})")
        print(f"   Delta (Moderation Effect): Coef. (InteractionTerm) = {delta:.4e} (p={p_delta:.4f})")

        if gamma > 0 and p_gamma < 0.1:
            print(
                "   FINDING (Gamma): The shock has a significant POSITIVE direct effect on leverage (as hypothesized).")
        if delta > 0 and p_delta < 0.1:
            print("   FINDING (Delta): The shock significantly WEAKENS the POT relationship (as hypothesized).")

    except Exception as e:
        print(f"   Error interpreting shock results: {e}")

