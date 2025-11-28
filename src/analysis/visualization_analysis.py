import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')


class VisualizationAnalysis:
    """
    Generate all necessary visualizations and summary tables
    for the Exploratory Data Analysis section, using previously
    calculated results.
    """

    def __init__(self):
        """
        Initializes the directory variables for reading results and
        saving the visualizations.
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = os.path.join(self.project_root, "data", "final", "analysis")
        self.visualization_dir = os.path.join(self.project_root, "data", "final", "visualization")

        # Create the directory to save the plots, if it doesn't exist
        os.makedirs(self.visualization_dir, exist_ok=True)
        print(f"Plots and tables will be saved in: {self.visualization_dir}")

    def load_data(self):
        """
        Loads all necessary CSV files for generating the plots and tables.
        Returns a dictionary of DataFrames.
        """
        try:
            dfs = {
                'final_data': pd.read_csv(os.path.join(self.results_dir, 'final_stationary_data.csv')),
                'stationarity_detailed': pd.read_csv(os.path.join(self.results_dir, 'stationarity_tests_detailed.csv')),
                'removal_log': pd.read_csv(os.path.join(self.results_dir, 'table5_series_removal_log.csv'))
            }
            print("Data files loaded successfully.")
            return dfs
        except FileNotFoundError as e:
            print(f"Error: File not found. Please ensure the 'analysis.py' script was run first.")
            print(f"Error detail: {e}")
            return None

    def generate_exploratory_plots(self, df):
        """
        Generates and saves the four main plots for the exploratory analysis.
        """
        if df is None or df.empty:
            print("DataFrame is empty. Skipping plot generation.")
            return

        print("Generating exploratory plots...")

        df['date'] = pd.to_datetime(df['date'])
        df['debt_to_equity'] = pd.to_numeric(df['debt_to_equity'], errors='coerce')
        df.dropna(subset=['debt_to_equity'], inplace=True)
        plt.style.use('seaborn-v0_8-whitegrid')

        sns.set_theme(style="ticks")
        academic_palette = ["#E0E0E0", "#FFFFFF"]
        custom_order = ['Brazil', 'USA']

        # --- Image 1: Debt to Assets ---
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='country',
            y='debt_to_assets',
            data=df,
            order=custom_order,  # Brasil à esquerda
            palette=academic_palette,
            linecolor='black',
            width=0.5,
            fliersize=3
        )
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()

        fig1_path = os.path.join(self.visualization_dir, 'fig_debt_to_assets.png')
        plt.savefig(fig1_path, dpi=300)
        plt.clf()
        print(f"Salvo: {fig1_path}")

        # --- Image 2: Debt to Equity ---
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='country',
            y='debt_to_equity',
            data=df,
            order=custom_order,
            palette=academic_palette,
            linecolor='black',
            width=0.5,
            fliersize=3
        )
        plt.ylim(bottom=df['debt_to_equity'].min() - 0.1, top=df['debt_to_equity'].quantile(0.95) * 1.2)
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()

        fig2_path = os.path.join(self.visualization_dir, 'fig_debt_to_equity.png')
        plt.savefig(fig2_path, dpi=300)
        plt.clf()
        print(f"Salvo: {fig2_path}")

        # --- Image 3: ROA ---
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='country',
            y='roa',
            data=df,
            order=custom_order,
            palette=academic_palette,
            linecolor='black',
            width=0.5,
            fliersize=3
        )
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()

        fig3_path = os.path.join(self.visualization_dir, 'fig_roa.png')
        plt.savefig(fig3_path, dpi=300)
        plt.clf()
        print(f"Salvo: {fig3_path}")

        # --- Figure 1: Comparative Box Plots ---
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
        fig1.suptitle('Figure 1: Comparative Distribution of Key Financial Ratios (Brazil vs. USA)', fontsize=16)
        sns.boxplot(x='country', y='debt_to_assets', data=df, ax=axes1[0], palette="Set2")
        axes1[0].set_title('Leverage (Debt-to-Assets)')
        sns.boxplot(x='country', y='debt_to_equity', data=df, ax=axes1[1], palette="Set2")
        axes1[1].set_title('Leverage (Debt-to-Equity)')
        axes1[1].set_ylim(bottom=df['debt_to_equity'].min() - 0.1, top=df['debt_to_equity'].quantile(0.95) * 1.2)
        sns.boxplot(x='country', y='roa', data=df, ax=axes1[2], palette="Set2")
        axes1[2].set_title('Profitability (Return on Assets)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1_path = os.path.join(self.visualization_dir, 'figure1_comparative_boxplots.png')
        plt.savefig(fig1_path)
        plt.clf()
        print(f"Figure 1 saved to: {fig1_path}")

        # --- Figure 2: Evolution of Average Leverage ---
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        avg_leverage_annual = df.groupby(['year', 'country'])['debt_to_assets'].mean().unstack()
        avg_leverage_annual = avg_leverage_annual[['Brazil', 'USA']]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(
            avg_leverage_annual.index,
            avg_leverage_annual['Brazil'],
            color='black',
            linestyle='-',
            linewidth=2,
            label='Brazil'
        )

        ax2.plot(
            avg_leverage_annual.index,
            avg_leverage_annual['USA'],
            color='#404040',
            linestyle='--',
            linewidth=2,
            label='USA'
        )

        ax2.set_title('')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(frameon=False, loc='best')
        ax2.grid(axis='y', linestyle=':', alpha=0.4, color='gray')

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()

        fig2_path = os.path.join(self.visualization_dir, 'leverage_time_series_annual.png')
        plt.savefig(fig2_path, dpi=300)
        plt.clf()
        print(f"Figure 2 saved: {fig2_path}")

        # --- Figure 3: Profitability vs. Leverage Relationship ---
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        scatter_params = {'alpha': 0.3, 'color': 'gray', 's': 20, 'edgecolor': 'none'}
        line_params = {'color': 'black', 'linewidth': 1.5}
        sns.regplot(
            x='roa',
            y='debt_to_assets',
            data=df[df['country'] == 'Brazil'],
            ax=axes3[0],
            scatter_kws=scatter_params,
            line_kws=line_params,
            truncate=False
        )
        axes3[0].set_title('Brazil', fontsize=14)
        sns.regplot(
            x='roa',
            y='debt_to_assets',
            data=df[df['country'] == 'USA'],
            ax=axes3[1],
            scatter_kws=scatter_params,
            line_kws=line_params,
            truncate=False
        )
        axes3[1].set_title('USA', fontsize=14)
        for ax in axes3:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

        fig3_path = os.path.join(self.visualization_dir, 'pot_scatter_plots.png')
        plt.savefig(fig3_path, dpi=300)
        plt.clf()
        print(f"Figure 3 Saved: {fig3_path}")

        # --- Figure 4: Debt-to-Equity Distribution ---
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=df, x='debt_to_equity', hue='country', fill=True, common_norm=False, palette="viridis",
                    alpha=0.5)
        ax4.set_title('Figure 4: Distribution of Debt-to-Equity Ratio', fontsize=16)
        ax4.set_xlabel('Debt-to-Equity Ratio')
        ax4.set_ylabel('Density')
        ax4.set_xlim(left=df['debt_to_equity'].min(), right=df['debt_to_equity'].quantile(0.95))
        plt.tight_layout()
        fig4_path = os.path.join(self.visualization_dir, 'figure4_d2e_density_plot.png')
        plt.savefig(fig4_path)
        plt.close('all')
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.kdeplot(
            data=df[df['country'] == 'Brazil'],
            x='debt_to_equity',
            color='black',
            fill=True,
            alpha=0.1,
            label='Brazil',
            ax=ax4,
            linestyle='-'
        )
        sns.kdeplot(
            data=df[df['country'] == 'USA'],
            x='debt_to_equity',
            color='#404040',
            fill=False,
            label='USA',
            ax=ax4,
            linestyle='--'
        )
        ax4.set_title('')
        ax4.set_xlabel('')
        ax4.set_ylabel('')
        ax4.set_xlim(left=df['debt_to_equity'].min(), right=df['debt_to_equity'].quantile(0.95))
        ax4.legend(frameon=False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.tight_layout()
        fig4_path = os.path.join(self.visualization_dir, 'd2e_density_plot.png')
        plt.savefig(fig4_path, dpi=300)
        plt.close('all')
        print(f"Figure 4 Saved: {fig4_path}")

    def generate_stationarity_tables(self, stationarity_df, removal_df):
        """
        Generates and saves summary tables of the stationarity process.
        """
        if stationarity_df is None or removal_df is None:
            print("Stationarity DataFrames not available. Skipping table generation.")
            return

        print("Generating stationarity summary tables...")

        # --- Table 5: Summary of Transformation Process ---
        original_series = stationarity_df[stationarity_df['Level'] == 'Original']

        total_series_br = len(
            original_series[original_series['Country'] == 'Brazil'][['Company', 'Variable']].drop_duplicates())
        total_series_us = len(
            original_series[original_series['Country'] == 'USA'][['Company', 'Variable']].drop_duplicates())

        non_stationary_br = \
        original_series[(original_series['Country'] == 'Brazil') & (original_series['Is_Stationary'] == False)].shape[0]
        non_stationary_us = \
        original_series[(original_series['Country'] == 'USA') & (original_series['Is_Stationary'] == False)].shape[0]

        stationary_after_transform_br = \
        removal_df[(removal_df['Country'] == 'Brazil') & (removal_df['Final_Stationary'] == True)].shape[0]
        stationary_after_transform_us = \
        removal_df[(removal_df['Country'] == 'USA') & (removal_df['Final_Stationary'] == True)].shape[0]

        summary_data = {
            'Country': ['Brazil', 'USA'],
            'Total Series Tested': [total_series_br, total_series_us],
            'Non-Stationary at Original Level': [non_stationary_br, non_stationary_us],
            'Stationary After Transformation': [stationary_after_transform_br, stationary_after_transform_us]
        }
        df_summary = pd.DataFrame(summary_data)

        table5_path = os.path.join(self.visualization_dir, 'table5_stationarity_summary.md')
        with open(table5_path, 'w') as f:
            f.write("Table 5: Summary of the Stationarity Transformation Process\n\n")
            f.write(df_summary.to_markdown(index=False))
        print(f"Table 5 saved to: {table5_path}")

        # --- Table 6: Removed Series ---
        removed_series = removal_df[removal_df['Action'].str.contains("Removed")].copy()
        if not removed_series.empty:
            removed_table = removed_series[['Country', 'Company', 'Variable']].copy()
            removed_table['Reason for Removal'] = 'Non-stationary after transformations'

            table6_path = os.path.join(self.visualization_dir, 'table6_series_removed.md')
            with open(table6_path, 'w') as f:
                f.write("Table 6: Series Removed from the Final Sample\n\n")
                f.write(removed_table.to_markdown(index=False))
            print(f"Table 6 saved to: {table6_path}")
        else:
            print("No series were removed. Table 6 will not be generated.")

    def run_visualization(self):
        """
        Executes the complete visualization pipeline.
        """
        print("Starting visualization generation...")
        data_dict = self.load_data()

        if data_dict:
            self.generate_exploratory_plots(data_dict['final_data'])
            self.generate_stationarity_tables(data_dict['stationarity_detailed'], data_dict['removal_log'])

        print("Visualization process completed.")
