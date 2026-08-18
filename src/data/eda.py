import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the script's own directory is in the path
# so that 'from ingest import ...' works regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest import load_and_validate_data


def perform_eda():
    """
    Executes full exploratory data analysis:
    - Structural profiling
    - Statistical summaries
    - Data visualization
    - Outlier detection
    - Class imbalance analysis
    """

    # ──────────────────────────────────────────────
    # 1. LOAD DATA USING THE INGESTION PIPELINE
    # ──────────────────────────────────────────────
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")
    df = load_and_validate_data(DATA_PATH)

    # ──────────────────────────────────────────────
    # 2. DATASET DIMENSIONS
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows (Samples): {df.shape[0]}")
    print(f"Total Columns (Metrics): {df.shape[1]}")

    # ──────────────────────────────────────────────
    # 3. FEATURE NAMES & DATA TYPES
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")
    print(df.dtypes)

    # ──────────────────────────────────────────────
    # 4. MISSING VALUES & DUPLICATES
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("--- 3. MISSING VALUES & DUPLICATES ---")
    missing_vals = df.isnull().sum()
    if missing_vals.sum() > 0:
        print("Missing Values per Column:\n", missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")

    duplicates = df.duplicated().sum()
    print(f"Duplicate Records Count: {duplicates}")

    # ──────────────────────────────────────────────
    # 5. SUMMARY STATISTICS (NUMERICAL FEATURES)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("--- 4. SUMMARY STATISTICS (NUMERICAL FEATURES) ---")
    print(df.describe())

    # ──────────────────────────────────────────────
    # 6. CLASS IMBALANCE ANALYSIS
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")
    if 'placement_status' in df.columns:
        class_counts = df['placement_status'].value_counts()
        class_percentages = df['placement_status'].value_counts(normalize=True) * 100
        print("Placement Status Counts:\n", class_counts)
        print("\nPlacement Status Percentages:\n", class_percentages)
    else:
        print("Target column 'placement_status' not found for class imbalance analysis.")

    # ──────────────────────────────────────────────
    # 7. VISUALIZATIONS & OUTLIER ANALYSIS
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("--- 6. GENERATING VISUALIZATIONS ---")

    # Set plot styling
    sns.set_theme(style="whitegrid")
    os.makedirs("reports/figures", exist_ok=True)

    # ── A. Correlation Matrix Heatmap ──
    plt.figure(figsize=(14, 10))
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Feature Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()
    print("-> Saved correlation heatmap to reports/figures/correlation_heatmap.png")

    # ── B. Scatter Plot (CGPA vs Salary) ──
    if 'cgpa' in df.columns and 'salary_package_lpa' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x='cgpa',
            y='salary_package_lpa',
            hue='placement_status',
            palette='Set1',
            alpha=0.7
        )
        plt.title("CGPA vs Salary Package (Colored by Placement Status)")
        plt.tight_layout()
        plt.savefig("reports/figures/scatter_cgpa_salary.png")
        plt.close()
        print("-> Saved scatter plot to reports/figures/scatter_cgpa_salary.png")

    # ── C. Pair Plot for Numerical Relationships ──
    try:
        # Using actual column names from YOUR dataset
        pairplot_cols = [
            'cgpa',
            'backlogs',
            'communication_skill_score',
            'internships_count',
            'salary_package_lpa'
        ]
        valid_pair_cols = [col for col in pairplot_cols if col in df.columns]
        if len(valid_pair_cols) > 1:
            pp = sns.pairplot(
                df[valid_pair_cols],
                diag_kind='kde',
                corner=True
            )
            pp.fig.suptitle(
                "Pairwise Relationships of Key Numerical Features",
                y=1.02
            )
            pp.savefig("reports/figures/pairplot_features.png")
            plt.close()
            print("-> Saved pair plot to reports/figures/pairplot_features.png")
    except Exception as e:
        print(f"Skipping pair plot due to layout constraints: {e}")

    # ── D. Outlier Detection via Boxplots ──
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=numerical_df,
        orient="h",
        palette="pastel"
    )
    plt.title("Outlier Identification via Boxplots (Numerical Features)")
    plt.tight_layout()
    plt.savefig("reports/figures/outliers_boxplot.png")
    plt.close()
    print("-> Saved outlier boxplot to reports/figures/outliers_boxplot.png")

    print("\nEDA Execution Complete.")
    print("Visualizations stored in 'reports/figures/'.")


if __name__ == "__main__":
    perform_eda()