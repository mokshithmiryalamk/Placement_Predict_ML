import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ──────────────────────────────────────────────
# PATH SETUP: Allow imports from src/data/
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():
    """
    Trains a Linear Regression model using the Standard Least Squares
    (Normal Equations) closed-form analytical solution.

    Configuration:
        L = 2 input dimensions (CGPA, Communication Skill Score)
        M = 1 output dimension (Salary Package LPA)
        P = 1 polynomial order (Linear)

    Solution: w = (Xᵀ X)⁻¹ Xᵀ y
    """

    # ──────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")
    df = load_and_validate_data(DATA_PATH)

    # ──────────────────────────────────────────────
    # 2. FILTER: Only placed students (salary > 0)
    # ──────────────────────────────────────────────
    # Rationale: Unplaced students have salary = 0, which would
    # heavily bias the regression plane downward. We train the
    # regressor only on students who were actually placed.
    df_clean = df[df['salary_package_lpa'] > 0].copy()

    # Also drop any potential NaN values
    df_clean = df_clean.dropna(
        subset=['cgpa', 'communication_skill_score', 'salary_package_lpa']
    )

    # ──────────────────────────────────────────────
    # 3. DEFINE INPUT (L=2) AND OUTPUT (M=1)
    # ──────────────────────────────────────────────
    feature_cols = ['cgpa', 'communication_skill_score']
    target_col = 'salary_package_lpa'

    X_raw = df_clean[feature_cols].values
    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]
    print(
        f"Loaded {N} data points with input dimension L = {X_raw.shape[1]} "
        f"and output dimension M = {y.shape[1]}"
    )

    # ──────────────────────────────────────────────
    # 4. DESIGN MATRIX (Add bias column of ones)
    # ──────────────────────────────────────────────
    # X_design = [1, x1, x2]  ← shape (N, 3)
    X_design = np.hstack([np.ones((N, 1)), X_raw])

    # ──────────────────────────────────────────────
    # 5. COMPUTE OPTIMAL WEIGHTS (Normal Equations)
    # ──────────────────────────────────────────────
    # w = (Xᵀ X)⁻¹ Xᵀ y
    XT_X = np.dot(X_design.T, X_design)

    try:
        XT_X_inv = np.linalg.inv(XT_X)
    except np.linalg.LinAlgError:
        # Fallback: Use Moore-Penrose pseudo-inverse if singular
        XT_X_inv = np.linalg.pinv(XT_X)

    XT_y = np.dot(X_design.T, y)
    w_optimal = np.dot(XT_X_inv, XT_y)

    print("\n--- Optimal Model Parameters (Weights & Bias) ---")
    print(f"Intercept (w0):                    {w_optimal[0, 0]:.4f}")
    print(f"Coefficient for {feature_cols[0]} (w1):        {w_optimal[1, 0]:.4f}")
    print(f"Coefficient for {feature_cols[1]} (w2): {w_optimal[2, 0]:.4f}")

    # ──────────────────────────────────────────────
    # 6. COMPUTE ERROR FUNCTION (E_w)
    # ──────────────────────────────────────────────
    y_pred = np.dot(X_design, w_optimal)
    E_w = 0.5 * np.sum((y_pred - y) ** 2)
    mse = np.mean((y_pred - y) ** 2)
    rmse = np.sqrt(mse)

    print(f"\nMinimized Error (E_w):    {E_w:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root MSE (RMSE):          {rmse:.4f} LPA")

    # ──────────────────────────────────────────────
    # 7. VISUALIZATION: 3D REGRESSION PLANE
    # ──────────────────────────────────────────────
    os.makedirs("reports/figures", exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')

    # For visual clarity, sample a subset (plotting 100k points is too dense)
    sample_size = min(2000, N)
    idx = np.random.choice(N, sample_size, replace=False)

    ax.scatter(
        X_raw[idx, 0],
        X_raw[idx, 1],
        y[idx, 0],
        color='blue',
        alpha=0.4,
        s=15,
        label='Actual Data Points (Sampled)'
    )

    # ── Regression Plane Meshgrid ──
    x1_surf = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 30)
    x2_surf = np.linspace(X_raw[:, 1].min(), X_raw[:, 1].max(), 30)
    x1_mesh, x2_mesh = np.meshgrid(x1_surf, x2_surf)

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        color='red',
        alpha=0.4,
        edgecolor='none'
    )

    ax.set_xlabel('CGPA (Feature 1)')
    ax.set_ylabel('Communication Skill Score (Feature 2)')
    ax.set_zlabel('Salary Package LPA (Target)')
    ax.set_title(
        'Linear Regression via Standard Least Squares\n'
        '(P=1, L=2, M=1) — Placed Students Only'
    )
    ax.legend()

    plt.tight_layout()
    output_path = "reports/figures/linear_regression_3d_plane.png"
    plt.savefig(output_path, dpi=100)
    plt.close()

    print(f"\n-> Successfully saved 3D regression plot to {output_path}")


if __name__ == "__main__":
    train_linear_regression_ls()