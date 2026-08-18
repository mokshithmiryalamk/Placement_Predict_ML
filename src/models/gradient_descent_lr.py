import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# ──────────────────────────────────────────────
# PATH SETUP: Allow imports from project root
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.data.ingest import load_and_validate_data


# ──────────────────────────────────────────────
# COST FUNCTION
# ──────────────────────────────────────────────
def compute_cost(X, y, w):
    """
    Computes the Mean Squared Error (MSE) cost function.

        E(w) = (1 / 2m) * sum((Xw - y)^2)
    """
    m = len(y)
    predictions = np.dot(X, w)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost


# ──────────────────────────────────────────────
# GRADIENT DESCENT OPTIMIZER
# ──────────────────────────────────────────────
def gradient_descent(X, y, w, alpha, num_iters):
    """
    Implements batch Gradient Descent from scratch.

        w := w - α * (1/m) * X^T (Xw - y)

    Returns:
        w              : Final learned weights
        cost_history   : Cost value at each iteration
    """
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        predictions = np.dot(X, w)
        errors = predictions - y

        # Gradient: (1/m) * X^T (Xw - y)
        gradient = (1 / m) * np.dot(X.T, errors)

        # Weight update rule
        w = w - alpha * gradient

        # Record cost for convergence analysis
        cost = compute_cost(X, y, w)
        cost_history.append(cost)

    return w, cost_history


# ──────────────────────────────────────────────
# MAIN EXPERIMENT
# ──────────────────────────────────────────────
def run_gradient_descent_experiment():
    """
    Full end-to-end pipeline:
      1. Load data
      2. Train/test split (80/20)
      3. Feature scaling
      4. Custom gradient descent with multiple learning rates
      5. Cost history visualization
      6. Comparison with scikit-learn
    """

    # ──────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")
    df = load_and_validate_data(DATA_PATH)

    # Filter to placed students only (salary > 0)
    df_clean = df[df['salary_package_lpa'] > 0].copy()

    feature_cols = ['cgpa']
    target_col = 'salary_package_lpa'

    df_clean = df_clean.dropna(subset=feature_cols + [target_col])

    X_raw = df_clean[feature_cols].values
    y_raw = df_clean[target_col].values.reshape(-1, 1)

    print(f"Total samples for training: {X_raw.shape[0]}")

    # ──────────────────────────────────────────────
    # 2. TRAIN/TEST SPLIT (80/20)
    # ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    # ──────────────────────────────────────────────
    # 3. FEATURE SCALING (Critical for GD stability)
    # ──────────────────────────────────────────────
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Add bias column (x0 = 1) → design matrix
    X_train_design = np.hstack(
        [np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    )
    X_test_design = np.hstack(
        [np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
    )

    # ──────────────────────────────────────────────
    # 4. LEARNING RATE EXPERIMENTATION
    # ──────────────────────────────────────────────
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    num_iterations = 1000

    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(12, 7))

    results = {}
    for alpha in learning_rates:
        # Initialize weights to zeros → shape [2, 1] (bias + 1 feature)
        w_init = np.zeros((X_train_design.shape[1], 1))

        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        results[alpha] = {
            'weights': w_opt,
            'history': cost_history,
            'final_cost': cost_history[-1]
        }

        # Plot cost convergence curve
        plt.plot(cost_history, label=f"α = {alpha}")

        print(f"α = {alpha:>6}  →  Final Cost = {cost_history[-1]:.6f}")

    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Cost Function E(w) — MSE", fontsize=12)
    plt.title(
        "Effect of Learning Rates on Gradient Descent Convergence",
        fontsize=14
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/gd_learning_rates_comparison.png", dpi=100)
    plt.close()
    print(
        "\n-> Saved learning rates cost graph to "
        "reports/figures/gd_learning_rates_comparison.png"
    )

    # ──────────────────────────────────────────────
    # 5. FINAL EVALUATION WITH BEST LEARNING RATE
    # ──────────────────────────────────────────────
    best_alpha = 0.1
    final_w = results[best_alpha]['weights']

    print("\n" + "=" * 60)
    print(f"--- CUSTOM GRADIENT DESCENT (α = {best_alpha}) ---")
    print(f"Intercept (w0):       {final_w[0, 0]:.6f}")
    print(f"Coefficient (w1):     {final_w[1, 0]:.6f}")

    # Compute train and test cost
    train_cost = compute_cost(X_train_design, y_train_scaled, final_w)
    test_cost = compute_cost(X_test_design, y_test_scaled, final_w)

    print(f"Training Cost (MSE):  {train_cost:.6f}")
    print(f"Testing Cost (MSE):   {test_cost:.6f}")

    # ──────────────────────────────────────────────
    # 6. COMPARISON WITH SCIKIT-LEARN
    # ──────────────────────────────────────────────
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train_scaled, y_train_scaled)

    print("\n" + "=" * 60)
    print("--- SCIKIT-LEARN LINEAR REGRESSION (Closed-Form) ---")
    print(f"Scikit-learn Intercept:   {sklearn_model.intercept_[0]:.6f}")
    print(f"Scikit-learn Coefficient: {sklearn_model.coef_[0, 0]:.6f}")

    # ──────────────────────────────────────────────
    # 7. PARAMETER PARITY CHECK
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("--- PARAMETER PARITY VERIFICATION ---")
    intercept_diff = abs(final_w[0, 0] - sklearn_model.intercept_[0])
    coef_diff = abs(final_w[1, 0] - sklearn_model.coef_[0, 0])
    print(f"Intercept Difference:   {intercept_diff:.6f}")
    print(f"Coefficient Difference: {coef_diff:.6f}")

    if intercept_diff < 0.01 and coef_diff < 0.01:
        print(
            "✅ SUCCESS: Custom Gradient Descent converged to the same "
            "solution as Scikit-learn's closed-form Normal Equations."
        )
    else:
        print(
            "⚠️  WARNING: Parameters differ. Consider more iterations "
            "or a different learning rate."
        )


if __name__ == "__main__":
    run_gradient_descent_experiment()