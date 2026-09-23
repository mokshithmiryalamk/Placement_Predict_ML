import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_ridge_experiment():
    """
    Demonstrates overfitting control using Ridge (L2) Regularization
    on high-degree polynomial features (Degree 15).
    """
    print("=" * 60)
    print("--- EXPERIMENT 6: RIDGE (L2) REGULARIZATION PATH ---")
    print("=" * 60)

    # 1. Simulate/Load Feature Data (CGPA vs Performance/Salary Curve)
    np.random.seed(42)
    X = np.sort(6 * np.random.rand(100, 1) + 4)  # CGPA between 4.0 and 10.0
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # Non-linear target + noise

    # 2. Split dataset into 80% Training and 20% Testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 3. Transform features to Degree 15 Polynomial (to induce heavy overfitting)
    poly_degree = 15
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 4. Scale features (crucial so regularization penalizes all weights equally)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # 5. Define range of lambda (alpha) values across 8 orders of magnitude
    lambdas = np.logspace(-4, 4, 200)
    train_errors = []
    test_errors = []

    # 6. Loop over each lambda, fit Ridge model, record MSE
    for lam in lambdas:
        ridge = Ridge(alpha=lam)
        ridge.fit(X_train_scaled, y_train)

        y_train_pred = ridge.predict(X_train_scaled)
        y_test_pred = ridge.predict(X_test_scaled)

        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    # Identify optimal lambda (minimizes test error)
    min_test_idx = np.argmin(test_errors)
    optimal_lambda = lambdas[min_test_idx]
    optimal_test_mse = test_errors[min_test_idx]

    print(f"Polynomial Degree: {poly_degree}")
    print(f"Optimal Lambda (λ / alpha): {optimal_lambda:.4f}")
    print(f"Minimum Test MSE:            {optimal_test_mse:.4f}")

    # 7. Plotting the Error Curves
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, train_errors, label='Training Error ($E_w$)', color='blue', linewidth=2)
    plt.plot(lambdas, test_errors, label='Testing Error ($E_w$)', color='red', linewidth=2, linestyle='--')
    plt.axvline(optimal_lambda, color='green', linestyle=':', label=f'Optimal λ ({optimal_lambda:.2f})')

    plt.xscale('log')
    plt.xlabel('Regularization Parameter (λ / alpha)', fontsize=12)
    plt.ylabel('Mean Squared Error ($E_w$)', fontsize=12)
    plt.title('Regularization Path: Ridge Regression Overfitting Control (Degree 15)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    output_path = "reports/figures/ridge_regularization_path.png"
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"\n-> Saved regularization path plot to {output_path}")


if __name__ == "__main__":
    run_ridge_experiment()