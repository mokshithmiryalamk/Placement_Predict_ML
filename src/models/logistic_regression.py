import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_binary_classification():
    """PART A: Binary Logistic Regression with Decision Boundary Plotting"""
    print("\n" + "=" * 60)
    print("--- PART A: BINARY LOGISTIC REGRESSION (PLACEMENT PREDICTION) ---")
    print("=" * 60)

    # 1. Generate Synthetic Binary Classification Dataset
    X_bin, y_bin = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_classes=2
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42
    )

    # 2. Train Binary Logistic Regression Model
    bin_model = LogisticRegression()
    bin_model.fit(X_train, y_train)

    # 3. Evaluate Binary Model
    y_pred = bin_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Binary Classification Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 4. Visualize Decision Boundary
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(8, 6))

    x_min, x_max = X_bin[:, 0].min() - 1, X_bin[:, 0].max() + 1
    y_min, y_max = X_bin[:, 1].min() - 1, X_bin[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = bin_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, edgecolors='k', cmap=plt.cm.coolwarm, alpha=0.8
    )
    plt.title("Binary Logistic Regression Decision Boundary", fontsize=13)
    plt.xlabel("Feature 1 (e.g., CGPA)")
    plt.ylabel("Feature 2 (e.g., Aptitude Score)")
    plt.tight_layout()

    out_path = "reports/figures/logistic_binary_boundary.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"-> Saved binary decision boundary plot to {out_path}")


def run_multiclass_classification():
    """PART B: Multiclass Logistic Regression (Multinomial vs OvR)"""
    print("\n" + "=" * 60)
    print("--- PART B: MULTICLASS LOGISTIC REGRESSION (BRANCH PREDICTION) ---")
    print("=" * 60)

    # 1. Generate Synthetic Multiclass Dataset (3 classes: e.g., CSE, ECE, ME)
    X_multi, y_multi = make_blobs(
        n_samples=1500, n_features=2, centers=3, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )

    # 2. Multinomial Softmax Model
    multi_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    multi_model.fit(X_train, y_train)
    y_multi_pred = multi_model.predict(X_test)
    multi_acc = accuracy_score(y_test, y_multi_pred)

    # 3. One-vs-Rest (OvR) Model
    ovr_model = LogisticRegression(multi_class='ovr', solver='lbfgs')
    ovr_model.fit(X_train, y_train)
    y_ovr_pred = ovr_model.predict(X_test)
    ovr_acc = accuracy_score(y_test, y_ovr_pred)

    print(f"Multinomial (Softmax) Test Accuracy: {multi_acc * 100:.2f}%")
    print(f"One-vs-Rest (OvR) Test Accuracy:      {ovr_acc * 100:.2f}%\n")

    print("Multinomial Classification Report:\n", classification_report(y_test, y_multi_pred))

    # 4. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_multi_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix - Multiclass Branch Prediction", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    out_path = "reports/figures/logistic_multiclass_cm.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"-> Saved multiclass confusion matrix to {out_path}")


if __name__ == "__main__":
    run_binary_classification()
    run_multiclass_classification()