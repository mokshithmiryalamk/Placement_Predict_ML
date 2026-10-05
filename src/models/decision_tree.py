from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Set relative paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'branch',
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]
TARGET = 'placement_status'


def load_data():
    """Loads and preprocesses placement dataset for Decision Tree training."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    df = df[FEATURES + [TARGET]].dropna()

    # Convert college_tier (e.g., 'Tier 1' -> 1)
    if not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = (
            df['college_tier']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

    # One-hot encode categorical features (branch)
    X = pd.get_dummies(df[FEATURES], columns=['branch'], dtype=float)

    # Encode binary target variable (Placed -> 1, Not Placed -> 0)
    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):
        labels = sorted(y.astype(str).unique())
        if len(labels) != 2:
            raise ValueError('placement_status must be binary')
        y = y.astype(str).map({labels[0]: 0, labels[1]: 1})

    return X, y


def run_decision_tree_experiment():
    """Executes Decision Tree training, visualization, and pruning analysis."""
    print("=" * 60)
    print("--- EXPERIMENT 6: DECISION TREE CLASSIFICATION & PRUNING ---")
    print("=" * 60)

    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ──────────────────────────────────────────────
    # 1. Gini vs. Entropy Splitting Criteria
    # ──────────────────────────────────────────────
    for name, criterion in [('gini', 'gini'), ('entropy', 'entropy')]:
        model = DecisionTreeClassifier(criterion=criterion, random_state=42)
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"Decision Tree ({name.upper()}) Test Accuracy: {acc:.4f}")

        # Plot Tree Visualization
        plt.figure(figsize=(18, 10))
        plot_tree(
            model,
            feature_names=X.columns.tolist(),
            class_names=[str(x) for x in sorted(y.unique())],
            filled=True,
            max_depth=4,
            fontsize=7
        )
        plt.title(f'Placement Decision Tree - {name.title()}', fontsize=14)
        plt.tight_layout()
        plot_path = OUT / f'decision_tree_{name}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"-> Saved tree diagram to {plot_path}")

    # ──────────────────────────────────────────────
    # 2. Cost Complexity Pruning (CCP) Analysis
    # ──────────────────────────────────────────────
    print("\n--- Running Cost Complexity Pruning (CCP) ---")
    path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_tr, y_tr)
    
    # Subsample alphas if path is too large for fast plotting
    alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 50)]
    
    rows = []
    for a in alphas:
        m = DecisionTreeClassifier(ccp_alpha=a, random_state=42).fit(X_tr, y_tr)
        rows.append([
            a,
            accuracy_score(y_tr, m.predict(X_tr)),
            accuracy_score(y_te, m.predict(X_te)),
            m.tree_.node_count
        ])

    pr = pd.DataFrame(rows, columns=['alpha', 'train_accuracy', 'test_accuracy', 'nodes'])
    best = pr.loc[pr.test_accuracy.idxmax()]

    print(f"Optimal CCP Alpha:   {best.alpha:.6f}")
    print(f"Pruned Test Accuracy: {best.test_accuracy:.4f}")
    print(f"Pruned Node Count:    {int(best.nodes)}")

    plt.figure(figsize=(9, 6))
    plt.plot(pr.alpha, pr.train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(pr.alpha, pr.test_accuracy, label='Testing Accuracy', color='red', linestyle='--')
    plt.xlabel('CCP Alpha (α)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Cost Complexity Pruning Path — Placement Prediction', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'decision_tree_ccp.png', dpi=150)
    plt.close()
    print("-> Saved CCP pruning path plot to reports/figures/decision_tree_ccp.png")

    # ──────────────────────────────────────────────
    # 3. Maximum Depth Tuning
    # ──────────────────────────────────────────────
    print("\n--- Analyzing Effect of Tree Depth ---")
    depths = range(1, 16)
    tr, te = [], []

    for depth in depths:
        m = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_tr, y_tr)
        tr.append(accuracy_score(y_tr, m.predict(X_tr)))
        te.append(accuracy_score(y_te, m.predict(X_te)))

    best_depth = list(depths)[int(np.argmax(te))]
    print(f"Optimal Max Depth:  {best_depth}")
    print(f"Best Test Accuracy: {max(te):.4f}")

    plt.figure(figsize=(9, 6))
    plt.plot(depths, tr, marker='o', label='Training Accuracy', color='blue')
    plt.plot(depths, te, marker='o', label='Testing Accuracy', color='red', linestyle='--')
    plt.xlabel('Maximum Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Effect of Tree Depth — Placement Prediction', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'decision_tree_depth.png', dpi=150)
    plt.close()
    print("-> Saved tree depth plot to reports/figures/decision_tree_depth.png")


if __name__ == '__main__':
    run_decision_tree_experiment()