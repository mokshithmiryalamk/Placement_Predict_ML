from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set relative paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

F = [
    'branch',
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]
T = 'placement_status'


def load():
    """Loads and cleans dataset for Random Forest training."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    df = df[F + [T]].dropna()

    # Convert college_tier string to numeric if needed
    if not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = (
            df['college_tier']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(df[F], columns=['branch'], dtype=float)

    # Map binary target
    y = df[T]
    if not pd.api.types.is_numeric_dtype(y):
        lab = sorted(y.astype(str).unique())
        y = y.astype(str).map({lab[0]: 0, lab[1]: 1})

    return X, y


def score(m, X, y):
    p = m.predict(X)
    return [
        round(accuracy_score(y, p), 4),
        round(precision_score(y, p, zero_division=0), 4),
        round(recall_score(y, p, zero_division=0), 4),
        round(f1_score(y, p, zero_division=0), 4)
    ]


def run():
    print("=" * 60)
    print("--- EXPERIMENT 7: RANDOM FOREST & ENSEMBLE LEARNING ---")
    print("=" * 60)

    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 1. Single Decision Tree vs Random Forest Baseline
    dt = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
    print('Decision Tree [Accuracy, Precision, Recall, F1]:', score(dt, Xte, yte))

    rf = RandomForestClassifier(
        n_estimators=100, max_features='sqrt', oob_score=True, random_state=42, n_jobs=-1
    ).fit(Xtr, ytr)
    print('Random Forest [Accuracy, Precision, Recall, F1]:', score(rf, Xte, yte))
    print(f'OOB Score: {round(rf.oob_score_, 4)} | OOB Error: {round(1 - rf.oob_score_, 4)}')

    # 2. Effect of Number of Trees
    print("\n--- Evaluating Number of Trees (N_estimators) ---")
    counts = [10, 25, 50, 100, 200]
    oob = []
    acc = []
    for n in counts:
        m = RandomForestClassifier(
            n_estimators=n, max_features='sqrt', oob_score=True, random_state=42, n_jobs=-1
        ).fit(Xtr, ytr)
        oob.append(1 - m.oob_score_)
        acc.append(accuracy_score(yte, m.predict(Xte)))

    plt.figure(figsize=(9, 6))
    plt.plot(counts, oob, marker='o', label='OOB Error', color='red')
    plt.plot(counts, acc, marker='o', label='Test Accuracy', color='blue')
    plt.xlabel('Number of Trees', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Effect of Number of Trees — Placement Prediction', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'random_forest_number_of_trees.png', dpi=150)
    plt.close()
    print("-> Saved plot to reports/figures/random_forest_number_of_trees.png")

    # 3. Effect of Feature Subsampling (max_features)
    print("\n--- Evaluating Feature Subsampling Strategies ---")
    opts = ['sqrt', 'log2', None]
    labs = ['sqrt', 'log2', 'all features']
    vals = []
    for o in opts:
        m = RandomForestClassifier(
            n_estimators=100, max_features=o, random_state=42, n_jobs=-1
        ).fit(Xtr, ytr)
        vals.append(accuracy_score(yte, m.predict(Xte)))

    print('Feature Subsampling Accuracy:', {k: round(v, 4) for k, v in zip(labs, vals)})

    plt.figure(figsize=(8, 6))
    plt.bar(labs, vals, color=['#2b5c8f', '#4682b4', '#70a1d7'])
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Feature Subsampling Strategies — Placement Prediction', fontsize=13)
    plt.ylim(0, 1.0)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / 'random_forest_feature_subsampling.png', dpi=150)
    plt.close()
    print("-> Saved plot to reports/figures/random_forest_feature_subsampling.png")


if __name__ == '__main__':
    run()