from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

# Suppress harmless deprecation and feature-name warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Features and Target
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
    """Load and preprocess placement dataset cleanly."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    df = df[F + [T]].dropna()

    # Convert Tier strings like 'Tier 1' -> 1
    if not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = (
            df['college_tier']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(df[F], columns=['branch'], dtype=float)

    # Encode binary target
    y = df[T]
    if not pd.api.types.is_numeric_dtype(y):
        lab = sorted(y.astype(str).unique())
        if len(lab) != 2:
            raise ValueError('placement_status must be binary')
        y = y.astype(str).map({lab[0]: 0, lab[1]: 1})

    return X, y


def shap_report(model, X_sample, feature_names, name):
    """
    Generate SHAP feature importance CSV and summary plot.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    # Handle different SHAP output formats across versions
    if isinstance(sv, list):
        vals = sv[1] if len(sv) > 1 else sv[0]
    elif len(getattr(sv, 'shape', ())) == 3:
        vals = sv[:, :, 1]
    else:
        vals = sv

    # Feature importance table
    imp = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': np.abs(vals).mean(axis=0)
    }).sort_values('Mean_Absolute_SHAP', ascending=False)

    imp.to_csv(OUT / f'{name}_shap_feature_importance.csv', index=False)

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(vals, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(OUT / f'{name}_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTop features ({name}):\n")
    print(imp.head(10).to_string(index=False))


def run():
    print("=" * 60)
    print("--- EXPERIMENT 8: XGBOOST vs LIGHTGBM + SHAP ---")
    print("=" * 60)

    X_df, y_df = load()
    feature_names = list(X_df.columns)

    Xtr, Xte, ytr, yte = train_test_split(
        X_df, y_df, test_size=0.2, stratify=y_df, random_state=42
    )

    rows = []

    # 1. XGBOOST
    print(f"\n{'=' * 40}")
    print("TRAINING XGBOOST...")
    print(f"{'=' * 40}")
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    t0 = time.perf_counter()
    xgb.fit(Xtr, ytr)
    sec_xgb = round(time.perf_counter() - t0, 4)

    pred_xgb = xgb.predict(Xte)
    acc_xgb = round(accuracy_score(yte, pred_xgb), 4)

    print(f"XGBOOST | Accuracy: {acc_xgb} | Training Time: {sec_xgb}s")
    print(classification_report(yte, pred_xgb))

    Xte_sample = Xte.sample(n=min(1000, len(Xte)), random_state=42)
    shap_report(xgb, Xte_sample, feature_names, 'xgboost')
    rows.append(['xgboost', acc_xgb, sec_xgb])

    # 2. LIGHTGBM (Windows Memory & Thread Safe)
    print(f"\n{'=' * 40}")
    print("TRAINING LIGHTGBM...")
    print(f"{'=' * 40}")

    lgb = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
        n_jobs=1
    )

    # Format arrays explicitly to C-contiguous float64 to prevent Windows C-DLL crashes
    Xtr_np = np.ascontiguousarray(Xtr.values, dtype=np.float64)
    ytr_np = np.ascontiguousarray(ytr.values, dtype=np.float64).ravel()
    Xte_np = np.ascontiguousarray(Xte.values, dtype=np.float64)

    t0 = time.perf_counter()
    lgb.fit(Xtr_np, ytr_np)
    sec_lgb = round(time.perf_counter() - t0, 4)

    pred_lgb = lgb.predict(Xte_np)
    acc_lgb = round(accuracy_score(yte, pred_lgb), 4)

    print(f"LIGHTGBM | Accuracy: {acc_lgb} | Training Time: {sec_lgb}s")
    print(classification_report(yte, pred_lgb))

    sample_idx = np.random.choice(len(Xte_np), min(1000, len(Xte_np)), replace=False)
    shap_report(lgb, Xte_np[sample_idx], feature_names, 'lightgbm')
    rows.append(['lightgbm', acc_lgb, sec_lgb])

    # Save Comparison Summary
    pd.DataFrame(
        rows,
        columns=['Model', 'Accuracy', 'Training_Time_Seconds']
    ).to_csv(OUT / 'xgb_lightgbm_comparison.csv', index=False)

    print("\n-> Saved comparison CSV to reports/figures/xgb_lightgbm_comparison.csv")


if __name__ == '__main__':
    run()