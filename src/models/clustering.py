from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Establish project root and relative paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Feature subset for clustering (unsupervised pattern discovery)
FEATURES = [
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]


def load_and_scale_data():
    """Loads dataset, extracts tier digits, and applies Standard Scaling."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()

    # Extract numeric value if tier is string (e.g., 'Tier 1' -> 1)
    if 'college_tier' in df.columns and not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = df['college_tier'].astype(str).str.extract(r'(\d+)')

    df_subset = df[FEATURES].apply(pd.to_numeric, errors='coerce').dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_subset)
    return scaled_data


def compute_clustering_scores(X, labels):
    """Calculates Silhouette Score and Davies-Bouldin Index safely."""
    non_noise_mask = labels != -1
    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]
    
    n_clusters = len(set(labels_clean))
    if n_clusters < 2:
        return np.nan, np.nan, n_clusters

    sample_sz = min(len(X_clean), 2000)
    sil = round(silhouette_score(X_clean, labels_clean, sample_size=sample_sz, random_state=42), 4)
    db = round(davies_bouldin_score(X_clean, labels_clean), 4)
    return sil, db, n_clusters


def run_experiment_12():
    print("=" * 65)
    print("--- EXPERIMENT 12: UNSUPERVISED CLUSTERING & PATTERN DISCOVERY ---")
    print("=" * 65)

    print("Loading and scaling placement features...")
    X_full = load_and_scale_data()
    n_samples = len(X_full)
    print(f"Dataset loaded with {n_samples} records across {X_full.shape[1]} features.")

    # 1. Evaluate K-Means across K = 2 to 10
    print("\nEvaluating K-Means over K = 2..10...")
    sample_sz = min(n_samples, 2000)
    ks = range(2, 11)
    inertias = []
    silhouettes = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_full)
        inertias.append(km.inertia_)
        sil_val = silhouette_score(X_full, labels, sample_size=sample_sz, random_state=42)
        silhouettes.append(sil_val)
        print(f"  Finished K = {k:<2} | Silhouette Score: {sil_val:.4f} | Inertia: {km.inertia_:.1f}")

    best_k = list(ks)[int(np.argmax(silhouettes))]
    print(f"\nOptimal K selected by highest Silhouette Score: K = {best_k}")

    # Plot 1: K-Means Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inertias, marker='o', color='#2b5c8f', linewidth=2)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=11)
    plt.title('Elbow Curve — Placement Dataset', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'clustering_kmeans_elbow.png', dpi=150)
    plt.close()
    print("-> Saved plot: reports/figures/clustering_kmeans_elbow.png")

    # Plot 2: K-Means Silhouette Curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), silhouettes, marker='o', color='#d95f02', linewidth=2)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Silhouette Score', fontsize=11)
    plt.title('Silhouette Analysis — Placement Dataset', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'clustering_kmeans_silhouette.png', dpi=150)
    plt.close()
    print("-> Saved plot: reports/figures/clustering_kmeans_silhouette.png")

    # 2. Algorithm Comparisons
    print("\nRunning algorithm benchmark (K-Means vs. Agglomerative vs. DBSCAN)...")
    
    # Subsample data if dataset is huge to prevent memory exhaustion on O(N^2) Hierarchical/DBSCAN matrix
    eval_sample_sz = min(n_samples, 5000)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_samples, eval_sample_sz, replace=False)
    X_eval = X_full[sample_idx]

    # Algorithm A: K-Means
    km_labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_eval)
    sil_km, db_km, c_km = compute_clustering_scores(X_eval, km_labels)

    # Algorithm B: Agglomerative Hierarchical
    ag_labels = AgglomerativeClustering(n_clusters=best_k, linkage='ward').fit_predict(X_eval)
    sil_ag, db_ag, c_ag = compute_clustering_scores(X_eval, ag_labels)

    # Plot 3: Hierarchical Dendrogram
    print("Generating Hierarchical Dendrogram...")
    dendro_sz = min(eval_sample_sz, 1000)
    dendro_sample = X_eval[:dendro_sz]
    Z = linkage(dendro_sample, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Dendrogram — Placement Dataset', fontsize=13)
    plt.xlabel('Cluster / Sample Index', fontsize=11)
    plt.ylabel('Euclidean Distance', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / 'hierarchical_dendrogram.png', dpi=150)
    plt.close()
    print("-> Saved plot: reports/figures/hierarchical_dendrogram.png")

    # Algorithm C: DBSCAN
    print("Running DBSCAN...")
    db_model = DBSCAN(eps=0.8, min_samples=5).fit(X_eval)
    db_labels = db_model.labels_
    sil_db, db_db, c_db = compute_clustering_scores(X_eval, db_labels)
    noise_points = int(np.sum(db_labels == -1))

    # Summary Table Output
    comparison_table = pd.DataFrame([
        ['K-Means', c_km, sil_km, db_km, 0],
        ['Agglomerative', c_ag, sil_ag, db_ag, 0],
        ['DBSCAN', c_db, sil_db, db_db, noise_points]
    ], columns=['Algorithm', 'Clusters', 'Silhouette_Score', 'Davies_Bouldin_Index', 'Noise_Points'])

    print("\n" + "=" * 65)
    print("CLUSTERING ALGORITHM BENCHMARK COMPARISON:")
    print("=" * 65)
    print(comparison_table.to_string(index=False))
    
    comparison_table.to_csv(OUT / 'clustering_comparison.csv', index=False)
    print(f"\n-> Saved comparison summary to: reports/figures/clustering_comparison.csv")


if __name__ == '__main__':
    run_experiment_12()