from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

# Set relative paths
ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / 'src' / 'data' / 'input_image.jpg'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def run():
    print("=" * 65)
    print("--- EXPERIMENT 11: K-MEANS IMAGE SEGMENTATION ---")
    print("=" * 65)

    if not IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found at '{IMAGE}'. "
            "Please place a JPG image at 'src/data/input_image.jpg'."
        )

    # 1. Load and preprocess image
    print(f"Loading image from: {IMAGE}")
    img = Image.open(IMAGE).convert('RGB').resize((300, 300))
    arr = np.array(img)
    
    # Reshape (300, 300, 3) image array into (90000, 3) matrix of RGB pixels
    X = arr.reshape(-1, 3).astype(float)

    ks = [2, 4, 6, 8]
    inertias = []
    segs = []

    # 2. Run K-Means for each K value
    for k in ks:
        print(f"Clustering image into K = {k} color segments...")
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)

        # Reconstruct segmented image using RGB cluster centroids
        seg = model.cluster_centers_[labels].reshape(arr.shape).astype('uint8')
        inertias.append(model.inertia_)
        segs.append(seg)

        # Save individual segmented output
        Image.fromarray(seg).save(OUT / f'segmented_k{k}.png')

    # 3. Side-by-side segmentation comparison plot
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(arr)
    ax[0].set_title('Original Image', fontsize=12)
    ax[0].axis('off')

    for a, k, s in zip(ax[1:], ks, segs):
        a.imshow(s)
        a.set_title(f'K = {k}', fontsize=12)
        a.axis('off')

    plt.tight_layout()
    comparison_path = OUT / 'image_segmentation_comparison.png'
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    print(f"-> Saved comparison plot to: {comparison_path}")

    # 4. Inertia / Elbow Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o', color='#2b5c8f', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
    plt.title('Elbow Curve — Image Segmentation', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    elbow_path = OUT / 'image_segmentation_elbow.png'
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"-> Saved elbow plot to: {elbow_path}")

    print("\nSegmentation completed successfully!")


if __name__ == '__main__':
    run()