"""Visualization helpers for anomaly analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_reconstruction_error(errors: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "reconstruction_error_hist.png"
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, color="#4c72b0", alpha=0.8)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_feature_space(raw: np.ndarray, corrected: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_space_pca.png"
    reducer = PCA(n_components=2, random_state=42)
    raw_2d = reducer.fit_transform(raw)
    corrected_2d = reducer.transform(corrected)
    plt.figure(figsize=(7, 5))
    plt.scatter(raw_2d[:, 0], raw_2d[:, 1], s=8, alpha=0.5, label="Raw")
    plt.scatter(
        corrected_2d[:, 0],
        corrected_2d[:, 1],
        s=8,
        alpha=0.5,
        label="Corrected",
    )
    plt.title("PCA Feature Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_downstream_auc(auc_raw: float, auc_corrected: float, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "downstream_auc.png"
    plt.figure(figsize=(5, 4))
    plt.bar(["Raw", "Corrected"], [auc_raw, auc_corrected], color=["#c44e52", "#55a868"])
    plt.title("Downstream ROC-AUC")
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
