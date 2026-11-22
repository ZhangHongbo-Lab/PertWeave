# utils.py

import numpy as np
import torch
import random
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
from umap import UMAP
from sklearn.manifold import TSNE
from typing import Dict, Any

def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_to_pickle(obj: Any, path: str):
    """Saves an object to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_from_pickle(path: str) -> Any:
    """Loads an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_tsne_umap_by_group(features: np.ndarray, labels: np.ndarray, save_path: str, num2label_map: Dict[int, str]):
    """Generates and saves TSNE/UMAP plots, highlighting one group at a time."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Plotting TSNE/UMAP for {len(num2label_map)} groups in {save_path}")

    # Generate TSNE and UMAP embeddings once
    tsne_embedding = TSNE(n_components=2, random_state=42).fit_transform(features)
    umap_embedding = UMAP(n_components=2, random_state=42).fit_transform(features)

    embeddings = {"tsne": tsne_embedding, "umap": umap_embedding}

    for plot_type, embedding in embeddings.items():
        plot_dir = os.path.join(save_path, plot_type)
        os.makedirs(plot_dir, exist_ok=True)
        unique_labels = sorted(num2label_map.keys())

        for label_id in unique_labels:
            group_name = num2label_map[label_id]
            idx = labels == label_id
            non_idx = labels != label_id

            plt.figure(figsize=(10, 10), dpi=200)
            plt.scatter(embedding[non_idx, 0], embedding[non_idx, 1], label="others", color="gray", s=2, alpha=0.5)
            plt.scatter(embedding[idx, 0], embedding[idx, 1], label=group_name, color="red", s=3)
            plt.legend()
            plt.title(f"{plot_type.upper()} - Group: {group_name}")
            plt.savefig(os.path.join(plot_dir, f"{plot_type}_{group_name.replace('/', '_')}.png"))
            plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, label_map: Dict[str, int]):
    """Calculates and plots confusion matrix."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    num_classes = len(label_map)
    labels = list(label_map.keys())
    
    cm_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
    confusion_matrix = cm_metric(torch.tensor(y_pred), torch.tensor(y_true)).numpy()

    plt.figure(figsize=(max(10, num_classes // 3), max(10, num_classes // 3)), dpi=200)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def write_log(filepath: str, content: str, mode: str = "a"):
    """Writes content to a log file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode) as f:
        f.write(content + "\n")