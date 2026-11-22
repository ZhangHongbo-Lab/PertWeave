# data_utils.py

import anndata as sc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from typing import Dict, Any, Tuple

# --- Data Loading ---

def load_features(scfound_path: str, geneformer_path: str, raw_path: str) -> Tuple[np.ndarray, np.ndarray, sc.AnnData]:
    """Loads scFoundation features, Geneformer features, and raw expression data."""
    print("Loading datasets...")
    scfound_features = np.load(scfound_path)
    with open(geneformer_path, "rb") as f:
        geneformer_features = pickle.load(f)
    raw_adata = sc.read_h5ad(raw_path)
    print(f"Loaded raw data shape: {raw_adata.shape}")
    print(f"Loaded scFoundation features shape: {scfound_features.shape}")
    print(f"Loaded Geneformer features shape: {geneformer_features.shape}")
    return scfound_features, geneformer_features, raw_adata

# --- Data Preprocessing and Filtering ---

def filter_data(
    scfound: np.ndarray, 
    geneformer: np.ndarray, 
    raw: sc.AnnData, 
    filter_least: int
) -> Tuple[np.ndarray, np.ndarray, sc.AnnData, np.ndarray, np.ndarray, sc.AnnData]:
    """Applies filtering logic to separate training data from test-rest data and filters by label frequency."""
    
    # 1. Separate "test_rest" data (unlabeled or complex labels like 'control', '-')
    raw_function = raw.obs['function']
    # Identify indices for training/validation set (those with clear function labels)
    train_val_idx = raw_function.str.contains("-") | raw_function.str.contains("control") | raw_function.str.contains("ctrl")
    test_rest_idx = ~train_val_idx

    # Create test_rest set
    test_rest_raw = raw[test_rest_idx]
    test_rest_scfound = scfound[test_rest_idx]
    test_rest_geneformer = geneformer[test_rest_idx]
    print(f"Separated test_rest data count: {test_rest_raw.shape[0]}")

    # Create main training/validation set
    raw = raw[train_val_idx]
    scfound = scfound[train_val_idx]
    geneformer = geneformer[train_val_idx]
    print(f"Initial training/validation data count: {raw.shape[0]}")

    # 2. Filter by minimum samples per class ('fea_ca')
    label = raw.obs['fea_ca']
    label_count = label.value_counts()
    label_count = label_count[label_count >= filter_least]
    print(f"Filtered label counts (>= {filter_least} samples):\n{label_count}")

    idx_where = label.isin(label_count.index)
    raw = raw[idx_where]
    scfound = scfound[idx_where]
    geneformer = geneformer[idx_where]
    print(f"Final filtered training/validation data count: {raw.shape[0]}")

    return scfound, geneformer, raw, test_rest_scfound, test_rest_geneformer, test_rest_raw

def create_label_maps(raw_adata: sc.AnnData) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Creates mappings from string labels to integer indices."""
    unique_label = raw_adata.obs['fea_ca'].unique()
    label_map = {label: i for i, label in enumerate(unique_label)}
    
    unique_func = raw_adata.obs['function'].unique()
    func_map = {func: i for i, func in enumerate(unique_func)}
    
    return label_map, func_map

def filter_ood_data(raw_out: sc.AnnData, scfound_out: np.ndarray, geneformer_out: np.ndarray, func_map: Dict[str, int]) -> Tuple[sc.AnnData, np.ndarray, np.ndarray]:
    """Filters OOD data to ensure its labels exist in the training function map."""
    raw_out_function = raw_out.obs["function"]
    idx_where = raw_out_function.isin(func_map.keys())
    
    raw_out = raw_out[idx_where]
    scfound_out = scfound_out[idx_where]
    geneformer_out = geneformer_out[idx_where]
    print(f"Filtered OOD data count based on function map: {raw_out.shape[0]}")
    return raw_out, scfound_out, geneformer_out

# --- Feature Engineering ---

def get_variance_features(raw_tensor: torch.Tensor, labels: pd.Series, num_features: int) -> torch.Tensor:
    """Select features based on variance across class means."""
    mean_per_class = torch.zeros(len(labels.unique()), raw_tensor.shape[1])
    for i in labels.unique():
        mean_per_class[i] = raw_tensor[labels == i].mean(axis=0)
    variance = mean_per_class.var(axis=0)
    indices = torch.argsort(variance)[-num_features:]
    return indices

def get_mean_features(raw_tensor: torch.Tensor, num_features: int) -> torch.Tensor:
    """Select features based on overall mean expression."""
    mean_expression = raw_tensor.mean(axis=0)
    indices = torch.argsort(mean_expression)[-num_features:]
    return indices

def select_features(raw_data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Sample columns from data tensor based on selected indices."""
    return raw_data[:, indices]

# --- PyTorch Dataset Class ---

class SimpleDataset(Dataset):
    def __init__(self, sc_features, gene_features, raw_expression, labels, functions, label_map, func_map, mode="train"):
        self.sc_features = sc_features
        self.gene_features = gene_features
        self.raw_expression = raw_expression # Already a tensor
        self.labels = labels # pd.Series or np.array
        self.functions = functions # pd.Series or np.array
        self.label_map = label_map
        self.func_map = func_map
        self.mode = mode

    def __len__(self):
        return len(self.sc_features)

    def __getitem__(self, idx):
        # sc = torch.tensor(self.sc_features[idx], dtype=torch.float32)
        # gene = torch.tensor(self.gene_features[idx], dtype=torch.float32)
        # raw_data = self.raw_expression[idx] # Assumes raw_expression is pre-tensored

        # if self.mode == "inference":
        #     return sc, gene, raw_data

        # func_name = self.functions.iloc[idx] if isinstance(self.functions, pd.Series) else self.functions[idx]
        # func_id = torch.tensor(self.func_map[func_name], dtype=torch.long)
        
        # if self.mode in ["train", "test", "out"]:
        #     label_id = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
        #     # If label_id is already an int (from label_map application) keep it, else map it.
        #     if isinstance(label_id, str):
        #         label_id = self.label_map[label_id]
        #     label_tensor = torch.tensor(label_id, dtype=torch.long)
            
        #     # 'out' mode might not have specific labels, but we structure it to return both.
        #     if self.mode == "out":
        #          return sc, gene, raw_data, func_id # Re-aligning with original logic for OOD data.
        #     else: # train/test
        #          return sc, gene, raw_data, label_tensor, func_id
        
        # Fallback for mode discrepancy, adjust as needed. Original code branches slightly differently.
        # Let's strictly follow original: out mode only returns func, not label.
        # Re-check original SimpleDataset: mode 'out' returns sc, gene, raw_data, func (4 items). 
        # Mode 'train'/'test' returns sc, gene, raw_data, label, func (5 items).
        
        # --- Revised __getitem__ based on strict interpretation of original code ---
        sc = torch.tensor(self.sc_features[idx], dtype=torch.float32)
        gene = torch.tensor(self.gene_features[idx], dtype=torch.float32)
        raw_data = self.raw_expression[idx]

        if self.mode == "inference":
            return sc, gene, raw_data

        func_name = self.functions.iloc[idx] if isinstance(self.functions, pd.Series) else self.functions[idx]
        func_tensor = torch.tensor(self.func_map[func_name], dtype=torch.long)

        if self.mode == "train" or self.mode == "test":
            label_val = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
            if isinstance(label_val, str):
                 label_tensor = torch.tensor(self.label_map[label_val], dtype=torch.long)
            else:
                 label_tensor = torch.tensor(label_val, dtype=torch.long) # Assumes it's already mapped int
            return sc, gene, raw_data, label_tensor, func_tensor
        else: # mode == "out"
            return sc, gene, raw_data, func_tensor
