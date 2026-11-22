# data_utils.py

import anndata as sc  # Only AnnData and read_h5ad are used by this trainer.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from typing import Dict, Any, Tuple

UNKNOWN_FUNCTION_ID = -1
UNKNOWN_FUNCTION_NAMES = frozenset({"unknow function", "unknown function"})


def normalize_function_name(function_name: Any) -> str:
    """Normalizes a scalar function label for robust comparisons."""
    if pd.isna(function_name):
        return ""
    return str(function_name).strip().casefold()


def is_unknown_function_name(function_name: Any) -> bool:
    """Returns whether a scalar label denotes the open-set Unknown function."""
    return normalize_function_name(function_name) in UNKNOWN_FUNCTION_NAMES


def get_unknown_function_mask(functions: pd.Series) -> pd.Series:
    """Returns a Boolean mask for all supported spellings/cases of Unknown function."""
    normalized = functions.astype("string").str.strip().str.casefold()
    return normalized.isin(UNKNOWN_FUNCTION_NAMES)


def _as_numpy_bool_mask(mask: pd.Series) -> np.ndarray:
    """Converts Pandas/extension Boolean masks for compatibility with older AnnData."""
    return np.asarray(mask.fillna(False), dtype=bool)


def _format_distribution(values: pd.Series, column_name: str) -> str:
    """Formats counts and percentages without truncating categories."""
    if values.empty:
        return "(none)"

    display_values = values.astype("string").str.strip().fillna("<missing>")
    display_values = display_values.mask(display_values.eq(""), "<empty>")
    counts = display_values.value_counts(dropna=False)
    distribution = counts.rename("count").to_frame()
    distribution["percentage"] = (distribution["count"] / len(display_values) * 100).round(2)
    distribution.index.name = column_name
    return distribution.reset_index().to_string(index=False)


def build_function_distribution_report(raw_adata: sc.AnnData, dataset_name: str) -> str:
    """Builds a pre-training function report, including Unknown's fea_ca makeup."""
    if "function" not in raw_adata.obs.columns:
        raise KeyError("raw_adata.obs must contain a 'function' column.")
    if "fea_ca" not in raw_adata.obs.columns:
        raise KeyError("raw_adata.obs must contain a 'fea_ca' column.")

    functions = raw_adata.obs["function"]
    normalized = functions.astype("string").str.strip().str.casefold()
    unknown_mask = get_unknown_function_mask(functions)
    missing_mask = functions.isna() | normalized.eq("")
    known_mask = ~(unknown_mask | missing_mask)

    total_count = len(functions)
    unknown_count = int(unknown_mask.sum())
    known_count = int(known_mask.sum())
    missing_count = int(missing_mask.sum())
    known_unique_count = int(functions[known_mask].nunique(dropna=True))

    lines = [
        f"=== Function distribution: {dataset_name} ===",
        f"Total samples: {total_count}",
        f"Known function samples: {known_count}",
        f"Unknown function samples: {unknown_count}",
        f"Missing/empty function samples: {missing_count}",
        f"Known function classes: {known_unique_count}",
        "Function label distribution (percentage of dataset):",
        _format_distribution(functions, "function"),
        "Unknown subset fea_ca distribution (percentage of Unknown subset):",
        _format_distribution(raw_adata.obs.loc[unknown_mask, "fea_ca"], "fea_ca"),
    ]
    return "\n".join(lines)


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
    
    # 1. Separate samples kept for training/validation from all remaining samples.
    raw_function = raw.obs['function']
    normalized_function = raw_function.astype("string").str.strip().str.casefold()
    unknown_function_mask = get_unknown_function_mask(raw_function)
    # Keep labeled functions, controls, and the explicit unknown-function class.
    train_val_idx = _as_numpy_bool_mask(
        normalized_function.str.contains("-", regex=False, na=False)
        | normalized_function.str.contains("control", regex=False, na=False)
        | normalized_function.str.contains("ctrl", regex=False, na=False)
        | unknown_function_mask
    )
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

    idx_where = _as_numpy_bool_mask(label.isin(label_count.index))
    raw = raw[idx_where]
    scfound = scfound[idx_where]
    geneformer = geneformer[idx_where]
    print(f"Final filtered training/validation data count: {raw.shape[0]}")

    return scfound, geneformer, raw, test_rest_scfound, test_rest_geneformer, test_rest_raw

def create_label_maps(raw_adata: sc.AnnData) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Creates label maps while excluding open-set Unknown from function targets."""
    unique_label = raw_adata.obs['fea_ca'].unique()
    label_map = {label: i for i, label in enumerate(unique_label)}

    functions = raw_adata.obs['function']
    unique_func = functions[~get_unknown_function_mask(functions)].unique()
    func_map = {func: i for i, func in enumerate(unique_func)}
    
    return label_map, func_map

def filter_ood_data(raw_out: sc.AnnData, scfound_out: np.ndarray, geneformer_out: np.ndarray, func_map: Dict[str, int]) -> Tuple[sc.AnnData, np.ndarray, np.ndarray]:
    """Keeps OOD samples with known function labels or the open-set Unknown label."""
    raw_out_function = raw_out.obs["function"]
    idx_where = _as_numpy_bool_mask(
        raw_out_function.isin(func_map.keys()) | get_unknown_function_mask(raw_out_function)
    )
    
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
        # Convert once here instead of copying/converting every sample in __getitem__.
        self.sc_features = torch.as_tensor(sc_features, dtype=torch.float32)
        self.gene_features = torch.as_tensor(gene_features, dtype=torch.float32)
        self.raw_expression = torch.as_tensor(raw_expression, dtype=torch.float32)
        self.label_map = label_map
        self.func_map = func_map
        self.mode = mode

        # OOD labels are intentionally unused and may not exist in label_map.
        self.labels = (
            self._encode_labels(labels)
            if labels is not None and mode in ("train", "test")
            else None
        )
        self.functions = self._encode_functions(functions) if functions is not None else None

    def _encode_labels(self, labels) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            return labels.to(dtype=torch.long)

        values = labels.to_numpy() if isinstance(labels, pd.Series) else np.asarray(labels)
        encoded = [self.label_map[value] if isinstance(value, (str, np.str_)) else int(value) for value in values]
        return torch.as_tensor(encoded, dtype=torch.long)

    def _encode_functions(self, functions) -> torch.Tensor:
        if isinstance(functions, torch.Tensor):
            return functions.to(dtype=torch.long)

        values = functions.to_numpy() if isinstance(functions, pd.Series) else np.asarray(functions)
        encoded = []
        for function_name in values:
            if is_unknown_function_name(function_name):
                encoded.append(UNKNOWN_FUNCTION_ID)
            else:
                try:
                    encoded.append(self.func_map[function_name])
                except KeyError as exc:
                    raise KeyError(
                        f"Function label {function_name!r} is neither in func_map nor recognized as Unknown."
                    ) from exc
        return torch.as_tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.sc_features.shape[0]

    def __getitem__(self, idx):
        sc = self.sc_features[idx]
        gene = self.gene_features[idx]
        raw_data = self.raw_expression[idx]

        if self.mode == "inference":
            return sc, gene, raw_data

        func_tensor = self.functions[idx]
        if self.mode in ("train", "test"):
            return sc, gene, raw_data, self.labels[idx], func_tensor
        return sc, gene, raw_data, func_tensor
