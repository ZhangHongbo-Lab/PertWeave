"""Independent components for additive Unknown-aware metric experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


UNKNOWN_FUNCTION_ID = -1


def _to_numpy(values, dtype=None):
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    elif hasattr(values, "to_numpy"):
        array = values.to_numpy()
    else:
        array = np.asarray(values)
    return np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)


def build_unknown_category_filter_mask(
    fine_labels,
    unknown_mask,
    min_samples: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep every known cell and Unknown classes with at least ``min_samples``.

    The filter is intended to run before train/test splitting.  The threshold
    is inclusive; use 101 when the scientific criterion is strictly greater
    than 100.  A non-positive value disables this additive filter.
    """
    fine = _to_numpy(fine_labels).reshape(-1)
    unknown = _to_numpy(unknown_mask, bool).reshape(-1)
    if fine.shape != unknown.shape:
        raise ValueError("fine_labels and unknown_mask must have equal shape.")
    threshold = int(min_samples)
    if threshold < 0:
        raise ValueError("min_samples must be non-negative.")

    unknown_values, unknown_counts = np.unique(fine[unknown], return_counts=True)
    count_map = {
        value.item() if isinstance(value, np.generic) else value: int(count)
        for value, count in zip(unknown_values, unknown_counts)
    }
    retained_values = {
        value for value, count in count_map.items()
        if threshold <= 0 or count >= threshold
    }
    keep = ~unknown
    if unknown.any():
        keep = keep | np.isin(fine, list(retained_values))

    rows = []
    for value, count in sorted(count_map.items(), key=lambda item: str(item[0])):
        retained = value in retained_values
        rows.append({
            "class_name": str(value),
            "num_cells": int(count),
            "retained": bool(retained),
        })
    report = {
        "threshold": threshold,
        "threshold_semantics": "inclusive_greater_or_equal",
        "enabled": threshold > 0,
        "known_cells_retained": int((~unknown).sum()),
        "unknown_cells_before": int(unknown.sum()),
        "unknown_cells_after": int((keep & unknown).sum()),
        "unknown_classes_before": int(len(count_map)),
        "unknown_classes_after": int(len(retained_values)),
        "unknown_classes_removed": int(len(count_map) - len(retained_values)),
        "classes": rows,
    }
    return keep.astype(bool, copy=False), report


@dataclass(frozen=True)
class UnifiedGeometryLabelEncoder:
    """Map known samples to function IDs and Unknown samples to fea_ca IDs."""

    known_function_count: int
    fine_class_count: int
    unknown_fine_ids: tuple[int, ...]
    unknown_function_id: int = UNKNOWN_FUNCTION_ID

    @classmethod
    def fit(
        cls,
        fine_labels,
        function_labels,
        fine_class_count: int | None = None,
        unknown_function_id: int = UNKNOWN_FUNCTION_ID,
    ) -> "UnifiedGeometryLabelEncoder":
        fine = _to_numpy(fine_labels, np.int64).reshape(-1)
        function = _to_numpy(function_labels, np.int64).reshape(-1)
        if fine.shape != function.shape:
            raise ValueError("fine_labels and function_labels must have equal shape.")
        if fine.size == 0:
            raise ValueError("Cannot fit a unified label encoder on an empty dataset.")
        if np.any(fine < 0):
            raise ValueError("Fine labels must be non-negative encoded IDs.")

        known = function != int(unknown_function_id)
        if not np.any(known):
            raise ValueError("At least one known-function sample is required.")
        if np.any(function[known] < 0):
            raise ValueError("Known function IDs must be non-negative.")
        known_ids = np.unique(function[known])
        expected = np.arange(int(known_ids.max()) + 1, dtype=np.int64)
        if not np.array_equal(known_ids, expected):
            raise ValueError(
                "Known function IDs must be contiguous from zero; got "
                f"{known_ids.tolist()}."
            )
        unknown_ids = tuple(int(value) for value in np.unique(fine[~known]))
        if not unknown_ids:
            raise ValueError("At least one Unknown-function fine class is required.")
        inferred_fine_count = int(fine.max()) + 1
        final_fine_count = inferred_fine_count if fine_class_count is None else int(fine_class_count)
        if final_fine_count < inferred_fine_count:
            raise ValueError("fine_class_count is smaller than the largest fine label.")
        return cls(
            known_function_count=len(known_ids),
            fine_class_count=final_fine_count,
            unknown_fine_ids=unknown_ids,
            unknown_function_id=int(unknown_function_id),
        )

    @property
    def num_classes(self) -> int:
        return self.known_function_count + len(self.unknown_fine_ids)

    @property
    def unknown_offset(self) -> int:
        return self.known_function_count

    def _lookup_numpy(self) -> np.ndarray:
        lookup = np.full(self.fine_class_count, -1, dtype=np.int64)
        for local_id, fine_id in enumerate(self.unknown_fine_ids):
            lookup[fine_id] = self.unknown_offset + local_id
        return lookup

    def encode_numpy(self, fine_labels, function_labels) -> np.ndarray:
        fine = _to_numpy(fine_labels, np.int64).reshape(-1)
        function = _to_numpy(function_labels, np.int64).reshape(-1)
        if fine.shape != function.shape:
            raise ValueError("fine_labels and function_labels must have equal shape.")
        if np.any((fine < 0) | (fine >= self.fine_class_count)):
            raise ValueError("Fine label is outside the fitted encoder range.")
        encoded = function.copy()
        unknown = function == self.unknown_function_id
        if np.any(unknown):
            encoded[unknown] = self._lookup_numpy()[fine[unknown]]
        if np.any(encoded < 0) or np.any(encoded >= self.num_classes):
            raise ValueError("Encountered an unknown fine class or invalid function ID.")
        return encoded

    def encode_torch(self, fine_labels: torch.Tensor, function_labels: torch.Tensor) -> torch.Tensor:
        if fine_labels.shape != function_labels.shape:
            raise ValueError("fine_labels and function_labels must have equal shape.")
        fine = fine_labels.to(dtype=torch.long)
        function = function_labels.to(dtype=torch.long)
        lookup = torch.as_tensor(
            self._lookup_numpy(), dtype=torch.long, device=fine.device
        )
        if torch.any((fine < 0) | (fine >= len(lookup))):
            raise ValueError("Fine label is outside the fitted encoder range.")
        unknown = function.eq(self.unknown_function_id)
        encoded = torch.where(unknown, lookup[fine], function)
        if torch.any((encoded < 0) | (encoded >= self.num_classes)):
            raise ValueError("Encountered an unknown fine class or invalid function ID.")
        return encoded

    def state_dict(self) -> dict[str, Any]:
        return {
            "known_function_count": self.known_function_count,
            "fine_class_count": self.fine_class_count,
            "unknown_fine_ids": list(self.unknown_fine_ids),
            "unknown_function_id": self.unknown_function_id,
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "UnifiedGeometryLabelEncoder":
        return cls(
            known_function_count=int(state["known_function_count"]),
            fine_class_count=int(state["fine_class_count"]),
            unknown_fine_ids=tuple(int(v) for v in state["unknown_fine_ids"]),
            unknown_function_id=int(state.get("unknown_function_id", UNKNOWN_FUNCTION_ID)),
        )

    def names(self, num2func: dict[int, str], num2label: dict[int, str]) -> dict[int, str]:
        names = {
            int(function_id): f"function::{name}"
            for function_id, name in num2func.items()
        }
        for local_id, fine_id in enumerate(self.unknown_fine_ids):
            names[self.unknown_offset + local_id] = (
                f"unknown_fea_ca::{num2label[int(fine_id)]}"
            )
        return dict(sorted(names.items()))


class UnknownPKBatchSampler(Sampler[list[int]]):
    """Reserve P × K positions for Unknown cells and fill the rest from the train split."""

    def __init__(
        self,
        fine_labels,
        function_labels,
        batch_size: int,
        categories_per_batch: int,
        samples_per_category: int,
        unknown_function_id: int = UNKNOWN_FUNCTION_ID,
        seed: int = 0,
    ):
        self.fine_labels = _to_numpy(fine_labels, np.int64).reshape(-1)
        self.function_labels = _to_numpy(function_labels, np.int64).reshape(-1)
        if len(self.fine_labels) != len(self.function_labels):
            raise ValueError("fine_labels and function_labels must have equal length.")
        if min(batch_size, categories_per_batch, samples_per_category) < 1:
            raise ValueError("Batch and P x K sizes must be positive.")
        if categories_per_batch * samples_per_category > batch_size:
            raise ValueError("P x K cannot exceed batch_size.")

        self.batch_size = int(batch_size)
        self.categories_per_batch = int(categories_per_batch)
        self.samples_per_category = int(samples_per_category)
        self.unknown_function_id = int(unknown_function_id)
        self.seed = int(seed)
        self.epoch = 0
        self.all_indices = np.arange(len(self.fine_labels), dtype=np.int64)
        self.batch_count = math.ceil(len(self.all_indices) / self.batch_size)

        category_to_indices: dict[int, list[int]] = defaultdict(list)
        for index in np.flatnonzero(self.function_labels == self.unknown_function_id):
            category_to_indices[int(self.fine_labels[index])].append(int(index))
        self.category_to_indices = {
            category: np.asarray(indices, dtype=np.int64)
            for category, indices in category_to_indices.items()
        }
        self.unknown_categories = np.asarray(sorted(self.category_to_indices), dtype=np.int64)
        if len(self.unknown_categories) == 0:
            raise ValueError("UnknownPKBatchSampler found no Unknown samples.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batch_count

    def _draw_any(self, candidates: np.ndarray, rng: np.random.Generator) -> list[int]:
        return rng.choice(
            candidates,
            size=self.samples_per_category,
            replace=len(candidates) < self.samples_per_category,
        ).astype(np.int64).tolist()

    def _draw_filler(self, selected: list[int], count: int, rng: np.random.Generator) -> list[int]:
        if count <= 0:
            return []
        selected_set = set(selected)
        available = np.asarray(
            [index for index in self.all_indices if int(index) not in selected_set],
            dtype=np.int64,
        )
        if len(available) >= count:
            return rng.choice(available, size=count, replace=False).astype(np.int64).tolist()
        filler = available.tolist()
        filler.extend(
            rng.choice(
                self.all_indices, size=count - len(filler), replace=True
            ).astype(np.int64).tolist()
        )
        return filler

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        replace_categories = len(self.unknown_categories) < self.categories_per_batch
        for _ in range(self.batch_count):
            categories = rng.choice(
                self.unknown_categories,
                size=self.categories_per_batch,
                replace=replace_categories,
            )
            selected: list[int] = []
            for category in categories:
                candidates = self.category_to_indices[int(category)]
                selected.extend(self._draw_any(candidates, rng))
            selected.extend(self._draw_filler(selected, self.batch_size - len(selected), rng))
            rng.shuffle(selected)
            yield selected


def _as_csr(matrix) -> sparse.csr_matrix:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(max(int(count), 0), len(values))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(values, kind="stable")[-count:].astype(np.int64)


class RawFoldPreprocessor:
    """Fold-local raw-count preprocessing without changing data_utils.py."""

    def __init__(
        self,
        mode: str,
        num_variance_features: int,
        num_mean_features: int,
        normalize_target_sum: float = 1.0e4,
        zscore_clip: float = 10.0,
    ):
        if mode != "legacy_counts":
            raise ValueError("This release uses legacy_counts.")
        if normalize_target_sum <= 0 or zscore_clip <= 0:
            raise ValueError("normalize_target_sum and zscore_clip must be positive.")
        self.mode = mode
        self.num_variance_features = int(num_variance_features)
        self.num_mean_features = int(num_mean_features)
        self.normalize_target_sum = float(normalize_target_sum)
        self.zscore_clip = float(zscore_clip)
        self.selected_indices: np.ndarray | None = None
        self.train_mean: np.ndarray | None = None
        self.train_std: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.selected_indices is not None

    def _select_features(self, prepared: sparse.csr_matrix, labels: np.ndarray) -> np.ndarray:
        labels = _to_numpy(labels, np.int64).reshape(-1)
        if len(labels) != prepared.shape[0]:
            raise ValueError("labels must align with raw-expression rows.")
        classes = np.unique(labels)
        class_means = np.empty((len(classes), prepared.shape[1]), dtype=np.float32)
        for row, class_id in enumerate(classes):
            class_means[row] = np.asarray(
                prepared[labels == class_id].mean(axis=0)
            ).reshape(-1)
        correction = 1 if len(classes) > 1 else 0
        class_variance = class_means.var(axis=0, ddof=correction)
        overall_mean = np.asarray(prepared.mean(axis=0)).reshape(-1)
        variance_indices = _top_indices(class_variance, self.num_variance_features)
        mean_indices = _top_indices(overall_mean, self.num_mean_features)
        combined = np.concatenate((variance_indices, mean_indices))
        return combined.astype(np.int64, copy=False)

    def fit(self, matrix, labels) -> "RawFoldPreprocessor":
        prepared = _as_csr(matrix)
        self.selected_indices = self._select_features(prepared, labels)
        if len(self.selected_indices) == 0:
            raise ValueError("Raw feature selection produced no columns.")
        self.train_mean = np.zeros(len(self.selected_indices), dtype=np.float32)
        self.train_std = np.ones(len(self.selected_indices), dtype=np.float32)
        return self

    def transform(self, matrix) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RawFoldPreprocessor must be fitted before transform().")
        prepared = _as_csr(matrix)
        values = prepared[:, self.selected_indices].toarray().astype(np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Raw preprocessing produced NaN or Inf.")
        return values

    def fit_transform(self, matrix, labels) -> np.ndarray:
        return self.fit(matrix, labels).transform(matrix)

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted RawFoldPreprocessor.")
        return {
            "mode": self.mode,
            "num_variance_features": self.num_variance_features,
            "num_mean_features": self.num_mean_features,
            "normalize_target_sum": self.normalize_target_sum,
            "zscore_clip": self.zscore_clip,
            "selected_indices": self.selected_indices.tolist(),
            "train_mean": self.train_mean.tolist(),
            "train_std": self.train_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "RawFoldPreprocessor":
        instance = cls(
            mode=state["mode"],
            num_variance_features=int(state["num_variance_features"]),
            num_mean_features=int(state["num_mean_features"]),
            normalize_target_sum=float(state["normalize_target_sum"]),
            zscore_clip=float(state["zscore_clip"]),
        )
        instance.selected_indices = np.asarray(state["selected_indices"], dtype=np.int64)
        instance.train_mean = np.asarray(state["train_mean"], dtype=np.float32)
        instance.train_std = np.asarray(state["train_std"], dtype=np.float32)
        return instance


@torch.no_grad()
def compute_unknown_clustering_metrics(
    features,
    fine_labels,
    function_labels,
    knn_k: int = 10,
    min_class_cells: int = 10,
    enrichment_threshold: float = 2.0,
    device: str | torch.device = "cpu",
    chunk_size: int = 256,
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    """Compute high-dimensional cosine-neighborhood metrics for Unknown only."""
    feats = torch.as_tensor(features, dtype=torch.float32, device=device)
    fine = torch.as_tensor(fine_labels, dtype=torch.long, device=device).reshape(-1)
    function = torch.as_tensor(function_labels, dtype=torch.long, device=device).reshape(-1)
    if feats.ndim != 2 or len(feats) != len(fine) or len(fine) != len(function):
        raise ValueError("features/fine_labels/function_labels are not row-aligned.")
    keep = function.eq(UNKNOWN_FUNCTION_ID)
    feats = feats[keep]
    fine = fine[keep]
    n = len(fine)
    if n < 2:
        return {"num_samples": int(n), "num_classes": int(torch.unique(fine).numel())}, []
    if not torch.isfinite(feats).all():
        raise FloatingPointError("Unknown metric features contain NaN or Inf.")
    k = min(int(knn_k), n - 1)
    if k < 1:
        return {"num_samples": int(n), "num_classes": int(torch.unique(fine).numel())}, []
    normalized = F.normalize(feats, dim=1)
    cell_purity = torch.empty(n, dtype=torch.float32, device=device)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        similarity = normalized[start:end] @ normalized.T
        row_indices = torch.arange(end - start, device=device)
        similarity[row_indices, torch.arange(start, end, device=device)] = float("-inf")
        neighbors = similarity.topk(k, dim=1).indices
        cell_purity[start:end] = (
            fine[neighbors].eq(fine[start:end, None]).float().mean(dim=1)
        )

    classes, counts = torch.unique(fine, sorted=True, return_counts=True)
    micro_random = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    rows: list[dict[str, float | int | bool]] = []
    eligible_enrichments: list[float] = []
    eligible_trends: list[bool] = []
    for class_id, count in zip(classes.tolist(), counts.tolist()):
        mask = fine.eq(int(class_id))
        purity = float(cell_purity[mask].mean())
        random_baseline = float((int(count) - 1) / (n - 1))
        enrichment = purity / random_baseline if random_baseline > 0 else float("nan")
        eligible = int(count) >= int(min_class_cells)
        trend = bool(eligible and np.isfinite(enrichment) and enrichment >= enrichment_threshold)
        if eligible:
            eligible_enrichments.append(float(enrichment))
            eligible_trends.append(trend)
        rows.append({
            "class_id": int(class_id),
            "num_cells": int(count),
            "knn_purity": purity,
            "random_baseline": random_baseline,
            "enrichment": float(enrichment),
            "eligible": eligible,
            "trend": trend,
        })

    report: dict[str, float | int] = {
        "num_samples": int(n),
        "num_classes": int(len(classes)),
        "eligible_classes": int(len(eligible_trends)),
        "knn_k": int(k),
        "micro_knn_purity": float(cell_purity.mean()),
        "micro_random_baseline": micro_random,
        "micro_enrichment": float(cell_purity.mean()) / micro_random if micro_random > 0 else float("nan"),
        "macro_knn_purity": float(np.mean([row["knn_purity"] for row in rows])),
        "median_class_enrichment": float(np.median(eligible_enrichments)) if eligible_enrichments else float("nan"),
        "trend_coverage": float(np.mean(eligible_trends)) if eligible_trends else 0.0,
    }
    return report, rows


def unknown_report_score(report: dict[str, Any]) -> tuple[float, float, float]:
    """Lexicographic checkpoint score: coverage, macro enrichment, purity."""
    return (
        float(report.get("trend_coverage", float("-inf"))),
        float(report.get("median_class_enrichment", float("-inf"))),
        float(report.get("micro_knn_purity", float("-inf"))),
    )


def classification_guardrail_pass(
    current: dict[str, float],
    baseline: dict[str, float] | None,
    max_drop: float = 0.05,
) -> tuple[bool, dict[str, float]]:
    """Check the four requested classification metrics against a baseline."""
    if baseline is None:
        return True, {}
    names = ("test_acc", "test_auc", "test_group_acc", "test_group_auc")
    drops: dict[str, float] = {}
    for name in names:
        if name not in current or name not in baseline:
            raise KeyError(f"Classification guardrail requires metric {name!r}.")
        drops[name] = float(baseline[name]) - float(current[name])
    # Decimal thresholds such as 0.05 are not represented exactly in binary;
    # preserve the stated inclusive boundary rather than rejecting 0.80-0.75.
    tolerance = 1e-12
    return all(
        drop <= float(max_drop) + tolerance for drop in drops.values()
    ), drops


