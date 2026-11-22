"""Run the Unknown experiment with reusable preprocessing and fold recovery."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import train as trainer


CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_ROOT = "unknown_metric_processed_cache"
GLOBAL_FILES = {
    "scfound_all": "scfound_all.npy",
    "geneformer_all": "geneformer_all.npy",
    "raw_all": "raw_all.h5ad",
    "source_row_indices": "source_row_indices.npy",
    "scfound_rest": "scfound_rest.npy",
    "geneformer_rest": "geneformer_rest.npy",
    "raw_rest": "raw_rest.h5ad",
    "scfound_out": "scfound_out.npy",
    "geneformer_out": "geneformer_out.npy",
    "raw_out": "raw_out.h5ad",
    "manifest": "manifest.json",
    "complete": "COMPLETE",
}
FOLD_RAW_FILES = {
    "raw_train": "raw_train.npy",
    "raw_test": "raw_test.npy",
    "raw_out": "raw_out.npy",
    "raw_rest": "raw_rest.npy",
    "preprocessor": "raw_preprocessor.json",
    "manifest": "manifest.json",
    "complete": "COMPLETE",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _array_sha256(values: Any, dtype: np.dtype | str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8).tobytes())
    return digest.hexdigest()


def _resolved_path(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def _file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Cache source file does not exist: {target}")
    stat = target.stat()
    return {
        "path": str(target),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_key(specification: dict[str, Any], length: int = 16) -> str:
    return _sha256_text(_canonical_json(specification))[:length]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _required_files_exist(root: Path, names: dict[str, str]) -> bool:
    return root.is_dir() and all((root / name).is_file() for name in names.values())


def _safe_remove_cache_directory(target: Path, cache_root: Path) -> None:
    target_resolved = target.resolve()
    root_resolved = cache_root.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to remove cache path outside cache root: {target_resolved}"
        ) from exc
    if target_resolved == root_resolved:
        raise RuntimeError("Refusing to remove the cache root itself.")
    if target.exists():
        shutil.rmtree(target)


def _acquire_build_lock(
    lock_path: Path,
    cache_is_valid: Callable[[], bool],
    timeout_seconds: float = 6 * 60 * 60,
) -> int | None:
    """Acquire an exclusive cache-build lock, or return None after another build."""

    started = time.monotonic()
    last_notice = float("-inf")
    while True:
        try:
            descriptor = os.open(
                str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            if cache_is_valid():
                return None
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for cache lock {lock_path}. If the builder "
                    "crashed, remove this single lock file after confirming no cache "
                    "builder is still running."
                )
            if elapsed - last_notice >= 30:
                print(f"[cache] Waiting for another process: {lock_path}")
                last_notice = elapsed
            time.sleep(2.0)
            continue
        payload = f"pid={os.getpid()} created={_utc_now_iso()}\n".encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        return descriptor


def _ensure_directory_cache(
    target: Path,
    *,
    cache_root: Path,
    validate: Callable[[Path], bool],
    build: Callable[[Path], None],
    rebuild: bool = False,
) -> bool:
    """Ensure one atomic directory cache. Return True for hit, False for build."""

    cache_root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if validate(target) and not rebuild:
        return True
    lock_path = target.with_name(f".{target.name}.lock")
    descriptor = _acquire_build_lock(lock_path, lambda: validate(target) and not rebuild)
    if descriptor is None:
        return True
    try:
        if validate(target) and not rebuild:
            return True
        if target.exists():
            if not rebuild:
                raise RuntimeError(
                    f"Incomplete or incompatible cache exists at {target}. Rerun with "
                    "--rebuild-processed-cache true after confirming no other process "
                    "uses it."
                )
            _safe_remove_cache_directory(target, cache_root)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            build(temporary)
            (temporary / "COMPLETE").write_text(
                f"completed={_utc_now_iso()}\n", encoding="utf-8"
            )
            os.replace(str(temporary), str(target))
        except BaseException:
            if temporary.exists():
                _safe_remove_cache_directory(temporary, cache_root)
            raise
        return False
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _save_array(path: Path, values: Any) -> None:
    array = np.asarray(values)
    if array.dtype == object:
        raise TypeError(f"Refusing to cache object array at {path}.")
    np.save(path, array, allow_pickle=False)


def _copy_preprocessor_state(
    destination: trainer.RawFoldPreprocessor,
    state: dict[str, Any],
) -> None:
    source = trainer.RawFoldPreprocessor.from_state_dict(state)
    if destination.mode != source.mode:
        raise RuntimeError(
            f"Cached raw mode {source.mode!r} != requested mode {destination.mode!r}."
        )
    destination.selected_indices = source.selected_indices.copy()
    destination.train_mean = source.train_mean.copy()
    destination.train_std = source.train_std.copy()


class ProcessedDataCache:
    """Global filtered-data cache plus fold-local raw-expression caches."""

    def __init__(
        self,
        params: dict[str, Any],
        cache_root: str | os.PathLike[str],
        *,
        rebuild: bool = False,
    ):
        self.params = dict(params)
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.rebuild = bool(rebuild)
        self.paths = dict(params["input_paths"])
        self.main_path_tuple = tuple(
            _resolved_path(self.paths[key])
            for key in ("scfoundation", "geneformer", "raw_h5ad")
        )
        self.out_path_tuple = tuple(
            _resolved_path(self.paths[key])
            for key in ("scfoundation_out", "geneformer_out", "raw_out_h5ad")
        )
        self.specification = self._global_specification()
        self.key = _cache_key(self.specification)
        self.root = self.cache_root / f"global_{self.key}"
        self.manifest: dict[str, Any] = {}
        self.unknown_filter_report: dict[str, Any] = {}

        self.scfound_all = None
        self.geneformer_all = None
        self.raw_all = None
        self.source_row_indices = None
        self.scfound_rest = None
        self.geneformer_rest = None
        self.raw_rest = None
        self.scfound_out = None
        self.geneformer_out = None
        self.raw_out = None

        hit = _ensure_directory_cache(
            self.root,
            cache_root=self.cache_root,
            validate=self._validate_global_directory,
            build=self._build_global_directory,
            rebuild=self.rebuild,
        )
        print(f"[cache] Global {'HIT' if hit else 'BUILT'}: {self.root}")
        self._load_global_directory()

    def _global_specification(self) -> dict[str, Any]:
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "inputs": {
                key: _file_identity(self.paths[key])
                for key in (
                    "scfoundation",
                    "geneformer",
                    "raw_h5ad",
                    "scfoundation_out",
                    "geneformer_out",
                    "raw_out_h5ad",
                )
            },
            "filter_least": int(trainer.cfg.FILTER_LEAST),
            "unknown_min_samples": int(self.params["unknown_min_samples"]),
            "required_obs_columns": ["fea_ca", "function"],
            "unknown_function_names": sorted(trainer.du.UNKNOWN_FUNCTION_NAMES),
        }

    def _validate_global_directory(self, root: Path) -> bool:
        if not _required_files_exist(root, GLOBAL_FILES):
            return False
        try:
            manifest = _read_json(root / GLOBAL_FILES["manifest"])
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest.get("cache_format_version") == CACHE_FORMAT_VERSION
            and manifest.get("cache_key") == self.key
            and manifest.get("specification") == self.specification
        )

    def _build_global_directory(self, root: Path) -> None:
        print("[cache] MISS: reading and filtering original multimodal inputs once...")
        original_load = trainer.du.load_features
        original_filter = trainer.du.filter_data
        original_filter_ood = trainer.du.filter_ood_data

        scfound_full, geneformer_full, raw_full = original_load(
            self.paths["scfoundation"],
            self.paths["geneformer"],
            self.paths["raw_h5ad"],
        )
        source_column = trainer.SOURCE_ROW_INDEX_COLUMN
        if source_column in raw_full.obs.columns:
            raise ValueError(
                f"Source H5AD already contains reserved column {source_column!r}."
            )
        raw_full.obs[source_column] = np.arange(raw_full.n_obs, dtype=np.int64)
        (
            scfound_all,
            geneformer_all,
            raw_all,
            scfound_rest,
            geneformer_rest,
            raw_rest,
        ) = original_filter(
            scfound_full, geneformer_full, raw_full, trainer.cfg.FILTER_LEAST
        )
        unknown_function_mask = np.asarray(
            [
                trainer.du.is_unknown_function_name(value)
                for value in raw_all.obs["function"].to_numpy()
            ],
            dtype=bool,
        )
        unknown_keep_mask, unknown_report = trainer.build_unknown_category_filter_mask(
            raw_all.obs["fea_ca"].astype(str).to_numpy(),
            unknown_function_mask,
            min_samples=int(self.params["unknown_min_samples"]),
        )
        if unknown_report["unknown_classes_after"] == 0:
            raise ValueError(
                "unknown_min_samples removed every Unknown category; lower the threshold."
            )
        if not unknown_keep_mask.all():
            scfound_all = scfound_all[unknown_keep_mask]
            geneformer_all = geneformer_all[unknown_keep_mask]
            raw_all = raw_all[unknown_keep_mask]

        scfound_out, geneformer_out, raw_out = original_load(
            self.paths["scfoundation_out"],
            self.paths["geneformer_out"],
            self.paths["raw_out_h5ad"],
        )
        _, func_map = trainer.du.create_label_maps(raw_all)
        raw_out, scfound_out, geneformer_out = original_filter_ood(
            raw_out, scfound_out, geneformer_out, func_map
        )

        # Materialize AnnData views once. Subsequent runs read these filtered
        # H5ADs and never touch the larger source H5ADs.
        raw_all = raw_all.copy()
        raw_rest = raw_rest.copy()
        raw_out = raw_out.copy()
        source_row_indices = raw_all.obs[source_column].to_numpy(
            dtype=np.int64, copy=True
        )
        del raw_all.obs[source_column]
        if source_column in raw_rest.obs.columns:
            del raw_rest.obs[source_column]

        _save_array(root / GLOBAL_FILES["scfound_all"], scfound_all)
        _save_array(root / GLOBAL_FILES["geneformer_all"], geneformer_all)
        _save_array(root / GLOBAL_FILES["source_row_indices"], source_row_indices)
        _save_array(root / GLOBAL_FILES["scfound_rest"], scfound_rest)
        _save_array(root / GLOBAL_FILES["geneformer_rest"], geneformer_rest)
        _save_array(root / GLOBAL_FILES["scfound_out"], scfound_out)
        _save_array(root / GLOBAL_FILES["geneformer_out"], geneformer_out)
        raw_all.write_h5ad(root / GLOBAL_FILES["raw_all"])
        raw_rest.write_h5ad(root / GLOBAL_FILES["raw_rest"])
        raw_out.write_h5ad(root / GLOBAL_FILES["raw_out"])

        arrays = {
            "scfound_all": np.asarray(scfound_all),
            "geneformer_all": np.asarray(geneformer_all),
            "scfound_rest": np.asarray(scfound_rest),
            "geneformer_rest": np.asarray(geneformer_rest),
            "scfound_out": np.asarray(scfound_out),
            "geneformer_out": np.asarray(geneformer_out),
        }
        manifest = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": self.key,
            "created_at": _utc_now_iso(),
            "specification": self.specification,
            "unknown_filter_report": unknown_report,
            "row_counts": {
                "all": int(raw_all.n_obs),
                "rest": int(raw_rest.n_obs),
                "out": int(raw_out.n_obs),
            },
            "array_shapes": {
                name: list(array.shape) for name, array in arrays.items()
            },
            "array_dtypes": {
                name: str(array.dtype) for name, array in arrays.items()
            },
            "source_row_indices_sha256": _array_sha256(
                source_row_indices, np.int64
            ),
            "obs_names_sha256": _sha256_text(
                "\n".join(raw_all.obs_names.astype(str).tolist())
            ),
        }
        _write_json(root / GLOBAL_FILES["manifest"], manifest)

        del (
            scfound_full,
            geneformer_full,
            raw_full,
            scfound_all,
            geneformer_all,
            raw_all,
            scfound_rest,
            geneformer_rest,
            raw_rest,
            scfound_out,
            geneformer_out,
            raw_out,
        )
        gc.collect()

    def _load_global_directory(self) -> None:
        self.manifest = _read_json(self.root / GLOBAL_FILES["manifest"])
        self.unknown_filter_report = dict(
            self.manifest["unknown_filter_report"]
        )
        # Copy-on-write mmap remains lazy but is writable from PyTorch's point
        # of view; model code never persists changes back into cache files.
        mmap_mode = "c"
        self.scfound_all = np.load(
            self.root / GLOBAL_FILES["scfound_all"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.geneformer_all = np.load(
            self.root / GLOBAL_FILES["geneformer_all"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.source_row_indices = np.load(
            self.root / GLOBAL_FILES["source_row_indices"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.scfound_rest = np.load(
            self.root / GLOBAL_FILES["scfound_rest"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.geneformer_rest = np.load(
            self.root / GLOBAL_FILES["geneformer_rest"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.scfound_out = np.load(
            self.root / GLOBAL_FILES["scfound_out"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.geneformer_out = np.load(
            self.root / GLOBAL_FILES["geneformer_out"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.raw_all = trainer.du.sc.read_h5ad(self.root / GLOBAL_FILES["raw_all"])
        self.raw_rest = trainer.du.sc.read_h5ad(
            self.root / GLOBAL_FILES["raw_rest"]
        )
        self.raw_out = trainer.du.sc.read_h5ad(self.root / GLOBAL_FILES["raw_out"])
        expected = self.manifest["row_counts"]
        checks = {
            "all": (
                self.raw_all.n_obs,
                len(self.scfound_all),
                len(self.geneformer_all),
                len(self.source_row_indices),
            ),
            "rest": (
                self.raw_rest.n_obs,
                len(self.scfound_rest),
                len(self.geneformer_rest),
            ),
            "out": (
                self.raw_out.n_obs,
                len(self.scfound_out),
                len(self.geneformer_out),
            ),
        }
        for name, counts in checks.items():
            if any(int(value) != int(expected[name]) for value in counts):
                raise RuntimeError(
                    f"Cached {name} row mismatch: observed={counts}, "
                    f"expected={expected[name]}."
                )
        if _array_sha256(self.source_row_indices, np.int64) != self.manifest[
            "source_row_indices_sha256"
        ]:
            raise RuntimeError("Cached source-row index integrity check failed.")
        obs_names_hash = _sha256_text(
            "\n".join(self.raw_all.obs_names.astype(str).tolist())
        )
        if obs_names_hash != self.manifest["obs_names_sha256"]:
            raise RuntimeError("Cached obs_names integrity check failed.")

    def _paths_match(self, values: Sequence[str], expected: tuple[str, ...]) -> bool:
        return tuple(_resolved_path(value) for value in values) == expected

    def cached_load_features(self, scfound_path: str, geneformer_path: str, raw_path: str):
        supplied = (scfound_path, geneformer_path, raw_path)
        if self._paths_match(supplied, self.main_path_tuple):
            print(f"[cache] Reusing filtered main data: {self.root}")
            return self.scfound_all, self.geneformer_all, self.raw_all
        if self._paths_match(supplied, self.out_path_tuple):
            print(f"[cache] Reusing filtered OOD data: {self.root}")
            return self.scfound_out, self.geneformer_out, self.raw_out
        raise RuntimeError(
            "The cached trainer received input paths outside its resolved cache identity: "
            f"{supplied}"
        )

    def cached_filter_data(self, scfound, geneformer, raw, filter_least):
        if scfound is not self.scfound_all or raw is not self.raw_all:
            raise RuntimeError("Unexpected main-data object passed to cached filter_data().")
        if int(filter_least) != int(trainer.cfg.FILTER_LEAST):
            raise RuntimeError(
                f"filter_least changed after cache resolution: {filter_least}"
            )
        raw.obs[trainer.SOURCE_ROW_INDEX_COLUMN] = np.asarray(
            self.source_row_indices, dtype=np.int64
        )
        return (
            self.scfound_all,
            self.geneformer_all,
            self.raw_all,
            self.scfound_rest,
            self.geneformer_rest,
            self.raw_rest,
        )

    def cached_filter_ood_data(self, raw_out, scfound_out, geneformer_out, func_map):
        if raw_out is not self.raw_out or scfound_out is not self.scfound_out:
            raise RuntimeError("Unexpected OOD object passed to cached filter_ood_data().")
        return self.raw_out, self.scfound_out, self.geneformer_out

    def cached_unknown_filter(self, fine_labels, unknown_mask, min_samples):
        if len(fine_labels) != int(self.manifest["row_counts"]["all"]):
            raise RuntimeError("Unexpected rows passed to cached Unknown filter.")
        if int(min_samples) != int(self.params["unknown_min_samples"]):
            raise RuntimeError("Unknown threshold changed after cache resolution.")
        return np.ones(len(fine_labels), dtype=bool), dict(
            self.unknown_filter_report
        )

    def _fold_specification(
        self,
        fold_params,
        train_index,
        test_index,
        y_fine,
        preprocessor_fit_index,
    ) -> dict[str, Any]:
        train_index = np.asarray(train_index, dtype=np.int64)
        test_index = np.asarray(test_index, dtype=np.int64)
        fit_index = (
            None
            if preprocessor_fit_index is None
            else np.asarray(preprocessor_fit_index, dtype=np.int64)
        )
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "global_cache_key": self.key,
            "train_index_sha256": _array_sha256(train_index, np.int64),
            "test_index_sha256": _array_sha256(test_index, np.int64),
            "preprocessor_fit_index_sha256": (
                None if fit_index is None else _array_sha256(fit_index, np.int64)
            ),
            "train_fine_labels_sha256": _array_sha256(
                np.asarray(y_fine)[train_index], np.int64
            ),
            "raw_preprocessor": {
                "mode": str(fold_params["raw_preprocess_mode"]),
                "num_variance_features": int(fold_params["num_of_gene_by_var"]),
                "num_mean_features": int(fold_params["num_of_gene_by_mean"]),
                "normalize_target_sum": float(
                    fold_params["raw_normalize_target_sum"]
                ),
                "zscore_clip": float(fold_params["raw_zscore_clip"]),
            },
        }

    def _validate_fold_directory(
        self, root: Path, key: str, specification: dict[str, Any]
    ) -> bool:
        if not _required_files_exist(root, FOLD_RAW_FILES):
            return False
        try:
            manifest = _read_json(root / FOLD_RAW_FILES["manifest"])
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest.get("cache_format_version") == CACHE_FORMAT_VERSION
            and manifest.get("cache_key") == key
            and manifest.get("specification") == specification
        )

    def cached_make_fold_datasets(
        self,
        original_make_fold_datasets,
        fold_params,
        raw_preprocessor,
        train_index,
        test_index,
        scfound_all,
        geneformer_all,
        raw_all,
        y_fine,
        y_function,
        label_map,
        func_map,
        scfound_out,
        geneformer_out,
        raw_out,
        y_function_out,
        scfound_rest,
        geneformer_rest,
        raw_rest,
        preprocessor_fit_index=None,
    ):
        specification = self._fold_specification(
            fold_params,
            train_index,
            test_index,
            y_fine,
            preprocessor_fit_index,
        )
        key = _cache_key(specification)
        root = self.root / "fold_raw" / f"foldraw_{key}"
        built_datasets: list[Any] = []

        def validate(candidate: Path) -> bool:
            return self._validate_fold_directory(candidate, key, specification)

        def build(candidate: Path) -> None:
            print(
                "[cache] Fold raw MISS: fitting train-only feature selection and "
                f"materializing dense arrays ({key})..."
            )
            datasets = original_make_fold_datasets(
                fold_params,
                raw_preprocessor,
                train_index,
                test_index,
                scfound_all,
                geneformer_all,
                raw_all,
                y_fine,
                y_function,
                label_map,
                func_map,
                scfound_out,
                geneformer_out,
                raw_out,
                y_function_out,
                scfound_rest,
                geneformer_rest,
                raw_rest,
                preprocessor_fit_index=preprocessor_fit_index,
            )
            built_datasets.extend(datasets)
            names = ("raw_train", "raw_test", "raw_out", "raw_rest")
            shapes: dict[str, list[int]] = {}
            for name, dataset in zip(names, datasets):
                raw_values = dataset.raw_expression.detach().cpu().numpy()
                _save_array(candidate / FOLD_RAW_FILES[name], raw_values)
                shapes[name] = list(raw_values.shape)
            _write_json(
                candidate / FOLD_RAW_FILES["preprocessor"],
                raw_preprocessor.state_dict(),
            )
            _write_json(
                candidate / FOLD_RAW_FILES["manifest"],
                {
                    "cache_format_version": CACHE_FORMAT_VERSION,
                    "cache_key": key,
                    "created_at": _utc_now_iso(),
                    "specification": specification,
                    "input_dim_raw": int(fold_params["input_dim_raw"]),
                    "shapes": shapes,
                },
            )

        hit = _ensure_directory_cache(
            root,
            cache_root=self.cache_root,
            validate=validate,
            build=build,
            rebuild=False,
        )
        if not hit:
            print(f"[cache] Fold raw BUILT: {root}")
            return tuple(built_datasets)

        print(f"[cache] Fold raw HIT: {root}")
        manifest = _read_json(root / FOLD_RAW_FILES["manifest"])
        preprocessor_state = _read_json(root / FOLD_RAW_FILES["preprocessor"])
        _copy_preprocessor_state(raw_preprocessor, preprocessor_state)
        fold_params["input_dim_raw"] = int(manifest["input_dim_raw"])
        raw_train = np.load(
            root / FOLD_RAW_FILES["raw_train"], mmap_mode="c", allow_pickle=False
        )
        raw_test = np.load(
            root / FOLD_RAW_FILES["raw_test"], mmap_mode="c", allow_pickle=False
        )
        raw_out_values = np.load(
            root / FOLD_RAW_FILES["raw_out"], mmap_mode="c", allow_pickle=False
        )
        raw_rest_values = np.load(
            root / FOLD_RAW_FILES["raw_rest"], mmap_mode="c", allow_pickle=False
        )
        expected_shapes = manifest["shapes"]
        observed = {
            "raw_train": list(raw_train.shape),
            "raw_test": list(raw_test.shape),
            "raw_out": list(raw_out_values.shape),
            "raw_rest": list(raw_rest_values.shape),
        }
        if observed != expected_shapes:
            raise RuntimeError(
                f"Fold raw cache shape mismatch: observed={observed}, "
                f"expected={expected_shapes}."
            )
        train_dataset = trainer.du.SimpleDataset(
            scfound_all[train_index],
            geneformer_all[train_index],
            raw_train,
            torch.as_tensor(y_fine[train_index]),
            torch.as_tensor(y_function[train_index]),
            label_map,
            func_map,
            mode="train",
        )
        test_dataset = trainer.du.SimpleDataset(
            scfound_all[test_index],
            geneformer_all[test_index],
            raw_test,
            torch.as_tensor(y_fine[test_index]),
            torch.as_tensor(y_function[test_index]),
            label_map,
            func_map,
            mode="test",
        )
        out_dataset = trainer.du.SimpleDataset(
            scfound_out,
            geneformer_out,
            raw_out_values,
            None,
            torch.as_tensor(y_function_out),
            label_map,
            func_map,
            mode="out",
        )
        rest_dataset = trainer.du.SimpleDataset(
            scfound_rest,
            geneformer_rest,
            raw_rest_values,
            None,
            None,
            label_map,
            func_map,
            mode="inference",
        )
        return train_dataset, test_dataset, out_dataset, rest_dataset

    @contextmanager
    def installed(self):
        original_load = trainer.du.load_features
        original_filter = trainer.du.filter_data
        original_filter_ood = trainer.du.filter_ood_data
        original_unknown_filter = trainer.build_unknown_category_filter_mask
        original_make_fold_datasets = trainer._make_fold_datasets
        original_run_fold = trainer.run_fold

        def cached_make_fold_datasets(*args, **kwargs):
            return self.cached_make_fold_datasets(
                original_make_fold_datasets, *args, **kwargs
            )

        def cleaned_run_fold(*args, **kwargs):
            try:
                return original_run_fold(*args, **kwargs)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[memory] Released fold-local objects and CUDA cache.")

        trainer.du.load_features = self.cached_load_features
        trainer.du.filter_data = self.cached_filter_data
        trainer.du.filter_ood_data = self.cached_filter_ood_data
        trainer.build_unknown_category_filter_mask = self.cached_unknown_filter
        trainer._make_fold_datasets = cached_make_fold_datasets
        trainer.run_fold = cleaned_run_fold
        try:
            yield
        finally:
            trainer.du.load_features = original_load
            trainer.du.filter_data = original_filter
            trainer.du.filter_ood_data = original_filter_ood
            trainer.build_unknown_category_filter_mask = original_unknown_filter
            trainer._make_fold_datasets = original_make_fold_datasets
            trainer.run_fold = original_run_fold


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--processed-cache-root",
        default=os.environ.get("PERTWEAVE_PROCESSED_CACHE", DEFAULT_CACHE_ROOT),
        help=(
            "Reusable processed-data cache root. The cache key includes source "
            "file size/mtime, FILTER_LEAST, and the Unknown class size threshold."
        ),
    )
    parser.add_argument(
        "--rebuild-processed-cache",
        type=trainer.str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Rebuild the matching global cache and its fold-raw subcaches.",
    )
    parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help="Build/load only the global filtered cache, then exit before training.",
    )


def _extract_cache_arguments(argv: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)
    _add_cache_arguments(parser)
    return parser.parse_known_args(list(argv))


def _combined_help() -> None:
    parser = trainer.build_parser()
    _add_cache_arguments(parser)
    parser.print_help()


def main(argv: Sequence[str] | None = None) -> None:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if "-h" in supplied or "--help" in supplied:
        _combined_help()
        return
    cache_args, training_argv = _extract_cache_arguments(supplied)

    parsed_training = trainer.build_parser().parse_args(training_argv)
    params = trainer.resolve_params(parsed_training)
    cache = ProcessedDataCache(
        params,
        cache_args.processed_cache_root,
        rebuild=cache_args.rebuild_processed_cache,
    )
    print("[mode] NumPy fold predictions; online plotting disabled.")
    if cache_args.prepare_cache_only:
        print(f"[cache] Global cache ready; training skipped: {cache.root}")
        return
    with cache.installed():
        trainer.main(training_argv)


if __name__ == "__main__":
    main()
