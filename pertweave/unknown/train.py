"""Training, evaluation and checkpoint export for function-unknown perturbations."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from . import config as cfg
from . import data as du
from .. import models
from .components import (
    UNKNOWN_FUNCTION_ID,
    RawFoldPreprocessor,
    UnifiedGeometryLabelEncoder,
    UnknownPKBatchSampler,
    build_unknown_category_filter_mask,
    classification_guardrail_pass,
    compute_unknown_clustering_metrics,
    unknown_report_score,
)


GUARDRAIL_METRICS = ("test_acc", "test_auc", "test_group_acc", "test_group_auc")
RESULT_KEYS = ("features", "logits", "preds", "labels", "group_logits",
               "group_preds", "group_labels", "feat_cls", "feat_group")
SOURCE_ROW_INDEX_COLUMN = "__unknown_prediction_source_row_index__"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def write_json(path: str | os.PathLike[str], payload: Any):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def write_rows_csv(path: str | os.PathLike[str], rows: list[dict[str, Any]]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int):
    """Match the frozen baseline seeding behavior without plotting dependencies."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_log(path: str | os.PathLike[str], content: str, mode: str = "a"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(content + "\n")


def str_to_bool(value: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, received {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PertWeave on annotated and function-unknown perturbations.")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument(
        "--recover-partial-fold",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help=(
            "Recover a non-empty foldN directory whose training reached the final "
            "epoch but post-training export did not complete. May be repeated. "
            "The existing last/best checkpoints, fold config, raw preprocessor, "
            "and final epoch metrics must all pass strict validation."
        ),
    )

    parser.add_argument("--scfoundation", default=cfg.SCFOUNDATION_NPY)
    parser.add_argument("--geneformer", default=cfg.GENEFORMER_PICKLE)
    parser.add_argument("--raw-h5ad", default=cfg.RAW_H5AD)
    parser.add_argument("--scfoundation-out", default=cfg.SCFOUNDATION_OUT_NPY)
    parser.add_argument("--geneformer-out", default=cfg.GENEFORMER_OUT_PICKLE)
    parser.add_argument("--raw-out-h5ad", default=cfg.RAW_OUT_H5AD)
    return parser


def resolve_params(args):
    params = dict(cfg.HYPERPARAMETERS)
    for key in ("epochs", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    params["seed"] = int(args.seed)
    params["device"] = str(args.device)
    params["input_paths"] = {
        "scfoundation": args.scfoundation,
        "geneformer": args.geneformer,
        "raw_h5ad": args.raw_h5ad,
        "scfoundation_out": args.scfoundation_out,
        "geneformer_out": args.geneformer_out,
        "raw_out_h5ad": args.raw_out_h5ad,
    }
    if params["epochs"] < 1 or params["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if params["unknown_metric_every"] < 1:
        raise ValueError("unknown_metric_every must be positive.")
    return params


def resolve_amp_dtype(params: dict[str, Any]):
    name = str(params["amp_dtype"]).casefold()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError("amp_dtype must be 'bfloat16' or 'float16'.")


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_finite(named_tensors: dict[str, torch.Tensor], context: str):
    for name, tensor in named_tensors.items():
        if tensor is not None and not torch.isfinite(tensor).all():
            finite = torch.isfinite(tensor)
            raise FloatingPointError(
                f"Non-finite tensor {name!r} at {context}: "
                f"shape={tuple(tensor.shape)}, bad={int((~finite).sum())}."
            )


def encode_function_labels(values, func_map: dict[Any, int]) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        if du.is_unknown_function_name(value):
            result.append(UNKNOWN_FUNCTION_ID)
        elif value in func_map:
            result.append(int(func_map[value]))
        else:
            raise KeyError(f"Function {value!r} is absent from func_map and is not Unknown.")
    return np.asarray(result, dtype=np.int64)


def make_metrics(num_classes: int, num_groups: int, device: torch.device):
    import torchmetrics

    metrics = {
        "train_loss": torchmetrics.MeanMetric(),
        "train_ce_loss": torchmetrics.MeanMetric(),
        "train_center_loss": torchmetrics.MeanMetric(),
        "train_ce_group_loss": torchmetrics.MeanMetric(),
        "train_center_loss_group": torchmetrics.MeanMetric(),
        "train_MCC_loss": torchmetrics.MeanMetric(),
        "train_MCC_loss_group": torchmetrics.MeanMetric(),
        "train_feat_MCC_loss": torchmetrics.MeanMetric(),
        "train_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "train_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        "train_group_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_groups),
        "train_group_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_groups),
        "test_loss": torchmetrics.MeanMetric(),
        "test_ce_loss": torchmetrics.MeanMetric(),
        "test_center_loss": torchmetrics.MeanMetric(),
        "test_ce_group_loss": torchmetrics.MeanMetric(),
        "test_center_loss_group": torchmetrics.MeanMetric(),
        "test_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "test_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        "test_group_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_groups),
        "test_group_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_groups),
        "out_group_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_groups),
        "out_group_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_groups),
    }
    return {name: metric.to(device) for name, metric in metrics.items()}


def metric_snapshot(metrics: dict[str, Any], names=None) -> dict[str, float]:
    selected = metrics.keys() if names is None else names
    result = {}
    for name in selected:
        if getattr(metrics[name], "_update_count", 1) == 0:
            result[name] = float("nan")
            continue
        value = metrics[name].compute()
        result[name] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return result


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def model_feature_dim(model: nn.Module) -> int:
    return int(unwrap_model(model).inner_proj.out_features)


def make_model(params: dict[str, Any], device: torch.device) -> nn.Module:
    model = models.UnknownPertWeave(
        input_dim_scfound=params["input_dim_scfound"],
        input_dim_gene=params["input_dim_geneformer"],
        input_dim_raw=params["input_dim_raw"],
        hidden_dim=params["hidden_dim"],
        output_dim=params["output_dim"],
        group_output_dim=params["group_output_dim"],
        feat_dim=params["feat_dim"],
        layer_num=params["layer_num"],
        layer_list=params["layer_list"],
        dropout=params["dropout"],
        dropout_num=params["drop_num"],
        linear=params["linear"],
        activation=params["activation"],
        reduction_factor=params["reduction_factor"],
        GLU_reduction=params["GLU_reduction"],
        data_mask=params["data_mask"],
        efficient_num=params["efficient_num"],
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def make_criteria(
    params: dict[str, Any],
    model: nn.Module,
    encoder: UnifiedGeometryLabelEncoder,
    device: torch.device,
) -> dict[str, nn.Module | None]:
    raw_dim = model_feature_dim(model)
    criteria: dict[str, nn.Module | None] = {
        "ce": nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"]),
        "ce_group": nn.CrossEntropyLoss(label_smoothing=params["label_smoothing_group"]),
        "MCC": models.MCCLoss(temperature=params["MCC_temperature"]),
        "legacy_fine": models.CenterLoss(params["output_dim"], raw_dim, device),
        "legacy_group": models.CenterLoss(params["group_output_dim"], raw_dim, device),
    }
    return criteria


def criterion_parameters(criteria: dict[str, nn.Module | None]):
    parameters = []
    for criterion in criteria.values():
        if criterion is not None:
            parameters.extend(list(criterion.parameters()))
    return parameters


def make_optimizer_scheduler(
    model,
    criteria,
    params,
    stage_epochs: int,
    lr_scale: float = 1.0,
):
    """Build a fresh optimizer/scheduler for one independently annealed stage."""
    stage_lr = float(params["raw_lr"]) * float(lr_scale)
    stage_final_lr = min(float(params["final_lr"]), stage_lr)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + criterion_parameters(criteria),
        lr=stage_lr,
        weight_decay=params["weight_decay"],
    )
    import timm.scheduler
    warmup_epochs = 10 if float(lr_scale) == 1.0 else min(5, max(int(stage_epochs) - 1, 0))
    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer,
        t_initial=int(stage_epochs),
        lr_min=stage_final_lr,
        warmup_t=warmup_epochs,
        warmup_lr_init=stage_final_lr,
    )
    return optimizer, scheduler


def compute_single_view_losses(
    output: torch.Tensor,
    output_group: torch.Tensor,
    feat_raw: torch.Tensor,
    label: torch.Tensor,
    function: torch.Tensor,
    criteria: dict[str, nn.Module | None],
    params: dict[str, Any],
    encoder: UnifiedGeometryLabelEncoder,
) -> dict[str, torch.Tensor]:
    output = output.float()
    output_group = output_group.float()
    feat_raw = feat_raw.float()
    known = function.ne(UNKNOWN_FUNCTION_ID)
    zero_raw = feat_raw.sum() * 0.0
    zero_group = output_group.sum() * 0.0

    ce = criteria["ce"](output, label)
    legacy_fine = (
        criteria["legacy_fine"](feat_raw, label)
        if criteria["legacy_fine"] is not None else zero_raw
    )
    if known.any():
        ce_group = criteria["ce_group"](output_group[known], function[known])
        legacy_group = (
            criteria["legacy_group"](feat_raw[known], function[known])
            if criteria["legacy_group"] is not None else zero_raw
        )
    else:
        ce_group = zero_group
        legacy_group = zero_raw

    supervised = (
        params["celoss_weight"] * ce
        + params["center_loss_weight"] * legacy_fine
        + params["celoss_group_weight"] * ce_group
        + params["center_loss_group_weight"] * legacy_group
    )
    return {
        "ce": ce,
        "legacy_fine": legacy_fine,
        "ce_group": ce_group,
        "legacy_group": legacy_group,
        "supervised": supervised,
        "known_mask": known,
    }


def compute_group_mcc(
    first: torch.Tensor,
    second: torch.Tensor,
    function: torch.Tensor,
    criterion: nn.Module,
    include_unknown: bool,
) -> torch.Tensor:
    mask = torch.ones_like(function, dtype=torch.bool) if include_unknown else function.ne(UNKNOWN_FUNCTION_ID)
    if mask.any():
        return criterion(first[mask].float(), second[mask].float())
    return (first.sum() + second.sum()) * 0.0


def train_epoch(
    model,
    loader,
    optimizer,
    criteria,
    metrics,
    params,
    encoder,
    device,
    scaler,
    fold,
    epoch,
):
    model.train()
    amp_enabled = params["use_amp"] and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(params)
    non_blocking = params["pin_memory"] and device.type == "cuda"
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)

    for batch_index, batch in enumerate(loader):
        sc_found, geneformer, raw_data, label, function = [
            tensor.to(device, non_blocking=non_blocking) for tensor in batch[:5]
        ]
        # Keep the frozen baseline optimizer behavior in legacy mode.
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            if amp_enabled and hasattr(model, "forward_two_views") and not isinstance(model, nn.DataParallel):
                raw_outputs1, raw_outputs2 = model.forward_two_views(
                    sc_found, geneformer, raw_data, data_mask=params["data_mask"]
                )
            else:
                raw_outputs1 = model(sc_found, geneformer, raw_data, data_mask=params["data_mask"])
                raw_outputs2 = model(sc_found, geneformer, raw_data, data_mask=params["data_mask"])
        output1, group1, feat1, _, _ = raw_outputs1
        output2, group2, feat2, _, _ = raw_outputs2
        losses1 = compute_single_view_losses(
            output1, group1, feat1, label, function, criteria, params, encoder
        )
        losses2 = compute_single_view_losses(
            output2, group2, feat2, label, function, criteria, params, encoder
        )
        mcc = criteria["MCC"](output1.float(), output2.float())
        mcc_group = compute_group_mcc(
            group1, group2, function, criteria["MCC"], params["unknown_in_mcc_group"]
        )
        mcc_feat = criteria["MCC"](feat1.float(), feat2.float())
        total = (
            (losses1["supervised"] + losses2["supervised"]) / 2.0
            + params["feat_MCC_weight"] * mcc_feat
            + params["MCC_weight"] * mcc
            + params["MCC_group_weight"] * mcc_group
        )
        if params["fail_on_nonfinite"]:
            ensure_finite(
                {
                    "output1": output1,
                    "group1": group1,
                    "feat1": feat1,
                    "total_loss": total,
                },
                f"fold={fold}, epoch={epoch}, batch={batch_index}",
            )
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics["train_loss"].update(total.detach())
        metrics["train_ce_loss"].update(losses1["ce"].detach())
        metrics["train_center_loss"].update(losses1["legacy_fine"].detach())
        metrics["train_ce_group_loss"].update(losses1["ce_group"].detach())
        metrics["train_center_loss_group"].update(losses1["legacy_group"].detach())
        metrics["train_MCC_loss"].update(mcc.detach())
        metrics["train_MCC_loss_group"].update(mcc_group.detach())
        metrics["train_feat_MCC_loss"].update(mcc_feat.detach())
        metrics["train_acc"].update(output1.float().detach(), label)
        metrics["train_auc"].update(output1.float().detach(), label)
        known = losses1["known_mask"]
        if known.any():
            metrics["train_group_acc"].update(group1.float()[known].detach(), function[known])
            metrics["train_group_auc"].update(group1.float()[known].detach(), function[known])


@torch.inference_mode()
def evaluate_epoch(
    model,
    loader,
    criteria,
    metrics,
    params,
    encoder,
    device,
    mode="test",
    collect_outputs=False,
    update_metrics=True,
):
    model.eval()
    amp_enabled = params["use_amp"] and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(params)
    non_blocking = params["pin_memory"] and device.type == "cuda"
    keys = (
        "feat_raw", "logits", "preds", "labels", "group_logits",
        "group_preds", "group_labels", "feat_cls", "feat_group",
    )
    collected = {key: [] for key in keys} if collect_outputs else None
    for batch_index, batch in enumerate(loader):
        label = None
        function = None
        if mode == "test":
            sc_found, geneformer, raw_data, label, function = batch
            label = label.to(device, non_blocking=non_blocking)
        elif len(batch) == 4:
            sc_found, geneformer, raw_data, function = batch
        else:
            sc_found, geneformer, raw_data = batch
        sc_found = sc_found.to(device, non_blocking=non_blocking)
        geneformer = geneformer.to(device, non_blocking=non_blocking)
        raw_data = raw_data.to(device, non_blocking=non_blocking)
        if function is not None:
            function = function.to(device, non_blocking=non_blocking)
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(sc_found, geneformer, raw_data, data_mask=params["data_mask"])
        output, group, feat_raw, feat_cls, feat_group = outputs
        output = output.float()
        group = group.float()
        feat_raw = feat_raw.float()
        if params["fail_on_nonfinite"]:
            ensure_finite(
                {
                    "output": output,
                    "group": group,
                    "feat_raw": feat_raw,
                    "feat_cls": feat_cls,
                    "feat_group": feat_group,
                },
                f"{mode} batch={batch_index}",
            )
        if collect_outputs:
            collected["feat_raw"].append(feat_raw.cpu())
            collected["logits"].append(output.cpu())
            collected["preds"].append(output.argmax(dim=1).cpu())
            collected["feat_cls"].append(feat_cls.float().cpu())
            collected["feat_group"].append(feat_group.float().cpu())
            if label is not None:
                collected["labels"].append(label.cpu())
            if function is not None:
                collected["group_logits"].append(group.cpu())
                collected["group_preds"].append(group.argmax(dim=1).cpu())
                collected["group_labels"].append(function.cpu())

        if function is None or not update_metrics:
            continue
        known = function.ne(UNKNOWN_FUNCTION_ID)
        if mode == "test":
            losses = compute_single_view_losses(
                output, group, feat_raw, label, function, criteria, params, encoder
            )
            metrics["test_loss"].update(losses["supervised"].detach())
            metrics["test_ce_loss"].update(losses["ce"].detach())
            metrics["test_center_loss"].update(losses["legacy_fine"].detach())
            metrics["test_ce_group_loss"].update(losses["ce_group"].detach())
            metrics["test_center_loss_group"].update(losses["legacy_group"].detach())
            metrics["test_acc"].update(output, label)
            metrics["test_auc"].update(output, label)
            if known.any():
                metrics["test_group_acc"].update(group[known], function[known])
                metrics["test_group_auc"].update(group[known], function[known])
        elif mode == "out" and known.any():
            metrics["out_group_acc"].update(group[known], function[known])
            metrics["out_group_auc"].update(group[known], function[known])

    if not collect_outputs:
        return {}
    results = {}
    for key, pieces in collected.items():
        if pieces:
            results[key] = np.concatenate([piece.numpy() for piece in pieces], axis=0)
        else:
            results[key] = np.array([])
    return results


def build_train_loader(
    dataset,
    params,
    device,
    fine_labels,
    function_labels,
    seed=0,
):
    worker_options = {
        "num_workers": params["num_workers"],
        "pin_memory": params["pin_memory"] and device.type == "cuda",
        "persistent_workers": params["persistent_workers"] and params["num_workers"] > 0,
    }
    sampler = UnknownPKBatchSampler(
        fine_labels=fine_labels,
        function_labels=function_labels,
        batch_size=params["batch_size"],
        categories_per_batch=params["unknown_categories_per_batch"],
        samples_per_category=params["unknown_samples_per_category"],
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, **worker_options)


def build_eval_loader(dataset, params, device):
    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"] and device.type == "cuda",
        persistent_workers=params["persistent_workers"] and params["num_workers"] > 0,
    )


def load_baseline_metrics(path: str | None, fold: int) -> dict[str, float] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = [
        payload,
        payload.get(f"fold{fold}", {}) if isinstance(payload, dict) else {},
        payload.get(str(fold), {}) if isinstance(payload, dict) else {},
        payload.get("folds", {}).get(str(fold), {}) if isinstance(payload, dict) else {},
        payload.get("metrics", {}) if isinstance(payload, dict) else {},
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and all(name in candidate for name in GUARDRAIL_METRICS):
            return {name: float(candidate[name]) for name in GUARDRAIL_METRICS}
    raise KeyError(
        f"Could not find {GUARDRAIL_METRICS} for fold {fold} in baseline file {path}."
    )


def sanitized_unknown_score(report: dict[str, Any]) -> tuple[float, float, float]:
    return tuple(value if np.isfinite(value) else float("-inf") for value in unknown_report_score(report))


def checkpoint_payload(
    model,
    criteria,
    optimizer,
    scheduler,
    scaler,
    epoch,
    params,
    encoder,
    raw_preprocessor,
    unknown_report,
    classification_metrics,
    guardrail_pass,
    guardrail_drops,
    training_stage="single",
    stage_epoch=None,
    transition_metadata=None,
):
    return {
        "epoch": int(epoch),
        "training_stage": str(training_stage),
        "stage_epoch": int(stage_epoch if stage_epoch is not None else epoch),
        "model_state_dict": unwrap_model(model).state_dict(),
        "criterion_state_dicts": {
            name: criterion.state_dict()
            for name, criterion in criteria.items() if criterion is not None
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "params": cfg.serializable_params(params),
        "unified_label_mapping": encoder.state_dict(),
        "raw_preprocessor": raw_preprocessor.state_dict(),
        "unknown_metrics": unknown_report,
        "classification_metrics": classification_metrics,
        "guardrail_pass": bool(guardrail_pass),
        "guardrail_drops": guardrail_drops,
        "stage_transition": transition_metadata,
    }


def save_checkpoint(path, **kwargs):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(**kwargs), target)


def load_model_checkpoint(model, path, device):
    # Loading a full checkpoint directly onto an already occupied CUDA device
    # briefly duplicates the model/optimizer tensors and can OOM during
    # post-training best-checkpoint evaluation. Deserialize on CPU and let
    # load_state_dict copy tensors into the existing model parameters.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def load_training_checkpoint(model, criteria, path, device):
    """Restore model and learnable loss centers for a stage transition."""
    checkpoint = load_model_checkpoint(model, path, device)
    for name, state in checkpoint.get("criterion_state_dicts", {}).items():
        criterion = criteria.get(name)
        if criterion is not None:
            criterion.load_state_dict(state)
    return checkpoint


def save_fold_arrays(root, results):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for key in RESULT_KEYS:
        np.save(root / f"{key}.npy", results["feat_raw" if key == "features" else key])


def validate_fold_artifacts(fold_root, require_last_embeddings=True):
    root = Path(fold_root)
    required = [root / name for name in ("best.pth", "last.pth", "raw_preprocessor.json",
                                         "fold_config.json")]
    if require_last_embeddings:
        required.extend(root / dataset / f"{key}.npy"
                        for dataset in ("test_data", "out_data", "test_rest_data")
                        for key in RESULT_KEYS)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete fold artifacts: {missing}")


def save_split_manifest(root, split_indices, sample_ids):
    arrays = {"sample_ids": np.asarray(sample_ids).astype(str)}
    json_manifest = {"num_samples": int(len(sample_ids)), "folds": {}}
    for fold, (train_indices, test_indices) in enumerate(split_indices):
        arrays[f"fold{fold}_train"] = np.asarray(train_indices, dtype=np.int64)
        arrays[f"fold{fold}_test"] = np.asarray(test_indices, dtype=np.int64)
        json_manifest["folds"][str(fold)] = {
            "train_count": int(len(train_indices)),
            "test_count": int(len(test_indices)),
            "train_key": f"fold{fold}_train",
            "test_key": f"fold{fold}_test",
        }
    np.savez_compressed(Path(root) / "split_indices.npz", **arrays)
    write_json(Path(root) / "split_manifest.json", json_manifest)


def append_label_names_to_unknown_rows(rows, num2label):
    named = []
    for row in rows:
        item = dict(row)
        item["class_name"] = str(num2label.get(int(item["class_id"]), item["class_id"]))
        named.append(item)
    return named


def _make_fold_datasets(
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
    if preprocessor_fit_index is None:
        raw_train = raw_preprocessor.fit_transform(
            raw_all[train_index].X, y_fine[train_index]
        )
    else:
        fit_index = np.asarray(preprocessor_fit_index, dtype=np.int64)
        raw_preprocessor.fit(raw_all[fit_index].X, y_fine[fit_index])
        raw_train = raw_preprocessor.transform(raw_all[train_index].X)
    raw_test = raw_preprocessor.transform(raw_all[test_index].X)
    raw_out_values = raw_preprocessor.transform(raw_out.X)
    raw_rest_values = raw_preprocessor.transform(raw_rest.X)
    fold_params["input_dim_raw"] = int(raw_train.shape[1])

    train_dataset = du.SimpleDataset(
        scfound_all[train_index], geneformer_all[train_index], raw_train,
        torch.as_tensor(y_fine[train_index]), torch.as_tensor(y_function[train_index]),
        label_map, func_map, mode="train",
    )
    test_dataset = du.SimpleDataset(
        scfound_all[test_index], geneformer_all[test_index], raw_test,
        torch.as_tensor(y_fine[test_index]), torch.as_tensor(y_function[test_index]),
        label_map, func_map, mode="test",
    )
    out_dataset = du.SimpleDataset(
        scfound_out, geneformer_out, raw_out_values, None,
        torch.as_tensor(y_function_out), label_map, func_map, mode="out",
    )
    rest_dataset = du.SimpleDataset(
        scfound_rest, geneformer_rest, raw_rest_values, None, None,
        label_map, func_map, mode="inference",
    )
    return train_dataset, test_dataset, out_dataset, rest_dataset


def _evaluate_and_export(
    tag,
    fold_root,
    model,
    loaders,
    criteria,
    metrics,
    params,
    encoder,
    device,
    num2label,
):
    output_root = Path(fold_root) if tag == "last" else Path(fold_root) / tag
    for metric in metrics.values():
        metric.reset()
    test_results = evaluate_epoch(
        model, loaders["test"], criteria, metrics, params, encoder, device,
        mode="test", collect_outputs=True, update_metrics=True,
    )
    out_results = evaluate_epoch(
        model, loaders["out"], criteria, metrics, params, encoder, device,
        mode="out", collect_outputs=True, update_metrics=True,
    )
    rest_results = evaluate_epoch(
        model, loaders["rest"], criteria, metrics, params, encoder, device,
        mode="inference", collect_outputs=True, update_metrics=False,
    )
    classification = metric_snapshot(metrics, GUARDRAIL_METRICS)
    unknown_report, unknown_rows = compute_unknown_clustering_metrics(
        test_results["feat_raw"],
        test_results["labels"],
        test_results["group_labels"],
        knn_k=params["unknown_metric_knn_k"],
        min_class_cells=params["unknown_metric_min_class_cells"],
        enrichment_threshold=params["unknown_metric_enrichment_threshold"],
        device=device,
    )
    if tag == "last":
        save_fold_arrays(output_root / "test_data", test_results)
        save_fold_arrays(output_root / "out_data", out_results)
        save_fold_arrays(output_root / "test_rest_data", rest_results)
    named_rows = append_label_names_to_unknown_rows(unknown_rows, num2label)
    write_json(output_root / "unknown_metrics.json", unknown_report)
    write_rows_csv(output_root / "unknown_class_metrics.csv", named_rows)
    write_json(output_root / "classification_metrics.json", classification)
    return (
        test_results, classification, unknown_report, named_rows
    )




def _canonical_recovery_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _recovery_param_view(params: dict[str, Any]) -> dict[str, Any]:
    return cfg.serializable_params(params)


def _read_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object at {target}, got {type(payload).__name__}.")
    return payload


def _assert_recovery_payload_equal(
    label: str,
    expected: Any,
    actual: Any,
) -> None:
    if _canonical_recovery_json(expected) == _canonical_recovery_json(actual):
        return
    changed_keys: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected).union(actual)):
            if _canonical_recovery_json(expected.get(key)) != _canonical_recovery_json(
                actual.get(key)
            ):
                changed_keys.append(str(key))
    suffix = f" Changed keys: {changed_keys}" if changed_keys else ""
    raise RuntimeError(f"Partial-fold recovery rejected: {label} mismatch.{suffix}")


def validate_partial_fold_recovery(
    fold_root: str | os.PathLike[str],
    fold: int,
    fold_params: dict[str, Any],
    raw_preprocessor: RawFoldPreprocessor,
    encoder: UnifiedGeometryLabelEncoder,
) -> dict[str, Any]:
    """Strictly validate that a partial fold is safe for postprocess-only recovery."""

    root = Path(fold_root)
    if (root / "completed.json").exists():
        raise RuntimeError(f"Fold {fold} is already complete and does not need recovery.")
    final_epoch_metric = (
        root / "epoch_metrics" / f"epoch_{int(fold_params['epochs']):03d}.json"
    )
    required = [
        root / "fold_config.json",
        root / "raw_preprocessor.json",
        root / "last.pth",
        root / "best.pth",
        final_epoch_metric,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Partial-fold recovery requires a completed training loop and both "
            f"checkpoints. Missing artifacts: {missing}"
        )

    saved_config = _read_json_object(root / "fold_config.json")
    current_config = cfg.serializable_params(fold_params)
    _assert_recovery_payload_equal(
        "current command versus saved fold config",
        _recovery_param_view(saved_config),
        _recovery_param_view(current_config),
    )
    saved_preprocessor = _read_json_object(root / "raw_preprocessor.json")
    current_preprocessor = raw_preprocessor.state_dict()
    _assert_recovery_payload_equal(
        "reconstructed versus saved raw preprocessor",
        saved_preprocessor,
        current_preprocessor,
    )
    expected_encoder = encoder.state_dict()

    checkpoint_report: dict[str, Any] = {}
    for tag in ("last", "best"):
        checkpoint_path = root / f"{tag}.pth"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        required_keys = {
            "epoch",
            "model_state_dict",
            "params",
            "raw_preprocessor",
            "unified_label_mapping",
        }
        absent = sorted(required_keys.difference(checkpoint))
        if absent:
            raise RuntimeError(
                f"Partial-fold recovery checkpoint {checkpoint_path} lacks keys {absent}."
            )
        epoch = int(checkpoint["epoch"])
        if tag == "last" and epoch != int(fold_params["epochs"]):
            raise RuntimeError(
                "Partial-fold recovery rejected: last.pth epoch "
                f"{epoch} != configured final epoch {fold_params['epochs']}."
            )
        if tag == "best" and not 1 <= epoch <= int(fold_params["epochs"]):
            raise RuntimeError(
                f"Partial-fold recovery rejected: best.pth epoch {epoch} is invalid."
            )
        _assert_recovery_payload_equal(
            f"{tag}.pth params versus saved fold config",
            _recovery_param_view(saved_config),
            _recovery_param_view(dict(checkpoint["params"])),
        )
        _assert_recovery_payload_equal(
            f"{tag}.pth raw preprocessor versus saved state",
            saved_preprocessor,
            checkpoint["raw_preprocessor"],
        )
        _assert_recovery_payload_equal(
            f"{tag}.pth unified label mapping versus current encoder",
            expected_encoder,
            checkpoint["unified_label_mapping"],
        )
        checkpoint_report[tag] = {
            "path": str(checkpoint_path.resolve()),
            "epoch": epoch,
            "size_bytes": int(checkpoint_path.stat().st_size),
        }
        del checkpoint
        gc.collect()

    final_metrics = _read_json_object(final_epoch_metric)
    if int(final_metrics.get("epoch", -1)) != int(fold_params["epochs"]):
        raise RuntimeError(
            f"Final epoch metric does not describe epoch {fold_params['epochs']}."
        )
    return {
        "mode": "postprocess_only",
        "fold": int(fold),
        "validated_at": datetime.now().astimezone().isoformat(),
        "configured_epochs": int(fold_params["epochs"]),
        "final_epoch_metric": str(final_epoch_metric.resolve()),
        "checkpoints": checkpoint_report,
        "configuration_validated": True,
        "raw_preprocessor_validated": True,
        "unified_label_mapping_validated": True,
    }


def _recover_history_from_epoch_metrics(fold_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = Path(fold_root)
    history_path = root / "history.json"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        if isinstance(history, list) and history:
            return history

    history: list[dict[str, Any]] = []
    for path in sorted((root / "epoch_metrics").glob("epoch_*.json")):
        payload = _read_json_object(path)
        record: dict[str, Any] = {
            "epoch": int(payload["epoch"]),
            "stage": payload.get("stage", "unknown"),
            "stage_epoch": int(payload.get("stage_epoch", payload["epoch"])),
            "recovered_from_epoch_metrics": True,
        }
        record.update(dict(payload.get("classification", {})))
        record.update(
            {
                f"unknown_{key}": value
                for key, value in dict(payload.get("unknown", {})).items()
            }
        )
        record["guardrail_pass"] = payload.get("guardrail_pass")
        record["guardrail_drops"] = payload.get("guardrail_drops", {})
        history.append(record)
    if not history:
        raise RuntimeError(
            f"Cannot reconstruct recovery history: no epoch metrics under {root}."
        )
    return history


def recover_partial_fold_outputs(
    fold,
    train_index,
    test_index,
    fold_root,
    fold_params,
    device,
    encoder,
    label_map,
    func_map,
    num2label,
    num2func,
    data,
    datasets,
    raw_preprocessor,
):
    """Finish last/best evaluation and artifact export without retraining."""

    recovery_start = time.time()
    recovery = validate_partial_fold_recovery(
        fold_root, fold, fold_params, raw_preprocessor, encoder
    )
    print(
        f"[recovery] Fold {fold}: validated final epoch "
        f"{fold_params['epochs']} and existing last/best checkpoints."
    )
    loaders = {
        "test": build_eval_loader(datasets[1], fold_params, device),
        "out": build_eval_loader(datasets[2], fold_params, device),
        "rest": build_eval_loader(datasets[3], fold_params, device),
    }
    model = make_model(fold_params, device)
    criteria = make_criteria(fold_params, model, encoder, device)
    metrics = make_metrics(
        fold_params["output_dim"], fold_params["group_output_dim"], device
    )
    baseline = load_baseline_metrics(fold_params["baseline_metrics_path"], fold)

    if fold_params["save_last_embeddings"]:
        last_checkpoint = load_training_checkpoint(
            model, criteria, Path(fold_root) / "last.pth", device
        )
        last_test, _, _, _ = _evaluate_and_export(
            "last",
            fold_root,
            model,
            loaders,
            criteria,
            metrics,
            fold_params,
            encoder,
            device,
            num2label,
        )
        del last_checkpoint, last_test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_checkpoint = load_training_checkpoint(
        model, criteria, Path(fold_root) / "best.pth", device
    )
    (
        best_test,
        best_classification,
        best_unknown,
        _,
    ) = _evaluate_and_export(
        "best",
        fold_root,
        model,
        loaders,
        criteria,
        metrics,
        fold_params,
        encoder,
        device,
        num2label,
    )
    guardrail, drops = classification_guardrail_pass(
        best_classification,
        baseline,
        fold_params["classification_guardrail_max_drop"],
    )
    acceptance = {
        "unknown_trend_coverage": best_unknown.get("trend_coverage", 0.0),
        "unknown_target_coverage": fold_params["unknown_metric_target_coverage"],
        "unknown_coverage_pass": best_unknown.get("trend_coverage", 0.0)
        >= fold_params["unknown_metric_target_coverage"],
        "classification_guardrail_available": baseline is not None,
        "classification_guardrail_pass": guardrail if baseline is not None else None,
        "classification_drops": drops,
    }
    recovery["finished_at"] = datetime.now().astimezone().isoformat()
    recovery["elapsed_seconds"] = float(time.time() - recovery_start)
    history = _recover_history_from_epoch_metrics(fold_root)
    summary = {
        "fold": int(fold),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_checkpoint_guardrail_pass": bool(best_checkpoint["guardrail_pass"]),
        "classification": best_classification,
        "unknown": best_unknown,
        "acceptance": acceptance,
        "elapsed_seconds": recovery["elapsed_seconds"],
        "recovery": {
            "mode": recovery["mode"],
            "postprocess_only": True,
            "training_was_not_repeated": True,
            "history_reconstructed_from_epoch_metrics": bool(
                history and history[0].get("recovered_from_epoch_metrics", False)
            ),
        },
    }
    write_json(Path(fold_root) / "history.json", history)
    write_json(Path(fold_root) / "summary.json", summary)
    validate_fold_artifacts(
        fold_root,
        fold_params["save_last_embeddings"],
    )
    write_json(Path(fold_root) / "recovery_report.json", recovery)
    write_json(
        Path(fold_root) / "completed.json",
        {
            "completed": True,
            "fold": int(fold),
            "recovered": True,
            "recovery_mode": "postprocess_only",
        },
    )
    print(
        f"[recovery] Fold {fold} completed without retraining; "
        f"best epoch={best_checkpoint['epoch']}."
    )
    return summary


def run_fold(
    fold,
    train_index,
    test_index,
    experiment_root,
    params,
    device,
    encoder,
    label_map,
    func_map,
    num2label,
    num2func,
    data,
    *,
    recover_partial=False,
):
    fold_root = Path(experiment_root) / f"fold{fold}"
    if (fold_root / "completed.json").exists():
        print(f"Fold {fold} is already complete; skipping.")
        return
    partial_exists = fold_root.exists() and any(fold_root.iterdir())
    if partial_exists and not recover_partial:
        raise FileExistsError(
            f"Partial fold directory exists at {fold_root}. Pass "
            f"--recover-partial-fold {fold} only if training reached the configured "
            "final epoch and the saved artifacts are intact; otherwise move it aside "
            "before rerunning."
        )
    if recover_partial and not partial_exists:
        raise FileNotFoundError(
            f"Recovery was requested for fold {fold}, but no non-empty partial "
            f"directory exists at {fold_root}."
        )
    if not recover_partial:
        fold_root.mkdir(parents=True, exist_ok=True)
    fold_params = dict(params)
    raw_preprocessor = RawFoldPreprocessor(
        mode=params["raw_preprocess_mode"],
        num_variance_features=params["num_of_gene_by_var"],
        num_mean_features=params["num_of_gene_by_mean"],
        normalize_target_sum=params["raw_normalize_target_sum"],
        zscore_clip=params["raw_zscore_clip"],
    )
    known_train_local = np.flatnonzero(
        data["y_function"][train_index] != UNKNOWN_FUNCTION_ID
    )
    if len(known_train_local) == 0:
        raise ValueError(f"Fold {fold} contains no known-function training cells.")
    datasets = _make_fold_datasets(
        fold_params, raw_preprocessor, train_index, test_index,
        data["scfound_all"], data["geneformer_all"], data["raw_all"],
        data["y_fine"], data["y_function"], label_map, func_map,
        data["scfound_out"], data["geneformer_out"], data["raw_out"],
        data["y_function_out"], data["scfound_rest"], data["geneformer_rest"],
        data["raw_rest"],
    )
    if recover_partial:
        return recover_partial_fold_outputs(
            fold,
            train_index,
            test_index,
            fold_root,
            fold_params,
            device,
            encoder,
            label_map,
            func_map,
            num2label,
            num2func,
            data,
            datasets,
            raw_preprocessor,
        )
    write_json(fold_root / "raw_preprocessor.json", raw_preprocessor.state_dict())
    write_json(fold_root / "fold_config.json", cfg.serializable_params(fold_params))

    train_loader = build_train_loader(
        datasets[0], fold_params, device,
        data["y_fine"][train_index], data["y_function"][train_index],
        seed=int(params["seed"]) + fold,
    )
    loaders = {
        "train": train_loader,
        "test": build_eval_loader(datasets[1], fold_params, device),
        "out": build_eval_loader(datasets[2], fold_params, device),
        "rest": build_eval_loader(datasets[3], fold_params, device),
    }
    model = make_model(fold_params, device)
    criteria = make_criteria(fold_params, model, encoder, device)
    optimizer, scheduler = make_optimizer_scheduler(
        model, criteria, fold_params, int(fold_params['epochs']),
    )
    amp_dtype = resolve_amp_dtype(fold_params)
    scaler = torch.cuda.amp.GradScaler(
        enabled=fold_params["use_amp"] and device.type == "cuda" and amp_dtype == torch.float16
    )
    metrics = make_metrics(fold_params["output_dim"], fold_params["group_output_dim"], device)
    baseline = load_baseline_metrics(fold_params["baseline_metrics_path"], fold)
    history: list[dict[str, Any]] = []
    best_score = (float("-inf"),) * 3
    best_any_score = (float("-inf"),) * 3
    best_epoch = None
    start = time.time()
    epoch_times = []

    for epoch in range(fold_params["epochs"]):
        stage = "single"
        stage_epoch = epoch + 1
        stage_total = fold_params["epochs"]
        epoch_start = time.perf_counter()
        for metric in metrics.values():
            metric.reset()
        train_epoch(
            model, train_loader, optimizer, criteria, metrics, fold_params,
            encoder, device, scaler, fold, epoch,
        )
        due = (
            ((epoch + 1) % fold_params["unknown_metric_every"] == 0)
            or stage_epoch == stage_total
            or epoch + 1 == fold_params["epochs"]
        )
        test_results = evaluate_epoch(
            model, loaders["test"], criteria, metrics, fold_params, encoder, device,
            mode="test", collect_outputs=due, update_metrics=True,
        )
        evaluate_epoch(
            model, loaders["out"], criteria, metrics, fold_params, encoder, device,
            mode="out", collect_outputs=False, update_metrics=True,
        )
        scheduler.step(stage_epoch - 1)
        snapshot = metric_snapshot(metrics)
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **snapshot,
        }

        if due:
            unknown_report, unknown_rows = compute_unknown_clustering_metrics(
                test_results["feat_raw"], test_results["labels"],
                test_results["group_labels"],
                knn_k=fold_params["unknown_metric_knn_k"],
                min_class_cells=fold_params["unknown_metric_min_class_cells"],
                enrichment_threshold=fold_params["unknown_metric_enrichment_threshold"],
                device=device,
            )
            current_classification = {name: snapshot[name] for name in GUARDRAIL_METRICS}
            guardrail, drops = classification_guardrail_pass(
                current_classification, baseline, fold_params["classification_guardrail_max_drop"]
            )
            score = sanitized_unknown_score(unknown_report)
            record.update({f"unknown_{key}": value for key, value in unknown_report.items()})
            record["guardrail_pass"] = guardrail
            record["guardrail_drops"] = drops
            epoch_metric_root = fold_root / "epoch_metrics"
            write_json(epoch_metric_root / f"epoch_{epoch + 1:03d}.json", {
                "epoch": epoch + 1,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "classification": current_classification,
                "unknown": unknown_report,
                "guardrail_pass": guardrail,
                "guardrail_drops": drops,
            })
            write_rows_csv(
                epoch_metric_root / f"epoch_{epoch + 1:03d}_classes.csv",
                append_label_names_to_unknown_rows(unknown_rows, num2label),
            )
            checkpoint_args = dict(
                model=model, criteria=criteria, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch + 1, params=fold_params, encoder=encoder,
                raw_preprocessor=raw_preprocessor, unknown_report=unknown_report,
                classification_metrics=current_classification, guardrail_pass=guardrail,
                guardrail_drops=drops,
                training_stage=stage, stage_epoch=stage_epoch,
            )
            if score > best_any_score:
                best_any_score = score
                save_checkpoint(fold_root / "best_unconstrained.pth", **checkpoint_args)
            elif not (fold_root / "best_unconstrained.pth").exists():
                save_checkpoint(fold_root / "best_unconstrained.pth", **checkpoint_args)
            if guardrail and (
                score > best_score or not (fold_root / "best.pth").exists()
            ):
                best_score = score
                best_epoch = epoch + 1
                save_checkpoint(fold_root / "best.pth", **checkpoint_args)

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_times.append(epoch_seconds)
        remaining = np.mean(epoch_times[-5:]) * (fold_params["epochs"] - epoch - 1)
        eta = datetime.now() + timedelta(seconds=float(remaining))
        record["epoch_seconds"] = epoch_seconds
        history.append(record)
        train_text = " | ".join(
            f"{name}: {snapshot[name]:.4f}" for name in (
                "train_loss", "train_ce_loss", "train_center_loss",
                "train_ce_group_loss", "train_center_loss_group",
                "train_MCC_loss",
                "train_MCC_loss_group", "train_feat_MCC_loss",
            )
        )
        val_text = " | ".join(f"{name}: {snapshot[name]:.4f}" for name in GUARDRAIL_METRICS)
        line = (
            f"Epoch {epoch + 1}/{fold_params['epochs']} | {stage} "
            f"{stage_epoch}/{stage_total} | LR: {optimizer.param_groups[0]['lr']:.2e} | {train_text}\n"
            f"Val Results: | {val_text}\n"
            f"Timing: epoch {format_duration(epoch_seconds)} | remaining {format_duration(remaining)} | ETA {eta:%Y-%m-%d %H:%M:%S}"
        )
        print(line)
        write_log(fold_root / "log.txt", line)

    final_unknown = history[-1]
    save_checkpoint(
        fold_root / "last.pth", model=model, criteria=criteria, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, epoch=fold_params["epochs"], params=fold_params,
        encoder=encoder, raw_preprocessor=raw_preprocessor,
        unknown_report={key.removeprefix("unknown_"): value for key, value in final_unknown.items() if key.startswith("unknown_")},
        classification_metrics={name: final_unknown[name] for name in GUARDRAIL_METRICS},
        guardrail_pass=bool(final_unknown.get("guardrail_pass", baseline is None)),
        guardrail_drops=final_unknown.get("guardrail_drops", {}),
        stage_epoch=fold_params["epochs"],
    )
    if best_epoch is None:
        shutil.copy2(fold_root / "best_unconstrained.pth", fold_root / "best.pth")
        fallback = torch.load(
            fold_root / "best.pth", map_location="cpu", weights_only=False
        )
        best_epoch = int(fallback["epoch"])
        write_json(fold_root / "guardrail_warning.json", {
            "message": "No checkpoint passed the supplied classification guardrail; best.pth is the clustering-best fallback.",
            "epoch": best_epoch,
        })

    if fold_params["save_last_embeddings"]:
        last_test, _, _, _ = _evaluate_and_export(
            "last", fold_root, model, loaders, criteria, metrics, fold_params,
            encoder, device, num2label,
        )
    best_checkpoint = load_model_checkpoint(model, fold_root / "best.pth", device)
    best_test, best_classification, best_unknown, _ = _evaluate_and_export(
        "best", fold_root, model, loaders, criteria, metrics, fold_params,
        encoder, device, num2label,
    )
    guardrail, drops = classification_guardrail_pass(
        best_classification, baseline, fold_params["classification_guardrail_max_drop"]
    )
    acceptance = {
        "unknown_trend_coverage": best_unknown.get("trend_coverage", 0.0),
        "unknown_target_coverage": fold_params["unknown_metric_target_coverage"],
        "unknown_coverage_pass": best_unknown.get("trend_coverage", 0.0) >= fold_params["unknown_metric_target_coverage"],
        "classification_guardrail_available": baseline is not None,
        "classification_guardrail_pass": guardrail if baseline is not None else None,
        "classification_drops": drops,
    }
    summary = {
        "fold": fold,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_checkpoint_guardrail_pass": bool(best_checkpoint["guardrail_pass"]),
        "classification": best_classification,
        "unknown": best_unknown,
        "acceptance": acceptance,
        "elapsed_seconds": time.time() - start,
    }
    write_json(fold_root / "history.json", history)
    write_json(fold_root / "summary.json", summary)
    validate_fold_artifacts(
        fold_root,
        fold_params["save_last_embeddings"],
    )
    write_json(fold_root / "completed.json", {"completed": True, "fold": fold})


def aggregate_fold_summaries(experiment_root):
    summaries = []
    for path in sorted(Path(experiment_root).glob("fold*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    if not summaries:
        return None
    aggregate = {
        "fold_count": len(summaries),
        "aggregation_note": "Fold metrics are summarized per fold; overlapping repeated hold-outs are not concatenated.",
        "classification": {},
        "unknown": {},
        "acceptance": {
            "folds_unknown_coverage_pass": int(sum(row["acceptance"]["unknown_coverage_pass"] for row in summaries)),
            "folds_classification_guardrail_available": int(
                sum(row["acceptance"]["classification_guardrail_available"] for row in summaries)
            ),
            "folds_classification_guardrail_pass": int(
                sum(row["acceptance"]["classification_guardrail_pass"] is True for row in summaries)
            ),
        },
    }
    for section, names in (
        ("classification", GUARDRAIL_METRICS),
        ("unknown", ("trend_coverage", "median_class_enrichment", "micro_knn_purity")),
    ):
        for name in names:
            values = np.asarray([row[section][name] for row in summaries], dtype=float)
            aggregate[section][name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "per_fold": values.tolist(),
            }
    write_json(Path(experiment_root) / "aggregate_metrics.json", aggregate)
    return aggregate




def main(argv=None):
    args = build_parser().parse_args(argv)
    params = resolve_params(args)
    device = torch.device(params["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu for a smoke run.")
    amp_enabled = params["use_amp"] and device.type == "cuda"
    if amp_enabled and resolve_amp_dtype(params) == torch.bfloat16:
        supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supported:
            raise RuntimeError("BF16 AMP was requested but this CUDA runtime does not support it.")
    set_seed(int(params["seed"]))
    paths = params["input_paths"]
    scfound_all, geneformer_all, raw_all = du.load_features(
        paths["scfoundation"], paths["geneformer"], paths["raw_h5ad"]
    )
    if SOURCE_ROW_INDEX_COLUMN in raw_all.obs.columns:
        raise ValueError(
            f"Source obs already contains reserved internal column {SOURCE_ROW_INDEX_COLUMN!r}."
        )
    raw_all.obs[SOURCE_ROW_INDEX_COLUMN] = np.arange(raw_all.n_obs, dtype=np.int64)
    scfound_all, geneformer_all, raw_all, scfound_rest, geneformer_rest, raw_rest = du.filter_data(
        scfound_all, geneformer_all, raw_all, cfg.FILTER_LEAST
    )
    unknown_function_mask = np.asarray([
        du.is_unknown_function_name(value)
        for value in raw_all.obs["function"].to_numpy()
    ], dtype=bool)
    unknown_filter_mask, unknown_filter_report = build_unknown_category_filter_mask(
        raw_all.obs["fea_ca"].astype(str).to_numpy(),
        unknown_function_mask,
        min_samples=int(params["unknown_min_samples"]),
    )
    if unknown_filter_report["unknown_classes_after"] == 0:
        raise ValueError(
            "unknown_min_samples removed every Unknown category; lower the threshold."
        )
    if not unknown_filter_mask.all():
        scfound_all = scfound_all[unknown_filter_mask]
        geneformer_all = geneformer_all[unknown_filter_mask]
        raw_all = raw_all[unknown_filter_mask].copy()
    scfound_out, geneformer_out, raw_out = du.load_features(
        paths["scfoundation_out"], paths["geneformer_out"], paths["raw_out_h5ad"]
    )
    label_map, func_map = du.create_label_maps(raw_all)
    raw_out, scfound_out, geneformer_out = du.filter_ood_data(
        raw_out, scfound_out, geneformer_out, func_map
    )
    num2label = {int(value): key for key, value in label_map.items()}
    num2func = {int(value): key for key, value in func_map.items()}
    y_fine = raw_all.obs["fea_ca"].map(label_map).to_numpy(dtype=np.int64)
    y_function = encode_function_labels(raw_all.obs["function"].to_numpy(), func_map)
    y_function_out = encode_function_labels(raw_out.obs["function"].to_numpy(), func_map)
    encoder = UnifiedGeometryLabelEncoder.fit(
        y_fine, y_function, fine_class_count=len(label_map)
    )
    params.update({
        "output_dim": len(label_map),
        "group_output_dim": len(func_map),
        "unified_geometry_num_classes": encoder.num_classes,
        "input_dim_scfound": int(scfound_all.shape[1]),
        "input_dim_geneformer": int(geneformer_all.shape[1]),
    })
    experiment_root = Path(cfg.experiment_directory(params, args.output_root))
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_json(experiment_root / "resolved_config.json", cfg.serializable_params(params))
    write_json(experiment_root / "unknown_category_filter.json", unknown_filter_report)
    label_mapping_payload = {
        "label_map": {str(key): int(value) for key, value in label_map.items()},
        "function_map": {str(key): int(value) for key, value in func_map.items()},
        "unified": encoder.state_dict(),
        "unified_names": encoder.names(num2func, num2label),
    }
    write_json(experiment_root / "label_mapping.json", label_mapping_payload)
    sample_ids_all = np.asarray(raw_all.obs_names.astype(str))
    splitter = StratifiedShuffleSplit(
        n_splits=params["foldn"], test_size=1.0 / params["foldn"],
        random_state=int(params["seed"]),
    )
    split_indices = list(splitter.split(scfound_all, y_fine))
    save_split_manifest(experiment_root, split_indices, sample_ids_all)
    selected_folds = list(range(len(split_indices)))
    recovery_folds = {int(value) for value in args.recover_partial_fold}
    invalid_recovery_folds = sorted(
        recovery_folds.difference(range(len(split_indices)))
    )
    if invalid_recovery_folds:
        raise ValueError(
            "recover-partial-fold values must be within "
            f"[0, {len(split_indices) - 1}]; got {invalid_recovery_folds}."
        )
    if args.only_fold is not None:
        if args.only_fold not in selected_folds:
            raise ValueError(f"only-fold must be within [0, {len(split_indices) - 1}].")
        selected_folds = [args.only_fold]
    if args.fold_limit is not None:
        if args.fold_limit < 1:
            raise ValueError("fold-limit must be positive.")
        selected_folds = selected_folds[: args.fold_limit]
    unselected_recovery_folds = sorted(recovery_folds.difference(selected_folds))
    if unselected_recovery_folds:
        raise ValueError(
            "Recovery was requested for folds excluded by --only-fold/--fold-limit: "
            f"{unselected_recovery_folds}."
        )
    data = {
        "scfound_all": scfound_all, "geneformer_all": geneformer_all, "raw_all": raw_all,
        "scfound_out": scfound_out, "geneformer_out": geneformer_out, "raw_out": raw_out,
        "scfound_rest": scfound_rest, "geneformer_rest": geneformer_rest, "raw_rest": raw_rest,
        "y_fine": y_fine, "y_function": y_function, "y_function_out": y_function_out,
        "sample_ids_all": sample_ids_all,
        "sample_ids_out": np.asarray(raw_out.obs_names.astype(str)),
        "sample_ids_rest": np.asarray(raw_rest.obs_names.astype(str)),
    }
    print(
        f"Experiment: {experiment_root}\n"
        f"Profile={params['profile']} | unified_classes={encoder.num_classes} "
        f"({encoder.known_function_count} known functions + {len(encoder.unknown_fine_ids)} Unknown fea_ca)\n"
        f"Folds={selected_folds} | center loss on feat_raw | "
        f"sampling={params['sampling_mode']} | raw={params['raw_preprocess_mode']}\n"
        f"Unknown min samples >= {params['unknown_min_samples']} | "
        f"Unknown classes retained={unknown_filter_report['unknown_classes_after']}/"
        f"{unknown_filter_report['unknown_classes_before']} | "
        f"recover_partial_folds={sorted(recovery_folds)}"
    )
    for fold in selected_folds:
        train_index, test_index = split_indices[fold]
        run_fold(
            fold, train_index, test_index, experiment_root, params, device,
            encoder, label_map, func_map, num2label, num2func, data,
            recover_partial=fold in recovery_folds,
        )
    aggregate_fold_summaries(experiment_root)
    print(f"Finished. Outputs: {experiment_root}")


if __name__ == "__main__":
    main()
