# train.py

import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import torchmetrics
import timm.scheduler

# Local application imports
from . import config as cfg
from . import data as du
from .. import models
from . import utils

# --- Metrics Initialization ---
def get_metrics_dict(num_classes: int, num_groups: int, device: str) -> dict:
    """Initializes torchmetrics objects for tracking performance."""
    metrics = {
        # Training metrics
        "train_loss": torchmetrics.MeanMetric(),
        "train_ce_loss": torchmetrics.MeanMetric(),
        "train_center_loss": torchmetrics.MeanMetric(),
        "train_ce_group_loss": torchmetrics.MeanMetric(),
        "train_center_loss_group": torchmetrics.MeanMetric(),
        "train_MCC_loss": torchmetrics.MeanMetric(),
        "train_MCC_loss_group": torchmetrics.MeanMetric(),
        "train_feat_MCC_loss": torchmetrics.MeanMetric(),
        "train_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "train_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device),
        "train_group_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_groups).to(device),
        "train_group_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_groups).to(device),
        
        # Validation metrics
        "test_loss": torchmetrics.MeanMetric(),
        "test_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "test_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device),
        "test_group_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_groups).to(device),
        "test_group_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_groups).to(device),

        # OOD metrics
        "out_group_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_groups).to(device),
        "out_group_auc": torchmetrics.AUROC(task="multiclass", num_classes=num_groups).to(device),
    }
    return metrics

# --- Training Step ---
def train_epoch(model, loader, optimizer, criteria, metrics, params, device):
    model.train()
    for sc_found, geneformer, raw_data, label, func in loader:
        sc_found, geneformer, raw_data, label, func = \
            sc_found.to(device), geneformer.to(device), raw_data.to(device), label.to(device), func.to(device)

        optimizer.zero_grad()

        # Forward pass 1 for MCC
        output1, output_group1, feat_raw1, _, _ = model(sc_found, geneformer, raw_data, data_mask=params["data_mask"])
        # Forward pass 2 for MCC
        output2, output_group2, feat_raw2, _, _ = model(sc_found, geneformer, raw_data, data_mask=params["data_mask"])

        # Calculate losses for pass 1
        loss_ce1 = criteria["ce"](output1, label)
        loss_center1 = criteria["center"](feat_raw1, label)
        loss_ce_group1 = criteria["ce_group"](output_group1, func)
        loss_center_group1 = criteria["center_group"](feat_raw1, func)
        loss1 = (params["celoss_weight"] * loss_ce1 +
                 params["center_loss_weight"] * loss_center1 +
                 params["celoss_group_weight"] * loss_ce_group1 +
                 params["center_loss_group_weight"] * loss_center_group1)

        # Calculate losses for pass 2
        loss_ce2 = criteria["ce"](output2, label)
        loss_center2 = criteria["center"](feat_raw2, label)
        loss_ce_group2 = criteria["ce_group"](output_group2, func)
        loss_center_group2 = criteria["center_group"](feat_raw2, func)
        loss2 = (params["celoss_weight"] * loss_ce2 +
                 params["center_loss_weight"] * loss_center2 +
                 params["celoss_group_weight"] * loss_ce_group2 +
                 params["center_loss_group_weight"] * loss_center_group2)

        # MCC losses
        loss_MCC = criteria["MCC"](output1, output2)
        loss_MCC_group = criteria["MCC"](output_group1, output_group2)
        loss_MCC_feat = criteria["MCC"](feat_raw1, feat_raw2) # Note: MCCLoss uses CE formulation, applying to features might need L1/L2/KL divergence instead. Sticking to original implementation structure.

        total_loss = ((loss1 + loss2) / 2 +
                      params["feat_MCC_weight"] * loss_MCC_feat +
                      params["MCC_weight"] * loss_MCC +
                      params["MCC_group_weight"] * loss_MCC_group)
        
        total_loss.backward()
        optimizer.step()

        # Update metrics (using pass 1 results for standard metrics)
        metrics["train_loss"].update(total_loss.item())
        metrics["train_ce_loss"].update(loss_ce1.item())
        metrics["train_center_loss"].update(loss_center1.item())
        metrics["train_ce_group_loss"].update(loss_ce_group1.item())
        metrics["train_center_loss_group"].update(loss_center_group1.item())
        metrics["train_MCC_loss"].update(loss_MCC.item())
        metrics["train_MCC_loss_group"].update(loss_MCC_group.item())
        metrics["train_feat_MCC_loss"].update(loss_MCC_feat.item())
        metrics["train_acc"].update(output1, label)
        metrics["train_auc"].update(output1, label)
        metrics["train_group_acc"].update(output_group1, func)
        metrics["train_group_auc"].update(output_group1, func)

# --- Evaluation Step ---
@torch.no_grad()
def evaluate_epoch(model, loader, criteria, metrics, params, device, mode="test"):
    model.eval()
    all_results = {
        "features": [], "logits": [], "preds": [], "labels": [],
        "group_logits": [], "group_preds": [], "group_labels": [],
        "feat_cls": [], "feat_group": []
    }

    for batch in loader:
        if mode == "test":
            sc_found, geneformer, raw_data, label, func = batch
            label = label.to(device)
            all_results["labels"].append(label.cpu())
        else: # mode == "out" or "inference"
            # OOD loader structure check: sc, gene, raw_data, func
            if len(batch) == 4:
                 sc_found, geneformer, raw_data, func = batch
            else: # inference loader structure: sc, gene, raw_data
                 sc_found, geneformer, raw_data = batch
                 func = None # No ground truth function label for inference set

        sc_found, geneformer, raw_data = sc_found.to(device), geneformer.to(device), raw_data.to(device)
        if func is not None:
            func = func.to(device)

        output, output_group, feat_raw, feat_cls, feat_group = model(sc_found, geneformer, raw_data, data_mask=params["data_mask"])

        all_results["features"].append(feat_raw.cpu())
        all_results["logits"].append(output.cpu())
        all_results["preds"].append(output.argmax(dim=1).cpu())
        all_results["feat_cls"].append(feat_cls.cpu())
        all_results["feat_group"].append(feat_group.cpu())
        
        if func is not None:
            all_results["group_logits"].append(output_group.cpu())
            all_results["group_preds"].append(output_group.argmax(dim=1).cpu())
            all_results["group_labels"].append(func.cpu())

            # Calculate validation/OOD loss and metrics only if labels are present
            if mode == "test":
                loss_ce = criteria["ce"](output, label)
                loss_center = criteria["center"](feat_raw, label)
                loss_ce_group = criteria["ce_group"](output_group, func)
                loss_center_group = criteria["center_group"](feat_raw, func)
                loss = (params["celoss_weight"] * loss_ce + params["center_loss_weight"] * loss_center +
                        params["celoss_group_weight"] * loss_ce_group + params["center_loss_group_weight"] * loss_center_group)
                
                metrics["test_loss"].update(loss.item())
                metrics["test_acc"].update(output, label)
                metrics["test_auc"].update(output, label)
                metrics["test_group_acc"].update(output_group, func)
                metrics["test_group_auc"].update(output_group, func)
            
            elif mode == "out":
                metrics["out_group_acc"].update(output_group, func)
                metrics["out_group_auc"].update(output_group, func)

    # Concatenate results from all batches
    for key, val_list in all_results.items():
        if val_list:
            all_results[key] = np.concatenate(val_list, axis=0)
        else:
            all_results[key] = np.array([]) # Handle empty lists if loader was empty or no data collected for key
            
    return all_results

# --- Main Orchestration ---
def main():
    # 1. Setup: Seed and Configuration
    utils.set_seed(cfg.SEED)
    params = cfg.HYPERPARAMETERS
    device = torch.device(cfg.DEVICE)

    # 2. Data Loading and Preprocessing
    scfound_all, geneformer_all, raw_all = du.load_features(cfg.SCFOUNDATION_NPY, cfg.GENEFORMER_PICKLE, cfg.RAW_H5AD)
    scfound_all, geneformer_all, raw_all, scfound_rest, geneformer_rest, raw_rest = du.filter_data(
        scfound_all, geneformer_all, raw_all, cfg.FILTER_LEAST
    )
    
    # Load OOD data
    scfound_out, geneformer_out, raw_out = du.load_features(cfg.SCFOUNDATION_OUT_NPY, cfg.GENEFORMER_OUT_PICKLE, cfg.RAW_OUT_H5AD)

    # Create label mappings based on training data
    label_map, func_map = du.create_label_maps(raw_all)
    num2label = {v: k for k, v in label_map.items()}
    num2func = {v: k for k, v in func_map.items()}
    
    # Filter OOD data based on training function map
    raw_out, scfound_out, geneformer_out = du.filter_ood_data(raw_out, scfound_out, geneformer_out, func_map)

    # Update params with dynamic dimensions
    params["output_dim"] = len(label_map)
    params["group_output_dim"] = len(func_map)
    params["input_dim_scfound"] = scfound_all.shape[1]
    params["input_dim_geneformer"] = geneformer_all.shape[1]
    raw_feature_count = params["num_of_gene_by_var"] + params["num_of_gene_by_mean"]
    params["input_dim_raw"] = raw_feature_count

    # Prepare data for cross-validation splitting
    X_raw_tensor = torch.tensor(raw_all.X.toarray(), dtype=torch.float32)
    y_label_series = raw_all.obs['fea_ca'].map(label_map)
    y_function_series = raw_all.obs['function']
    
    # Prepare OOD data tensors
    X_raw_out_tensor = torch.tensor(raw_out.X.toarray(), dtype=torch.float32)
    y_label_out_series = raw_out.obs['fea_ca'] # Note: might contain labels not in label_map
    y_function_out_series = raw_out.obs['function']
    
    # Prepare test_rest data tensors
    X_raw_rest_tensor = torch.tensor(raw_rest.X.toarray(), dtype=torch.float32)

    # 3. Cross-validation Training Loop
    sss = StratifiedShuffleSplit(n_splits=params["foldn"], test_size=1.0/params["foldn"], random_state=cfg.SEED)
    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(sss.split(scfound_all, y_label_series)):
        print(f"\n--- Starting Fold {fold + 1}/{params['foldn']} ---")
        save_root = os.path.join(cfg.get_experiment_paths(params, cfg.SEED), f"fold{fold}")
        log_file = os.path.join(save_root, "log.txt")
        if os.path.exists(save_root):
            print(f"Fold directory {save_root} already exists, skipping...")
            continue
        os.makedirs(save_root, exist_ok=True)
        utils.write_log(log_file, f"Fold: {fold}\nSave root: {save_root}\nParams: {params}", mode="w")

        # --- Data Splitting and Feature Selection for current fold ---
        X_scfound_train, X_scfound_test = scfound_all[train_index], scfound_all[test_index]
        X_geneformer_train, X_geneformer_test = geneformer_all[train_index], geneformer_all[test_index]
        X_raw_train, X_raw_test = X_raw_tensor[train_index], X_raw_tensor[test_index]
        y_train, y_test = y_label_series.iloc[train_index], y_label_series.iloc[test_index]
        y_function_train, y_function_test = y_function_series.iloc[train_index], y_function_series.iloc[test_index]

        # Feature selection based on training data of this fold
        var_idx = du.get_variance_features(X_raw_train, y_train, params["num_of_gene_by_var"])
        mean_idx = du.get_mean_features(X_raw_train, params["num_of_gene_by_mean"])
        # selected_indices = torch.unique(torch.cat((var_idx, mean_idx)))
        selected_indices = torch.cat((var_idx, mean_idx))
        print(f"Selected raw feature count: {len(selected_indices)}")

        X_raw_train_sampled = du.select_features(X_raw_train, selected_indices)
        X_raw_test_sampled = du.select_features(X_raw_test, selected_indices)
        X_raw_out_sampled = du.select_features(X_raw_out_tensor, selected_indices)
        X_raw_rest_sampled = du.select_features(X_raw_rest_tensor, selected_indices)

        # --- Data Loaders ---
        train_dataset = du.SimpleDataset(X_scfound_train, X_geneformer_train, X_raw_train_sampled, y_train, y_function_train, label_map, func_map, mode="train")
        test_dataset = du.SimpleDataset(X_scfound_test, X_geneformer_test, X_raw_test_sampled, y_test, y_function_test, label_map, func_map, mode="test")
        out_dataset = du.SimpleDataset(scfound_out, geneformer_out, X_raw_out_sampled, y_label_out_series, y_function_out_series, label_map, func_map, mode="out")
        rest_dataset = du.SimpleDataset(scfound_rest, geneformer_rest, X_raw_rest_sampled, None, None, label_map, func_map, mode="inference")

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=8)
        out_loader = DataLoader(out_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=8)
        rest_loader = DataLoader(rest_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=8)

        # --- Model Initialization ---
        # Update input_dim_raw based on actual selected features count for this fold
        params["input_dim_raw"] = X_raw_train_sampled.shape[1] 
        model = models.PertWeave(
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
            efficient_num=params["efficient_num"]
        ).to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model = nn.DataParallel(model)

        utils.write_log(log_file, str(model))

        # --- Loss Functions and Optimizer ---
        # Calculate feature dimension after reductions for CenterLoss initialization
        feat_dim_centerloss = params["hidden_dim"] * sum([int(b) for b in params["data_mask"]])
        for _ in range(params["layer_num"]):
            feat_dim_centerloss //= params["reduction_factor"]

        criteria = {
            "ce": nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"]),
            "ce_group": nn.CrossEntropyLoss(label_smoothing=params["label_smoothing_group"]),
            "center": models.CenterLoss(num_classes=params["output_dim"], feat_dim=feat_dim_centerloss, device=device),
            "center_group": models.CenterLoss(num_classes=params["group_output_dim"], feat_dim=feat_dim_centerloss, device=device),
            "MCC": models.MCCLoss(temperature=params["MCC_temperature"]),
        }
        
        model_params = list(model.parameters()) + list(criteria["center"].parameters()) + list(criteria["center_group"].parameters())
        optimizer = torch.optim.AdamW(model_params, lr=params["raw_lr"], weight_decay=params["weight_decay"])
        scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=params["epochs"], lr_min=params["final_lr"], warmup_t=10, warmup_lr_init=params["final_lr"])

        # --- Training Loop ---
        metrics_dict = get_metrics_dict(params["output_dim"], params["group_output_dim"], device)
        history = {key: [] for key in metrics_dict.keys() if "auc" not in key and "acc" not in key} # Simplify history tracking

        for epoch in range(params["epochs"]):
            # Reset metrics for epoch
            for metric in metrics_dict.values():
                metric.reset()

            # Train step
            train_epoch(model, train_loader, optimizer, criteria, metrics_dict, params, device)
            
            # Validation step
            test_results = evaluate_epoch(model, test_loader, criteria, metrics_dict, params, device, mode="test")
            
            # OOD evaluation step
            _ = evaluate_epoch(model, out_loader, criteria, metrics_dict, params, device, mode="out")

            scheduler.step(epoch)
            
            # Logging results for the epoch
            log_str = f"Epoch {epoch+1}/{params['epochs']} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
            for name, metric in metrics_dict.items():
                if "train_" in name and "loss" in name: # Log train losses immediately
                    log_str += f"{name}: {metric.compute().item():.4f} | "
            print(log_str)

            test_log_str = "Val Results: | "
            for name, metric in metrics_dict.items():
                if "test_" in name or "out_" in name or "train_acc" in name or "train_auc" in name:
                    value = metric.compute().item()
                    history.setdefault(name, []).append(value)
                    test_log_str += f"{name}: {value:.4f} | "
            print(test_log_str)
            utils.write_log(log_file, log_str + "\n" + test_log_str)

        # --- Post-Training Saving and Evaluation for Fold ---
        print(f"Training finished for fold {fold}. Saving results...")
        torch.save(model.state_dict(), os.path.join(save_root, "model.pth"))
        utils.save_to_pickle(history, os.path.join(save_root, "history.pickle"))
        utils.save_to_pickle(label_map, os.path.join(save_root, "label_map.pickle"))
        utils.save_to_pickle(func_map, os.path.join(save_root, "func_map.pickle"))

        # Final evaluation and data saving
        test_results = evaluate_epoch(model, test_loader, criteria, metrics_dict, params, device, mode="test")
        out_results = evaluate_epoch(model, out_loader, criteria, metrics_dict, params, device, mode="out")
        rest_results = evaluate_epoch(model, rest_loader, criteria, metrics_dict, params, device, mode="inference")

        for results_dict, name in [(test_results, "test_data"), (out_results, "out_data"), (rest_results, "test_rest_data")]:
            results_path = os.path.join(save_root, name)
            os.makedirs(results_path, exist_ok=True)
            for key, value in results_dict.items():
                np.save(os.path.join(results_path, f"{key}.npy"), value)

        # Plotting confusion matrix for this fold
        utils.plot_confusion_matrix(test_results["group_labels"], test_results["group_preds"], os.path.join(save_root, "cm_test_group.png"), func_map)
        utils.plot_confusion_matrix(out_results["group_labels"], out_results["group_preds"], os.path.join(save_root, "cm_out_group.png"), func_map)

        # Plotting t-SNE/UMAP for this fold
        utils.plot_tsne_umap_by_group(test_results["features"], test_results["group_labels"], os.path.join(save_root, "plots", "test"), num2func)
        utils.plot_tsne_umap_by_group(out_results["features"], out_results["group_labels"], os.path.join(save_root, "plots", "out"), num2func)

    print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    
    # 4. Final results aggregation across folds (add a separate function if needed)
    aggregate_and_report_results(cfg.get_experiment_paths(params, cfg.SEED), params["foldn"], num2label, num2func)


def aggregate_and_report_results(root_path: str, foldn: int, num2label: dict, num2func: dict):
    """Loads results from all folds, aggregates them, calculates final metrics, and saves reports."""
    print("\n--- Aggregating results across all folds ---")
    all_fold_results = {
        "test_logits": [], "test_labels": [], "test_group_logits": [], "test_group_labels": [],
        "out_group_logits": [], "out_group_labels": []
    }
    
    for fold in range(foldn):
        fold_root = os.path.join(root_path, f"fold{fold}")
        if not os.path.exists(os.path.join(fold_root, "test_data")):
            print(f"Warning: Results for fold {fold} not found. Skipping aggregation for this fold.")
            continue
        
        all_fold_results["test_logits"].append(np.load(os.path.join(fold_root, "test_data", "logits.npy")))
        all_fold_results["test_labels"].append(np.load(os.path.join(fold_root, "test_data", "labels.npy")))
        all_fold_results["test_group_logits"].append(np.load(os.path.join(fold_root, "test_data", "group_logits.npy")))
        all_fold_results["test_group_labels"].append(np.load(os.path.join(fold_root, "test_data", "group_labels.npy")))
        all_fold_results["out_group_logits"].append(np.load(os.path.join(fold_root, "out_data", "group_logits.npy")))
        all_fold_results["out_group_labels"].append(np.load(os.path.join(fold_root, "out_data", "group_labels.npy")))

    if not all_fold_results["test_labels"]:
        print("Error: No fold results found to aggregate.")
        return

    # Concatenate results
    final_results = {}
    for key, value_list in all_fold_results.items():
        final_results[key] = np.concatenate(value_list, axis=0)

    # Calculate final metrics
    test_preds = final_results["test_logits"].argmax(axis=1)
    test_group_preds = final_results["test_group_logits"].argmax(axis=1)
    out_group_preds = final_results["out_group_logits"].argmax(axis=1)
    test_probs = torch.softmax(torch.tensor(final_results["test_logits"]), dim=1)
    test_group_probs = torch.softmax(torch.tensor(final_results["test_group_logits"]), dim=1)
    out_group_probs = torch.softmax(torch.tensor(final_results["out_group_logits"]), dim=1)

    final_metrics_report = ""
    num_classes = len(num2label)
    num_groups = len(num2func)

    # Test Metrics (fine-grained label)
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)(torch.tensor(test_preds), torch.tensor(final_results["test_labels"]))
    test_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)(test_probs, torch.tensor(final_results["test_labels"]))
    final_metrics_report += f"Final Test Accuracy: {test_acc.item():.4f}\n"
    final_metrics_report += f"Final Test AUC: {test_auc.item():.4f}\n"

    # Test Metrics (group label)
    test_group_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_groups)(torch.tensor(test_group_preds), torch.tensor(final_results["test_group_labels"]))
    test_group_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_groups)(test_group_probs, torch.tensor(final_results["test_group_labels"]))
    final_metrics_report += f"Final Test Group Accuracy: {test_group_acc.item():.4f}\n"
    final_metrics_report += f"Final Test Group AUC: {test_group_auc.item():.4f}\n"

    # OOD Metrics (group label)
    out_group_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_groups)(torch.tensor(out_group_preds), torch.tensor(final_results["out_group_labels"]))
    out_group_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_groups)(out_group_probs, torch.tensor(final_results["out_group_labels"]))
    final_metrics_report += f"Final OOD Group Accuracy: {out_group_acc.item():.4f}\n"
    final_metrics_report += f"Final OOD Group AUC: {out_group_auc.item():.4f}\n"

    print(final_metrics_report)
    utils.write_log(os.path.join(root_path, "final_results.txt"), final_metrics_report, mode="w")

    # Plot final confusion matrices
    utils.plot_confusion_matrix(final_results["test_group_labels"], test_group_preds, os.path.join(root_path, "final_test_group_confusion_matrix.png"), {v: k for k, v in num2func.items()})
    utils.plot_confusion_matrix(final_results["out_group_labels"], out_group_preds, os.path.join(root_path, "final_out_group_confusion_matrix.png"), {v: k for k, v in num2func.items()})


if __name__ == "__main__":
    main()
