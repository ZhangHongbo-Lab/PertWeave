<p align="center">
  <img src="assets/pertweave_banner.png" alt="PertWeave" width="100%">
</p>

# PertWeave

PertWeave predicts perturbation identity and functional group from single-cell
gene expression, scFoundation embeddings and Geneformer CLS embeddings.
Run the commands below from the repository root. Choose the original experiment
for function-annotated perturbations or the supplementary experiment to include
perturbations of unknown function.

## Installation and inputs

Both experiments use the same dependencies. From the repository root, choose
**one** environment below (64-bit Linux or Windows, x86-64). Keep the YAML
and `requirements.txt` together. A fresh environment is recommended.

**NVIDIA GPU training** (official PyTorch CUDA 12.1 wheels):

```bash
conda env create -f environment.yml
conda activate pertweave-cu121
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

A compatible NVIDIA driver is required. Verify that CUDA is available before
training; the default Unknown configuration also requires BF16 support.
The wheel supplies the CUDA runtime; these scripts do not compile CUDA kernels.
If your server requires another PyTorch build, consult the
[official PyTorch 2.5.1 installation matrix](https://pytorch.org/get-started/previous-versions/#v251)
and change the index and both wheel suffixes together.

**CPU installation and synthetic smoke test** (no data download or GPU):

```bash
conda env create -f environment-cpu.yml
conda activate pertweave-cpu
python -m pip check
python -c "import torch, torchvision, timm, anndata, umap; print(torch.__version__, anndata.__version__)"
python tests/test_smoke.py
```

The public launchers default to GPU training. For a real CPU run, set
`DEVICE = "cpu"` in `pertweave/known/config.py` or pass `--device cpu`
to the Unknown launcher. Full training on CPU can be slow.

**Already using venv or another manager?** With Python 3.11, select the torch
build first, then install the same pinned requirements. For example, for CPU:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

The recipes pin Python, direct dependencies and selected compatibility-sensitive
dependencies; they are not fully hashed platform lockfiles. They target the
clean release, not an exact export of the original training server. The pinned
Zarr 2 backend avoids the NumPy 2 requirement of newer Zarr releases.

### Input files

Both workflows require a main dataset and an external evaluation dataset.
For each dataset, supply raw expression counts in an H5AD file, scFoundation
embeddings in a NumPy file and Geneformer CLS embeddings in a pickle file.
The embedding rows must follow the cell order of the corresponding H5AD.
Preserve the original observation metadata, including perturbation identity
(`fea_ca`) and functional annotation (`function`).

## Original experiment

Set `PERTWEAVE_DATA` to the parent of the original file layout in [pertweave/known/config.py](pertweave/known/config.py),
or edit the main input paths (`RAW_H5AD`, `SCFOUNDATION_NPY`,
`GENEFORMER_PICKLE`) and the three external paths (`RAW_OUT_H5AD`,
`SCFOUNDATION_OUT_NPY`, `GENEFORMER_OUT_PICKLE`). Use the function-annotated
main dataset. Launch training with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

The default configuration runs five stratified train/test splits for 250 epochs
each. Checkpoints, predictions and metrics are written under
`github_ver_refactored/` using the experiment path defined in the known experiment's configuration.

## Supplementary experiment with Unknown KOs

Use a main H5AD containing both annotated and function-unknown cells.
Set `PERTWEAVE_UNKNOWN_DATA` to the directory containing these six files or
symbolic links:

```text
main/cells.h5ad
main/scfoundation.npy
main/geneformer_cls.pickle
external/cells.h5ad
external/scfoundation.npy
external/geneformer_cls.pickle
```

```bash
export PERTWEAVE_UNKNOWN_DATA=/absolute/path/to/data_root
CUDA_VISIBLE_DEVICES=0 python train_unknown_metric_cached.py --output-root runs/unknown_functions
```

For files with existing names, supply their paths directly using
`--raw-h5ad`, `--scfoundation`, `--geneformer`, `--raw-out-h5ad`,
`--scfoundation-out` and `--geneformer-out`. Run
`python train_unknown_metric_cached.py --help` to view the options.

The defaults in [pertweave/unknown/final_settings.json](pertweave/unknown/final_settings.json) specify the final single-stage
experiment. It uses five stratified 80:20 splits, 100 epochs and seed 3407.
Unknown perturbations with at least 100 cells participate in perturbation-level
training. Functional supervision uses annotated cells. Each batch of 512
reserves four cells from each of 24 uniformly sampled Unknown categories and
fills the remaining 416 positions from the full training pool. The decoder
hidden width is 3072, both center loss weights are 0.5, and the shared-feature
consistency weight is 1.0. The default precision setting uses BF16 on CUDA.

Both workflows write final-epoch predictions to each fold's `test_data/`,
`out_data/` and `test_rest_data/`, using the same nine NumPy filenames.
`features.npy` contains the shared embedding; `feat_cls.npy` and
`feat_group.npy` contain the decoder embeddings. `logits.npy` and
`group_logits.npy` are the perturbation and function scores. Empty arrays
indicate unavailable labels or outputs for that dataset, just as in the
original trainer. In `test_data/`, select rows where `group_labels.npy == -1`
for Unknown cells, then apply row-wise softmax to their function logits.
Use the saved label mapping for the column order and split indices for cell IDs.
These are candidate known functions, not verified annotations.

There is no all-Unknown inference pass or prediction H5AD; H5AD remains an
input and internal preprocessing cache format. The Unknown workflow keeps
its best/last checkpoints, fold diagnostics and recovery metadata. The
`best/` subdirectory holds metrics, not a second copy of prediction arrays.
The repository includes neither datasets nor trained weights.

The cached entry point reuses preprocessing in `unknown_metric_processed_cache/`.
Append `--only-fold 0` for a single split or `--processed-cache-root PATH` to
choose a cache location. If a split finished its final epoch but export was
interrupted, repeat the same command with `--recover-partial-fold N`, where
`N` is the zero-based split index.

## Repository layout

Keep using the two launch commands above from the repository root. The launchers
delegate to `pertweave/known/` (original) and `pertweave/unknown/` (supplementary);
both use [the shared model](pertweave/models.py). Data filters and label handling
remain separate because the two experiments use different supervision.
