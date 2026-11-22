# config.py

import os

# --- Basic Configuration ---
SEED = 3407
DEVICE = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Preprocessing Parameters ---
SAMPLE_LEAST = 50 # Note: This variable is defined but not used in the original filtering logic.
FILTER_LEAST = 0  # Minimum number of samples per class to keep a class.
GENEFORMER_FEATURE_TYPE = "cls" # "cls" or other feature types from Geneformer output.

# --- File Paths ---
# Input data paths
BASE_DATA_PATH = os.environ.get("PERTWEAVE_DATA", "data")
SCFOUNDATION_NPY = os.path.join(BASE_DATA_PATH, "scFoundation-main/model/deepcdr_01B-resolution_singlecell_cell_embedding_t4mito_crispr_raw_nc_filter_MoreThanNum50_cat706_filtCtrl_3k_tohuman_uni.npy")
GENEFORMER_PICKLE = os.path.join(BASE_DATA_PATH, f"new_geneformer/models/mito_crispr_raw_nc_filter_MoreThanNum50_cat706_filtCtrl_3k_tohuman-20L95M-sdpa-random_proj-eval/feature_extraction/all_features_{GENEFORMER_FEATURE_TYPE}.pickle")
RAW_H5AD = os.path.join(BASE_DATA_PATH, "data/paperdata/mito_crispr_raw_nc_filter_MoreThanNum50_cat706_filtCtrl_3k_tohuman.h5ad")

# Out-of-distribution (OOD) / external validation data paths
RAW_OUT_H5AD = os.path.join(BASE_DATA_PATH, "data/paperdata/mito_crispr_raw_single_sgrna_cells_no_ctrl_with_ko_state_numLessThan50_tohuman.h5ad")
SCFOUNDATION_OUT_NPY = os.path.join(BASE_DATA_PATH, "scFoundation-main/model/deepcdr_01B-resolution_singlecell_cell_embedding_t4paperdata_mito_crispr_raw_single_sgrna_cells_no_ctrl_with_ko_state_numLessThan50_tohuman_uni.npy")
GENEFORMER_OUT_PICKLE = os.path.join(BASE_DATA_PATH, f"new_geneformer/models/mito_crispr_raw_single_sgrna_cells_no_ctrl_with_ko_state_numLessThan50_tohuman.dataset-20L95M-sdpa-random_proj/feature_extraction/all_features_{GENEFORMER_FEATURE_TYPE}.pickle")

# --- Model Hyperparameters ---
HYPERPARAMETERS = {
    # Data structure related (to be filled by data loader)
    "input_dim_scfound": 0,
    "input_dim_geneformer": 0,
    "input_dim_raw": 0,
    "output_dim": 0,
    "group_output_dim": 0,
    
    # Feature engineering
    "num_of_gene_by_var": 3712,
    "num_of_gene_by_mean": 256,

    # Model architecture
    "hidden_dim": 3584,
    "feat_dim": 2688,
    "layer_num": 3,
    "layer_list": [1, 1, 1], # List length should match layer_num potentially
    "activation": 'sigmoid', # 'tanh', 'relu', 'silu', 'mish', etc.
    "reduction_factor": 2,
    "GLU_reduction": 1,
    "efficient_num": 4,
    "data_mask": "111", # "scFound|geneformer|raw" binary mask

    # Training parameters
    "epochs": 250,
    "batch_size": 512,
    "raw_lr": 0.001,
    "final_lr": 1e-6,
    "weight_decay": 0.002,
    "foldn": 5,

    # Loss weights and regularization
    "dropout": 0.3,
    "drop_num": 1, # Number of dropout layers in MC-Dropout style within GLU_efficient
    "linear": True, # Use linear activation in final layers
    "label_smoothing": 0.2,
    "label_smoothing_group": 0.2,
    "celoss_weight": 1.0,
    "center_loss_weight": 0.1,
    "celoss_group_weight": 0.5,
    "center_loss_group_weight": 0.1,
    "feat_MCC_weight": 1.0,
    "MCC_weight": 0.1,
    "MCC_group_weight": 0.1,
    "MCC_temperature": 1.0,
}

def get_experiment_paths(params: dict, seed: int) -> str:
    """Generates a unique directory name based on hyperparameters."""
    layer_list_str = "".join([str(i) for i in params['layer_list']])
    
    tier0_prefix = "github_ver_refactored"
    first_prefix1 = f"ctrl3k-stat-test250530-noiseDATAout-scale-gene_eval-GLUefficient{params['efficient_num']}-innerfc"
    first_prefix2 = f"re-old2NEWpa22s-F{params['foldn']}-heno_fin_lrelu-{GENEFORMER_FEATURE_TYPE}genef-scf-cat-{FILTER_LEAST}"
    second_prefix = (f"rdrce{params['MCC_weight']}-rdrt{params['MCC_temperature']}-f{params['feat_MCC_weight']}-g{params['MCC_group_weight']}"
                     f"-drn{params['drop_num']}-GeVar{params['num_of_gene_by_var']}Max{params['num_of_gene_by_mean']}"
                     f"-mask{params['data_mask']}-seed{seed}-act{params['activation'][:4]}-cos{params['final_lr']}-nosam-up{params['feat_dim']}")
    second2_prefix = (f"ce{params['celoss_weight']}-cen{params['center_loss_weight']}-func-ce{params['celoss_group_weight']}"
                      f"-cen{params['center_loss_group_weight']}-bs{params['batch_size']}-ls{params['label_smoothing']}-lsg{params['label_smoothing_group']}")
    third_prefix = (f"l{params['layer_num']}-ll{layer_list_str}-hid{params['hidden_dim']}-rd{params['reduction_factor']}"
                    f"-glurd{params['GLU_reduction']}-lr{params['raw_lr']}-wd{params['weight_decay']}-ep{params['epochs']}-drop{params['dropout']}")

    root = os.path.join(tier0_prefix, first_prefix1, first_prefix2, second_prefix, second2_prefix, third_prefix)
    return root
