"""Self-contained settings for the released Unknown training experiment."""
import json
import os
from pathlib import Path

with Path(__file__).with_name("final_settings.json").open(encoding="utf-8") as f:
    FINAL_SETTINGS = json.load(f)

SEED = FINAL_SETTINGS["seed"]
DEVICE = FINAL_SETTINGS["device"]
FILTER_LEAST = 0
GENEFORMER_FEATURE_TYPE = "cls"
_data = Path(os.environ.get("PERTWEAVE_UNKNOWN_DATA", "data"))
RAW_H5AD = str(_data / "main" / "cells.h5ad")
SCFOUNDATION_NPY = str(_data / "main" / "scfoundation.npy")
GENEFORMER_PICKLE = str(_data / "main" / "geneformer_cls.pickle")
RAW_OUT_H5AD = str(_data / "external" / "cells.h5ad")
SCFOUNDATION_OUT_NPY = str(_data / "external" / "scfoundation.npy")
GENEFORMER_OUT_PICKLE = str(_data / "external" / "geneformer_cls.pickle")

# Dataset and label dimensions are populated from the supplied data at runtime.
HYPERPARAMETERS = {
    **FINAL_SETTINGS,
    **dict.fromkeys(("input_dim_scfound", "input_dim_geneformer", "input_dim_raw",
                     "output_dim", "group_output_dim"), 0),
}
 
def serializable_params(params):
    return dict(params)


def experiment_directory(params, output_root):
    """Keep differently configured runs in separate directories."""
    import hashlib
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    return Path(output_root or "unknown_metric_runs") / f"legacy_unknown_numpy_{digest}"

