import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

from modelscape.backend.cli import parse_args
from modelscape.backend.job_iterator import main as run_job_iterator
from modelscape.data.ntk_coeffs import get_relu_level_coeff_fn
from modelscape.model import MLP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import get_binarized_dataset, get_matrix_hermites, preprocess
from feature_decomp import Monomial, generate_hea_monomials
from scripts.get_spaced_eigvals import select_indices_with_geometric_decay
from utils import ensure_numpy, ensure_torch

from FileManager import FileManager

from mupify import mupify, rescale


def _post_init_mupify(model, opt, gamma=1.0, mup_param="mup", **_):
    mupify(model, opt, param=mup_param)
    rescale(model, gamma)
    return model, opt


def load_config(path):
    """
    Load a configuration file based on its extension.
    Supports .yaml/.yml and .json.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    with open(path, "r") as f:
        if ext in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to read YAML configs. "
                    "Install it or use a .json config."
                ) from exc
            return yaml.safe_load(f) or {}
        if ext == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported configuration file extension: {ext}")


DEFAULT_INDICES_OF_INTEREST = [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 100, 180]


def cifar_batch_fn(X_data, pca_data, monomials, bsz, X=None, y=None, gen=None, **_):
    """
    Fixed CIFAR data batch function compatible with modelscape.run_job:
    - If X and y are provided, always returns that fixed pair.
    - Otherwise samples a random minibatch from preprocessed CIFAR data.

    Target y is defined by the requested monomial on PCA-normalized coordinates.
    """
    X_data, pca_data = map(ensure_torch, (X_data, pca_data))
    if bsz <= 0:
        raise ValueError(f"bsz must be positive, got {bsz}")

    if monomials is None:
        raise ValueError("monomials must be provided by iterator_names/job")

    # Build labels once per job/monomial.
    y_full = get_matrix_hermites(X=pca_data, monomials=monomials, previously_normalized=True)
    y_full = ensure_torch(y_full)
    if y_full.ndim == 2:
        y_full = y_full.sum(dim=1) / y_full.shape[1]
    y_full = y_full * (X_data.shape[0] ** 0.5)

    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            return ensure_torch(X), ensure_torch(y)

        n_total = X_data.shape[0]
        if bsz >= n_total:
            return X_data, y_full

        idx = torch.randint(0, n_total, (bsz,), generator=gen, device=X_data.device)
        return X_data[idx], y_full[idx]

    return batch_fn


def _get_spacing_config(args):
    cfg = {
        "indices_of_interest": DEFAULT_INDICES_OF_INTEREST,
        "all_indices": False,
        "cutoff_hea_eigval": 1e-5,
        "geometric_spacing": 0.92,
        "cutoff_hea_mode": 1000,
        "kmax": 6,
        "monomial_degree": None,
        "num_targets": 100,
    }

    for key in list(cfg.keys()):
        if hasattr(args, key):
            cfg[key] = getattr(args, key)
        upper_key = key.upper()
        if hasattr(args, upper_key):
            cfg[key] = getattr(args, upper_key)
    return cfg


def _get_spaced_targets(data_eigvals, datasethps, spacing_cfg):
    if data_eigvals is None:
        raise ValueError("data_eigvals must be provided for CIFAR target generation")

    data_eigvals = ensure_numpy(data_eigvals)
    d = int(len(data_eigvals))

    weight_variance = float(datasethps.get("weight_variance", 1.0))
    bias_variance = float(datasethps.get("bias_variance", 1.0))
    level_coeff_fn = get_relu_level_coeff_fn(
        data_eigvals=data_eigvals,
        weight_variance=weight_variance,
        bias_variance=bias_variance,
    )

    indices_of_interest = spacing_cfg.get("indices_of_interest") or DEFAULT_INDICES_OF_INTEREST
    if spacing_cfg.get("all_indices"):
        data_indices = np.arange(0, d, 1, dtype=int)
    else:
        data_indices = np.asarray(indices_of_interest, dtype=int)

    if data_indices.size == 0:
        raise ValueError("indices_of_interest must contain at least one index")
    if data_indices.max() >= d:
        raise ValueError(f"indices_of_interest has entries >= d ({d})")

    gammas_of_interest = data_eigvals[data_indices]
    hea_eigvals, monomials = generate_hea_monomials(
        gammas_of_interest,
        num_monomials=int(datasethps.get("cutoff_mode", 40000)),
        eval_level_coeff=level_coeff_fn,
        kmax=int(spacing_cfg.get("kmax", 6)),
    )

    selected_indices = select_indices_with_geometric_decay(
        hea_eigvals, spacing_cfg.get("geometric_spacing", 0.92)
    )
    selected_indices = [
        i for i in selected_indices if hea_eigvals[i] > spacing_cfg.get("cutoff_hea_eigval", 1e-6)
    ]
    if spacing_cfg.get("monomial_degree") is not None:
        allowed_degrees = set(spacing_cfg["monomial_degree"])
        selected_indices = [i for i in selected_indices if monomials[i].degree() in allowed_degrees]
    selected_indices = selected_indices[: int(spacing_cfg.get("cutoff_hea_mode", 1000))]

    index_map = np.asarray(data_indices, dtype=int)
    selected_monomials = []
    for i in selected_indices:
        basis = monomials[i].basis()
        mapped = {int(index_map[int(k)]): int(v) for k, v in basis.items()}
        selected_monomials.append(Monomial(mapped))

    selected_eigvals = np.asarray(hea_eigvals)[selected_indices]

    num_targets = spacing_cfg.get("num_targets", None)
    if num_targets is not None:
        num_targets = int(num_targets)
        selected_monomials = selected_monomials[:num_targets]
        selected_eigvals = selected_eigvals[:num_targets]
        selected_indices = selected_indices[:num_targets]

    meta = {
        "selected_indices": selected_indices,
        "data_indices_of_interest": data_indices.tolist(),
        "spacing_cfg": spacing_cfg,
    }
    return selected_monomials, selected_eigvals, meta


def _load_and_preprocess_cifar(args):
    datasethps = dict(getattr(args, "datasethps", {}) or {})
    dataset = str(datasethps.get("dataset", "cifar5m")).lower()
    if dataset != "cifar5m":
        raise ValueError(f"Expected dataset='cifar5m', got {dataset!r}")

    classes = datasethps.get("classes", None)
    center = bool(datasethps.get("center", True))
    normalize = bool(datasethps.get("normalize", False))
    zca_strength = float(datasethps.get("zca_strength", 0.0))

    X_raw, _ = get_binarized_dataset(dataset, classes, int(args.N_TOT))
    X_full = ensure_torch(X_raw)
    X_full = preprocess(X_full, center=center, normalize=normalize, zca_strength=zca_strength)

    U, lambdas, _ = torch.linalg.svd(X_full, full_matrices=False)
    total_energy = torch.sum(lambdas ** 2)
    X_full = X_full * torch.sqrt(torch.tensor(float(args.N_TOT), device=X_full.device) / total_energy)

    # Global PCA coordinates used for monomial target labels.
    pca_full = (float(args.N_TOT) ** 0.5) * U
    data_eigvals = (lambdas ** 2) / total_energy

    return X_full, pca_full, data_eigvals, datasethps


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    args = parse_args()

    expt_dir = os.path.dirname(__file__)
    if pre_args.config:
        if os.path.isabs(pre_args.config):
            config_path = pre_args.config
        else:
            candidate = os.path.join(expt_dir, pre_args.config)
            config_path = candidate if os.path.exists(candidate) else pre_args.config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {pre_args.config}")
    else:
        config_candidates = [
            "mlp_cifar_config.yaml",
            "mlp_cifar_config.yml",
            "mlp_cifar_config.json",
            "mlp_config.yaml",
            "mlp_config.yml",
            "mlp_config.json",
            "config.yaml",
            "config.yml",
            "config.json",
        ]
        config_path = next(
            (
                os.path.join(expt_dir, name)
                for name in config_candidates
                if os.path.exists(os.path.join(expt_dir, name))
            ),
            None,
        )

    config = load_config(config_path) if config_path else {}
    for key, value in config.items():
        setattr(args, key, value)

    args.MODEL_CLASS = MLP
    args.post_init_fn = _post_init_mupify

    ttl_threshold = float(getattr(args, "TTL_THRESHOLD", 0.1))
    args.LOSS_CHECKPOINTS = [ttl_threshold]
    args.ONLYTHRESHOLDS = True

    args.N_TOT = int(args.N_TEST) + int(args.N_TRAIN)

    datapath = os.getenv("DATASETPATH")
    exptpath = os.getenv("RESULTPATH")
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $RESULTPATH environment variable")

    X_full, pca_full, data_eigvals, datasethps = _load_and_preprocess_cifar(args)

    spacing_cfg = _get_spacing_config(args)
    target_monomials, target_hea_eigvals, target_meta = _get_spaced_targets(
        data_eigvals=data_eigvals,
        datasethps=datasethps,
        spacing_cfg=spacing_cfg,
    )
    args.TARGET_MONOMIALS = target_monomials

    iterators = [args.N_SAMPLES, range(args.NUM_TRIALS), args.TARGET_MONOMIALS]
    iterator_names = ["ntrain", "trial", "monomials"]

    expt_subdir = config.get("expt_dir") or "hermite-eigenstructure-ansatz/mlp/cifar5m"
    expt_dir = os.path.join(exptpath, expt_subdir)
    dir_suffix = config.get("dir_suffix")
    if dir_suffix:
        expt_dir = f"{expt_dir}_{dir_suffix}"

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")

    args.DIM = int(X_full.shape[1])

    bfn_config = {
        "X_data": X_full,
        "pca_data": pca_full,
        "base_bfn": cifar_batch_fn,
    }

    global_config = args.__dict__.copy()
    grabs = {}
    global_config.update({"otherreturns": grabs})

    mp.set_start_method("spawn", force=True)

    result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)

    monomial_strs = [str(m) for m in target_monomials]
    monomial_bases = [m.basis() for m in target_monomials]
    target_map = {str(m): float(ev) for m, ev in zip(target_monomials, target_hea_eigvals)}
    result.update(
        {
            "ttl_threshold": ttl_threshold,
            "loss_checkpoints": list(args.LOSS_CHECKPOINTS),
            "target_info": {
                "monomial_strs": monomial_strs,
                "monomial_bases": monomial_bases,
                "hea_eigvals": [float(x) for x in target_hea_eigvals],
                "monomial_str_to_eigval": target_map,
                **target_meta,
            },
            "training_config": {
                "LR": float(args.LR),
                "MAX_ITER": int(args.MAX_ITER),
                "EMA_SMOOTHER": float(args.EMA_SMOOTHER),
                "ONLINE": bool(args.ONLINE),
                "N_TRAIN": int(args.N_TRAIN),
                "N_TEST": int(args.N_TEST),
                "N_SAMPLES": list(args.N_SAMPLES),
                "NUM_TRIALS": int(args.NUM_TRIALS),
            },
            "dataset_config": {
                "dataset": datasethps.get("dataset", "cifar5m"),
                "center": bool(datasethps.get("center", True)),
                "normalize": bool(datasethps.get("normalize", False)),
                "zca_strength": float(datasethps.get("zca_strength", 0.0)),
                "classes": datasethps.get("classes", None),
            },
        }
    )

    print(f"Results saved to {expt_dir}")
    expt_fm.save(result, "result_ttl.pickle")
    torch.cuda.empty_cache()
