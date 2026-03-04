#!/usr/bin/env python3
import argparse
import ast
import os
import re
import sys
from itertools import product

import numpy as np
import torch

from modelscape.backend.cli import parse_args
from modelscape.model import MLP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from FileManager import FileManager
from data import get_synthetic_X
from mlp_learning import (
    _get_spaced_targets,
    _get_spacing_config,
    _post_init_mupify,
    load_config,
)


PROGRESS_RE = re.compile(
    r"train\s+(?P<train>[-+0-9.eE]+)\s+\|\s+"
    r"test\s+(?P<test>[-+0-9.eE]+)\s+\|\s+"
    r"timekey\s+(?P<timekey>\[[^\]]*\]|[-+0-9.eE]+)\s+\|\s+"
    r"ntrain=(?P<ntrain>-?\d+)\s+\|\s+"
    r"trial=(?P<trial>-?\d+)\s+\|\s+"
    r"monomials=(?P<monomials>\{.*?\})"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _resolve_config_path(pre_config):
    expt_dir = os.path.dirname(__file__)
    if pre_config:
        if os.path.isabs(pre_config):
            config_path = pre_config
        else:
            candidate = os.path.join(expt_dir, pre_config)
            config_path = candidate if os.path.exists(candidate) else pre_config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {pre_config}")
        return config_path

    config_candidates = [
        "mlp_config.yaml",
        "mlp_config.yml",
        "mlp_config.json",
        "config.yaml",
        "config.yml",
        "config.json",
    ]
    return next(
        (
            os.path.join(expt_dir, name)
            for name in config_candidates
            if os.path.exists(os.path.join(expt_dir, name))
        ),
        None,
    )


def _make_trace(var_names):
    return {"var_names": list(var_names), "config2outcome": {}, "outcome_shape": None}


def _coerce_key_component(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, str):
        return value
    return str(value)


def _trace_store(trace, config, outcome):
    config = tuple(_coerce_key_component(v) for v in config)
    out_array = np.asarray(outcome)
    if not np.issubdtype(out_array.dtype, np.number):
        raise ValueError(f"Non-numeric outcome for {config}: {outcome}")

    shape = out_array.shape
    if trace["outcome_shape"] is None:
        trace["outcome_shape"] = shape
    elif shape != trace["outcome_shape"]:
        raise ValueError(
            f"Outcome shape {shape} != expected {trace['outcome_shape']} for config {config}"
        )

    existing = trace["config2outcome"].get(config)
    if existing is not None:
        if np.array_equal(np.asarray(existing), out_array):
            return False
    trace["config2outcome"][config] = outcome
    return True


def _parse_timekey(raw):
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            arr = np.asarray(parsed, dtype=float)
        except (SyntaxError, ValueError):
            arr = np.fromstring(raw.strip("[]"), sep=" ", dtype=float)
            if arr.size == 0:
                raise ValueError(f"Could not parse timekey: {raw}")
        if arr.ndim == 0:
            arr = np.asarray([float(arr)], dtype=float)
        return arr
    return np.asarray([float(raw)], dtype=float)


def _parse_records(path):
    with open(path, "r", errors="replace") as f:
        text = f.read()

    text = ANSI_RE.sub("", text).replace("\r", "\n")

    records = []
    for match in PROGRESS_RE.finditer(text):
        gd = match.groupdict()
        records.append(
            {
                "ntrain": int(gd["ntrain"]),
                "trial": int(gd["trial"]),
                "monomials": gd["monomials"].strip(),
                "train": float(gd["train"]),
                "test": float(gd["test"]),
                "timekey": _parse_timekey(gd["timekey"]),
            }
        )
    return records


def _build_result_from_records(records, args, target_monomials, target_hea_eigvals, target_meta, ttl_threshold):
    iterator_names = ["ntrain", "trial", "monomials"]
    result = {
        "jobs": list(product(args.N_SAMPLES, range(args.NUM_TRIALS), target_monomials)),
        "var_axes": iterator_names,
        "losses": _make_trace(iterator_names),
        "timekeys": _make_trace(iterator_names),
        "extras": {},
    }

    duplicates = 0
    overwritten = 0
    for rec in records:
        key = (rec["ntrain"], rec["trial"], rec["monomials"])

        loss_added = _trace_store(result["losses"], key, float(rec["test"]))
        time_added = _trace_store(result["timekeys"], key, rec["timekey"])

        if not loss_added and not time_added:
            duplicates += 1
        elif (not loss_added) != (not time_added):
            overwritten += 1

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
        }
    )
    return result, duplicates, overwritten


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--path", type=str, required=True, help="Path to nohup output file")
    pre_parser.add_argument("--config", type=str, default=None, help="Path to config file (yaml/json)")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_args()

    config_path = _resolve_config_path(pre_args.config)
    config = load_config(config_path) if config_path else {}
    for key, value in config.items():
        setattr(args, key, value)

    args.MODEL_CLASS = MLP
    args.post_init_fn = _post_init_mupify

    ttl_threshold = float(getattr(args, "TTL_THRESHOLD", 0.1))
    args.LOSS_CHECKPOINTS = [ttl_threshold]
    args.ONLYTHRESHOLDS = True
    args.N_TOT = args.N_TEST + args.N_TRAIN

    spacing_cfg = _get_spacing_config(args)

    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(args.SEED)
    _, data_eigvals = get_synthetic_X(**args.datasethps, N=args.N_TOT, gen=gen)
    target_monomials, target_hea_eigvals, target_meta = _get_spaced_targets(
        data_eigvals=data_eigvals,
        datasethps=args.datasethps,
        spacing_cfg=spacing_cfg,
    )
    args.TARGET_MONOMIALS = target_monomials

    exptpath = os.getenv("RESULTPATH")
    if exptpath is None:
        raise ValueError("must set $RESULTPATH environment variable")

    expt_dir = os.path.join(exptpath, config.get("expt_dir"))
    dir_suffix = config.get("dir_suffix")
    if dir_suffix:
        expt_dir = f"{expt_dir}_{dir_suffix}"

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")

    nohup_path = os.path.abspath(pre_args.path)
    if not os.path.exists(nohup_path):
        raise FileNotFoundError(f"Nohup output file not found: {nohup_path}")

    records = _parse_records(nohup_path)
    if not records:
        raise ValueError(f"No parsable run records found in {nohup_path}")

    result, duplicates, overwritten = _build_result_from_records(
        records,
        args,
        target_monomials,
        target_hea_eigvals,
        target_meta,
        ttl_threshold,
    )

    parsed_unique = len(result["timekeys"]["config2outcome"])
    expected_total = len(result["jobs"])
    print(
        f"Parsed {len(records)} progress records ({parsed_unique} unique jobs; expected {expected_total})."
    )
    if duplicates:
        print(f"Ignored {duplicates} duplicate redraw records.")
    if overwritten:
        print(f"Updated {overwritten} partially duplicated records (loss/timekey mismatch).")

    print(f"Results saved to {expt_dir}")
    expt_fm.save(result, "result_ttl.pickle")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
