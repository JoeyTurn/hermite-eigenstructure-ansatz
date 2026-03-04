#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
from itertools import product

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from modelscape.backend.cli import parse_args
from modelscape.backend.job import run_job
from modelscape.backend.job_iterator import normalize_bfn_config, sanitize_expt_trace
from modelscape.backend.worker import worker
from modelscape.model import MLP

from FileManager import FileManager

# Reuse logic from mlp_learning for consistency
from mlp_learning import (
    load_config,
    polynomial_batch_fn,
    _post_init_mupify,
    _get_spacing_config,
    _get_spaced_targets,
)
from data import get_synthetic_X


DEFAULT_RESULT_NAME = "result_ttl.pickle"


def _make_trace(var_names):
    if not isinstance(var_names, list):
        var_names = list(var_names)
    if "outcome" in var_names:
        raise ValueError("variable name 'outcome' disallowed")
    return {"var_names": var_names, "config2outcome": {}, "outcome_shape": None}


def _trace_set(trace, config, outcome):
    if not isinstance(trace, dict):
        raise TypeError("trace must be a dict")
    if trace.get("outcome_shape") is None:
        out_array = np.asarray(outcome)
        if not np.issubdtype(out_array.dtype, np.number):
            raise ValueError("measurement outcome must be numeric")
        trace["outcome_shape"] = out_array.shape
    elif np.shape(outcome) != trace["outcome_shape"]:
        raise ValueError(f"outcome shape {np.shape(outcome)} != expected {trace['outcome_shape']}")

    config = (config,) if not isinstance(config, tuple) else config
    var_names = trace.get("var_names")
    if var_names is not None and len(config) != len(var_names):
        raise ValueError(f"len config {len(config)} != num vars {len(var_names)}")
    allowed_types = (int, float, str, tuple, np.integer, np.floating)
    if not all(isinstance(c, allowed_types) for c in config):
        raise ValueError(f"config {config} elements must be one of {allowed_types}")

    config2outcome = trace.setdefault("config2outcome", {})
    if config in config2outcome:
        raise ValueError(f"config {config} already exists. overwriting not supported")
    config2outcome[config] = outcome
    return trace


def _normalize_key(key):
    if not isinstance(key, tuple):
        key = (key,)
    out = []
    for v in key:
        if isinstance(v, (np.integer, int)):
            out.append(int(v))
        elif isinstance(v, (np.floating, float)):
            out.append(float(v))
        elif isinstance(v, str):
            out.append(v)
        else:
            out.append(str(v))
    return tuple(out)


def _existing_keys(result):
    if not isinstance(result, dict):
        return set(), None
    trace = None
    for name in ("losses", "timekeys"):
        candidate = result.get(name)
        if isinstance(candidate, dict) and "config2outcome" in candidate:
            trace = candidate
            break
    if trace is None:
        return set(), None
    keys = set(_normalize_key(k) for k in trace.get("config2outcome", {}).keys())
    var_names = trace.get("var_names")
    return keys, list(var_names) if var_names is not None else None


def _job_key(job, iterator_names, key_order=None):
    if key_order is None:
        key_order = iterator_names
    mapping = {name: val for name, val in zip(iterator_names, job)}
    parts = []
    for name in key_order:
        val = mapping[name]
        if name == "monomials":
            if hasattr(val, "basis"):
                parts.append(str(val.basis()))
            else:
                parts.append(str(val))
        else:
            parts.append(val)
    return _normalize_key(tuple(parts))


def _load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _run_jobs_single(missing_jobs, iterator_names, global_config, bfn_config, grab_aliases,
                     loss_trace, time_trace, extras_traces):
    total = len(missing_jobs)
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        for job_index, job in enumerate(missing_jobs):
            try:
                payload = run_job(0, job, global_config, bfn_config, iterator_names, job_index=job_index)
                kind, out = "ok", payload
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                kind, out = "err", (job, repr(e), tb)

            if kind == "ok":
                job, timekeys, train_losses, test_losses, *others = out
                job = sanitize_expt_trace(job)

                _trace_set(loss_trace, job, test_losses)
                _trace_set(time_trace, job, timekeys)
                for kidx, k in enumerate(grab_aliases):
                    _trace_set(extras_traces[k], job, others[kidx])

                if not global_config["ONLYTHRESHOLDS"]:
                    train_losses = train_losses[-1]
                    test_losses = test_losses[-1]

                job_str = " | ".join(
                    [f"{name}={val}" for name, val in zip(iterator_names, job)]
                )
                pbar.set_postfix_str(
                    f"train {train_losses:.3g} | test {test_losses:.3g} | timekey {timekeys} | {job_str}",
                    refresh=False,
                )
            else:
                print(f"[ERROR] {out[0]}: {out[1:]}")

            pbar.update(1)


def _run_jobs_mp(missing_jobs, iterator_names, global_config, bfn_config, grab_aliases,
                 loss_trace, time_trace, extras_traces):
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[WARN] No CUDA devices detected; falling back to single-process CPU.")
        return _run_jobs_single(
            missing_jobs, iterator_names, global_config, bfn_config, grab_aliases,
            loss_trace, time_trace, extras_traces,
        )

    for job_index, job in enumerate(missing_jobs):
        job_queue.put(("job", job_index, job))
    for _ in range(num_gpus):
        job_queue.put(None)

    procs = [
        ctx.Process(
            target=worker,
            args=(dev, job_queue, result_queue, global_config, bfn_config, iterator_names),
        )
        for dev in range(num_gpus)
    ]
    for p in procs:
        p.start()

    total = len(missing_jobs)
    done = 0
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        while done < total:
            kind, payload = result_queue.get()
            if kind == "ok":
                job, timekeys, train_losses, test_losses, *others = payload
                job = job[:-1] + (str(job[-1]),)

                _trace_set(loss_trace, job, test_losses)
                _trace_set(time_trace, job, timekeys)
                for kidx, k in enumerate(grab_aliases):
                    _trace_set(extras_traces[k], job, others[kidx])

                if not global_config["ONLYTHRESHOLDS"]:
                    train_losses = train_losses[-1]
                    test_losses = test_losses[-1]

                job_str = " | ".join(
                    [f"{name}={val}" for name, val in zip(iterator_names, job)]
                )
                pbar.set_postfix_str(
                    f"train {test_losses:.3g} | test {train_losses:.3g} | timekey {timekeys} | {job_str}",
                    refresh=False,
                )
            elif kind == "err":
                job, err = payload
                print(f"[ERROR] {job}: {err}")
            elif kind == "bootstrap_err":
                print(f"[ERROR] worker bootstrap: {payload}")
            else:
                print(f"[WARN] Unknown worker message: {kind} {payload}")

            done += 1
            pbar.update(1)

    for p in procs:
        p.join()


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_parser.add_argument("--result", type=str, default=None, help="Path to result_ttl.pickle")
    pre_parser.add_argument("--dry-run", action="store_true", help="List missing jobs and exit")
    pre_parser.add_argument("--no-mp", action="store_true", help="Force single-process execution")
    pre_parser.add_argument("--use-mp", action="store_true", help="Force multiprocessing execution")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if pre_args.no_mp and pre_args.use_mp:
        raise ValueError("Use only one of --no-mp or --use-mp")

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

    args.N_TOT = args.N_TEST + args.N_TRAIN

    spacing_cfg = _get_spacing_config(args)

    gen = torch.Generator(device="cuda").manual_seed(args.SEED)
    X_full, data_eigvals = get_synthetic_X(**args.datasethps, N=args.N_TOT, gen=gen)

    target_monomials, target_hea_eigvals, target_meta = _get_spaced_targets(
        data_eigvals=data_eigvals,
        datasethps=args.datasethps,
        spacing_cfg=spacing_cfg,
    )

    args.TARGET_MONOMIALS = target_monomials

    iterators = [args.N_SAMPLES, range(args.NUM_TRIALS), args.TARGET_MONOMIALS]
    iterator_names = ["ntrain", "trial", "monomials"]

    datapath = os.getenv("DATASETPATH")
    exptpath = os.getenv("RESULTPATH")
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
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

    result_path = pre_args.result or os.path.join(expt_dir, DEFAULT_RESULT_NAME)
    if not os.path.isabs(result_path):
        result_path = os.path.join(expt_dir, result_path)
    print(f"Result file: {result_path}")

    existing_result = _load_pickle(result_path)
    existing_keys, existing_var_names = _existing_keys(existing_result)

    all_jobs = list(product(*iterators))
    key_order = existing_var_names or iterator_names
    all_keys = set(_job_key(job, iterator_names, key_order=key_order) for job in all_jobs)
    overlap_keys = existing_keys & all_keys
    extra_keys = existing_keys - all_keys
    missing_jobs = [job for job in all_jobs if _job_key(job, iterator_names, key_order=key_order) not in existing_keys]

    print(f"Found {len(existing_keys)} completed runs out of {len(all_jobs)} total.")
    print(f"Completed that match this config: {len(overlap_keys)}")
    if extra_keys:
        print(f"Completed keys not in this config: {len(extra_keys)}")
    print(f"Missing runs: {len(missing_jobs)}")

    if pre_args.dry_run:
        if missing_jobs:
            preview = ", ".join([str(_job_key(j, iterator_names, key_order=key_order)) for j in missing_jobs[:10]])
            if len(missing_jobs) > 10:
                preview += ", ..."
            print(f"Missing preview: {preview}")
        if extra_keys:
            extra_preview = ", ".join([str(k) for k in list(extra_keys)[:10]])
            if len(extra_keys) > 10:
                extra_preview += ", ..."
            print(f"Extra key preview: {extra_preview}")
        return

    if not missing_jobs:
        print("Nothing to run. Exiting.")
        return

    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]
    args.DIM = dim

    bfn_config = dict(
        lambdas=lambdas,
        Vt=Vt,
        data_eigvals=data_eigvals,
        N=args.N_TOT,
        base_bfn=polynomial_batch_fn,
    )

    global_config = args.__dict__.copy()
    grabs = {}
    global_config.update({"otherreturns": grabs})

    use_mp = None
    if pre_args.no_mp:
        use_mp = False
    elif pre_args.use_mp:
        use_mp = True
    else:
        mp.set_start_method("spawn", force=True)
        start_method = mp.get_start_method(allow_none=True)
        use_mp = (start_method == "spawn")

    if use_mp and torch.cuda.device_count() == 0:
        print("[WARN] No CUDA devices detected; falling back to single-process CPU.")
        use_mp = False

    bfn_config = normalize_bfn_config(bfn_config, use_mp=use_mp)

    result = existing_result or {}
    result["jobs"] = all_jobs
    result["var_axes"] = iterator_names

    loss_trace = result.get("losses") or _make_trace(iterator_names)
    time_trace = result.get("timekeys") or _make_trace(iterator_names)

    extras_traces = result.get("extras")
    if not isinstance(extras_traces, dict):
        extras_traces = {}
    for alias in grabs.keys():
        if alias not in extras_traces:
            extras_traces[alias] = _make_trace(iterator_names)

    if use_mp:
        _run_jobs_mp(missing_jobs, iterator_names, global_config, bfn_config, list(grabs.keys()),
                     loss_trace, time_trace, extras_traces)
    else:
        _run_jobs_single(missing_jobs, iterator_names, global_config, bfn_config, list(grabs.keys()),
                         loss_trace, time_trace, extras_traces)

    result["losses"] = loss_trace
    result["timekeys"] = time_trace
    result["extras"] = extras_traces

    monomial_strs = [str(m) for m in target_monomials]
    monomial_bases = [m.basis() for m in target_monomials]
    target_map = {str(m): float(ev) for m, ev in zip(target_monomials, target_hea_eigvals)}
    result.update({
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
    })

    _save_pickle(result, result_path)
    print(f"Results saved to {result_path}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
