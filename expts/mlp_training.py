import numpy as np

import torch
from torch.multiprocessing import get_context
import torch.multiprocessing as mp
from tqdm import tqdm
import gc

import sys, os
import json

sys.path.append("../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from kernels import ReluNTK
from feature_decomp import Monomial
from utils import ensure_torch, derive_seed, seed_everything, tuple_to_numpy
from mlps import MLP, train_network
from tools import trial_count_fn

from ExptTrace import ExptTrace
from FileManager import FileManager

import argparse
from pathlib import Path

## ------- ARG PARSING --------

def int_from_any(x: str) -> int:
    """Accept '10000', '1e4', '5.0' → int."""
    try:
        v = int(float(x))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid int: {x}") from e
    return v

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("expected a boolean")

class GrabRunner:
    def __init__(self, base_fn_name, source="in", concat_outside=True, base_kwargs=None):
        self.base_fn_name = base_fn_name
        self.source = source
        self.concat_outside = bool(concat_outside)
        self.base_kwargs = dict(base_kwargs or {})

    def __call__(self, model, *_, **__):
        import mlp_grabs as m
        W, b = m.get_Win(model) if self.source == "in" else m.get_Wout(model)
        G = m.get_W_gram(W, concatenate_outside=self.concat_outside)
        fn = getattr(m, self.base_fn_name)
        return fn(G, **self.base_kwargs)


def build_other_grabs(spec, *, default_source="in", concat_outside=True,
                      per_alias_gram=None, per_alias_kwargs=None):
    """
    spec: {"Wii1":"get_W_ii", "Wii2":"get_W_ii", "Wtr":"get_W_trace"}
    per_alias_gram: {"Wii1":{"source":"out","concat_outside":True}, "Wii2":{"source":"in","concat_outside":False}}
    per_alias_kwargs: {"Wii1":{"i":1}, "Wii2":{"i":2}}
    """
    per_alias_gram   = per_alias_gram   or {}
    per_alias_kwargs = per_alias_kwargs or {}
    out = {}

    for alias, fn_name in spec.items():
        cfg = {"source": default_source, "concat_outside": concat_outside}
        cfg.update(per_alias_gram.get(alias, {}))
        kw = dict(per_alias_kwargs.get(alias, {}))

        out[alias] = GrabRunner(
            base_fn_name=fn_name,
            source=cfg["source"],
            concat_outside=cfg["concat_outside"],
            base_kwargs=kw,
        )

    return out


def load_json(path: str):
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text())

def parse_args():
    p = argparse.ArgumentParser(description="Config for MLP training")
    p.add_argument("--ONLINE", type=bool, default=True, help="Whether to use online training (full dataset at once) or fixed dataset.")
    p.add_argument("--N_TRAIN", type=int_from_any, default=4000, help="Number of training samples.")
    p.add_argument("--N_TEST", type=int_from_any, default=10_000, help="Number of test samples.")
    p.add_argument("--DATASET", type=str, default="synthetic", help="Dataset to use, currently: synthetic (gaussian) or cifar10.")
    p.add_argument("--TARGET_FUNCTION_TYPE", type=str, default="monomial", help="Type of target function to learn.")
    p.add_argument("--TARGET_MONOMIALS", type=json.loads, default=None, help="List of target monomials as JSON string.")
    p.add_argument("--ONLYTHRESHOLDS", type=str2bool, default=True, help="If True, only record last loss instead of full curve.")
    p.add_argument("--N_SAMPLES", nargs="+", type=int, default=[1024], help="Number of samples.")
    # p.add_argument("--kerneltype", type=str, default="ReluNTK", help="Kernel type to use for synthetic data generation.")
    p.add_argument("--NUM_TRIALS", type=int_from_any, default=1, help="Number of independent trials.")
    
    p.add_argument("--MAX_ITER", type=int_from_any, default=1e5, help="Steps per trial.")
    p.add_argument("--LR", type=float, default=1e-2, help="Learning rate.")
    p.add_argument("--DEPTH", type=int_from_any, default=3, help="Number of hidden layers+1.")
    p.add_argument("--WIDTH", type=int_from_any, default=1024, help="Width of hidden layers.")
    p.add_argument("--GAMMA", type=float, default=1.0, help="Richness parameter for training.")
    p.add_argument("--DEVICES", type=int, nargs="+", default=[0], help="GPU ids, e.g. --DEVICES 2 4")
    
    p.add_argument("--SEED", type=int, default=42, help="RNG seed.")
    p.add_argument("--LOSS_CHECKPOINTS", type=float, nargs="+", default=[0.15, 0.1], help="Loss checkpoints to record.")
    p.add_argument("--EMA_SMOOTHER", type=float, default=0.9, help="EMA smoother for loss tracking.")
    p.add_argument("--DETERMINSITIC", type=str2bool, default=True, help="Whether to use deterministic training.")
    p.add_argument("--VERBOSE", type=str2bool, default=False, help="Whether to print out training info.")
    
    p.add_argument("--EXPT_NAME", type=str, default="mlp-learning-curves", help="Where to save results.")

    # p.add_argument("--DATASETPATH", type=str, default=str(Path.home() / "data"), help="Path to dataset root.")
    p.add_argument("--datasethps", type=json.loads,
                    default='{"normalized": true, "cutoff_mode": 40000, "d": 200, "offset": 6, "alpha": 2.0, "noise_size": 1, "yoffset": 1.2, "beta": 1.2, "classes": null, "binarize": false, "weight_variance": 1, "bias_variance": 1}',
                    help="Dataset hyperparameters as JSON string.")
    p.add_argument("--datasethps_path", help="Path to datasethps .json")
    p.add_argument("--target_monomials_path", help="Path to target monomials .json")
    
    p.add_argument("--other_model_grabs", help="Dict of {name, fn} pairs for grabbing model parameters", type=json.loads, default={})
    p.add_argument("--W_source",  choices=["in","out"], default="in", help="Default base matrix: Win or Wout")
    p.add_argument("--concat_outside", type=str2bool, default="True", help="Choice of having the gram matrix be W^T W or W W^T")
    p.add_argument("--other_model_gram", type=json.loads, default={}, help='Per-alias overrides, e.g. {"Wii1":{"source":"out","concat_outside":"True"}}')
    p.add_argument("--other_model_kwargs", type=json.loads, default={}, help='Per-alias kwargs, e.g. {"Wii1":{"i":1}}')
    return p.parse_args()

## ----- Technicals below -----

## --- Multiprocessing setup ---

def run_job(device_id, job, global_config, bfn_config=None):
    """
    job: (target, n, trial)
    global_config: anything read-only you want to avoid capturing from globals
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    base_seed = global_config.get("SEED", None)
    job_seed  = derive_seed(base_seed, device_id)
    GEN, RNG = seed_everything(job_seed, device_id)

    torch.set_num_threads(1)  # avoid CPU contention when many procs

    if global_config["TARGET_FUNCTION_TYPE"] == "monomial":
        from data import polynomial_batch_fn
        batch_function = lambda target_monomial, n, X, y: polynomial_batch_fn(**bfn_config, monomials=target_monomial, bsz=n, gen=GEN, X=X, y=y)
    target, n, trial = job
    X_te, y_te = batch_function(target, global_config["N_TEST"], X=None, y=None)(0)
    
    X_tr, y_tr = batch_function(target, n, X=None, y=None)(trial) if not global_config["ONLINE"] else None, None

    bfn = batch_function(target, n, X=X_tr, y=y_tr)
    
    model = MLP(d_in=global_config["DIM"], depth=global_config["DEPTH"],
                d_out=1, width=global_config["WIDTH"]).to(device)

    outdict = train_network(
        model=model,
        batch_function=bfn,
        lr=global_config["LR"],
        max_iter=int(global_config["MAX_ITER"]),
        loss_checkpoints=global_config["LOSS_CHECKPOINTS"],
        gamma=global_config["GAMMA"],
        ema_smoother=global_config["EMA_SMOOTHER"],
        only_thresholds=global_config["ONLYTHRESHOLDS"],
        verbose=global_config["VERBOSE"],
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_te, y_te=y_te,
        otherreturns=global_config.get("otherreturns", {}),
    )

    timekeys = outdict["timekeys"]
    train_losses = outdict["train_losses"]
    test_losses = outdict["test_losses"]
    otherouts = [outdict[k] for k in global_config.get("otherreturns", {}).keys()]
    # Cleanup GPU memory
    del model, X_tr, y_tr, X_te, y_te, outdict
    torch.cuda.empty_cache()
    gc.collect()

    return (n, str(target), int(trial), timekeys, train_losses, test_losses, *otherouts)#, others) #timekeys is numpy

def worker(device_id, job_queue, result_queue, global_config, bfn_config):
    torch.cuda.set_device(device_id)

    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            payload = run_job(device_id, job, global_config, bfn_config)
            payload = tuple_to_numpy(payload)
        
            result_queue.put(("ok", payload))
        except Exception as e:
            result_queue.put(("err", (job, repr(e))))


## --- Multiprocessing execution ---
def main(args):
    if args.TARGET_MONOMIALS is not None:
        args.TARGET_MONOMIALS = [Monomial(m) for m in args.TARGET_MONOMIALS]
    elif args.target_monomials_path:
        target_monomials_json = load_json(args.target_monomials_path)
        args.TARGET_MONOMIALS = [Monomial(m) for m in target_monomials_json]
    elif args.TARGET_MONOMIALS is None:
        args.TARGET_MONOMIALS = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3}),]
    
    
    if args.datasethps_path:
        args.datasethps = load_json(args.datasethps_path)

    
    args.N_TOT = args.N_TEST+args.N_TRAIN
    # cfg = vars(args)

    ## --- Grab targets and such ---
    if args.DATASET == "synthetic":
        from notebook_fns import get_synthetic_dataset

        X_full, _, H, monomials, hea_eigvals, _, data_eigvals = get_synthetic_dataset(**args.datasethps, N=args.N_TOT, kerneltype=ReluNTK,
                                                                                           gen=torch.Generator(device='cuda').manual_seed(args.SEED))

    elif args.DATASET == "cifar10":
        from ImageData import ImageData
        PIXEL_NORMALIZED =  False # Don't normalize pixels, normalize samples
        classes = args.datasethps['classes']
        normalized = args.datasethps['normalized']

        if classes is not None:
            imdata = ImageData('cifar10', "../data", classes=classes, onehot=len(classes)!=2, format="N")
        else:
            imdata = ImageData('cifar10', "../data", classes=classes, onehot=False, format="N")
        X_train, y_train = imdata.get_dataset(args.N_TRAIN, **args.datasethps, get='train',
                                            centered=True, normalize=PIXEL_NORMALIZED)
        X_test, y_test = imdata.get_dataset(args.N_TEST, **args.datasethps, get='test',
                                            centered=True, normalize=PIXEL_NORMALIZED)
        X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        X_train, y_train, X_test, y_test = [t/torch.linalg.norm(t) for t in (X_train, y_train, X_test, y_test)] if normalized else (X_train, y_train, X_test, y_test)
        if normalized:
            X_train *= args.N_TRAIN**(0.5); X_test *= args.N_TEST**(0.5)
            y_train *= args.N_TRAIN**(0.5); y_test *= args.N_TEST**(0.5)
        X_full = torch.cat((X_train, X_test), dim=0)
        y_full = torch.cat((y_train, y_test), dim=0)
        data_eigvals = torch.linalg.svdvals(X_full)**2
        data_eigvals /= data_eigvals.sum()


    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]

    ## --- Target function defs ---
    if args.TARGET_FUNCTION_TYPE == "monomial":
        target_monomials = args.TARGET_MONOMIALS
        targets = target_monomials
        bfn_config = dict(lambdas=lambdas, Vt=Vt, data_eigvals=data_eigvals, N=args.N_TOT)
    
    global_config = dict(DEPTH=args.DEPTH, WIDTH=args.WIDTH, LR=args.LR, GAMMA=args.GAMMA,
        EMA_SMOOTHER=args.EMA_SMOOTHER, MAX_ITER=args.MAX_ITER,
        LOSS_CHECKPOINTS=args.LOSS_CHECKPOINTS, N_TEST=args.N_TEST,
        SEED=args.SEED, ONLYTHRESHOLDS=args.ONLYTHRESHOLDS, DIM=dim,
        TARGET_FUNCTION_TYPE=args.TARGET_FUNCTION_TYPE,
        ONLINE=args.ONLINE, VERBOSE=args.VERBOSE
        )

    # m = __import__("mlp_grabs")
    # grabs = {k: getattr(m, v) for k, v in args.other_model_grabs.items()}
    grabs = build_other_grabs(args.other_model_grabs, default_source=args.W_source, concat_outside=args.concat_outside,
        per_alias_gram=args.other_model_gram, per_alias_kwargs=args.other_model_kwargs,)
    # import inspect
    # print(inspect.signature(grabs['W1tr']))
    # print(inspect.signature(grabs['W2tr']))
    # print(grabs.values())
    global_config.update({"otherreturns": grabs})
    

    var_axes = ["target", "ntrain", "trial"]
    et_losses = ExptTrace(var_axes)
    et_timekeys = ExptTrace(var_axes)
    grab_aliases = list(args.other_model_grabs.keys())
    et_extras = {alias: ExptTrace(var_axes) for alias in grab_aliases}

    jobs = [(target, n, trial)
            for target in targets
            for n in args.N_SAMPLES
            for trial in range(args.NUM_TRIALS)]

    ctx = get_context("spawn")

    job_queue    = ctx.Queue()
    result_queue = ctx.Queue()

    NUM_GPUS = torch.cuda.device_count()
    for job in jobs:
        job_queue.put(job)
    for _ in range(NUM_GPUS):
        job_queue.put(None)

    procs = [ctx.Process(target=worker, args=(dev, job_queue, result_queue, global_config, bfn_config))
            for dev in range(NUM_GPUS)]
    for p in procs: p.start()

    total = len(jobs)
    done = 0
    
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        while done < total:
            kind, payload = result_queue.get()
            if kind == "ok":
                n, tstr, trial, timekeys, train_losses, test_losses, *others = payload
                et_losses[tstr, n, trial] = test_losses
                et_timekeys[tstr, n, trial] = timekeys
                for kidx, k in enumerate(grabs.keys()):
                    et_extras[k][tstr, n, trial] = others[kidx]
                if not(args.ONLYTHRESHOLDS):
                    train_losses = train_losses[-1]
                    test_losses = test_losses[-1]
                pbar.set_postfix_str(
                f"train {train_losses:.3g} | test {test_losses:.3g} | timekey {timekeys} | n={n} | target={tstr} | trial={trial}",
                refresh=False
            )
            else:
                job, err = payload
                print(f"[ERROR] {job}: {err}")
            done += 1
            pbar.update(1)

    for p in procs: p.join()

    result = {
        "targets": targets,
        "monomials": monomials,
        "hea_eigvals": hea_eigvals.cpu(),
        "losses": et_losses.serialize(),
        "timekeys": et_timekeys.serialize(),
        "extras": {name: et_extras[name].serialize() for name in grab_aliases},
    }

    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    ## --- Setup file management ---

    datapath = os.getenv("DATASETPATH")
    exptpath = os.getenv("EXPTPATH")
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $EXPTPATH environment variable")
    expt_dir = os.path.join(exptpath, "phlab", args.EXPT_NAME, args.DATASET)

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")
    hypers = dict(expt_name=args.EXPT_NAME, dataset=args.DATASET, target_function_type=args.TARGET_FUNCTION_TYPE,
                width=args.WIDTH, depth=args.DEPTH, gamma=args.GAMMA, lr=args.LR, seed=args.SEED, n_train=args.N_TRAIN, n_test=args.N_TEST)
    with open(expt_fm.get_filename("hypers.json"), 'w') as f:
        json.dump(hypers, f, indent=4)

    ## --- Begin experiment ---
    mp.set_start_method("spawn", force=True)
    main(args)