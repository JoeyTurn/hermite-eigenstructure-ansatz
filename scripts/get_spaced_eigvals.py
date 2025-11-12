import numpy as np

import sys, os
import json
import argparse
from pathlib import Path

sys.path.append("../")

from kernels import ReluNTK
from feature_decomp import generate_hea_monomials
from data import get_powerlaw

def parse_args():
    p = argparse.ArgumentParser(description="Config for MLP training")
    p.add_argument("--normalized", type=bool, default=True,)
    p.add_argument("--cutoff_mode", type=int, default=40000,)
    p.add_argument("--d", type=int, default=200,)
    p.add_argument("--offset", type=int, default=6,)
    p.add_argument("--alpha", type=float, default=2.0,)
    p.add_argument("--noise_size", type=int, default=1,)
    p.add_argument("--yoffset", type=float, default=1.2,)
    p.add_argument("--beta", type=float, default=1.2,)
    p.add_argument("--classes", type=int, default=None,)
    p.add_argument("--binarize", type=bool, default=False,)
    p.add_argument("--indices_of_interest", type=int, nargs='+', default=[0,1,2,3,5,10,20,40,60,100,150],)
    p.add_argument("--cutoff_hea_eigval", type=float, default=1e-6,)
    p.add_argument("--geometric_spacing", type=float, default=0.9)
    p.add_argument("--cutoff_hea_mode", type=int, default=1000,)
    p.add_argument("--monomial_degree", type=int, nargs='+', default=None,
                   help="If provided, only keep monomials whose .degree() is in this list.")
    # p.add_argument("--weight_variance", type=float, default=1,) #these should always be 1
    # p.add_argument("--bias_variance", type=float, default=1,) #these should always be 1
    p.add_argument("--outdir", type=Path, default=None,
                   help="Directory to write outputs (creates if needed).")
    p.add_argument("--out-hps", type=Path, default=None,
                   help="Path to write dataset hparams JSON.")
    p.add_argument("--out-main", type=Path, default=None,
                   help="Path to write main output JSON (metrics/results).")
    return p.parse_args()

def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)

def select_indices_with_geometric_decay(hea_eigvals, ratio=.9):
    # assert that hea_eigvals is (a) positive and (b) already sorted
    assert np.all(hea_eigvals > 0)
    assert np.all(np.diff(hea_eigvals) <= 0)

    selected_indices = []

    cur_eigval_thresh = hea_eigvals[0] + 1

    for i in range(len(hea_eigvals)):
        if hea_eigvals[i] < cur_eigval_thresh:
            selected_indices.append(i)
            cur_eigval_thresh = hea_eigvals[i] * ratio
    return selected_indices

def main():
    args = parse_args()
    datasethps = vars(args)
    data_eigvals = get_powerlaw(P=datasethps['d'], exp=datasethps['alpha'], offset=datasethps['offset'], normalize=True) #aka data_eigvals
    level_coeff_fn = ReluNTK.get_level_coeff_fn(data_eigvals=data_eigvals, bias_variance=1, weight_variance=1)
    hea_eigvals, monomials = generate_hea_monomials(data_eigvals, datasethps['cutoff_mode'], level_coeff_fn, kmax=6)

    data_indices_of_interest = args.indices_of_interest
    # data_indices_of_interest = [0, 1, 2, 3, 5, 10, 20, 40, 60, 100, 150]
    gammas_of_interest = data_eigvals[data_indices_of_interest]
    hea_eigvals, monomials = generate_hea_monomials(gammas_of_interest, datasethps['cutoff_mode'], level_coeff_fn, kmax=6)

    selected_indices = select_indices_with_geometric_decay(hea_eigvals, args.geometric_spacing)
    selected_indices = [i for i in selected_indices if hea_eigvals[i] > args.cutoff_hea_eigval]
    # optional monomial order filter
    if args.monomial_degree is not None:
        allowed_degrees = set(args.monomial_degree)
        selected_indices = [i for i in selected_indices if monomials[i].degree() in allowed_degrees]
    selected_indices = selected_indices[:args.cutoff_hea_mode]

    selected_hea_eigvals = hea_eigvals[selected_indices]
    # selected_monomials = [monomials[i].basis() for i in selected_indices] #in basis form, what json can serialize

    index_map = np.array(data_indices_of_interest, dtype=int)
    selected_monomials = [
        {str(int(index_map[int(k)])): int(v)
        for k, v in monomials[i].basis().items()}
        for i in selected_indices
    ]

    # f"selected {len(selected_indices)} HEA eigenmodes."
    # monomials_as_dicts = [monomial.basis() for monomial in monomials]
    # mapped_monomials_as_dicts = []
    # for monomial_dict in monomials_as_dicts:
    #     mapped_dict = {}
    #     for key, value in monomial_dict.items():
    #         mapped_key = data_indices_of_interest[key]
    #         mapped_dict[mapped_key] = value
    #     mapped_monomials_as_dicts.append(mapped_dict)
    # mapped_monomials_as_dicts = mapped_monomials_as_dicts[:args.cutoff_hea_mode]

    outdir = args.outdir
    out_hps = args.out_hps
    out_main = args.out_main

    if outdir and not out_hps:
        out_hps = outdir / "datasethps.json"
    if outdir and not out_main:
        out_main = outdir / "output.json"

    # Write both if paths provided
    if out_hps:
        keys = ["normalized","cutoff_mode","d","offset","alpha",
                "noise_size","yoffset","beta","classes","binarize"]
        hps = {k: getattr(args, k) for k in keys if hasattr(args, k)}
        write_json(hps, Path(out_hps))
        print(f"[info] wrote hparams → {out_hps}", file=sys.stderr)
    if out_main:
        write_json(selected_monomials, Path(out_main))
        print(f"[info] wrote results → {out_main}", file=sys.stderr)

    # return mapped_monomials_as_dicts

if __name__ == "__main__":
    main()