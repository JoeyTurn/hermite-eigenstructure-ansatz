import argparse
import os
import pickle
import ast
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = ['xkcd:black', 'xkcd:red', 'xkcd:orange yellow', 'xkcd:green', 'xkcd:blue', 'xkcd:purple', 'xkcd:violet']
labels = ['constant', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic']

from misc import rcsetup

rcsetup()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _load_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _threshold_index(loss_checkpoints, ttl_threshold):
    thresholds = np.asarray(loss_checkpoints, dtype=float)
    thresholds = np.sort(thresholds)[::-1]
    idx = int(np.argmin(np.abs(thresholds - ttl_threshold)))
    if not np.isclose(thresholds[idx], ttl_threshold, rtol=1e-6, atol=1e-9):
        raise ValueError(f"ttl_threshold={ttl_threshold} not found in loss_checkpoints={thresholds.tolist()}")
    return idx, thresholds


def _infer_suffix_from_path(path):
    if path is None:
        return None
    p = os.path.normpath(path)
    if p.endswith(".pickle"):
        p = os.path.dirname(p)
    base = os.path.basename(p)
    if "_" not in base:
        return None
    return base.split("_", 1)[1]


def _bin_logspace(x, y, yerr, n_bins=30, pad_low=0.9, pad_high=1.1):
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    if x.size == 0:
        return x, y, yerr
    x_min = x.min() * pad_low
    x_max = x.max() * pad_high
    bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)
    # assign each x to nearest bin in log space
    logx = np.log10(x)
    logbins = np.log10(bins)
    idx = np.abs(logx[:, None] - logbins[None, :]).argmin(axis=1)
    xb = []
    yb = []
    eb = []
    for b in range(len(bins)):
        mask = idx == b
        if not np.any(mask):
            continue
        xb.append(bins[b])
        yb.append(float(np.mean(y[mask])))
        eb.append(float(np.mean(yerr[mask])))
    return np.asarray(xb), np.asarray(yb), np.asarray(eb)


def _extract_times(result, ttl_threshold, lr_override=None):
    timekeys = result["timekeys"]["config2outcome"]
    losses = result["losses"]["config2outcome"]

    loss_checkpoints = result.get("loss_checkpoints", [ttl_threshold])
    ttl_idx, _ = _threshold_index(loss_checkpoints, ttl_threshold)

    training_cfg = result.get("training_config", {})
    lr = float(lr_override) if lr_override is not None else float(training_cfg.get("LR", 1.0))

    target_map = None
    alt_target_map = None
    if "target_info" in result:
        target_map = result["target_info"].get("monomial_str_to_eigval")
        bases = result["target_info"].get("monomial_bases")
        eigs = result["target_info"].get("hea_eigvals")
        if bases is not None and eigs is not None:
            alt_target_map = {str(basis): float(ev) for basis, ev in zip(bases, eigs)}
    if target_map is None and "target_map" in result:
        target_map = result["target_map"]

    if target_map is None and alt_target_map is None:
        raise ValueError("No monomial_str_to_eigval mapping found in result.")

    order_map = None
    if "target_info" in result:
        bases = result["target_info"].get("monomial_bases")
        monomial_strs = result["target_info"].get("monomial_strs")
        if bases is not None:
            order_map = {}
            for i, b in enumerate(bases):
                deg = int(sum(b.values()))
                # dict-style string (e.g., "{0: 1, 3: 1}")
                order_map[str(b)] = deg
                # latex/pretty string (e.g., "$x_{0}$") if available
                if monomial_strs is not None and i < len(monomial_strs):
                    order_map[str(monomial_strs[i])] = deg

    # Aggregate steps by monomial across trials
    monomial_steps = {}
    for config, tk in timekeys.items():
        monomial_str = config[-1]
        tk_arr = np.asarray(tk)
        if tk_arr.ndim == 0:
            step_idx = int(tk_arr)
        else:
            step_idx = int(tk_arr[ttl_idx])

        # Ignore trials that never crossed the threshold (timekey==0)
        if step_idx == 0:
            continue

        final_loss = losses.get(config, None)
        if final_loss is not None:
            final_loss_val = float(np.asarray(final_loss))
            if final_loss_val > ttl_threshold:
                continue  # threshold not reached

        # Convert step index -> number of gradient steps executed
        steps = step_idx + 1
        monomial_steps.setdefault(monomial_str, []).append(steps)

    inv_eigs = []
    eff_times = []
    eff_stds = []
    orders = []
    for monomial_str, steps_list in monomial_steps.items():
        eigval = None
        order = None
        if target_map is not None and monomial_str in target_map:
            eigval = float(target_map[monomial_str])
        elif alt_target_map is not None and monomial_str in alt_target_map:
            eigval = float(alt_target_map[monomial_str])
        else:
            # Try parsing dict-like strings into canonical dict repr
            try:
                parsed = ast.literal_eval(monomial_str)
                if isinstance(parsed, dict):
                    parsed = {int(k): int(v) for k, v in parsed.items()}
                    key = str(parsed)
                    if alt_target_map is not None and key in alt_target_map:
                        eigval = float(alt_target_map[key])
                    if order_map is not None and key in order_map:
                        order = int(order_map[key])
                    if order is None:
                        order = int(sum(parsed.values()))
            except Exception:
                pass
        if order is None and order_map is not None and monomial_str in order_map:
            order = int(order_map[monomial_str])
        if order is None:
            order = 0
        if eigval is None or eigval <= 0:
            continue
        inv_eigs.append(1.0 / eigval)
        eff_times.append(lr * float(np.mean(steps_list)))
        eff_stds.append(lr * float(np.std(steps_list, ddof=0)))
        orders.append(order)

    return np.asarray(inv_eigs), np.asarray(eff_times), np.asarray(eff_stds), np.asarray(orders), lr


def make_triangle(ax, x0=7, y0=5, dlogx=1.3, slope=0.5):
    #triangle
    dlogy = slope * dlogx
    x1 = x0 * 10**dlogx
    y1 = y0 * 10**dlogy
    dx = x1 - x0
    dy = y1 - y0
    x_coords = [x0, x1, x1]
    y_coords = [y0, y0, y1]
    ax.plot(x_coords, y_coords, 'k-', linewidth=0.5)
    ax.plot([x0, x1], [y0, y1], 'k-', linewidth=0.5)
    ax.text(x0+dx*1.1, y0+dy*(np.log10(2)), r'$1/2$',
            va='center', ha='left', fontsize=13, fontweight="light")

from matplotlib.lines import Line2D

def make_color_handles(colors, max_handle=None):
        label_list = labels

        # Determine the last included index (inclusive)
        if max_handle is None:
            last = min(len(label_list) - 1, len(colors) - 1)
        else:
            if max_handle < 0:
                return []  # nothing to show
            last = min(max_handle, len(label_list) - 1, len(colors) - 1)

        pairs = [(lbl, c) for lbl, c in zip(label_list[:last+1], colors[:last+1])]

        color_handles = [
            Line2D([0], [0], marker='o', linestyle='none',
                markerfacecolor=c, markeredgecolor='black', markersize=10,
                label=lbl)
            for lbl, c in pairs
        ]
        return color_handles


def add_degree_and_shape_legends(ax, color_handles, colors=colors, legloc = "upper left"):

    all_handles = color_handles
    ax.legend(
        handles=all_handles,
        ncol=1,#2,
        loc=legloc,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
        # title="color = degree, shape = index pattern"
    )


def main():
    p = argparse.ArgumentParser(description="Plot effective time vs inverse HEA eigvals for MLP TTL.")
    p.add_argument("--result", type=str, default=None, help="Path to result_ttl.pickle")
    p.add_argument("--results", type=str, default=None, help="Path to results dir or result_ttl.pickle")
    p.add_argument("--out", type=str, default=None, help="Path to save figure (png/pdf).")
    p.add_argument("--ttl-threshold", type=float, default=None, help="EMA test loss threshold (default from result).")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate used in effective time.")
    args = p.parse_args()

    result_path = args.result
    if result_path is None and args.results is not None:
        result_path = args.results

    if result_path is None:
        datapath = os.getenv("RESULTPATH")
        if datapath is None:
            raise ValueError("Provide --result or set $RESULTPATH")
        expt_dir = os.path.join(datapath, "DPR/mlp/synthetic")
        result_path = os.path.join(expt_dir, "result_ttl.pickle")

    if os.path.isdir(result_path):
        result_path = os.path.join(result_path, "result_ttl.pickle")
    args.result = result_path

    result = _load_result(args.result)

    ttl_threshold = float(args.ttl_threshold) if args.ttl_threshold is not None else float(result.get("ttl_threshold", 0.1))

    inv_eigs, eff_times, eff_stds, orders, lr = _extract_times(result, ttl_threshold, lr_override=args.lr)
    
    if inv_eigs.size == 0:
        raise RuntimeError("No valid (threshold-crossing) points found.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    max_order = min(len(labels), len(colors)) - 1
    for order_idx, (label, color) in enumerate(zip(labels, colors)):
        mask = orders == order_idx
        if not np.any(mask):
            continue
        xb, yb, eb = _bin_logspace(inv_eigs[mask], eff_times[mask], eff_stds[mask])
        ax.errorbar(xb, yb, yerr=eb,
                    fmt="o", ms=4.5, color=color, ecolor=color, elinewidth=0.8,
                    capsize=2.5, alpha=0.9, markeredgecolor="none", label=f"{label}: {color}")

    # Higher-order terms (> max_order) fall back to last color
    mask_hi = orders > max_order
    if np.any(mask_hi):
        xb, yb, eb = _bin_logspace(inv_eigs[mask_hi], eff_times[mask_hi], eff_stds[mask_hi])
        ax.errorbar(xb, yb, yerr=eb,
                    fmt="o", ms=4.5, color=colors[-1], ecolor=colors[-1], elinewidth=0.8,
                    capsize=2.5, alpha=0.9, markeredgecolor="none", label=f"higher: {colors[-1]}")
    x_min = float(inv_eigs.min()) * 0.9
    x_max = float(inv_eigs.max()) * 1.1
    xaxis = np.logspace(np.log10(x_min), np.log10(x_max), 30)
    ax.plot(xaxis, 10**(0.65)*(xaxis)**(0.5), color='k', linestyle='--', alpha=0.8)# label=f"")

    s=4
    #triangle
    x0, y0 = 7, 5
    dlogx = 1.3
    dlogy = 0.5 * dlogx
    x1 = x0 * 10**dlogx
    y1 = y0 * 10**dlogy
    dx = x1 - x0
    dy = y1 - y0
    x_coords = [x0, x1, x1]
    y_coords = [y0, y0, y1]
    ax.plot(x_coords, y_coords, 'k-', linewidth=0.5)
    ax.plot([x0, x1], [y0, y1], 'k-', linewidth=0.5)
    ax.text(x0+dx*1.1, y0+dy*(np.log10(2)), r'$1/2$',
            va='center', ha='left', fontsize=13, fontweight="light")

    suffix = _infer_suffix_from_path(args.results or args.result)
    title = "MLP @ Synthetic" if suffix is None else f"MLP @ Synthetic ({suffix})"
    ax.text(0.02, 0.97, title, ha="left", va="top",
            transform=ax.transAxes, fontsize=16, color="xkcd:dark gray")

    color_handles = make_color_handles(colors, max_handle=max_order)
    all_handles = color_handles
    ax.legend(
            handles=all_handles,
            ncol=1,#2,
            loc="lower right",
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.6,
        )

    ax.set_xlabel(r'Inverse of HEA eigenvalue $\lambda_{\boldsymbol{\alpha}}^{-1}$', fontsize=14)
    ax.set_ylabel("Eff. opt. time $\eta\cdot n_\mathrm{{iter}}$", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if args.out is None:
        suffix = _infer_suffix_from_path(args.results or args.result)
        if suffix:
            args.out = f"mlp_ttl_vs_hea_{suffix}.png"
        else:
            args.out = "mlp_ttl_vs_hea.png"

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out} (lr={lr})")


if __name__ == "__main__":
    main()
