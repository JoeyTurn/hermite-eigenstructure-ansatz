import ast
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

import os, sys
sys.path.append("/home/user/feature_recombination/")

from kernels import ReluNTK
from feature_decomp import Monomial
from utils import ensure_torch
from misc import rcsetup
from FileManager import FileManager
from ExptTrace import ExptTrace

rcsetup()

colors = ['xkcd:black', 'xkcd:red', 'xkcd:orange yellow', 'xkcd:green', 'xkcd:blue', 'xkcd:purple']
# colors = ['xkcd:black', 'xkcd:lilac', 'xkcd:blue', 'xkcd:green', 'xkcd:orange yellow', 'xkcd:red']
#['xkcd:purple', 'xkcd:raspberry', 'xkcd:cerulean', 'xkcd:tangerine', 'xkcd:black', 'xkcd:forest green', 'xkcd:crimson']
markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']

### plotting
def make_triangle(ax, x=7, y0=5, dlogx=1.3, slope=0.5):
    #triangle        
    dlogy = slope * dlogx 
    x1 = x * 10**dlogx
    y1 = y0 * 10**dlogy
    dx = x1 - x
    dy = y1 - y0
    x_coords = [x, x1, x1]
    y_coords = [y0, y0, y1]
    ax.plot(x_coords, y_coords, 'k-', linewidth=0.5)
    ax.plot([x, x1], [y0, y1], 'k-', linewidth=0.5)  
    ax.text(x+dx*1.1, y0+dy*(np.log10(2)), r'$1/2$', 
            va='center', ha='left', fontsize=13, fontweight="light")
from matplotlib.lines import Line2D

def make_color_handles(colors, max_handle=None, s=None):
        labels = ["constant", "linear", "quadratic", "cubic", "quartic"]

        # Determine the last included index (inclusive)
        if max_handle is None:
            last = min(len(labels) - 1, len(colors) - 1)
        else:
            if max_handle < 0:
                return []  # nothing to show
            last = min(max_handle, len(labels) - 1, len(colors) - 1)

        pairs = [(lbl, c) for lbl, c in zip(labels[:last+1], colors[:last+1])]

        color_handles = [
            Line2D([0], [0], marker='o', linestyle='none',
                markerfacecolor=c, markeredgecolor='black', markersize=10,
                label=lbl)
            for lbl, c in pairs
        ]
        return color_handles

def add_degree_and_shape_legends(ax, color_handles, colors=colors, legloc = "upper left"):

    # shape_handles = [
    #     Line2D([0], [0], marker=m, linestyle='none',
    #            markerfacecolor='white', markeredgecolor='black', markersize=9,
    #            label=lbl)
    #     for m, lbl in [
    #         ('x', 'constant mode'),
    #         ('s', 'no repeated indices'),
    #         ('o', '1 repeated index'),
    #         ('^', '2 repeated indices'),
    #         ('D', '3 repeated indices'),
    #     ]
    # ]

    # One legend, two columns, top-right
    all_handles = color_handles# + shape_handles
    ax.legend(
        handles=all_handles,
        ncol=1,#2,
        loc=legloc,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
        # title="color = degree, shape = index pattern"
    )
import matplotlib.colors as mcolors
# colors = ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]
def lighten(color, amount=0.5):
    rgb = mcolors.to_rgb(mcolors.XKCD_COLORS.get(color, color))
    if amount >= 0:
        new_rgb = tuple(c + (1 - c) * amount for c in rgb)
    else:
        new_rgb = tuple(c * (1 + amount) for c in rgb)
    # Clip values to [0, 1]
    new_rgb = tuple(min(max(x, 0), 1) for x in new_rgb)
    return new_rgb

def plot_ttl_across_max_deg(fig, axes, timekeys_mean, timekeys_std, targets, monomials, eigvals, errorbar_on=True, colors=None, s=6): #fra_eigvals

    target_monomials = targets
    if type(target_monomials[0]) == str:
        target_monomials = np.array([Monomial.from_repr(target_monomial) for target_monomial in target_monomials])
    target_degrees = np.array([target_monomial.degree() for target_monomial in target_monomials])
    target_max_degrees = np.array([target_monomial.max_degree() for target_monomial in target_monomials])
    degrees = np.array([monomial.degree() if monomial in target_monomials else 0 for monomial in monomials], dtype=int)
    max_degrees = np.array([monomial.max_degree() if monomial in target_monomials else 0 for monomial in monomials], dtype=int)


    def _plot_within_order(ax, order=0, opacity=0.5):
        degmaxlocs = torch.tensor(np.where(max_degrees == order)[0])
        degtargetmaxlocs = torch.tensor(np.where(target_max_degrees == order)[0])

        xaxis = (eigvals[degmaxlocs.long()].cpu())**(-1.)
        ys = ensure_torch(timekeys_mean)
        yerr = ensure_torch(timekeys_std)
        slope, intercept = get_log_log_linear_fit(xaxis, ys[degtargetmaxlocs.long()])
        inside_degrees=degrees[degmaxlocs]

        for deg in np.unique(inside_degrees):
            idxs = (inside_degrees == deg)
            if errorbar_on:
                ax.errorbar((eigvals.cpu()[degmaxlocs][idxs])**(-1), timekeys_mean[degtargetmaxlocs][idxs],
                            yerr=yerr[degtargetmaxlocs][idxs],
                            color=colors[deg % len(colors)], alpha=opacity, fmt=markers[order], s=s)
            else:
                ax.scatter((eigvals.cpu()[degmaxlocs][idxs])**(-1), timekeys_mean[degtargetmaxlocs][idxs], marker=markers[order],
                        color=colors[deg % len(colors)], alpha=opacity, s=s)
        ax.set_title(f"Max Degree {order}")
        ax.plot(xaxis, 10**(intercept)*xaxis**(slope), color='k', label=f"log(TTL) = {slope:.2f}*$\\log(\\lambda_{{HEA}}^{{-1}})$+{intercept:.1f}")

    flatax = np.ravel(axes)
    for i, ax in enumerate(flatax, start=1):  # axes[0] -> i=1
        _plot_within_order(ax, order=i)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

def get_log_log_linear_fit(x, y, fixed_slope: float | None = None):
    """
    Fit log10(y) = slope * log10(x) + intercept.
    If fixed_slope is provided, only estimate the intercept.
    """
    xt = ensure_torch(x)
    yt = ensure_torch(y)

    log_x = torch.log10(xt).reshape(-1, 1)
    z = torch.log10(yt).reshape(-1, 1)

    if fixed_slope is not None:
        m = float(fixed_slope)
        b = float((z - m * log_x).mean())
        return m, b

    A = torch.column_stack((log_x, torch.ones_like(log_x)))
    sol = torch.linalg.lstsq(A, z).solution.squeeze()  # [slope, intercept]
    return float(sol[0]), float(sol[1])
def plot_time_to_learn_eigenvalue(eigvals, timekeys, target_monomials, scale='log', **kwargs):
    target_monomials = np.asarray(target_monomials, dtype=object)
    if target_monomials.size == 0:
        return
    if type(target_monomials[0]) == str:
        target_monomials = np.array([Monomial.from_repr(target_monomial) for target_monomial in target_monomials])
    degrees = np.array([target_monomial.degree() for target_monomial in target_monomials])
    colors = kwargs.get("colors", ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"])
    #marker stuff
    markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']
    max_degrees  = np.array([target_monomial.max_degree() for target_monomial in target_monomials])
    max_deg = int(np.asarray(max_degrees).max())
    uniq_md = np.arange(max_deg + 1)
    md2marker   = {md: markers[i % len(markers)] for i, md in enumerate(uniq_md)}

    pairs = np.unique(np.stack([degrees, max_degrees], axis=1), axis=0)

    for degree, md in pairs:
        idxs = np.flatnonzero((degrees == degree) & (max_degrees == md))
        if kwargs.get("errorbar", False):
            plt.errorbar((eigvals[idxs])**(-1), timekeys[idxs], yerr=kwargs.get("yerr")[idxs], color=colors[degree % len(colors)], alpha=kwargs.get("alpha", 1), fmt=md2marker[md])
        else:
            plt.scatter((eigvals[idxs])**(-1), timekeys[idxs], marker=md2marker[md], color=colors[degree % len(colors)], alpha=kwargs.get("alpha", 1))
    plt.xscale(scale)
    plt.yscale(scale)
    plt.xlabel(f"HEA Eigval $\\lambda^{{-1}}$")
    plt.ylabel(f"Time to learn "+kwargs.get("breakpoint", ""))
    # plt.title(f"Time to learn vs FRA Eigval")

def plot_time_to_learn_eigenvalue_lightened(eigvals, timekeys, target_monomials, scale='log', lightenamount=0, **kwargs):
    target_monomials = np.asarray(target_monomials, dtype=object)
    if target_monomials.size == 0:
        return
    if type(target_monomials[0]) == str:
        target_monomials = np.array([Monomial.from_repr(target_monomial) for target_monomial in target_monomials])
    degrees = np.array([target_monomial.degree() for target_monomial in target_monomials])
    colors = kwargs.get("colors", ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"])
    #marker stuff
    markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']
    max_degrees  = np.array([target_monomial.max_degree() for target_monomial in target_monomials])
    max_deg = int(np.asarray(max_degrees).max())
    uniq_md = np.arange(max_deg + 1)
    md2marker   = {md: markers[i % len(markers)] for i, md in enumerate(uniq_md)}

    pairs = np.unique(np.stack([degrees, max_degrees], axis=1), axis=0)

    for degree, md in pairs:
        idxs = np.flatnonzero((degrees == degree) & (max_degrees == md))
        if kwargs.get("errorbar", False):
            plt.errorbar((eigvals[idxs])**(-1), timekeys[idxs], yerr=kwargs.get("yerr")[idxs], color=lighten(colors[degree % len(colors)], amount=lightenamount), alpha=kwargs.get("alpha", 1), fmt=md2marker[md])
        else:
            plt.scatter((eigvals[idxs])**(-1), timekeys[idxs], marker=md2marker[md], color=lighten(colors[degree % len(colors)], amount=lightenamount), alpha=kwargs.get("alpha", 1))
    plt.xscale(scale)
    plt.yscale(scale)
    # plt.title(f"Time to learn vs FRA Eigval")

def plot_time_to_learn_eigenvalue_lightened_ax(ax, eigvals, timekeys, target_monomials, scale='log', lightenamount=0, **kwargs):
    target_monomials = np.asarray(target_monomials, dtype=object)
    if target_monomials.size == 0:
        return
    if type(target_monomials[0]) == str:
        target_monomials = np.array([Monomial.from_repr(target_monomial) for target_monomial in target_monomials])
    degrees = np.array([target_monomial.degree() for target_monomial in target_monomials])
    colors = kwargs.get("colors", ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"])
    #marker stuff
    markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']
    max_degrees  = np.array([target_monomial.max_degree() for target_monomial in target_monomials])
    max_deg = int(np.asarray(max_degrees).max())
    uniq_md = np.arange(max_deg + 1)
    md2marker   = {md: markers[i % len(markers)] for i, md in enumerate(uniq_md)}

    pairs = np.unique(np.stack([degrees, max_degrees], axis=1), axis=0)

    for degree, md in pairs:
        idxs = np.flatnonzero((degrees == degree) & (max_degrees == md))
        if kwargs.get("errorbar", False):#fmt=md2marker[md]
            ax.errorbar((eigvals[idxs])**(-1), timekeys[idxs], yerr=kwargs.get("yerr")[idxs], color=lighten(colors[degree % len(colors)], amount=lightenamount),
                        alpha=kwargs.get("alpha", 1), fmt=markers[kwargs.get('marker_index', 2)], ms=kwargs.get("s"))
        else:#md2marker[md]
            ax.scatter((eigvals[idxs])**(-1), timekeys[idxs], color=lighten(colors[degree % len(colors)], amount=lightenamount),
                       alpha=kwargs.get("alpha", 1), marker=markers[kwargs.get('marker_index', 2)], ms=kwargs.get("s"))
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    # plt.title(f"Time to learn vs FRA Eigval")

def plot_time_to_learn_eigenvalue_ax(ax, eigvals, timekeys, target_monomials, scale='log', **kwargs):
    target_monomials = np.asarray(target_monomials, dtype=object)
    if target_monomials.size == 0:
        return
    if type(target_monomials[0]) == str:
        target_monomials = np.array([Monomial.from_repr(target_monomial) for target_monomial in target_monomials])
    degrees = np.array([target_monomial.degree() for target_monomial in target_monomials])
    colors = kwargs.get("colors", ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"])
    #marker stuff
    markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']
    max_degrees  = np.array([target_monomial.max_degree() for target_monomial in target_monomials])
    max_deg = int(np.asarray(max_degrees).max())
    uniq_md = np.arange(max_deg + 1)
    md2marker   = {md: markers[i % len(markers)] for i, md in enumerate(uniq_md)}

    pairs = np.unique(np.stack([degrees, max_degrees], axis=1), axis=0)

    for degree, md in pairs:
        idxs = np.flatnonzero((degrees == degree) & (max_degrees == md))
        if kwargs.get("errorbar", False):
            ax.errorbar((eigvals[idxs])**(-1), timekeys[idxs], yerr=kwargs.get("yerr")[idxs], color=colors[degree % len(colors)], alpha=kwargs.get("alpha", 1), fmt=md2marker[md])
        else:
            ax.scatter((eigvals[idxs])**(-1), timekeys[idxs], marker=md2marker[md], color=colors[degree % len(colors)], alpha=kwargs.get("alpha", 1))
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    # plt.title(f"Time to learn vs FRA Eigval")

def plot_time_to_learn_eigenvalue_ax(ax, eigvals, timekeys, target_monomials, scale='log', **kwargs):
    target_monomials = np.asarray(target_monomials, dtype=object)
    if target_monomials.size == 0:
        return
    if type(target_monomials[0]) == str:
        target_monomials = np.array([Monomial.from_repr(target_monomial) for target_monomial in target_monomials])
    degrees = np.array([target_monomial.degree() for target_monomial in target_monomials])
    colors = kwargs.get("colors", ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"])
    #marker stuff
    markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']
    max_degrees  = np.array([target_monomial.max_degree() for target_monomial in target_monomials])
    max_deg = int(np.asarray(max_degrees).max())
    uniq_md = np.arange(max_deg + 1)
    md2marker   = {md: markers[i % len(markers)] for i, md in enumerate(uniq_md)}

    pairs = np.unique(np.stack([degrees, max_degrees], axis=1), axis=0)

    for degree, md in pairs:
        idxs = np.flatnonzero((degrees == degree) & (max_degrees == md))
        if kwargs.get("errorbar", False):
            ax.errorbar((eigvals[idxs])**(-1), timekeys[idxs], yerr=kwargs.get("yerr")[idxs], color=colors[degree % len(colors)], alpha=kwargs.get("alpha", 1), fmt=md2marker[md])
        else:
            ax.scatter((eigvals[idxs])**(-1), timekeys[idxs], marker=md2marker[md], color=colors[degree % len(colors)], alpha=kwargs.get("alpha", 1))
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    # plt.title(f"Time to learn vs FRA Eigval")

### load results grabbing
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

def _coerce_monomial(value):
    if isinstance(value, Monomial):
        return value.copy()
    if isinstance(value, dict):
        return Monomial({int(k): int(v) for k, v in value.items()})
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return Monomial({int(k): int(v) for k, v in parsed.items()})
        return Monomial.from_repr(text)
    raise TypeError(f"Unsupported monomial value: {type(value)!r}")


def _build_plot_compat(result, targets, monomials, hea_eigvals, eff_times, eff_stds, lr):
    target_info = result.setdefault("target_info", {})
    target_info["monomials"] = [m.copy() for m in monomials]

    pseudo_timekeys = np.asarray(eff_times, dtype=float)
    if lr != 0:
        pseudo_timekeys = pseudo_timekeys / lr
    pseudo_timekeys = pseudo_timekeys.reshape(1, -1, 1)

    return {
        "timekeys": pseudo_timekeys,
        "targets": np.asarray(targets, dtype=object),
        "num_trials": pseudo_timekeys.shape[-1],
        "monomials": np.asarray(monomials, dtype=object),
        "hea_eigvals": torch.as_tensor(np.asarray(hea_eigvals, dtype=float)),
        "breakpoints_means": torch.as_tensor(np.asarray(eff_times, dtype=float)),
        "breakpoints_std": torch.as_tensor(np.asarray(eff_stds, dtype=float)),
    }


def _extract_times(result, ttl_threshold, lr_override=None):
    timekeys = result["timekeys"]["config2outcome"]
    losses = result["losses"]["config2outcome"]

    loss_checkpoints = result.get("loss_checkpoints", [ttl_threshold])
    ttl_idx, _ = _threshold_index(loss_checkpoints, ttl_threshold)

    training_cfg = result.get("training_config", {})
    lr = float(lr_override) if lr_override is not None else float(training_cfg.get("LR", 1.0))

    target_info = result.get("target_info", {})
    target_map = target_info.get("monomial_str_to_eigval") or result.get("target_map") or {}
    bases = target_info.get("monomials") or target_info.get("monomial_bases") or []
    monomial_strs = target_info.get("monomial_strs") or []
    stored_eigs = target_info.get("hea_eigvals") or []

    monomial_lookup = {}
    eig_lookup = {}
    order_lookup = {}

    for idx, basis in enumerate(bases):
        monomial = _coerce_monomial(basis)
        eigval = float(stored_eigs[idx]) if idx < len(stored_eigs) else None
        keys = {str(basis), str(monomial)}
        if idx < len(monomial_strs):
            keys.add(str(monomial_strs[idx]))
        for key in keys:
            monomial_lookup[key] = monomial
            if eigval is not None:
                eig_lookup[key] = eigval
            order_lookup[key] = monomial.degree()

    for key, eigval in target_map.items():
        try:
            monomial = _coerce_monomial(key)
        except Exception:
            continue
        monomial_lookup.setdefault(key, monomial)
        monomial_lookup.setdefault(str(monomial), monomial)
        eig_lookup[key] = float(eigval)
        eig_lookup.setdefault(str(monomial), float(eigval))
        order_lookup.setdefault(key, monomial.degree())
        order_lookup.setdefault(str(monomial), monomial.degree())

    if not eig_lookup:
        raise ValueError("No monomial-to-eigenvalue mapping found in result.")

    # Aggregate steps by monomial across trials
    monomial_steps = {}
    for config, tk in timekeys.items():
        raw_monomial = config[-1]
        try:
            monomial = monomial_lookup.get(raw_monomial, _coerce_monomial(raw_monomial))
        except Exception:
            monomial = None
        if monomial is None:
            continue
        monomial_key = str(monomial)
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
        monomial_steps.setdefault(monomial_key, []).append(steps)

    inv_eigs = []
    eff_times = []
    eff_stds = []
    orders = []
    filtered_targets = []
    filtered_monomials = []
    filtered_hea_eigvals = []
    for monomial_key, steps_list in monomial_steps.items():
        monomial = monomial_lookup.get(monomial_key, _coerce_monomial(monomial_key))
        eigval = eig_lookup.get(monomial_key)
        if eigval is None:
            eigval = eig_lookup.get(str(monomial))
        order = order_lookup.get(monomial_key, monomial.degree())
        if eigval is None or eigval <= 0:
            continue
        inv_eigs.append(1.0 / eigval)
        eff_times.append(lr * float(np.mean(steps_list)))
        eff_stds.append(lr * float(np.std(steps_list, ddof=0)))
        orders.append(order)
        filtered_targets.append(str(monomial))
        filtered_monomials.append(monomial)
        filtered_hea_eigvals.append(eigval)

    compat = _build_plot_compat(
        result,
        targets=filtered_targets,
        monomials=filtered_monomials,
        hea_eigvals=filtered_hea_eigvals,
        eff_times=eff_times,
        eff_stds=eff_stds,
        lr=lr,
    )
    return np.asarray(inv_eigs), np.asarray(eff_times), np.asarray(eff_stds), np.asarray(orders), lr, compat
dataset_path = '/mnt/private/results/hermite-eigenstructure-ansatz/mlp/synthetic'
result_path = os.path.join(dataset_path, "result_ttl.pickle")
result = _load_result(result_path)

ttl_threshold = float(result.get("ttl_threshold", 0.1))

inv_eigs, eff_times, eff_stds, orders, lr, compat = _extract_times(result, ttl_threshold)
import json

from feature_decomp import Monomial

c = {
    0: 0.9423311143775628,
    1: 0.7585548159036320,
    2: 0.3062938307898845,
    3: 0.1837762984739307,
    4: 0.4492309518251639,
    5: 1.3749189737679260,
    6: 5.9352937877506510,
}

dataset_path = '/mnt/private/results/hermite-eigenstructure-ansatz/mlp/synthetic'
result_path = os.path.join(dataset_path, "result_ttl.pickle")
result = _load_result(result_path)

ttl_threshold = float(result.get("ttl_threshold", 0.1))

inv_eigs, eff_times, eff_stds, orders, lr, compat = _extract_times(result, ttl_threshold)

# --- here ---

timekeys = compat["timekeys"]
targets = compat["targets"]
num_trials = compat["num_trials"]
monomials = compat["monomials"]
hea_eigvals = compat["hea_eigvals"]
locs = torch.zeros(len(targets))

for i, monomial in enumerate(targets):
    monomial = Monomial.from_repr(monomial)
    # print(np.where(np.array(monomials) == monomial))
    if i >= len(targets):
        break
    # print(loc)
    loc = np.where(np.array(monomials) == monomial)[0][0]
    locs[i] = loc
tags = [{}, {1:1}, {5:1}, {100:1}, {204:1},
        {0:2}, {0:1,2:1}, {2:1, 3:1}, {1:1,10:1}, #{1:1,40:1},
        {1:3}, {0:1, 1:1, 2:1}, {3:1, 5:2},
        {0:4}, {0:1, 1:1, 2:1, 3:1}]
noteable_targets = list(map(Monomial, tags))
t2 = list(map(Monomial.from_repr, targets))
noteable_locs = torch.zeros(len(noteable_targets))
noteable_y = torch.zeros(len(noteable_targets))

for i, monomial in enumerate(noteable_targets):
    if i >= len(targets):
        break
    if monomial in monomials:
        loc = np.where(np.array(monomials) == monomial)[0][0]
        noteable_locs[i] = loc
    if monomial in t2:
        loc = np.where(np.array(t2) == monomial)[0][0]
        noteable_y[i] = loc
# noteable_targets
s = 6
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
breakpoints_means = compat["breakpoints_means"]
breakpoints_std = compat["breakpoints_std"]

xaxis = ((hea_eigvals[locs.long()].cpu()))**(-1.)

plot_time_to_learn_eigenvalue_lightened_ax(ax, hea_eigvals[locs.long()].cpu(), breakpoints_means.cpu(), targets,
                                scale='log', breakpoint="(error < 30% starting err)", alpha=0.2, errorbar=True, lightenamount=0,
                                yerr=breakpoints_std, colors=colors, s=s)
# solid line along emp points
x_min = float(inv_eigs.min()) * 0.9
x_max = float(inv_eigs.max()) * 1.1
xaxis = np.logspace(np.log10(x_min), np.log10(x_max), 30)
ax.plot(xaxis, 10**(0.65)*(xaxis)**(0.5), color='k', linestyle='--', alpha=0.8)

#points
point_locs = [3, 3, 3, 3,
        3, 0, 3, 3, 0,
        0, 3, 0,
        0, 3]

for idx, xi, yi in zip(noteable_locs, hea_eigvals[noteable_locs.long()]**(-1.), breakpoints_means[noteable_y.long()]):
    m = Monomial(monomials[idx.int().cpu().numpy()])
    if m in tags:
        color = colors[m.degree() % len(colors)]
        ax.scatter(xi, yi, color = color, alpha=1, zorder=10, s=s**2.)
        i = tags.index(monomials[idx.int().cpu().numpy()])
        kwargs = {"ha": "left" if point_locs[i] < 2 else "right",
                  "va": "bottom" if point_locs[i] % 2 == 1 else "top"}
        z = 0.08
        off = (
            z if point_locs[i] < 2 else -z,
            z if point_locs[i] % 2 == 1 else -z
        )
        if m == Monomial({3:2, 5:1}):
            kwargs["ha"] = "center"
            off = (off[0]+0.28, off[1]+0.04)
        elif m == Monomial({3:1, 5:2}):
            kwargs["ha"] = "center"
            off = (off[0], off[1]-.14)
        elif m == Monomial({2:1, 3:1}):
            kwargs["ha"] = "right"
            off = (off[0]+0.08, off[1]+0.08)
        elif m == Monomial({1:1, 10:1}):
            kwargs["ha"] = "left"
            kwargs["va"] = "top"
            off = (off[0]+.07, off[1]-.04)
        elif m == Monomial({0:1, 1:1, 2:1, 3:1}):
            kwargs["ha"] = "right"
            off = (off[0], off[1]-.08)
        ax.annotate(str(m).replace('x', 'z'), (xi*(1+off[0]), yi*(1+off[1])), fontsize=13,
                    color=lighten(color, -0.2), alpha=0.7, **kwargs)
        

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

ax.text(0.02, 0.97, f"MLP @ Gaussian Data", ha="left", va="top",
        transform=ax.transAxes, fontsize=16, color="xkcd:dark gray")

color_handles = make_color_handles(colors, max_handle=4, s=s)
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
ax.set_ylabel("Eff. opt. time $\\eta\\cdot n_\\mathrm{{iter}}$", fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
# plt.legend()
# ax.set_title(f"MLP @ Gaussian Data", fontsize=24)
plt.show()
