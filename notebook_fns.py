import torch
import numpy as np
from utils import ensure_torch
from feature_decomp import generate_hea_monomials
from data import get_powerlaw, compute_hermite_basis, get_synthetic_X

def get_synthetic_dataset(X=None, data_eigvals=None, d=500, N=15000, offset=3, alpha=1.5, cutoff_mode=10000,
                          noise_size=0.1, yoffset=3, beta=1.2, normalized=True, gen=None, **kwargs):
    """
    noise_size: total noise size of the N-dim target vector y
    """
    if X is None:
        X, data_eigvals = get_synthetic_X(d=d, N=N, offset=offset, alpha=alpha, gen=gen)

    kernel_width = kwargs.get("kernel_width", 2)
    kerneltype = kwargs.get("kerneltype", None)
    fra_eigvals, monomials = generate_hea_monomials(data_eigvals, cutoff_mode, kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals, **kwargs), kmax=kwargs.get('kmax', 9))
    H = ensure_torch(compute_hermite_basis(X, monomials))
    fra_eigvals = ensure_torch(fra_eigvals)
    v_true = ensure_torch(get_powerlaw(H.shape[1], beta/2, offset=yoffset, normalize=normalized))
    v_true = v_true if not normalized else v_true/torch.linalg.norm(v_true)* N**(0.5)
    y = ensure_torch(H) @ v_true + ensure_torch(torch.normal(0., noise_size, (H.shape[0],), generator=gen, device=H.device))#/H.shape[0]**(0.5)
    return X, y, H, monomials, fra_eigvals, v_true, data_eigvals

def get_standard_tools(X, kerneltype, kernel_width, top_mode_idx=3000, data_eigvals=None, kmax=20):
    
    if data_eigvals is None:
        N, _ = X.shape
        S = torch.linalg.svdvals(X)
        data_eigvals = S**2 / (S**2).sum()

    kernel = kerneltype(X, kernel_width=kernel_width)
    eval_level_coeff = kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_hea_monomials(data_eigvals, top_mode_idx, eval_level_coeff, kmax=kmax)
    H = ensure_torch(compute_hermite_basis(X, monomials))

    return monomials, kernel, H, fra_eigvals, data_eigvals

# under development

# def get_all_targets(target_monomials, monomials, H):
#     N = H.shape[0]
#     y_all = torch.zeros((N, len(target_monomials)))
#     locs = torch.zeros(len(target_monomials))

#     for i, monomial in enumerate(target_monomials):
#         loc = np.where(np.array(monomials) == monomial)[0][0]
#         locs[i] = loc
#         y_all[:, i] = H[:, loc]

#     return y_all, locs

# def find_good_hea_eigenmodes(hea_eigvals):
#     # heuristic: look for geometric decay in eigenvalues
#     # i.e. if eigvals are decreasing by at least 10% each time, keep going
#     # once they start decreasing by less than 10%, stop

#     def select_indices_with_geometric_decay(values, ratio=.9):

#         # assert that values is (a) positive and (b) already sorted
#         assert np.all(values > 0)
#         assert np.all(np.diff(values) <= 0)

#         selected_indices = []

#         cur_eigval_thresh = values[0] + 1
#         ratio = .9

#         for i in range(len(hea_eigvals)):
#             if values[i] < cur_eigval_thresh:
#                 selected_indices.append(i)
#                 cur_eigval_thresh = values[i] * ratio

#         return selected_indices

#     from data import get_powerlaw
#     data_eigvals = get_powerlaw(P=datasethps['d'], exp=datasethps['alpha'], offset=datasethps['offset'], normalize=True) #aka data_eigvals
#     level_coeff_fn = ReluNTK.get_level_coeff_fn(data_eigvals=data_eigvals, bias_variance=1, weight_variance=1)
#     hea_eigvals, monomials = generate_hea_monomials(data_eigvals, datasethps['cutoff_mode'], level_coeff_fn, kmax=6) #don't touch kmax, not going above order 6

#     data_indices_of_interest = [0, 1, 2, 3, 5, 10, 20, 40, 60, 100, 150]
#     gammas_of_interest = data_eigvals.cpu().numpy()[data_indices_of_interest]
#     hea_eigvals, monomials = generate_hea_monomials(gammas_of_interest, datasethps['cutoff_mode'], level_coeff_fn, kmax=4)

#     selected_indices = select_indices_with_geometric_decay(hea_eigvals, .9)
#     hea_eigenval_cutoff = 1e-6
#     selected_indices = [i for i in selected_indices if hea_eigvals[i] > hea_eigenval_cutoff]

#     selected_hea_eigvals = hea_eigvals[selected_indices]
#     selected_monomials = [monomials[i] for i in selected_indices]

#     # f"selected {len(selected_indices)} HEA eigenmodes."
#     monomials_as_dicts = [monomial.basis() for monomial in monomials]
#     mapped_monomials_as_dicts = []
#     for monomial_dict in monomials_as_dicts:
#         mapped_dict = {}
#         for key, value in monomial_dict.items():
#             mapped_key = data_indices_of_interest[key]
#             mapped_dict[mapped_key] = value
#         mapped_monomials_as_dicts.append(mapped_dict)
#     return selected_hea_eigvals
