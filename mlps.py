import torch.nn as nn
import torch
import numpy as np
from collections import deque
from mupify import mupify, rescale
import copy

class CenteredWrapper(nn.Module):
    """
    Wrap any nn.Module so that forward(x) returns model(x) - baseline(x),
    where baseline is a frozen snapshot taken at wrap time.
    """
    def __init__(self, model: nn.Module, baseline_dtype=None):
        super().__init__()
        self.model = model
        self.baseline = copy.deepcopy(model)  # snapshot at wrap time
        for p in self.baseline.parameters():
            p.requires_grad = False
        self.baseline.eval()
        self._baseline_dtype = baseline_dtype
        if baseline_dtype is not None:
            self.baseline.to(dtype=baseline_dtype)

    @torch.no_grad()
    def recenter(self):
        """Reset baseline to current model weights (still frozen)."""
        self.baseline.load_state_dict(self.model.state_dict())
        if self._baseline_dtype is not None:
            self.baseline.to(dtype=self._baseline_dtype)
        self.baseline.eval()

    def forward(self, x):
        y = self.model(x)  # grads here
        with torch.inference_mode():
            y0 = self.baseline(x)  # no grads / low mem
        return y - y0


def centeredMLP(model: nn.Module, baseline_dtype=None) -> nn.Module:
    """
    Usage:
        centerednet = centeredMLP(baseline_net)     # wrap for centering
    """
    return CenteredWrapper(model, baseline_dtype=baseline_dtype)


class MLP(nn.Module):
    def __init__(self, d_in=1, width=4096, depth=2, d_out=1, bias=True, nonlinearity=None, forcezeros=False):
        super().__init__()
        self.d_in, self.width, self.depth, self.d_out = d_in, width, depth, d_out

        self.input_layer = nn.Linear(d_in, width, bias)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width, bias) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(width, d_out, bias)
        if forcezeros:
            with torch.no_grad():
                self.output_layer.weight.zero_()
                if self.output_layer.bias is not None:
                    self.output_layer.bias.zero_()
        self.nonlin = nonlinearity if nonlinearity is not None else nn.ReLU()
        
    def forward(self, x):
        h = self.nonlin(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
        out = self.output_layer(h)
        return out

    def get_activations(self, x):
        h_acts = []
        h = self.nonlin(self.input_layer(x))
        h_acts.append(h)
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
            h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out


def train_network(model, batch_function, lr=1e-2, max_iter=int(1e3), loss_checkpoints=None, percent_thresholds=None,
                  gamma=1., ema_smoother=0.0, X_tr=None, y_tr=None, X_te=None, y_te=None, stopper=None, only_thresholds=False,
                  verbose=False, **kwargs):
    """
    Returns:
        model, losses, timekeys

    - timekeys[j]: first gradient step where the loss drops below the j-th threshold.
    - If RELATIVE mode, thresholds are `percent_thresholds * init_loss`.
    - If ABSOLUTE mode, thresholds are raw absolutes and the comparison metric is raw loss.
    - If only_thresholds is True, only returns the timekeys and not the full loss curves.
    """

    def return_statement(model, ema_tr, ema_te, timekeys, tr_losses=None, te_losses=None, extras={}):
        if only_thresholds:
            extras.update({"model": model, "train_losses": ema_tr, "test_losses": ema_te, "timekeys": timekeys})
            return extras
        # if otherreturns is not None:
        #     for name, _ in items:
        #         extras[name] = np.array(extras[name])
        extras.update({"model": model, "train_losses": tr_losses, "test_losses": te_losses, "timekeys": timekeys})
        return extras 

    def fill_losses(tr_losses, te_losses, extras, i):
        tr_losses[i:] = tr_losses[i]
        te_losses[i:] = te_losses[i]
        if otherreturns is not None:
            for name, _ in list(otherreturns.items()):
                extras[name][i:] = extras[name][i]
        return tr_losses, te_losses, extras
    
    #checking if losses are absolute or relative
    has_abs = (loss_checkpoints is not None) and len(loss_checkpoints) > 0
    is_relative = (percent_thresholds is not None) and len(percent_thresholds) > 0
    if has_abs and is_relative or not has_abs and not is_relative:
        raise ValueError("Provide exactly one of loss_checkpoints OR percent_thresholds.")

    # model stuff
    lr = lr * gamma if gamma >= 1 else lr * (gamma**2.)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mupify(model, opt, param="mup")
    rescale(model, gamma)
    model = centeredMLP(model).to(next(model.parameters()).device)
    loss_fn = torch.nn.MSELoss()

    # thresholding
    thresholds = np.asarray(percent_thresholds if is_relative else loss_checkpoints, dtype=float)
    thresholds = np.sort(thresholds)[::-1] # descending
    timekeys = np.full(thresholds.shape, 0, dtype=int)

    otherreturns = kwargs.get("otherreturns", None)
    if type(otherreturns) is dict:
        items = list(otherreturns.items())
    extras = {}

    if not(only_thresholds):
        tr_losses = np.empty(max_iter, dtype=float)
        te_losses = np.empty(max_iter, dtype=float)
        # if otherreturns is not None:
        #     for name, _ in items:
                # extras[name] = np.empty((max_iter, 200, 200), dtype=float) #temporary, this is gonna need to be CHANGED
                # extras[name] = []#np.empty(max_iter, dtype=float)
    else:
        tr_losses = te_losses = None
        if otherreturns is not None:
            for name, _ in items:
                extras[name] = None
    ema = None
    pointer = 0   
    X_tr_not_provided = X_tr is None

    # training loop 
    # print("Starting training loop...")
    for i in range(int(max_iter)):
        X_tr, y_tr = batch_function(i)
        
        opt.zero_grad()
        out = model(X_tr).squeeze()
        loss = loss_fn(out, y_tr.squeeze())
        loss.backward()
        opt.step()

        tr_loss_val = float(loss.item())
        if X_tr_not_provided:
            te_loss_val = tr_loss_val
        else:
            with torch.no_grad():
                out = model(X_te).squeeze()
                loss = loss_fn(out, y_te.squeeze())
                te_loss_val = float(loss.item())
        # initialize thresholds & loss trace baseline at first step
        if i == 0:
            ema_tr = tr_loss_val
            ema_te = te_loss_val
            if is_relative:
                thresholds *= tr_loss_val
            # prefill losses after init val calculated
            if not(only_thresholds):
                if otherreturns is not None:
                    for name, fn in items:
                        m = kwargs.get("monomial", None)
                        val = np.array(fn(model=model.model, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, monomial=m)) #the model.model to unwrap from centeredMLP
                        shape = (max_iter,) + val.shape
                        extras[name] = np.empty(shape, dtype=val.dtype)
                        # extras[name][:]= val
                tr_losses, te_losses, extras=fill_losses(tr_losses, te_losses, extras, i)

        ema_tr = (ema_smoother * ema_tr + (1.0 - ema_smoother) * tr_loss_val)
        ema_te = (ema_smoother * ema_te + (1.0 - ema_smoother) * te_loss_val)
        if not(only_thresholds):
            tr_losses[i] = ema_tr
            te_losses[i] = ema_te
            if otherreturns is not None:
                for name, fn in items:
                    val = fn(model=model.model, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, monomial=kwargs.get("monomial", None)) #the model.model to unwrap from centeredMLP
                    extras[name][i] = val
                    
        if verbose:
            print(f"Step {i}: ema train loss {ema_tr}, ema test loss {ema_te}")

        #if a plateauchecker exists, check against it
        if stopper is not None:
            stop, _ = stopper.update(i, te_loss_val)
            if not(only_thresholds):
                tr_losses, te_losses, extras=fill_losses(tr_losses, te_losses, extras, i)
            if stop:
                return return_statement(model, ema_tr, ema_te, timekeys, tr_losses, te_losses, extras)

        # if a threshold is hit, update pointer
        while pointer < len(thresholds) and ema_tr < thresholds[pointer]:
            timekeys[pointer] = i
            if not(only_thresholds):
                tr_losses, te_losses, extras=fill_losses(tr_losses, te_losses, extras, i)
            pointer += 1

        # early exit if all thresholds crossed
        if pointer == len(thresholds):
            if otherreturns is not None and only_thresholds:
                for name, fn in items:
                    val = fn(model=model.model, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, monomial=kwargs.get("monomial", None))
                    extras[name] = val#float(val)
            return return_statement(model, ema_tr, ema_te, timekeys, tr_losses, te_losses, extras)

    return return_statement(model, ema_tr, ema_te, timekeys, tr_losses, te_losses, extras)


# for checking when training has plateaued; discard if we want the full training run
# as this takes longer to evaluate than just running to completion 
class PlateauStopper:
    def __init__(self, min_iter=200, min_drop_frac=0.1, window=50, eval_every=10, patience=2, slope_eps=5e-4, rel_improve_eps=5e-4,
                 ema_smoother=0.0, use_log_time=True):
        self.min_iter = int(min_iter)
        self.min_drop_frac = float(min_drop_frac)
        self.window = int(window)
        self.eval_every = int(eval_every)
        self.patience = int(patience)
        self.slope_eps = float(slope_eps)
        self.rel_improve_eps = float(rel_improve_eps)
        self.ema_smoother = float(ema_smoother)
        self.use_log_time = bool(use_log_time)

        self.init_loss = None
        self.best_loss = np.inf
        self.started = False
        self.history = deque(maxlen=self.window)   # (step, loss) pairs
        self.loss = None
        self._last_check_step = -np.inf
        self._flat_hits = 0

    def update(self, step: int, test_loss: float):
        if self.loss is None:
            self.loss = float(test_loss)
        self.loss = self.ema_smoother * self.loss + (1 - self.ema_smoother) * float(test_loss)

        if self.init_loss is None:
            self.init_loss = self.loss
        self.best_loss = min(self.best_loss, self.loss)

        # don't start tracking until some time has passed/some improvement in losss
        if step < self.min_iter:
            self.history.append((step, self.loss))
            return False, {"reason": "warmup", "step": step}
        if not self.started:
            self.started = (self.best_loss <= (1 - self.min_drop_frac) * self.init_loss)

        self.history.append((step, self.loss))

        # only check every eval_every steps with a full window
        if step - self._last_check_step < self.eval_every or len(self.history) < self.window:
            return False, {"reason": "not_due_or_short_window", "step": step}

        self._last_check_step = step

        # build x=log(step) or x=step, y=loss
        steps = np.array([s for s, _ in self.history], dtype=float)
        ys    = np.array([y for _, y in self.history], dtype=float)

        x = np.log(steps + 1.0) if self.use_log_time else steps
        # simple linear regression slope on window
        x_mean = x.mean()
        y_mean = ys.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = 0.0 if denom == 0 else np.sum((x - x_mean) * (ys - y_mean)) / denom

        # relative improvement across the window
        y0, yT = ys[0], ys[-1]
        rel_improve = max(0.0, (y0 - yT) / max(abs(y0), 1e-8))

        plateau = self.started and (abs(slope) < self.slope_eps) and (rel_improve < self.rel_improve_eps)

        if plateau:
            self._flat_hits += 1
        else:
            self._flat_hits = 0

        should_stop = self._flat_hits >= self.patience

        info = {
            "slope": slope,
            "rel_improve": rel_improve,
            "flat_hits": self._flat_hits,
            "started": self.started,
            "loss": self.loss,
            "step": step
        }
        return should_stop, info

    def reset(self):
        self.init_loss = None
        self.best_loss = np.inf
        self.started = False
        self.history.clear()
        self.ema_loss = None
        self._last_check_step = -np.inf
        self._flat_hits = 0
        return self