"""A compact shared multi-item neural conditional model; optional CUDA path."""
from __future__ import annotations

import time
import numpy as np
from phoenix_core import normalize


def neural_tables(layout, folds, deadline, config):
    try:
        import torch
        from torch import nn
        if not torch.cuda.is_available():
            return {}, {}, {'status': 'cuda_unavailable'}
    except ImportError:
        return {}, {}, {'status': 'torch_unavailable'}
    vis, hid = layout.visible, layout.hidden
    if len(vis) < 60 or not layout.targets or not layout.given:
        return {}, {}, {'status': 'insufficient_visible_rows'}
    torch.manual_seed(1729)
    device = torch.device('cuda:0')
    j = len(layout.targets)
    widths = [len(layout.opts[k]) for k in layout.targets]
    km = max(widths)
    offsets = np.r_[0, np.cumsum(layout.widths + 1)]
    X = np.zeros((len(layout.frame), int(offsets[-1])), dtype=np.float32)
    for col, w in enumerate(layout.widths):
        codes = np.where(layout.X[:, col] >= 0, layout.X[:, col], w)
        X[np.arange(len(X)), offsets[col] + codes] = 1.
    xt = torch.as_tensor(X, device=device)
    Y = np.column_stack([layout.Y[k][vis] for k in layout.targets])
    yt = torch.as_tensor(Y, dtype=torch.long, device=device)
    invalid = torch.as_tensor(np.arange(km)[None, :] >= np.array(widths)[:, None], device=device)
    oo = {k: np.full((len(vis), w), np.nan) for k, w in zip(layout.targets, widths)}
    hh = {k: np.zeros((len(hid), w)) for k, w in zip(layout.targets, widths)}
    done = 0
    for f in np.unique(folds)[:config.get('neural_max_folds', 3)]:
        if time.monotonic() >= deadline - 3:
            break
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(),
                            nn.Dropout(.1), nn.Linear(128, j * km)).to(device)
        bias = np.full((j, km), -15., dtype=np.float32)
        for z, (k, w) in enumerate(zip(layout.targets, widths)):
            p = np.bincount(Y[tr, z], minlength=w).astype(float) + .5
            bias[z, :w] = np.log(p / p.sum())
        with torch.no_grad():
            net[-1].bias.copy_(torch.as_tensor(bias.ravel(), device=device))
            net[-1].weight.mul_(.02)
        optim = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.03)
        rng = np.random.default_rng(100 + int(f))
        epochs = 0
        for epoch in range(config.get('neural_epochs', 40)):
            if time.monotonic() >= deadline - 3:
                break
            net.train()
            order = rng.permutation(tr)
            for begin in range(0, len(order), 512):
                ids = order[begin:begin + 512]
                logits = net(xt[vis[ids]]).reshape(-1, j, km).masked_fill(invalid, -1e4)
                loss = nn.functional.cross_entropy(logits.transpose(1, 2), yt[ids])
                if not torch.isfinite(loss):
                    return {}, {}, {'status': 'nonfinite_loss'}
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.)
                optim.step()
            epochs += 1
        if epochs < 3:
            break
        net.eval()
        rows = np.concatenate((vis[va], hid))
        chunks = []
        with torch.inference_mode():
            for begin in range(0, len(rows), 512):
                logits = net(xt[rows[begin:begin + 512]]).reshape(-1, j, km).masked_fill(invalid, -1e4)
                chunks.append(logits.softmax(-1).cpu().numpy())
        p = np.concatenate(chunks)
        for z, (k, w) in enumerate(zip(layout.targets, widths)):
            oo[k][va] = normalize(p[:len(va), z, :w])
            hh[k] += p[len(va):, z, :w]
        done += 1
        del net, optim
    if not done:
        return {}, {}, {'status': 'time_budget'}
    return oo, {k: normalize(v / done) for k, v in hh.items()}, {'status': 'finished', 'folds': done}
