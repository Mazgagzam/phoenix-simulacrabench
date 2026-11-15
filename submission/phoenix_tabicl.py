"""Bounded cross-fitting of a bundled, immutable TabICLv2 checkpoint.

No model download is attempted by this module. It uses the upstream BSD-3
implementation, independently of any other competitor's submission source.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from phoenix_core import normalize

HERE = Path(__file__).resolve().parent
CKPT = HERE / 'weights' / 'tabicl-classifier-v2-20260212.ckpt'
_ESTIMATOR = None
_DISABLED = None


def _estimator(config):
    global _ESTIMATOR, _DISABLED
    if _ESTIMATOR is not None:
        return _ESTIMATOR
    if _DISABLED is not None:
        return None
    if not CKPT.is_file():
        _DISABLED = 'checkpoint_absent'
        return None
    try:
        # Some validators temporarily expose the submission directory only
        # during main.py import. Resolve the bundle's vendor path here too.
        sys.path.insert(0, str(HERE / 'vendor'))
        import torch
        from tabicl import TabICLClassifier
        if not torch.cuda.is_available():
            _DISABLED = 'cuda_unavailable'
            return None
        class BundledClassifier(TabICLClassifier):
            def _load_model(self):
                if hasattr(self, 'model_'):
                    self.model_.clear_cache()
                    return
                super()._load_model()
        _ESTIMATOR = BundledClassifier(
            device='cuda:0', n_estimators=config.get('tabicl_estimators', 1),
            random_state=1729, model_path=str(CKPT), allow_auto_download=False,
            use_fa3=False, n_jobs=1)
        return _ESTIMATOR
    except (ImportError, OSError, RuntimeError, TypeError):
        _DISABLED = 'dependency_or_device_unavailable'
        return None


def foundation_tables(layout, folds, deadline, config):
    estimator = _estimator(config)
    if estimator is None:
        return {}, {}, {'status': _DISABLED}
    vis, hid = layout.visible, layout.hidden
    if len(vis) < 60 or not len(layout.given):
        return {}, {}, {'status': 'insufficient_visible_rows'}
    X = pd.DataFrame({k: pd.Categorical(layout.X[:, j])
                      for j, k in enumerate(layout.given)})
    out, hidden = {}, {}
    failures, fits = 0, 0
    durations = []
    # Alternate gated/ordinary items so any timeout does not systematically
    # exclude one questionnaire section. Entropy supplies useful work per item.
    def priority(item):
        y = layout.Y[item][vis]
        p = np.bincount(y, minlength=len(layout.opts[item])).astype(float) + .5
        p /= p.sum()
        return -float(-np.sum(p * np.log(p)))
    order = sorted(layout.targets, key=priority)
    for item in order:
        if time.monotonic() >= deadline - 5:
            break
        y, k = layout.Y[item][vis], len(layout.opts[item])
        oo = np.full((len(vis), k), np.nan)
        hh = np.zeros((len(hid), k))
        completed = 0
        for f in np.unique(folds)[:config.get('tabicl_max_folds', 3)]:
            reserve = max(3.0, 1.4 * np.median(durations)) if durations else 5.0
            if time.monotonic() + reserve >= deadline:
                break
            tr, va = folds != f, folds == f
            labels = np.unique(y[tr])
            if not len(labels):
                continue
            rows = np.concatenate((vis[va], hid))
            start = time.monotonic()
            try:
                if len(labels) == 1:
                    prob = np.zeros((len(rows), k))
                    prob[:, labels[0]] = 1.0
                else:
                    # A single estimator is refitted; upstream manages its
                    # pretrained network and clears data-specific state on fit.
                    estimator.fit(X.iloc[vis[tr]], y[tr])
                    pred = np.asarray(estimator.predict_proba(X.iloc[rows]), dtype=float)
                    prob = np.full((len(rows), k), 1e-9)
                    prob[:, np.asarray(estimator.classes_, dtype=int)] = pred
                prob = normalize(prob)
                if not np.isfinite(prob).all():
                    raise ValueError('Nonfinite foundation prediction')
                n = va.sum()
                oo[va], hh = prob[:n], hh + prob[n:]
                completed += 1
                fits += 1
            except Exception:
                # Optional model failure is local to this item/fold. Statistical
                # predictions remain valid and no hidden records are logged.
                failures += 1
                if failures >= 3:
                    break
            durations.append(time.monotonic() - start)
        if completed:
            out[item], hidden[item] = oo, normalize(hh / completed)
        if failures >= 3:
            break
    return out, hidden, {'status': 'finished', 'items': len(out),
                         'fits': fits, 'failures': failures}
