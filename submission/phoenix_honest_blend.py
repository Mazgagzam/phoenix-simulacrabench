"""In-memory TRAIN holdout and L2-regularized probability blending.

This helper never writes respondent data or labels. Predictors receive
only the masked holdout frame; the original competition query rows are omitted.
It fits a statistical mixture coefficient without modifying neural weights.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Holdout:
    masked: pd.DataFrame
    truth: dict
    context_positions: np.ndarray
    validation_positions: np.ndarray


def prepare_holdout(frame, schema, visible, seed=1729, fraction=.2,
                    min_context=720, min_validation=200):
    visible = np.asarray(visible, dtype=int)
    if not 0 < fraction < 1:
        raise ValueError('Invalid holdout fraction')
    if len(np.unique(visible)) != len(visible) or (visible < 0).any() or (visible >= len(frame)).any():
        raise ValueError('Invalid visible-row positions')
    if len(visible) < min_context + min_validation:
        return None
    targets = [name for name, item in schema['items'].items() if item['class'] == 'PREDICT']
    if not targets or not set(targets).issubset(frame.columns):
        raise ValueError('Invalid target columns')
    if not frame.iloc[visible][targets].notna().all().all():
        raise ValueError('Visible TRAIN answers must be complete')
    count = min(max(min_validation, int(fraction * len(visible))), len(visible)-min_context)
    shuffled = np.random.default_rng(seed).permutation(visible)
    context, validation = shuffled[:-count], shuffled[-count:]
    order = np.concatenate([context, validation])
    masked = frame.iloc[order].copy().reset_index(drop=True)
    truth = {item: masked[item].iloc[len(context):].to_numpy(copy=True) for item in targets}
    for item in targets:
        masked[item] = masked[item].astype(object)
        masked.loc[len(context):, item] = None
    assert masked.loc[len(context):, targets].isna().all().all()
    assert set(context).isdisjoint(validation)
    return Holdout(masked, truth, context, validation)


def choose_weight(baseline, alternate, truth, *, floor=.001,
                  weights=(0., .25, .5, .75, 1.), l2=.001):
    """Choose one instrument-wide weight from validation answers only.

    `truth` holds schema-encoded validation labels, not competition query labels.
    L2 shrinkage toward baseline reduces selection noise from a finite holdout.
    The scoring floor is the organizer's affine mixture, applied exactly once.
    """
    if l2 < 0 or floor < 0 or not weights or any(w < 0 or w > 1 for w in weights):
        raise ValueError('Invalid mixture controls')
    if not truth or set(baseline) != set(truth) or set(alternate) != set(truth):
        raise ValueError('Validation tables do not match')
    prepared = {}
    rows = None
    for item, labels in truth.items():
        y = np.asarray(labels)
        a, b = np.asarray(baseline[item], dtype=float), np.asarray(alternate[item], dtype=float)
        if a.ndim != 2 or a.shape != b.shape or y.ndim != 1 or len(y) != len(a) or not len(y):
            raise ValueError('Invalid validation shapes')
        if rows is not None and rows != len(y):
            raise ValueError('Validation respondents differ between items')
        rows = len(y)
        if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= a.shape[1]).any():
            raise ValueError('Invalid validation labels')
        if a.shape[1]*floor >= 1:
            raise ValueError('Invalid scoring floor')
        for table in (a,b):
            if not np.isfinite(table).all() or (table < 0).any() or (table.sum(axis=1) <= 0).any():
                raise ValueError('Invalid probability table')
        prepared[item] = (a/a.sum(axis=1,keepdims=True), b/b.sum(axis=1,keepdims=True), y)
    losses = {}
    for weight in weights:
        by_item = []
        for a, b, y in prepared.values():
            q = (1-weight)*a + weight*b
            p = floor + (1-q.shape[1]*floor)*q
            by_item.append(-np.log(p[np.arange(len(y)),y]))
        losses[float(weight)] = np.mean(by_item,axis=0)
    chosen = min(losses, key=lambda w:(float(losses[w].mean())+l2*w*w,w))
    return dict(weight=chosen,validation_log_loss=float(losses[chosen].mean()),
                baseline_log_loss=float(losses[0.].mean()) if 0. in losses else None,
                respondents=rows,l2=l2,
                curves={str(w):float(loss.mean()) for w,loss in losses.items()})
