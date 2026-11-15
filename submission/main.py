"""Phoenix 2026: calibrated categorical models with optional TabICLv2.

The predictor never opens network connections or writes data. Complete visible
answers supply empirical estimates following the official crowd baseline API.
Pretrained artifacts, if bundled, are loaded from this submission only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

_START = time.monotonic()
_SEEN = set()
_HERE = Path(__file__).resolve().parent
with (_HERE / 'config.json').open(encoding='utf-8') as _f:
    CONFIG = json.load(_f)

from phoenix_core import (Layout, statistical_tables, gate_tables, stack_tables)
from phoenix_neural import neural_tables
from phoenix_tabicl import foundation_tables
from phoenix_world_bank_pfn import probabilities, is_world_bank
from phoenix_adaptive_pfn import select, blend

LAST_DIAGNOSTICS = {}


def _deadline(layout):
    split = layout.schema.get('split', {})
    final = len(layout.visible) >= split.get('n_train', 10**9) + split.get('n_dev', 0) / 2
    total = (3600 if final else 900) * CONFIG.get('budget_fraction', 0.8)
    total = float(os.environ.get('PHOENIX_TOTAL_SECONDS', total))
    j = len(layout.targets)
    _SEEN.add(j)
    # Reserve for the other instruments regardless of the grader's call order.
    costs = {12: 2 * 12 * 14885, 76: 4 * 76 * 5195, 213: 213 * 8279}
    cost = costs.get(j, max(1, j * len(layout.visible)))
    others = sum(c for k, c in costs.items() if k not in _SEEN) if j in costs else 0
    left = max(0, total - (time.monotonic() - _START))
    return time.monotonic() + left * cost / (cost + others)


def _baseline_tables(layout, deadline):
    oof, query, models, folds = statistical_tables(layout, deadline, CONFIG)
    extras_o, extras_h = {}, {}
    if CONFIG.get('gates', True):
        go, gh = gate_tables(layout, models, oof, folds, deadline)
        for k in go:
            extras_o.setdefault(k, {})['gate'] = (go[k], gh[k])
    tab_meta = {}
    neural_meta = {}
    # Scratch neural fitting is a public-data research comparator. Deployment
    # uses pretrained inference until the README's offline-training wording is
    # clarified; public benchmark tools explicitly enable this research path.
    if (CONFIG.get('neural', False) and os.environ.get('PHOENIX_PUBLIC_RESEARCH') == '1'
            and time.monotonic() < deadline):
        # Keep most of the optional-model budget for pretrained inference.
        neural_deadline = time.monotonic() + max(0, deadline - time.monotonic()) * .25
        no, nh, neural_meta = neural_tables(layout, folds, neural_deadline, CONFIG)
        for k in no:
            extras_o.setdefault(k, {})['neural'] = (no[k], nh[k])
    if CONFIG.get('tabicl', False) and time.monotonic() < deadline:
        to, th, tab_meta = foundation_tables(layout, folds, deadline, CONFIG)
        for k in to:
            extras_o.setdefault(k, {})['tabicl'] = (to[k], th[k])
    tables, meta = stack_tables(layout, oof, query, models, extras_o, extras_h)
    return tables, dict(meta, tabicl=tab_meta, neural=neural_meta)


def predict(frame, schema):
    global LAST_DIAGNOSTICS
    layout = Layout.build(frame, schema)
    deadline = _deadline(layout)
    eligible = (is_world_bank(layout) and CONFIG.get('pfn_world_bank', False)
                and CONFIG.get('pfn_honest_blend', False) and len(layout.hidden) > 0)
    selection, weight = dict(status='not_world_bank', weight=None), None
    if eligible:
        weight, selection = select(layout, deadline, CONFIG, _baseline_tables, probabilities)
    baseline_deadline = (time.monotonic() + .65 * max(0., deadline-time.monotonic())
                         if weight is not None else deadline)
    tables, meta = _baseline_tables(layout, baseline_deadline)
    if eligible:
        tables, pfn = blend(layout, tables, deadline, CONFIG, meta, weight, probabilities)
        pfn['selection'] = selection
    else:
        pfn = dict(status='not_world_bank', weight=None)
    LAST_DIAGNOSTICS = dict(meta, pfn=pfn, elapsed=time.monotonic() - _START)
    return layout.vectors(tables)
