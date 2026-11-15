"""Categorical probability models for whole-respondent survey completion.

Only declared GIVEN fields are features. PREDICT answers enter fitting only
for complete visible respondents; identifiers and row order are not features.
All model selection uses cross-fitted probabilities and the grader's floor.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

FLOOR = 0.001
ALPHAS = np.array([1.0, 4.0, 16.0, 64.0, 256.0, 1024.0])


def normalize(p):
    p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    sums = p.sum(axis=-1, keepdims=True)
    return np.divide(p, sums, out=np.full_like(p, 1.0 / p.shape[-1]),
                     where=sums > 0)


def floored(p):
    p = normalize(p)
    return FLOOR + (1.0 - FLOOR * p.shape[-1]) * p


def losses(p, y):
    return -np.log(floored(p)[np.arange(len(y)), y])


def temperature(p, value):
    if value == 1.0:
        return normalize(p)
    z = np.log(np.maximum(p, 1e-12)) / value
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def options(schema, item):
    r = schema['items'][item]
    return list(r['values']) + ([schema.get('gated_value', 'NA_GATED')]
                              if r.get('gate') else [])


@dataclass
class Layout:
    names: list
    given: list
    targets: list
    opts: dict
    X: np.ndarray
    widths: np.ndarray
    Y: dict
    visible: np.ndarray
    hidden: np.ndarray
    frame: pd.DataFrame
    schema: dict

    @classmethod
    def build(cls, frame, schema):
        names = [k for k, r in schema['items'].items()
                 if r['class'] in ('GIVEN', 'PREDICT') and k in frame]
        given = [k for k in names if schema['items'][k]['class'] == 'GIVEN']
        targets = [k for k in names if schema['items'][k]['class'] == 'PREDICT']
        opts = {k: options(schema, k) for k in names}
        for k in names:
            if (frame[k].notna() & ~frame[k].isin(opts[k])).any():
                raise ValueError('A supplied answer is not in its declared option list')
        codes = {k: pd.Categorical(frame[k], categories=opts[k]).codes.astype(np.int32)
                 for k in names}
        for k in names:
            if np.any((codes[k] < 0) & frame[k].notna().to_numpy()):
                raise ValueError('A supplied answer is not in its declared option list')
        X = np.column_stack([codes[k] for k in given]) if given else np.empty((len(frame), 0), dtype=np.int32)
        complete = (np.all(np.column_stack([codes[k] >= 0 for k in targets]), axis=1)
                    if targets else np.ones(len(frame), dtype=bool))
        return cls(names, given, targets, opts, X,
                   np.array([len(opts[k]) for k in given], dtype=np.int32),
                   {k: codes[k] for k in targets}, np.flatnonzero(complete),
                   np.flatnonzero(~complete), frame, schema)

    def vectors(self, tables):
        """Canonical order follows shipped schema keys, then frame rows."""
        cells = np.argwhere(self.frame[self.names].isna().to_numpy())
        return [tables[self.names[col]][row].tolist() for row, col in cells]


def folds_for(layout, n_folds=3, seed=1729):
    """Seeded respondent folds, balancing declared country/wave strata."""
    n = len(layout.visible)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    n_folds = min(n_folds, n)
    keys = [k for k in ('Country', 'user.country', 's1_2_country_asylum_lbl', 'wave')
            if k in layout.given]
    groups = {}
    for local, row in enumerate(layout.visible):
        key = tuple(layout.X[row, layout.given.index(k)] for k in keys)
        groups.setdefault(key, []).append(local)
    rng = np.random.default_rng(seed)
    f = np.empty(n, dtype=np.int32)
    offset = 0
    for key in sorted(groups):
        rows = rng.permutation(groups[key])
        f[rows] = (np.arange(len(rows)) + offset) % n_folds
        offset = (offset + len(rows)) % n_folds
    return f


def _counts(x, y, levels, k):
    good = (x >= 0) & (y >= 0)
    return np.bincount(x[good] * k + y[good], minlength=levels * k).reshape(levels, k).astype(float)


def information(counts):
    n = counts.sum()
    if n < 2:
        return 0.0
    r, c = counts.sum(1), counts.sum(0)
    expected = r[:, None] * c[None, :] / n
    good = counts > 0
    mi = np.sum(counts[good] * np.log(counts[good] / expected[good])) / n
    df = max(0, (np.count_nonzero(r) - 1) * (np.count_nonzero(c) - 1))
    return max(0.0, float(mi - df / (2 * n)))


def eb_alpha(counts, prior):
    """Dirichlet-multinomial empirical-Bayes shrinkage on training counts."""
    used = counts.sum(1) > 0
    c = counts[used]
    if not len(c):
        return 64.0
    p = np.maximum(np.broadcast_to(prior, counts.shape)[used], 1e-7)
    p = p / p.sum(1, keepdims=True)
    n = c.sum(1)
    values = []
    for a in ALPHAS:
        ap = a * p
        ll = np.sum(gammaln(a) - gammaln(a + n)
                    + np.sum(gammaln(ap + c) - gammaln(ap), axis=1))
        values.append(ll)
    return float(ALPHAS[np.argmax(values)])


def _table(counts, prior, alpha=None):
    p = np.broadcast_to(prior, counts.shape)
    a = eb_alpha(counts, p) if alpha is None else alpha
    return normalize(counts + a * p)


def _lookup(table, codes, default):
    result = np.tile(default, (len(codes), 1))
    good = (codes >= 0) & (codes < len(table))
    result[good] = table[codes[good]]
    return result


class CategoricalModel:
    """Complementary, inexpensive conditional probability estimates."""
    components = ('marginal', 'single', 'backoff', 'evidence', 'context',
                  'weighted', 'hierarchy')

    def __init__(self, X, y, widths, k, max_cells=4096, contexts=()):
        counts = np.bincount(y[y >= 0], minlength=k).astype(float)
        self.prior = (counts + 0.5) / (counts.sum() + 0.5 * k)
        self.widths = widths
        self.singles = []
        self.mi = []
        for j, w in enumerate(widths):
            c = _counts(X[:, j], y, int(w), k)
            self.singles.append(_table(c, self.prior))
            self.mi.append(information(c))
        self.rank = np.argsort(-np.asarray(self.mi), kind='stable')[:min(6, len(widths))]
        self.chain = []
        self.pairs = []
        self.contexts = []
        if not len(self.rank):
            return
        a = int(self.rank[0])
        for b in self.rank[1:4]:
            b = int(b)
            levels = int(widths[a] * widths[b])
            if levels > max_cells:
                continue
            code = X[:, a] * widths[b] + X[:, b]
            code[(X[:, a] < 0) | (X[:, b] < 0)] = -1
            c = _counts(code, y, levels, k)
            # Back off a sparse pair to the two partially pooled main effects.
            prior = normalize((np.repeat(self.singles[a], widths[b], axis=0)
                               + np.tile(self.singles[b], (widths[a], 1))) / 2)
            tab = _table(c, prior)
            self.pairs.append((a, b, tab, information(c)))
        self.pairs.sort(key=lambda z: -z[3])
        # Hierarchical counts preserve the most reliable main effect while
        # pooling successively finer intersections back to their parent group.
        previous = self.prior[None, :]
        code = np.zeros(len(X), dtype=int)
        good = np.ones(len(X), dtype=bool)
        levels = 1
        for col in self.rank[:4]:
            col = int(col)
            width = int(widths[col])
            levels *= width
            if levels > max_cells:
                break
            code = code * width + np.maximum(X[:, col], 0)
            good &= X[:, col] >= 0
            encoded = np.where(good, code, -1)
            c = _counts(encoded, y, levels, k)
            previous = _table(c, np.repeat(previous, width, axis=0))
            self.chain.append((col, previous))
        candidates = list(contexts)
        if len(self.rank) >= 3:
            candidates.append(tuple(int(v) for v in self.rank[:3]))
        for cols in dict.fromkeys(candidates):
            if any(c >= len(widths) for c in cols):
                continue
            levels = int(np.prod(widths[list(cols)]))
            if levels > max_cells:
                continue
            code = np.zeros(len(X), dtype=int)
            good = np.ones(len(X), dtype=bool)
            for col in cols:
                code = code * widths[col] + np.maximum(X[:, col], 0)
                good &= X[:, col] >= 0
            code[~good] = -1
            c = _counts(code, y, levels, k)
            self.contexts.append((cols, _table(c, self.prior)))

    def predict(self, X):
        prior = np.tile(self.prior, (len(X), 1))
        if not len(self.rank):
            return {k: prior.copy() for k in self.components}
        probs = {int(j): _lookup(self.singles[int(j)], X[:, int(j)], self.prior)
                 for j in self.rank}
        single = probs[int(self.rank[0])]
        backoff = single
        if self.pairs:
            a, b, tab, _ = self.pairs[0]
            code = X[:, a] * self.widths[b] + X[:, b]
            good = (X[:, a] >= 0) & (X[:, b] >= 0)
            backoff = single.copy()
            backoff[good] = tab[code[good]]
        # Tempering/stacking corrects duplicated information in correlated GIVENs.
        evidence = np.log(np.maximum(prior, 1e-12))
        for j in self.rank[:4]:
            if self.mi[int(j)] > 0:
                evidence += (np.log(np.maximum(probs[int(j)], 1e-12))
                             - np.log(np.maximum(prior, 1e-12)))
        evidence = np.exp(evidence - logsumexp(evidence, axis=1, keepdims=True))
        weighted = np.log(np.maximum(prior, 1e-12))
        reliability = np.sqrt(np.maximum(np.asarray(self.mi)[self.rank], 0.))
        if reliability.sum() > 0:
            reliability *= 2. / reliability.sum()
            for j, weight in zip(self.rank, reliability):
                weighted += weight * (np.log(np.maximum(probs[int(j)], 1e-12))
                                      - np.log(np.maximum(prior, 1e-12)))
        weighted = np.exp(weighted - logsumexp(weighted, axis=1, keepdims=True))
        hierarchy = prior.copy()
        code = np.zeros(len(X), dtype=int)
        good = np.ones(len(X), dtype=bool)
        for col, table in self.chain:
            code = code * self.widths[col] + np.maximum(X[:, col], 0)
            good &= X[:, col] >= 0
            hierarchy[good] = table[code[good]]
        context = []
        for cols, table in self.contexts:
            code = np.zeros(len(X), dtype=int)
            good = np.ones(len(X), dtype=bool)
            for col in cols:
                code = code * self.widths[col] + np.maximum(X[:, col], 0)
                good &= X[:, col] >= 0
            code[~good] = -1
            context.append(_lookup(table, code, self.prior))
        context = sum(context) / len(context) if context else backoff
        return dict(marginal=prior, single=single, backoff=backoff,
                    evidence=evidence, context=context, weighted=weighted,
                    hierarchy=hierarchy)


def mixture_weights(probabilities, y, center=None, strength=0.005):
    """Regularized simplex fit on cross-fitted true-category probabilities."""
    m = len(probabilities)
    if m == 1 or not len(y):
        return np.ones(m) / m
    center = np.ones(m) / m if center is None else normalize(np.asarray(center))
    truth = np.column_stack([floored(p)[np.arange(len(y)), y] for p in probabilities])
    def objective(w):
        q = np.maximum(truth @ w, 1e-12)
        loss = -np.log(q).mean() + strength * np.sum((w - center) ** 2)
        grad = -(truth / q[:, None]).mean(0) + 2 * strength * (w - center)
        return loss, grad
    result = minimize(objective, center, jac=True, method='SLSQP',
                      bounds=[(0.0, 1.0)] * m,
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                   'jac': lambda w: np.ones(m)},
                      options={'maxiter': 60, 'ftol': 1e-8})
    return normalize(result.x) if result.success else center


def calibrate(oof, y, query, prior):
    """Temperature and marginal shrinkage, selected without query targets."""
    if len(y) < 100:
        return normalize(query), (1.0, 0.0)
    # A separate calibration confirmation split, with a conservative fallback.
    rng = np.random.default_rng(991)
    perm = rng.permutation(len(y))
    tune, confirm = perm[::2], perm[1::2]
    base = losses(oof, y)
    best = (float(base[tune].mean()), 1.0, 0.0)
    for t in (0.7, 0.85, 1.0, 1.15, 1.4, 1.8):
        p = temperature(oof, t)
        for shrink in (0.0, 0.02, 0.05, 0.1):
            trial = (1 - shrink) * p + shrink * prior
            score = float(losses(trial[tune], y[tune]).mean())
            if score < best[0] - 1e-5:
                best = (score, t, shrink)
    _, t, shrink = best
    trial = (1 - shrink) * temperature(oof, t) + shrink * prior
    gains = base[confirm] - losses(trial[confirm], y[confirm])
    se = gains.std(ddof=1) / np.sqrt(len(gains)) if len(gains) > 1 else np.inf
    if gains.mean() <= max(0.0002, se):
        return normalize(query), (1.0, 0.0)
    return normalize((1 - shrink) * temperature(query, t) + shrink * prior), (t, shrink)


def statistical_tables(layout, deadline, config):
    vis, hid = layout.visible, layout.hidden
    Xv, Xh = layout.X[vis], layout.X[hid]
    folds = folds_for(layout, config.get('folds', 3), config.get('seed', 1729))
    names = CategoricalModel.components
    oof, query = {}, {}
    models = {}
    context_names = [('Country', 'Q04', 'Q05'),
                     ('s1_2_country_asylum_lbl', 'wave', 's2_1c_education_level_lbl'),
                     ('user.country', 'user.employmentStatus', 'user.educationLevel')]
    contexts = [tuple(layout.given.index(k) for k in names)
                for names in context_names if all(k in layout.given for k in names)]
    for item in layout.targets:
        y, k = layout.Y[item][vis], len(layout.opts[item])
        expired = time.monotonic() >= deadline
        full = CategoricalModel(Xv[:, :0] if expired else Xv, y,
                                layout.widths[:0] if expired else layout.widths, k,
                                contexts=contexts if not expired else ())
        models[item] = full
        query[item] = full.predict(Xh[:, :0] if expired else Xh)
        oof[item] = {name: np.tile(full.prior, (len(vis), 1)) for name in names}
        if len(vis) < 12 or time.monotonic() >= deadline:
            # No validation evidence: keep all components at the crowd fallback.
            query[item] = {name: query[item]['marginal'].copy() for name in names}
            continue
        for fold in np.unique(folds):
            train, valid = folds != fold, folds == fold
            model = CategoricalModel(Xv[train], y[train], layout.widths, k, contexts=contexts)
            pred = model.predict(Xv[valid])
            for name in names:
                oof[item][name][valid] = pred[name]
    return oof, query, models, folds


def gate_tables(layout, models, oof, folds, deadline):
    """Marginalize hidden parent categories; learn observed routing deviations."""
    gv = layout.schema.get('gated_value', 'NA_GATED')
    vis, hid = layout.visible, layout.hidden
    result_o, result_h = {}, {}
    for child in layout.targets:
        gate = layout.schema['items'][child].get('gate')
        if not gate or time.monotonic() >= deadline:
            continue
        parent = gate['parent']
        if parent not in layout.targets or parent == child:
            continue
        yp, yc = layout.Y[parent][vis], layout.Y[child][vis]
        kp, kc = len(layout.opts[parent]), len(layout.opts[child])
        if kp > 32 or len(vis) < 30:
            continue
        oo = np.zeros((len(vis), kc))
        for fold in np.unique(folds):
            tr, va = folds != fold, folds == fold
            prior = np.bincount(yc[tr], minlength=kc).astype(float) + 0.5
            prior /= prior.sum()
            c = _counts(yp[tr], yc[tr], kp, kc)
            table = _table(c, prior)
            # Parent probabilities are cross-fitted; never use validation truth.
            p_parent = oof[parent]['backoff'][va]
            oo[va] = p_parent @ table
        c = _counts(yp, yc, kp, kc)
        table = _table(c, models[child].prior)
        parent_h = models[parent].predict(layout.X[hid])['backoff']
        hh = parent_h @ table
        # Country/wave can change coding. Conditioning the child directly remains
        # a separate component; stacking can reject this simpler gate model.
        result_o[child], result_h[child] = normalize(oo), normalize(hh)
    return result_o, result_h


def stack_tables(layout, oof, query, models, extras_o=None, extras_h=None):
    """Shared blend with item-specific shrinkage and final probability calibration."""
    extras_o, extras_h = extras_o or {}, extras_h or {}
    vis, hid = layout.visible, layout.hidden
    names = list(CategoricalModel.components)
    m = len(names)
    # Pooled blend: instrument-wide weights prevent tiny-item selection noise.
    pooled = [[] for _ in names]
    ys = []
    for item in layout.targets:
        y = layout.Y[item][vis]
        for j, name in enumerate(names):
            pooled[j].append(floored(oof[item][name])[np.arange(len(y)), y])
        ys.append(y)
    if not ys or not len(vis):
        global_w = np.r_[1., np.zeros(m - 1)]
    else:
        truths = np.column_stack([np.concatenate(p) for p in pooled])
        # K differs by item, so optimize already-floored true probabilities.
        def obj(w):
            q = np.maximum(truths @ w, 1e-12)
            return -np.log(q).mean(), -(truths / q[:, None]).mean(0)
        r = minimize(obj, np.ones(m) / m, jac=True, method='SLSQP',
                     bounds=[(0, 1)] * m,
                     constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                     options={'maxiter': 60, 'ftol': 1e-8})
        global_w = normalize(r.x) if r.success else np.ones(m) / m
    tables, details = {}, {}
    for item in layout.targets:
        y, prior = layout.Y[item][vis], models[item].prior
        ps = [oof[item][n] for n in names]
        ph = [query[item][n] for n in names]
        w = mixture_weights(ps, y, global_w, strength=max(0.005, 50 / max(1, len(y))))
        po, pq = sum(a * p for a, p in zip(w, ps)), sum(a * p for a, p in zip(w, ph))
        extra_weights = {}
        for label, (eo, eh) in extras_o.get(item, {}).items():
            # Foundation models may finish only some folds. Select on exactly
            # those cross-fitted rows; never treat unavailable predictions as real.
            available = np.isfinite(eo).all(1)
            if available.sum() < 40:
                continue
            ww = mixture_weights([po[available], eo[available]], y[available],
                                 center=np.array([0.8, 0.2]), strength=0.02)
            a = float(ww[1])
            po[available] = (1 - a) * po[available] + a * eo[available]
            pq = (1 - a) * pq + a * eh
            extra_weights[label] = a
        pq, cal = calibrate(po, y, pq, prior) if len(y) else (pq, (1., 0.))
        table = np.tile(prior, (len(layout.frame), 1))
        table[hid] = pq
        tables[item] = table
        details[item] = dict(weights=w.tolist(), extras=extra_weights,
                             temperature=cal[0], shrink=cal[1])
    for item in layout.given:
        c = layout.frame[item].value_counts()
        p = np.array([c.get(v, 0) + 0.5 for v in layout.opts[item]])
        tables[item] = np.tile(normalize(p), (len(layout.frame), 1))
    return tables, dict(global_weights=global_w.tolist(), items=details)
