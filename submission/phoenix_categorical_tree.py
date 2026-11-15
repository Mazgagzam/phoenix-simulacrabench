"""Research-only categorical decoder for a class-limited inference provider.

The provider gets complete TRAIN labels and GIVEN features, and must return
(classes, probabilities). It may use a frozen pretrained in-context model.
This module does not train neural weights and is not imported by a submission.
No performance on confidential competition respondents is established here.
"""
import math
import time

import numpy as np


def categorical_probabilities(x_train, y_train, x_query, output_width, provider,
                              *, max_classes=10, prior_strength=4., deadline=None):
    """Reconstruct P(answer | GIVEN) using direct or conditional group heads.

All schema slots are preserved, including categories absent from the context.
If a head fails, the function raises rather than returning partial predictions.
Deadlines are cooperative: the provider must also enforce its inference budget.
"""
    if (not isinstance(output_width, (int, np.integer)) or output_width < 1 or
            not isinstance(max_classes, (int, np.integer)) or max_classes < 2):
        raise ValueError('Invalid categorical limits')
    if not np.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError('Invalid prior strength')
    x = np.asarray(x_train)
    q = np.asarray(x_query)
    y = np.asarray(y_train)
    if (x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1] or
            y.ndim != 1 or len(y) != len(x)):
        raise ValueError('Invalid GIVEN/context shapes')
    if (not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all() or
            np.any(y != np.floor(y)) or np.any(y < 0) or
            np.any(y >= output_width)):
        raise ValueError('Context labels must be complete schema indices')
    y = y.astype(np.int64)
    meta = dict(provider_calls=0, maximum_provider_classes=0, depth=0,
                output_width=int(output_width), prior_strength=float(prior_strength))

    def check_time():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Categorical inference budget exhausted')

    def direct(rows, labels, outcomes):
        check_time()
        width = len(outcomes)
        position = np.searchsorted(outcomes, labels)
        counts = np.bincount(position, minlength=width).astype(float)
        prior = (counts + .5) / (len(rows) + .5 * width)
        if not len(rows):
            return np.broadcast_to(prior, (len(q), width)).copy()
        observed = np.unique(labels)
        p = np.zeros((len(q), width), dtype=float)
        if len(observed) == 1:
            p[:, np.searchsorted(outcomes, observed[0])] = 1.
        else:
            if len(observed) > max_classes:
                raise ValueError('Provider class limit exceeded')
            encoded = np.searchsorted(observed, labels)
            classes, predicted = provider(x[rows], encoded, q)
            check_time()
            classes = np.asarray(classes)
            predicted = np.asarray(predicted, dtype=float)
            expected = np.arange(len(observed))
            if (classes.ndim != 1 or len(classes) != len(observed) or
                    not np.array_equal(np.sort(classes), expected) or
                    predicted.shape != (len(q), len(observed)) or
                    not np.isfinite(predicted).all() or (predicted < 0).any() or
                    not np.allclose(predicted.sum(axis=1), 1., atol=1e-6)):
                raise ValueError('Invalid categorical provider output')
            p[:, np.searchsorted(outcomes, observed[classes.astype(int)])] = predicted
            meta['provider_calls'] += 1
            meta['maximum_provider_classes'] = max(
                meta['maximum_provider_classes'], len(observed))
        weight = len(rows) / (len(rows) + prior_strength)
        return weight * p + (1 - weight) * prior

    def node(rows, outcomes, depth):
        check_time()
        meta['depth'] = max(meta['depth'], depth)
        labels = y[rows]
        if len(np.unique(labels)) <= max_classes:
            return direct(rows, labels, outcomes)
        # Groups depend on canonical schema indices, not query labels or order.
        groups = np.array_split(outcomes, min(max_classes,
                                              math.ceil(len(outcomes) / max_classes)))
        group_of = np.empty(output_width, dtype=np.int64)
        for i, group in enumerate(groups):
            group_of[group] = i
        memberships = group_of[labels]
        group_p = direct(rows, memberships, np.arange(len(groups)))
        result = np.empty((len(q), len(outcomes)), dtype=float)
        for i, group in enumerate(groups):
            child = node(rows[memberships == i], group, depth + 1)
            result[:, np.searchsorted(outcomes, group)] = group_p[:, i, None] * child
        return result

    if not len(q):
        return np.empty((0, output_width)), meta
    result = node(np.arange(len(y)), np.arange(output_width), 0)
    if (not np.isfinite(result).all() or (result < 0).any() or
            not np.allclose(result.sum(axis=1), 1., atol=1e-6)):
        raise ValueError('Invalid reconstructed probabilities')
    return result, meta
