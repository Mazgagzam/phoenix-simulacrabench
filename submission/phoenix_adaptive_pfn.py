"""Select a statistical blend on an honest TRAIN holdout; freeze neural weights."""
import time

import numpy as np

from phoenix_core import Layout, normalize
from phoenix_honest_blend import prepare_holdout, choose_weight


def complete_baseline(layout, meta, config):
    if not config.get('tabicl', False):
        return True
    tab = meta.get('tabicl', {})
    return (tab.get('items') == len(layout.targets) and tab.get('fits') == 3*len(layout.targets)
            and tab.get('failures', 0) == 0)


def select(layout, deadline, config, baseline_fn, probability_fn):
    info = dict(status='not_started', weight=None, l2=config.get('pfn_blend_l2', .001))
    try:
        holdout = prepare_holdout(layout.frame, layout.schema, layout.visible,
            seed=config.get('pfn_holdout_seed', 1729), fraction=.2)
        if holdout is None:
            return None, dict(info, status='insufficient_visible_context')
        inner = Layout.build(holdout.masked, layout.schema)
        assert len(inner.visible) == len(holdout.context_positions)
        assert len(inner.hidden) == len(holdout.validation_positions)
        info.update(context_rows=len(inner.visible), validation_rows=len(inner.hidden))
        now = time.monotonic()
        if deadline-now < 12:
            return None, dict(info, status='validation_budget_fallback')
        base, base_meta = baseline_fn(inner, now + .32*(deadline-now))
        info['baseline_tabicl'] = base_meta.get('tabicl', {})
        if not complete_baseline(inner, base_meta, config):
            return None, dict(info, status='incomplete_validation_baseline')
        now = time.monotonic()
        alternate, meta = probability_fn(inner, now + .40*max(0., deadline-now))
        info['validation_pfn'] = meta
        if (alternate is None or meta.get('status') != 'finished'
                or meta.get('items') != len(inner.targets) or meta.get('failures', 0)):
            return None, dict(info, status='incomplete_validation_pfn')
        truth = {item:layout.Y[item][holdout.validation_positions].copy() for item in inner.targets}
        baseline = {item:base[item][inner.hidden] for item in inner.targets}
        selected = choose_weight(baseline, alternate, truth, l2=info['l2'])
        # Validation curves and respondent labels are not retained in diagnostics.
        return selected['weight'], dict(info, status='validated', weight=selected['weight'])
    except Exception:
        return None, dict(info, status='validation_failure')


def blend(layout, tables, deadline, config, baseline_meta, weight, probability_fn):
    info = dict(status='baseline_retained', weight=weight)
    if weight is None:
        return tables, info
    if not complete_baseline(layout, baseline_meta, config):
        return tables, dict(info, status='incomplete_original_baseline')
    try:
        alternate, meta = probability_fn(layout, deadline)
        info['query_pfn'] = meta
        if (alternate is None or meta.get('status') != 'finished'
                or meta.get('items') != len(layout.targets) or meta.get('failures', 0)):
            return tables, dict(info, status='incomplete_optional_inference')
        for item in layout.targets:
            q = alternate.get(item)
            if (q is None or q.shape != (len(layout.hidden),len(layout.opts[item]))
                    or not np.isfinite(q).all() or (q < 0).any()
                    or not np.allclose(q.sum(axis=1), 1., atol=1e-6)):
                return tables, dict(info, status='invalid_optional_probabilities')
        # Execute the full optional pass even when the weight is zero. This makes
        # the resource audit cover the worst-case inference path without overrides.
        if weight == 0.:
            return tables, dict(info, status='validation_prefers_baseline', changed_targets=0)
        result = dict(tables)
        changed = entries = 0
        absolute = 0.
        for item in layout.targets:
            old = tables[item][layout.hidden]
            q = normalize((1-weight)*old + weight*alternate[item])
            difference = np.abs(q-old)
            changed += int(np.any(difference > 1e-10))
            entries += difference.size
            absolute += float(difference.sum())
            result[item] = tables[item].copy()
            result[item][layout.hidden] = q
        return result, dict(info, status='adaptive_blend_applied', changed_targets=changed,
                            mean_absolute_probability_change=absolute/max(1,entries))
    except Exception:
        return tables, dict(info, status='optional_inference_failure')
