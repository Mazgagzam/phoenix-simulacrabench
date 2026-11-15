"""World Bank-only fixed blend using immutable offline TabPFN-3.5 Fast weights.

Built with PriorLabs-TabPFN. Native categorical heads support up to 160 classes. Visible labels form inference context; query features contain GIVEN only.
An incomplete optional pass retains the whole original baseline.
"""
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
from phoenix_core import normalize
from phoenix_categorical_tree import categorical_probabilities

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / 'weights/tabpfn-v3.5-fast-20260909.safetensors'
CHECKPOINT_SHA = '14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
_MODEL = None
_MODEL_WIDTH = None
_DISABLED = None


def is_world_bank(layout):
    return (len(layout.given) == 9 and
            {'group', 'user.age', 'user.country', 'user.employmentStatus'}.issubset(layout.given))


def forbidden_download(*args, **kwargs):
    raise RuntimeError('A verified bundled checkpoint is required')


def estimator(width):
    global _MODEL, _MODEL_WIDTH, _DISABLED
    if _MODEL is not None:
        return _MODEL if _MODEL_WIDTH == width else None
    if _DISABLED is not None:
        return None
    try:
        if not CHECKPOINT.is_file() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA:
            _DISABLED = 'checkpoint_unavailable'
            return None
        sys.path.insert(0, str(HERE / 'vendor'))
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['DO_NOT_TRACK'] = '1'
        import torch
        from tabpfn import TabPFNClassifier, model_loading
        if not torch.cuda.is_available():
            _DISABLED = 'cuda_unavailable'
            return None
        model_loading.download_model = forbidden_download
        torch.set_grad_enabled(False)
        _MODEL = TabPFNClassifier(model_path=CHECKPOINT, device='cuda:0', n_estimators=2,
            categorical_features_indices=list(range(width)), random_state=1729,
            softmax_temperature=1., n_preprocessing_jobs=1, fit_mode='fit_with_cache',
            memory_saving_mode='auto', show_progress_bar=False)
        _MODEL_WIDTH = width
        return _MODEL
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        _DISABLED = 'dependency_or_device_unavailable'
        return None


def probabilities(layout, deadline):
    meta = dict(status='not_started', items=0, provider_calls=0, failures=0,
                maximum_provider_classes=0, high_cardinality_targets=0,
                checkpoint_version='v3.5-fast', native_class_limit=160)
    if time.monotonic() >= deadline - 3:
        return None, dict(meta, status='time_budget_fallback')
    model = estimator(len(layout.given))
    if model is None:
        return None, dict(meta, status=_DISABLED or 'estimator_unavailable')
    rows = np.random.default_rng(131071).permutation(layout.visible)
    x = layout.X[rows].astype(np.float32)
    q = layout.X[layout.hidden].astype(np.float32)
    x[x < 0] = np.nan
    q[q < 0] = np.nan
    meta['context_rows'] = len(rows)
    meta['estimators'] = getattr(model, 'n_estimators', None)
    durations = []

    def provider(train_given, encoded_labels, query_given):
        reserve = max(3., 1.4 * float(np.median(durations))) if durations else 3.
        if time.monotonic() + reserve >= deadline:
            raise TimeoutError('Optional inference deadline')
        before = time.monotonic()
        model.fit(train_given, encoded_labels)
        predicted = []
        for offset in range(0, len(query_given), 256):
            if time.monotonic() >= deadline:
                raise TimeoutError('Optional inference deadline')
            predicted.append(model.predict_proba(query_given[offset:offset + 256]))
        durations.append(time.monotonic() - before)
        return model.classes_, np.concatenate(predicted)

    out = {}
    try:
        for item in layout.targets:
            width = len(layout.opts[item])
            out[item], info = categorical_probabilities(x, layout.Y[item][rows], q,
                width, provider, max_classes=160, prior_strength=4., deadline=deadline)
            meta['items'] += 1
            meta['provider_calls'] += info['provider_calls']
            meta['maximum_provider_classes'] = max(meta['maximum_provider_classes'],
                                                    info['maximum_provider_classes'])
            meta['high_cardinality_targets'] += int(width > 10)
    except TimeoutError:
        return None, dict(meta, status='time_budget_fallback')
    except Exception:
        # Do not expose exceptions, labels, records, or partial optional output.
        return None, dict(meta, status='pretrained_failure', failures=1)
    return out, dict(meta, status='finished')


def improve(layout, tables, deadline, config, baseline_meta):
    info = dict(status='not_world_bank', weight=None)
    if not is_world_bank(layout):
        return tables, info
    if (not config.get('pfn_world_bank', False) or not len(layout.hidden) or
            len(layout.visible) < 720):
        return tables, dict(info, status='baseline_retained')
    tab = baseline_meta.get('tabicl', {})
    if config.get('tabicl', False) and (tab.get('items') != len(layout.targets) or
            tab.get('fits') != 3 * len(layout.targets) or tab.get('failures', 0)):
        return tables, dict(info, status='incomplete_original_baseline')
    if float(config.get('pfn_fixed_weight', 1.)) != 1.:
        raise ValueError('This frozen experiment requires the selected all-context model with weight one')
    alternate, optional = probabilities(layout, deadline)
    info['query_pfn'] = optional
    if alternate is None:
        return tables, dict(info, status=optional['status'])
    if (optional.get('status') != 'finished' or optional.get('items') != len(layout.targets) or
            optional.get('failures', 0)):
        return tables, dict(info, status='incomplete_optional_inference')
    for item in layout.targets:
        p = alternate.get(item)
        if (p is None or p.shape != (len(layout.hidden), len(layout.opts[item])) or
                not np.isfinite(p).all() or (p < 0).any() or
                not np.allclose(p.sum(axis=1), 1., atol=1e-6)):
            return tables, dict(info, status='invalid_optional_probabilities')
    result = dict(tables)
    changed_targets = changed = entries = 0
    absolute = 0.
    for item in layout.targets:
        old = tables[item][layout.hidden]
        new = normalize(alternate[item])
        difference = np.abs(new - old)
        count = int(np.count_nonzero(difference > 1e-10))
        changed_targets += int(count > 0)
        changed += count
        entries += difference.size
        absolute += float(difference.sum())
        result[item] = tables[item].copy()
        result[item][layout.hidden] = new
    info.update(status='fixed_blend_applied', weight=1., changed_targets=changed_targets,
                changed_probability_fraction=changed / max(1, entries),
                mean_absolute_probability_change=absolute / max(1, entries))
    return result, info
