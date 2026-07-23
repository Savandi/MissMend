"""Persistent injection store on D: drive.

For each (dataset_slug, rate, seed) the store keeps three files on disk:

    /mnt/d/eval_injection_runs/<dataset_slug>/rate_<rate>/seed_<seed>/
        events.pkl              # pickled list of post-injection events
        ground_truth.json       # {str(idx): original_label} for every injected event
        meta.json               # {n_total, n_labelled, n_injected, sensors, activities, source_path}

Only events with non-empty concept_name are eligible for injection. The original
label is stored in ground_truth.json before being replaced by None on the
events stored in events.pkl. The pipeline therefore reads the post-injection
events directly, and the evaluator reads ground_truth.json to score recoveries.
"""
from __future__ import annotations
import json
import pickle
import random
from pathlib import Path
from typing import Tuple, Dict, Set, List

PERSIST_ROOT = Path('/mnt/d/eval_injection_runs')

def _run_dir(dataset_slug: str, rate: float, seed: int) -> Path:
    return PERSIST_ROOT / dataset_slug / f'rate_{rate:.2f}' / f'seed_{seed}'

def run_exists(dataset_slug: str, rate: float, seed: int) -> bool:
    import time
    d = _run_dir(dataset_slug, rate, seed)
    for attempt in range(2):
        try:
            return (d / 'events.pkl').exists() and (d / 'ground_truth.json').exists()
        except OSError as e:
            if e.errno == 19 and attempt == 0:
                print(f'[persisted_injection] D: dropped on run_exists({dataset_slug}, {rate}, {seed}); sleeping 60s then retrying...', flush=True)
                time.sleep(60)
                continue
            raise SystemExit(
                f'[persisted_injection] D: drive lost on run_exists({dataset_slug}, {rate}, {seed}) after retry. '
                f'Aborting script. Remount with: sudo mount -t drvfs D: /mnt/d, then relaunch.'
            )

def persist_run(
    dataset_slug: str,
    source_events: list,
    rate: float,
    seed: int,
    *,
    sensor_vocab: list | None = None,
    activity_vocab: list | None = None,
    source_path: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a persisted (rate, seed) injection run for the dataset.

    The injection is applied to a shallow copy: each injected event has its
    concept_name set to None, but `source_events` itself is not mutated. The
    persisted events.pkl IS the post-injection list ready for streaming.
    """
    run_dir = _run_dir(dataset_slug, rate, seed)
    if run_exists(dataset_slug, rate, seed) and not overwrite:
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    labelled_indices = [
        i for i, e in enumerate(source_events)
        if e.concept_name and str(e.concept_name).strip()
    ]
    n_inject = int(len(labelled_indices) * rate)
    selected = rng.sample(labelled_indices, min(n_inject, len(labelled_indices)))

    ground_truth: Dict[int, str] = {}
    for idx in selected:
        ground_truth[idx] = source_events[idx].concept_name
        source_events[idx].concept_name = None
    try:
        with open(run_dir / 'events.pkl', 'wb') as f:
            pickle.dump(source_events, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        for idx, original_label in ground_truth.items():
            source_events[idx].concept_name = original_label

    with open(run_dir / 'ground_truth.json', 'w') as f:
        json.dump({str(k): v for k, v in ground_truth.items()}, f, indent=0)

    meta = {
        'dataset_slug': dataset_slug,
        'rate': rate,
        'seed': seed,
        'n_total': len(source_events),
        'n_labelled': len(labelled_indices),
        'n_injected': len(ground_truth),
        'sensor_vocab': list(sensor_vocab) if sensor_vocab is not None else [],
        'activity_vocab': list(activity_vocab) if activity_vocab is not None else [],
        'source_path': source_path,
    }
    with open(run_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return run_dir

def load_run(dataset_slug: str, rate: float, seed: int, drop_natural_missing: bool = False):
    """Load a persisted (rate, seed) run.

    When ``drop_natural_missing`` is True, events whose label is empty in the
    SOURCE log (i.e. natural-missing, not injected) are filtered out of the
    returned stream. The injected events keep their (None) label so they
    remain recovery targets. Original indices are remapped so
    ``injected_indices`` and ``ground_truth`` refer to the post-filter event
    list. Useful for datasets like Cotton (11.48% natural-missing) and Chess
    (15.6%) to test whether cluster geometry improves when natural-missing
    events do not contribute to BFR sufficient statistics or to the warmup
    buffer.

    D-drive resilience: if D: drops mid-run (WSL drvfs OSError 19), this
    function sleeps and retries once. If the second attempt also fails with
    OSError 19, it raises SystemExit so the caller's whole script aborts
    cleanly — instead of grinding through the rest of the queue producing
    errors on every cell while D: stays disconnected.

    Returns:
        events: post-injection list of events (concept_name=None on injected indices).
        ground_truth: dict mapping idx -> original label.
        injected_indices: set of indices whose label was injected.
        meta: dict from meta.json (includes sensor_vocab, activity_vocab, etc.).
    """
    import time
    import sys
    run_dir = _run_dir(dataset_slug, rate, seed)
    for attempt in range(2):
        try:
            if not run_exists(dataset_slug, rate, seed):
                raise FileNotFoundError(
                    f'No persisted run for dataset={dataset_slug} rate={rate} seed={seed} '
                    f'(expected under {run_dir}).')
            break
        except OSError as e:
            if e.errno == 19 and attempt == 0:
                print(f'[persisted_injection] D: dropped on {dataset_slug} rate={rate} seed={seed}; sleeping 60s then retrying...', flush=True)
                time.sleep(60)
                continue
            raise SystemExit(
                f'[persisted_injection] D: drive lost on {dataset_slug} rate={rate} seed={seed} after retry. '
                f'Aborting script. Remount with: sudo mount -t drvfs D: /mnt/d, then relaunch.'
            )
    with open(run_dir / 'events.pkl', 'rb') as f:
        events = pickle.load(f)
    with open(run_dir / 'ground_truth.json') as f:
        gt_raw = json.load(f)
    ground_truth = {int(k): v for k, v in gt_raw.items()}
    injected_indices = set(ground_truth.keys())
    meta = {}
    if (run_dir / 'meta.json').exists():
        with open(run_dir / 'meta.json') as f:
            meta = json.load(f)

    if drop_natural_missing:
        keep_mask = []
        for i, e in enumerate(events):
            has_label = e.concept_name and str(e.concept_name).strip()
            keep_mask.append(has_label or i in injected_indices)
        old_to_new = {}
        new_events = []
        for i, e in enumerate(events):
            if keep_mask[i]:
                old_to_new[i] = len(new_events)
                new_events.append(e)
        events = new_events
        ground_truth = {old_to_new[i]: v for i, v in ground_truth.items()
                        if i in old_to_new}
        injected_indices = set(ground_truth.keys())
        dropped = sum(1 for k in keep_mask if not k)
        meta = dict(meta)
        meta['_natural_missing_dropped'] = dropped
        meta['_events_after_drop'] = len(events)

    return events, ground_truth, injected_indices, meta
