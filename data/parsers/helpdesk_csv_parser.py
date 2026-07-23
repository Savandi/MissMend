"""Adapter for the Italian Help Desk CSV log (BPI / PM benchmark).

Source columns (from ``finale.csv``):
    Case ID, Activity, Resource, Complete Timestamp, Variant, Variant index,
    Variant, seriousness, customer, product, responsible_section,
    seriousness_2, service_level, service_type, support_section, workgroup

The first ``Variant`` column is duplicated in the CSV (it appears twice) —
pandas auto-renames the second one. We map only the first occurrence.

Mapping to DataStreamXESEvent:
    Case ID            → case_id
    Activity           → concept_name
    Complete Timestamp → timestamp
    Resource           → resource
    (no lifecycle column in this log; left None)
    All remaining columns → attributes dict (skipping the duplicated Variant)
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from data.parsers.datastream_xes_parser import DataStreamXESEvent

def _parse_helpdesk_timestamp(value) -> Optional[datetime]:
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value, format='%Y/%m/%d %H:%M:%S.%f').to_pydatetime()
    except Exception:
        try:
            return pd.to_datetime(value).to_pydatetime()
        except Exception:
            return None

def iter_events(csv_path) -> Iterator[DataStreamXESEvent]:
    """Yield events from the Help desk CSV in source (row) order. Each row is
    one event; the case_id groups them implicitly."""
    df = pd.read_csv(csv_path)
    if 'Variant.1' in df.columns:
        df = df.drop(columns=['Variant.1'])

    case_col = 'Case ID'
    activity_col = 'Activity'
    timestamp_col = 'Complete Timestamp'
    resource_col = 'Resource'
    skip_cols = {case_col, activity_col, timestamp_col, resource_col}

    for _, row in df.iterrows():
        ev = DataStreamXESEvent()
        ev.case_id = str(row[case_col]) if pd.notna(row[case_col]) else ''
        act = row[activity_col]
        ev.concept_name = str(act) if pd.notna(act) and str(act).strip() else None
        ev.timestamp = _parse_helpdesk_timestamp(row[timestamp_col])
        if pd.notna(row[resource_col]):
            ev.resource = str(row[resource_col])
        for col in df.columns:
            if col in skip_cols:
                continue
            val = row[col]
            if pd.notna(val):
                ev.attributes[col] = str(val)
        ev.file_path = str(csv_path)
        yield ev

def load_events(csv_path, max_events: Optional[int] = None):
    out = []
    for i, ev in enumerate(iter_events(csv_path)):
        if max_events is not None and i >= max_events:
            break
        out.append(ev)
    return out
