"""Streaming parser for plain XES event logs from BPIC/PM benchmarks.

The existing ``DataStreamXESParser`` is tied to the DataStream XES extension
(IoT sensor blocks). BPIC logs are plain XES with no sensor data. This module
provides a lightweight, gzip-aware iterative parser that emits
``DataStreamXESEvent`` objects with empty ``sensor_readings`` so the rest of
the framework can consume them without modification.

Supports both ``.xes`` and ``.xes.gz`` inputs. The parser uses ``ET.iterparse``
to keep memory bounded — BPIC2017 (1.2M events) parses without loading the
whole tree.

Standard XES attribute mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    trace <string key="concept:name" .../>   → event.case_id
    event <string key="concept:name" .../>   → event.concept_name (None when empty)
    event <date   key="time:timestamp" .../> → event.timestamp
    event <string key="lifecycle:transition" .../> → event.lifecycle
    event <string key="org:resource" .../>   → event.resource
    every other event-level attribute        → event.attributes[key] = value
"""
from __future__ import annotations
import gzip
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from data.parsers.datastream_xes_parser import DataStreamXESEvent

def _strip_ns(tag: str) -> str:
    if tag.startswith('{'):
        return tag.split('}', 1)[1]
    return tag

def _resolve_xes_path(path: Path) -> Path:
    """Some Windows extractions of XES files produce a nested directory of the
    same name (``foo.xes/foo.xes``) or an empty stub directory next to a real
    ``foo.xes.gz``. Detect and unwrap both cases so callers can pass either the
    directory path or the file path."""
    if path.is_dir():
        inner = path / path.name
        if inner.is_file():
            return inner
        for candidate in path.iterdir():
            if candidate.is_file() and (candidate.suffix in ('.xes', '.gz')
                                        or candidate.name.endswith('.xes.gz')):
                return candidate
        sibling = path.parent / (path.name + '.gz')
        if sibling.is_file():
            return sibling
        raise FileNotFoundError(f'No .xes file found inside or beside directory {path}')
    return path

def _open_xes(path: Path):
    path = _resolve_xes_path(path)
    if path.suffix == '.gz' or path.name.endswith('.xes.gz'):
        return gzip.open(path, 'rb')
    return open(path, 'rb')

def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith('Z'):
            value = value[:-1] + '+00:00'
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None

def _event_from_xml(event_elem) -> DataStreamXESEvent:
    """Convert an <event> element into a DataStreamXESEvent. The element's
    children are <string>/<date>/<int>/<float>/<boolean> elements with
    ``key="..."`` and ``value="..."`` attributes per the XES standard."""
    ev = DataStreamXESEvent()
    for child in event_elem:
        key = child.attrib.get('key')
        val = child.attrib.get('value')
        if key is None:
            continue
        if key == 'concept:name':
            ev.concept_name = val if val and val.strip() else None
        elif key == 'time:timestamp':
            ev.timestamp = _parse_timestamp(val)
        elif key == 'lifecycle:transition':
            ev.lifecycle = val
        elif key == 'org:resource':
            ev.resource = val
        elif val is not None:
            ev.attributes[key] = val
    return ev

def iter_events(file_path) -> Iterator[DataStreamXESEvent]:
    """Yield DataStreamXESEvent in source order from a plain XES file.

    The yield order is the file order: events appear grouped by trace
    in the source XES, so the iterator emits trace 1's events, then trace
    2's events, etc. Each event has its case_id set from the enclosing trace.
    """
    path = Path(file_path)
    with _open_xes(path) as fh:
        current_case_id: Optional[str] = None
        context = ET.iterparse(fh, events=('start', 'end'))
        for ev_type, elem in context:
            tag = _strip_ns(elem.tag)
            if ev_type == 'start' and tag == 'trace':
                current_case_id = None
            elif ev_type == 'end' and tag == 'string' and elem.attrib.get('key') == 'concept:name':
                parent_tag = None
                if current_case_id is None:
                    current_case_id = elem.attrib.get('value') or ''
            elif ev_type == 'end' and tag == 'event':
                event = _event_from_xml(elem)
                event.case_id = current_case_id or ''
                event.file_path = str(path)
                yield event
                elem.clear()
            elif ev_type == 'end' and tag == 'trace':
                elem.clear()

def load_events(file_path, max_events: Optional[int] = None):
    """Convenience: read everything into a list (returns the same event
    objects iter_events yields). Use only for small logs where the memory
    profile is fine."""
    events = []
    for i, ev in enumerate(iter_events(file_path)):
        if max_events is not None and i >= max_events:
            break
        events.append(ev)
    return events
