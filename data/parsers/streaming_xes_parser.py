import yaml
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from collections import defaultdict, OrderedDict
import os

from data.parsers.datastream_xes_parser import (
    DataStreamXESEvent,
    DataStreamXESParser,
)

class StreamingXESParser:

    def __init__(self, dataset_path, max_files=None, max_events_per_file=None,
                 case_buffer_size=10000, file_pattern_priority=None):
        self.dataset_path = Path(dataset_path)
        self.max_files = max_files
        self.max_events_per_file = max_events_per_file
        self.case_buffer_size = case_buffer_size
        self.file_pattern_priority = file_pattern_priority or ('*.xes', '*.xes.yaml', '*.yaml')

        self.sensor_vocabulary = set()
        self.activity_vocabulary = set()

        self._delegate = DataStreamXESParser(dataset_path)

    def _iter_files(self):
        if self.dataset_path.is_file():
            yield self.dataset_path
            return
        seen = set()
        count = 0
        for pattern in self.file_pattern_priority:
            for f in self.dataset_path.rglob(pattern):
                if f in seen:
                    continue
                seen.add(f)
                yield f
                count += 1
                if self.max_files and count >= self.max_files:
                    return

    def _is_yaml_file(self, file_path):
        s = str(file_path).lower()
        return s.endswith('.xes.yaml') or s.endswith('.yaml')

    def _is_xml_file(self, file_path):
        s = str(file_path).lower()
        return s.endswith('.xes') and not s.endswith('.xes.yaml')

    def _stream_yaml_file(self, file_path):
        trace_info = {}
        buffer_lines = []
        emitted = 0

        try:
            f = open(file_path, 'r', encoding='utf-8')
        except OSError:
            return

        try:
            for line in f:
                if line.strip() == '---':
                    if buffer_lines:
                        for event in self._parse_yaml_doc(buffer_lines, trace_info, file_path):
                            if event is None:
                                continue
                            if isinstance(event, dict) and event.get('_trace_update'):
                                trace_info = event['trace_info']
                            else:
                                yield event
                                emitted += 1
                                if self.max_events_per_file and emitted >= self.max_events_per_file:
                                    return
                        buffer_lines = []
                else:
                    buffer_lines.append(line)
            if buffer_lines:
                for event in self._parse_yaml_doc(buffer_lines, trace_info, file_path):
                    if event is None or (isinstance(event, dict) and event.get('_trace_update')):
                        continue
                    yield event
        finally:
            f.close()

    def _parse_yaml_doc(self, lines, trace_info, file_path):
        doc = ''.join(lines).strip()
        if not doc:
            return
        try:
            data = yaml.safe_load(doc)
        except yaml.YAMLError:
            return
        if not data:
            return

        if 'log' in data:
            trace = data['log'].get('trace', {})
            new_trace_info = {
                'concept_name': trace.get('concept:name'),
                'cpee_name': trace.get('cpee:name'),
                'cpee_instance': trace.get('cpee:instance'),
            }
            yield {'_trace_update': True, 'trace_info': new_trace_info}
            return

        if 'event' not in data:
            return

        evt_data = data['event']
        event = DataStreamXESEvent()
        event.case_id = evt_data.get('concept:instance', trace_info.get('concept_name'))
        event.subprocess_id = trace_info.get('cpee_instance')
        event.concept_name = evt_data.get('concept:name', None)
        event.timestamp = self._delegate._parse_timestamp(evt_data.get('time:timestamp'))
        event.lifecycle = evt_data.get('cpee:lifecycle:transition',
                                       evt_data.get('lifecycle:transition', ''))
        event.resource = evt_data.get('org:resource', None)
        event.file_path = str(file_path)

        skip_keys = {'concept:name', 'concept:instance', 'time:timestamp',
                     'lifecycle:transition', 'cpee:lifecycle:transition',
                     'org:resource', 'stream:datastream', 'cpee:description',
                     'cpee:activity', 'cpee:instance', 'cpee:state',
                     'cpee:uuid', 'id:id'}
        for key, val in evt_data.items():
            if key not in skip_keys:
                event.attributes[key] = val

        if 'stream:datastream' in evt_data:
            event.sensor_readings = self._delegate._extract_sensor_readings_yaml(
                evt_data['stream:datastream'])
            for k in event.sensor_readings:
                self.sensor_vocabulary.add(k)

        if event.concept_name and str(event.concept_name).strip():
            self.activity_vocabulary.add(str(event.concept_name))

        yield event

    def _stream_xml_file(self, file_path):
        ns = {'xes': 'http://code.deckfour.org/xes',
              'stream': 'https://cpee.org/datastream/datastream.xesext'}
        emitted = 0

        try:
            context = ET.iterparse(str(file_path), events=('start', 'end'))
        except (ET.ParseError, OSError):
            return

        case_id = None
        trace_attrs = {}
        inside_event = False

        for ev_kind, elem in context:
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if ev_kind == 'start':
                if tag == 'event':
                    inside_event = True
                continue

            if tag == 'trace':
                case_id = None
                trace_attrs = {}
                inside_event = False
                elem.clear()
                continue

            if tag == 'event':
                event = self._parse_xml_event_element(elem, file_path, case_id, ns)
                if event is not None:
                    yield event
                    emitted += 1
                    if self.max_events_per_file and emitted >= self.max_events_per_file:
                        elem.clear()
                        return
                elem.clear()
                inside_event = False
                continue

            if tag in ('string', 'int', 'float', 'date'):
                if inside_event:
                    continue
                k = elem.get('key', '')
                v = elem.get('value', '')
                if k == 'concept:name':
                    case_id = v
                elif k:
                    trace_attrs[k] = v

    def _parse_xml_event_element(self, elem, file_path, case_id, ns):
        event = DataStreamXESEvent()
        event.case_id = case_id
        event.file_path = str(file_path)

        for attr in elem:
            attr_tag = attr.tag.split('}')[-1] if '}' in attr.tag else attr.tag
            k = attr.get('key', '')
            v = attr.get('value', '')

            if k == 'concept:name':
                event.concept_name = v if v and v.strip() else None
            elif k == 'time:timestamp':
                event.timestamp = self._delegate._parse_timestamp(v)
            elif k == 'lifecycle:transition':
                event.lifecycle = v
            elif k == 'org:resource':
                event.resource = v
            elif attr_tag == 'list' and k == 'stream:datastream':
                pass
            elif k and k not in ('concept:instance',):
                event.attributes[k] = v

        event.sensor_readings = self._delegate._extract_sensor_readings_xml(elem, ns)
        for k in event.sensor_readings:
            self.sensor_vocabulary.add(k)

        if not event.lifecycle:
            event.lifecycle = 'complete'

        if event.concept_name and str(event.concept_name).strip():
            self.activity_vocabulary.add(str(event.concept_name))

        return event

    def stream_events(self, event_filter=None):
        for f in self._iter_files():
            if self._is_yaml_file(f):
                gen = self._stream_yaml_file(f)
            elif self._is_xml_file(f):
                gen = self._stream_xml_file(f)
            else:
                continue

            for event in gen:
                if event_filter and not event_filter(event):
                    continue
                yield event

    def stream(self, event_filter=None):
        return self.stream_events(event_filter)

    def discover_vocabulary(self, max_events=10000):
        count = 0
        for event in self.stream_events():
            count += 1
            if count >= max_events:
                break
        return {
            'sensors': sorted(self.sensor_vocabulary),
            'activities': sorted(self.activity_vocabulary),
            'events_scanned': count,
        }

if __name__ == '__main__':
    import sys
    import time

    dataset = sys.argv[1] if len(sys.argv) > 1 else 'chess'
    DATASET_PATHS = {
        'chess': '/mnt/d/Chess Pieces Production/turmv4_batch4/turmv4_batch6/',
        'cottoncandy': '/mnt/d/Cotton Candy XES YAML (Power Consumption, Environment, Temperatures, Metrology)/cotton-candy/batch-0/',
        'cyberseciot': '/mnt/d/cybersec_iot_datastream_xes/',
        'mimiciv': '/mnt/d/mimiciv-v3/datastream_xes_v2/',
        'smartfactory': '/mnt/d/An IoT-Enriched Event Log for Process Mining in Smart Factories/Data Quality Issues Event Log/Data Quality Issues Event Log/',
        'vienna': '/mnt/d/',
    }

    if dataset not in DATASET_PATHS:
        print(f"Unknown dataset: {dataset}; available: {list(DATASET_PATHS.keys())}")
        sys.exit(1)

    path = DATASET_PATHS[dataset]
    max_events = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    print(f"Streaming {dataset} from {path}")
    print(f"Reading first {max_events} events to verify streaming behaviour...")

    parser = StreamingXESParser(path, max_files=10 if dataset == 'cyberseciot' else None)
    start = time.time()
    n = 0
    for event in parser.stream_events():
        n += 1
        if n <= 3:
            print(f"  [{n}] case={event.case_id}, activity={event.concept_name}, "
                  f"sensors={len(event.sensor_readings)}, ts={event.timestamp}")
        if n >= max_events:
            break
    elapsed = time.time() - start

    print(f"\nStreamed {n} events in {elapsed:.2f}s ({n/max(elapsed, 0.01):.0f} events/sec)")
    print(f"Sensor vocabulary discovered so far: {len(parser.sensor_vocabulary)}")
    print(f"Activity vocabulary discovered so far: {len(parser.activity_vocabulary)}")
