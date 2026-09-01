import yaml
import xml.etree.ElementTree as ET
import os
import gzip
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def _open_text(file_path):
    if str(file_path).lower().endswith('.gz'):
        return gzip.open(file_path, 'rt', encoding='utf-8')
    return open(file_path, 'r', encoding='utf-8')

class DataStreamXESEvent:
    __slots__ = [
        'case_id', 'subprocess_id', 'concept_name', 'timestamp',
        'lifecycle', 'resource', 'attributes', 'sensor_readings',
        'file_path'
    ]

    def __init__(self):
        self.case_id = None
        self.subprocess_id = None
        self.concept_name = None
        self.timestamp = None
        self.lifecycle = None
        self.resource = None
        self.attributes = {}
        self.sensor_readings = {}
        self.file_path = None

class DataStreamXESParser:

    YAML_DATASETS = {'chess', 'cottoncandy', 'vienna'}
    XML_DATASETS = {'cyberseciot', 'mimiciv', 'smartfactory'}

    def __init__(self, dataset_path, dataset_type=None, event_filter=None, include_all=True):
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type or self._detect_type()
        self.event_filter = event_filter
        self.include_all = include_all
        self.sensor_vocabulary = set()
        self.activity_vocabulary = set()

    def _detect_type(self):
        yaml_files = list(self.dataset_path.rglob('*.xes.yaml'))
        xml_files = list(self.dataset_path.rglob('*.xes'))
        if yaml_files and not xml_files:
            return 'yaml'
        if xml_files and not yaml_files:
            return 'xml'
        if yaml_files and xml_files:
            return 'yaml'
        return 'yaml'

    def _parse_timestamp(self, ts_str):
        if not ts_str:
            return None
        if isinstance(ts_str, datetime):
            return ts_str
        ts_str = str(ts_str)
        for fmt in (
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f+00:00',
        ):
            try:
                return datetime.strptime(ts_str[:32], fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    def _sort_key(self, e):
        if e.timestamp is None:
            return datetime.min
        if e.timestamp.tzinfo is not None:
            return e.timestamp.replace(tzinfo=None)
        return e.timestamp

    def _extract_sensor_readings_yaml(self, datastream):
        readings = {}
        if not datastream:
            return readings
        for item in datastream:
            if not isinstance(item, dict):
                continue
            if 'stream:point' in item:
                point = item['stream:point']
                sensor_id = point.get('stream:id', '')
                value = point.get('stream:value', None)
                source = point.get('stream:source', '')
                key = f"{source}/{sensor_id}" if source else sensor_id
                self.sensor_vocabulary.add(key)
                try:
                    readings[key] = float(value)
                except (TypeError, ValueError):
                    readings[key] = value
        return readings

    def _parse_yaml_file(self, file_path):
        with _open_text(file_path) as f:
            content = f.read()

        docs = content.split('---')
        trace_info = {}
        events = []

        for doc in docs:
            doc = doc.strip()
            if not doc:
                continue
            try:
                data = yaml.safe_load(doc)
            except yaml.YAMLError:
                continue
            if not data:
                continue

            if 'log' in data:
                trace = data['log'].get('trace', {})
                trace_info = {
                    'concept_name': trace.get('concept:name'),
                    'cpee_name': trace.get('cpee:name'),
                    'cpee_instance': trace.get('cpee:instance'),
                }
                continue

            if 'event' not in data:
                continue

            evt_data = data['event']
            event = DataStreamXESEvent()
            event.case_id = evt_data.get('concept:instance', trace_info.get('concept_name'))
            event.subprocess_id = trace_info.get('cpee_instance')
            event.concept_name = evt_data.get('concept:name', None)
            event.timestamp = self._parse_timestamp(evt_data.get('time:timestamp'))
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
                event.sensor_readings = self._extract_sensor_readings_yaml(
                    evt_data['stream:datastream'])

            if event.concept_name and str(event.concept_name).strip():
                self.activity_vocabulary.add(str(event.concept_name))

            events.append(event)

        return events

    def _get_xml_attr(self, element, key, ns=None):
        ns = ns or {'xes': 'http://code.deckfour.org/xes',
                     'stream': 'https://cpee.org/datastream/datastream.xesext'}
        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child.get('key') == key:
                return child.get('value')
        return None

    def _get_stream_attr(self, elem, name):
        NS = 'https://cpee.org/datastream/datastream.xesext'
        return elem.get(f'stream:{name}') or elem.get(f'{{{NS}}}{name}') or ''

    def _extract_sensor_readings_xml(self, event_elem, ns):
        readings = {}
        for datastream in event_elem:
            tag = datastream.tag.split('}')[-1] if '}' in datastream.tag else datastream.tag
            if tag == 'list' and datastream.get('key') == 'stream:datastream':
                for point in datastream:
                    point_tag = point.tag.split('}')[-1] if '}' in point.tag else point.tag
                    if point_tag == 'list' and point.get('key') == 'stream:point':
                        sensor_id = ''
                        value = None
                        source = self._get_stream_attr(point, 'source')
                        if not source:
                            source = self._get_stream_attr(point, 'system')
                        observation = self._get_stream_attr(point, 'observation')

                        for attr in point:
                            attr_key = attr.get('key', '')
                            attr_val = attr.get('value', None)
                            s_val = self._get_stream_attr(attr, 'value')
                            s_id = self._get_stream_attr(attr, 'id')

                            if attr_key == 'stream:id':
                                sensor_id = attr_val or ''
                            elif attr_key == 'stream:value':
                                value = attr_val
                            elif s_val:
                                value = s_val
                            elif s_id:
                                sensor_id = s_id

                        if not sensor_id and observation:
                            sensor_id = observation.split('#')[-1] if '#' in observation else observation

                        key = f"{source.split('#')[-1] if '#' in source else source}/{sensor_id}" if source and sensor_id else (sensor_id or source)
                        if not key or key == '/':
                            continue
                        self.sensor_vocabulary.add(key)
                        try:
                            readings[key] = float(value)
                        except (TypeError, ValueError):
                            readings[key] = value
        return readings

    def _parse_xml_file(self, file_path):
        ns = {'xes': 'http://code.deckfour.org/xes',
              'stream': 'https://cpee.org/datastream/datastream.xesext'}

        try:
            tree = ET.parse(file_path)
        except (ET.ParseError, OSError):
            return []

        root = tree.getroot()
        events = []

        for trace in root.iter():
            tag = trace.tag.split('}')[-1] if '}' in trace.tag else trace.tag
            if tag != 'trace':
                continue

            case_id = None
            trace_attrs = {}
            for child in trace:
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag in ('string', 'int', 'float', 'date'):
                    k = child.get('key', '')
                    v = child.get('value', '')
                    if k == 'concept:name':
                        case_id = v
                    else:
                        trace_attrs[k] = v

            for child in trace:
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag != 'event':
                    continue

                event = DataStreamXESEvent()
                event.case_id = case_id
                event.file_path = str(file_path)

                for attr in child:
                    attr_tag = attr.tag.split('}')[-1] if '}' in attr.tag else attr.tag
                    k = attr.get('key', '')
                    v = attr.get('value', '')

                    if k == 'concept:name':
                        event.concept_name = v if v and v.strip() else None
                    elif k == 'time:timestamp':
                        event.timestamp = self._parse_timestamp(v)
                    elif k == 'lifecycle:transition':
                        event.lifecycle = v
                    elif k == 'org:resource':
                        event.resource = v
                    elif attr_tag == 'list' and k == 'stream:datastream':
                        pass
                    elif k and k not in ('concept:instance',):
                        event.attributes[k] = v

                event.sensor_readings = self._extract_sensor_readings_xml(child, ns)

                if not event.lifecycle:
                    event.lifecycle = 'complete'

                if event.concept_name and str(event.concept_name).strip():
                    self.activity_vocabulary.add(str(event.concept_name))

                events.append(event)

        return events

    def _is_yaml_file(self, file_path):
        s = str(file_path).lower()
        if s.endswith('.gz'):
            s = s[:-3]
        return s.endswith('.xes.yaml') or s.endswith('.yaml')

    def _is_xml_file(self, file_path):
        s = str(file_path).lower()
        if s.endswith('.gz'):
            s = s[:-3]
        return s.endswith('.xes') and not s.endswith('.xes.yaml')

    def _parse_file(self, file_path):
        if self._is_yaml_file(file_path):
            return self._parse_yaml_file(file_path)
        elif self._is_xml_file(file_path):
            return self._parse_xml_file(file_path)
        return []

    def _find_files(self):
        all_files = []
        for pattern in ('*.xes.yaml', '*.yaml', '*.xes',
                        '*.xes.yaml.gz', '*.yaml.gz', '*.xes.gz'):
            for f in sorted(self.dataset_path.rglob(pattern)):
                if f not in all_files:
                    all_files.append(f)
        return all_files

    def parse_all(self, filter_events=True):
        files = self._find_files()
        all_events = []

        for f in files:
            events = self._parse_file(f)
            if filter_events and self.event_filter:
                events = [e for e in events if self.event_filter(e)]
            all_events.extend(events)

        all_events.sort(key=self._sort_key)
        return all_events

    def stream(self, filter_events=True):
        events = self.parse_all(filter_events)
        for event in events:
            yield event

    def build_sensor_vocabulary(self):
        if not self.sensor_vocabulary:
            self.parse_all(filter_events=False)
        return sorted(self.sensor_vocabulary)

    def get_stats(self, events=None):
        if events is None:
            events = self.parse_all()

        total = len(events)
        with_label = sum(1 for e in events if e.concept_name and str(e.concept_name).strip())
        with_sensors = sum(1 for e in events if e.sensor_readings)
        missing_labels = total - with_label

        cases = defaultdict(int)
        for e in events:
            cases[e.case_id] += 1

        return {
            'total_events': total,
            'events_with_label': with_label,
            'events_missing_label': missing_labels,
            'missing_label_rate': missing_labels / total if total > 0 else 0,
            'events_with_sensors': with_sensors,
            'sensor_rate': with_sensors / total if total > 0 else 0,
            'num_cases': len(cases),
            'num_activities': len(self.activity_vocabulary),
            'num_sensors': len(self.sensor_vocabulary),
            'avg_events_per_case': total / len(cases) if cases else 0,
        }

DATASET_PATHS = {
    'chess': 'data/logs/chess/',
    'cottoncandy': 'data/logs/cottoncandy/',
    'vienna': 'data/logs/vienna/',
    'cyberseciot': 'data/logs/cyberseciot/',
    'mimiciv': 'data/logs/mimiciv/',
    'smartfactory': 'data/logs/smartfactory/',
}

if __name__ == '__main__':
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else 'chess'

    if dataset not in DATASET_PATHS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(DATASET_PATHS.keys())}")
        sys.exit(1)

    path = DATASET_PATHS[dataset]
    print(f"Parsing {dataset} from {path}...")

    parser = DataStreamXESParser(path)
    events = parser.parse_all(filter_events=False)

    stats = parser.get_stats(events)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nSensor vocabulary ({len(parser.sensor_vocabulary)} sensors):")
    for s in sorted(parser.sensor_vocabulary)[:10]:
        print(f"  {s}")

    print(f"\nActivity vocabulary ({len(parser.activity_vocabulary)} activities):")
    for a in sorted(parser.activity_vocabulary)[:10]:
        print(f"  {a}")

    print(f"\nStreaming simulation test (first 5 events):")
    parser2 = DataStreamXESParser(path)
    for i, event in enumerate(parser2.stream(filter_events=False)):
        if i >= 5:
            break
        print(f"  [{i}] case={event.case_id}, activity={event.concept_name}, "
              f"sensors={len(event.sensor_readings)}, ts={event.timestamp}")
