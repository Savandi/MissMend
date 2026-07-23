import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')
from data.parsers.datastream_xes_parser import DataStreamXESParser, DATASET_PATHS

def preprocess_dataset(name, path, output_dir, max_events=None):
    print(f"Pre-processing {name} from {path}...")

    parser = DataStreamXESParser(path)
    events = parser.parse_all(filter_events=False)

    if max_events and len(events) > max_events:
        events = events[:max_events]

    print(f"  Parsed {len(events)} events")
    print(f"  Activities: {len(parser.activity_vocabulary)}")
    print(f"  Sensors: {len(parser.sensor_vocabulary)}")

    rows = []
    for e in events:
        ts = None
        if e.timestamp:
            ts = e.timestamp.replace(tzinfo=None) if e.timestamp.tzinfo else e.timestamp

        row = {
            'case_id': str(e.case_id) if e.case_id else '',
            'concept_name': str(e.concept_name) if e.concept_name else '',
            'timestamp': ts,
            'lifecycle': str(e.lifecycle) if e.lifecycle else '',
            'resource': str(e.resource) if e.resource else '',
            'subprocess_id': str(e.subprocess_id) if e.subprocess_id else '',
        }

        for k, v in e.attributes.items():
            if isinstance(v, str):
                row[f'attr_{k}'] = v
            elif isinstance(v, (int, float)):
                row[f'attr_{k}'] = str(v)

        for sensor_key, reading in e.sensor_readings.items():
            row[f'sensor_{sensor_key}'] = str(reading)

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_file = output_path / f"{name}.csv.gz"
    df.to_csv(parquet_file, index=False, compression='gzip')

    meta = {
        'name': name,
        'total_events': len(events),
        'activities': sorted(parser.activity_vocabulary),
        'sensors': sorted(parser.sensor_vocabulary),
        'attribute_keys': sorted(set(k for k in df.columns if k.startswith('attr_'))),
        'sensor_keys': sorted(set(k for k in df.columns if k.startswith('sensor_'))),
    }
    meta_file = output_path / f"{name}_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {parquet_file} ({parquet_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Meta: {meta_file}")
    print(f"  Columns: {len(df.columns)}")
    print()

if __name__ == '__main__':
    output_dir = '/mnt/c/Users/drana/Downloads/iot-event-log-quality-analysis/data/preprocessed'

    dataset = sys.argv[1] if len(sys.argv) > 1 else 'all'
    max_events = int(sys.argv[2]) if len(sys.argv) > 2 else None

    datasets_to_process = {
        'chess': ('chess', DATASET_PATHS['chess']),
        'cottoncandy': ('cottoncandy', DATASET_PATHS['cottoncandy']),
        'smartfactory': ('smartfactory', DATASET_PATHS['smartfactory']),
    }

    if dataset == 'all':
        for name, path in datasets_to_process.values():
            preprocess_dataset(name, path, output_dir, max_events)
    elif dataset in datasets_to_process:
        name, path = datasets_to_process[dataset]
        preprocess_dataset(name, path, output_dir, max_events)
    else:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(datasets_to_process.keys())} or 'all'")
