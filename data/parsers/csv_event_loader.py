import pandas as pd
from datetime import datetime
from data.parsers.datastream_xes_parser import DataStreamXESEvent

def load_events_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    events = []
    sensor_vocab = set()
    activity_vocab = set()

    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    attr_cols = [c for c in df.columns if c.startswith('attr_')]

    for _, row in df.iterrows():
        event = DataStreamXESEvent()
        event.case_id = str(row.get('case_id', ''))
        cn = row.get('concept_name', '')
        event.concept_name = str(cn) if pd.notna(cn) and str(cn).strip() else None
        ts = row.get('timestamp')
        if pd.notna(ts):
            try:
                event.timestamp = pd.to_datetime(ts).to_pydatetime()
            except Exception:
                event.timestamp = None
        event.lifecycle = str(row.get('lifecycle', '')) if pd.notna(row.get('lifecycle', '')) else ''
        event.resource = str(row.get('resource', '')) if pd.notna(row.get('resource', '')) else None
        event.subprocess_id = str(row.get('subprocess_id', '')) if pd.notna(row.get('subprocess_id', '')) else None

        for col in attr_cols:
            val = row.get(col)
            if pd.notna(val):
                key = col[5:]
                event.attributes[key] = str(val)

        for col in sensor_cols:
            val = row.get(col)
            if pd.notna(val):
                key = col[7:]
                try:
                    event.sensor_readings[key] = float(val)
                except (ValueError, TypeError):
                    event.sensor_readings[key] = val
                sensor_vocab.add(key)

        if event.concept_name and str(event.concept_name).strip():
            activity_vocab.add(str(event.concept_name))

        events.append(event)

    return events, sorted(sensor_vocab), sorted(activity_vocab)
