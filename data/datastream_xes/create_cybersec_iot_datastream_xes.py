
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import os
import gc
import ast

TRAIN_PATH = "/mnt/c/Users/drana/Downloads/cybersec_iot_spinet_data/train_data"
TEST_PATH = "/mnt/c/Users/drana/Downloads/cybersec_iot_spinet_data/test_data"
OUTPUT_DIR = Path("/mnt/d/cybersec_iot_datastream_xes")
BATCH_SIZE = 500

BASE_TIMESTAMP = datetime(2023, 4, 12, 0, 0, 0)

SECURITY_COUNTERS = [
    'USER_AUTH', 'USER_MGMT_COUNT', 'CRED_COUNT', 'USER_ERR_COUNT',
    'USYS_CONFIG_COUNT', 'CHID_COUNT', 'SELINUX_ERR_COUNT', 'SYSTEM_COUNT',
    'SERVICE_COUNT', 'DAEMON_COUNT', 'NETFILTER_COUNT', 'SECCOMP_COUNT',
    'AVC_COUNT', 'ANOM_COUNT', 'INTEGRITY_COUNT', 'KERNEL_COUNT',
    'RESP_COUNT', 'SELINUX_MGMT_COUNT'
]

CSV_USECOLS = [
    'SYSCALL_timestamp', 'SYSCALL_syscall', 'SYSCALL_success', 'SYSCALL_exit',
    'PROCESS_comm', 'PROCESS_exe', 'PROCESS_PATH',
    'CUSTOM_openFiles', 'CUSTOM_libs', 'CUSTOM_openSockets',
    'PROCESS_uid', 'SYSCALL_pid',
    'USER_AUTH', 'USER_MGMT_COUNT', 'CRED_COUNT', 'USER_ERR_COUNT',
    'USYS_CONFIG_COUNT', 'CHID_COUNT', 'SELINUX_ERR_COUNT', 'SYSTEM_COUNT',
    'SERVICE_COUNT', 'DAEMON_COUNT', 'NETFILTER_COUNT', 'SECCOMP_COUNT',
    'AVC_COUNT', 'ANOM_COUNT', 'INTEGRITY_COUNT', 'KERNEL_COUNT',
    'RESP_COUNT', 'SELINUX_MGMT_COUNT'
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBPROCESS_DIR = OUTPUT_DIR / "subprocesses"
SUBPROCESS_DIR.mkdir(parents=True, exist_ok=True)


def xml_escape(text):
    if text is None:
        return ""
    text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")


def format_timestamp(ts_ms):
    if ts_ms is None or pd.isna(ts_ms):
        return BASE_TIMESTAMP.strftime("%Y-%m-%dT%H:%M:%S.%f")
    try:
        absolute_ts = BASE_TIMESTAMP + timedelta(milliseconds=int(ts_ms))
        return absolute_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except:
        return BASE_TIMESTAMP.strftime("%Y-%m-%dT%H:%M:%S.%f")


def parse_list_field(field_value):
    if pd.isna(field_value) or field_value == '[]':
        return []
    try:
        if isinstance(field_value, list):
            return field_value
        return ast.literal_eval(str(field_value))
    except:
        return []


def create_main_xes_header():
    return '''<?xml version="1.0" encoding="utf-8" ?>
<log xes.version="1849-2016" xes.features="nested-attributes"
     xmlns="http://www.xes-standard.org/"
     xmlns:stream="https://cpee.org/datastream/datastream.xesext">
    <extension name="Concept" prefix="concept" uri="http://www.xes-standard.org/concept.xesext" />
    <extension name="Lifecycle" prefix="lifecycle" uri="http://www.xes-standard.org/lifecycle.xesext" />
    <extension name="Organizational" prefix="org" uri="http://www.xes-standard.org/org.xesext" />
    <extension name="Time" prefix="time" uri="http://www.xes-standard.org/time.xesext" />
    <extension name="Sensorstream" prefix="stream" uri="https://cpee.org/datastream/datastream.xesext" />
    <global scope="trace">
        <string key="concept:name" value="unknown"/>
    </global>
    <global scope="event">
        <string key="concept:name" value="unknown"/>
        <date key="time:timestamp" value="1970-01-01T00:00:00.000+00:00"/>
    </global>
    <string key="origin" value="SPINET Cybersecurity IoT Competition" />
    <string key="creator" value="Cybersecurity IoT DataStream XES Generator v2.0" />
    <string key="description" value="IoT-enriched event log with system call events and embedded state sensors (openFiles, libs, sockets) and security counters" />
'''


def create_subprocess_xes(subprocess_id, traces_data):

    xes = f'''<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xmlns:stream="https://cpee.org/datastream/datastream.xesext"
     xmlns="http://code.deckfour.org/xes" xes.creator="Cybersec-IoT-v2.1">
    <extension name="Concept" prefix="concept" uri="http://code.deckfour.org/xes/concept.xesext"/>
    <extension name="Time" prefix="time" uri="http://code.deckfour.org/xes/time.xesext"/>
    <extension name="Organizational" prefix="org" uri="http://code.deckfour.org/xes/org.xesext"/>
    <extension name="Sensorstream" prefix="stream" uri="https://cpee.org/datastream/datastream.xesext"/>
    <string key="SubProcessID" value="{subprocess_id}"/>
'''

    for pid, events in traces_data.items():
        xes += f'''    <trace>
        <string key="concept:name" value="{xml_escape(str(pid))}"/>
        <string key="SYSCALL_pid" value="{xml_escape(str(pid))}"/>
'''

        for event in events:
            activity = event['activity']
            timestamp = format_timestamp(event['timestamp'])
            xes += f'''        <event>
'''
            if activity is not None:
                xes += f'''            <string key="concept:name" value="{xml_escape(activity)}"/>
'''
            else:
                xes += f'''            <string key="concept:name" value=""/>
'''
            xes += f'''            <date key="time:timestamp" value="{timestamp}"/>
            <string key="lifecycle:transition" value="complete"/>
            <string key="org:resource" value="{xml_escape(event.get('process_comm', ''))}"/>
            <string key="process:exe" value="{xml_escape(event.get('process_exe', ''))}"/>
            <string key="process:path" value="{xml_escape(event.get('process_path', ''))}"/>
            <string key="syscall:exit" value="{event.get('syscall_exit', '')}"/>
'''

            iot_points = event.get('iot_points', [])
            if iot_points:
                xes += '''            <list key="stream:datastream">
'''
                for point in iot_points:
                    source = xml_escape(point['source'])
                    system_type = point['system_type']
                    interaction_type = point.get('interaction_type', 'sosa:Observation')
                    sensor_id = xml_escape(point.get('id', ''))
                    point_ts = format_timestamp(event['timestamp'])
                    point_value = xml_escape(str(point.get('value', '')))
                    category = xml_escape(point.get('category', ''))
                    procedure_type = point.get('procedure_type', 'discrete').replace('stream:', '')

                    xes += f'''                <list key="stream:point" stream:source="{source}" stream:system_type="{system_type}" stream:interaction_type="{interaction_type}">
                    <string key="stream:id" value="{sensor_id}"/>
                    <date key="stream:timestamp" value="{point_ts}"/>
                    <string key="stream:value" value="{point_value}"/>
                    <list key="stream:meta">
                        <string key="category" value="{category}"/>
                        <string key="procedure_type" value="{procedure_type}"/>
                    </list>
                </list>
'''
                xes += '''            </list>
'''

            xes += '''        </event>
'''

        xes += '''    </trace>
'''

    xes += '''</log>
'''
    return xes


def process_csv_file(csv_path, dataset):

    df = pd.read_csv(csv_path, usecols=CSV_USECOLS)
    traces_data = {}
    missing_label_count = 0

    for _, row in df.iterrows():
        pid = row['SYSCALL_pid']

        syscall = row.get('SYSCALL_syscall', 'unknown')
        user = row.get('PROCESS_uid', 'unknown')
        success = row.get('SYSCALL_success', None)
        if pd.isna(success):
            activity = None
            missing_label_count += 1
        else:
            activity = f"{syscall}_{user}_{success}"

        iot_points = []

        open_files = parse_list_field(row.get('CUSTOM_openFiles', '[]'))
        if open_files:
            iot_points.append({
                'id': 'openFiles',
                'source': 'cybersec:state/openFiles',
                'system_type': 'sosa:Sensor',
                'category': 'file_handle_list',
                'procedure_type': 'stream:discrete',
                'interaction_type': 'sosa:Observation',
                'value': str(open_files)[:500]
            })

        libs = parse_list_field(row.get('CUSTOM_libs', '[]'))
        if libs:
            iot_points.append({
                'id': 'loadedLibs',
                'source': 'cybersec:state/loadedLibs',
                'system_type': 'sosa:Sensor',
                'category': 'library_list',
                'procedure_type': 'stream:discrete',
                'interaction_type': 'sosa:Observation',
                'value': str(libs)[:500]
            })

        sockets = parse_list_field(row.get('CUSTOM_openSockets', '[]'))
        if sockets:
            iot_points.append({
                'id': 'openSockets',
                'source': 'cybersec:state/openSockets',
                'system_type': 'sosa:Sensor',
                'category': 'socket_list',
                'procedure_type': 'stream:discrete',
                'interaction_type': 'sosa:Observation',
                'value': str(sockets)[:500]
            })

        security_values = {}
        for counter in SECURITY_COUNTERS:
            val = row.get(counter, 0)
            if pd.notna(val) and val != 0:
                security_values[counter] = int(val)

        if security_values:
            iot_points.append({
                'id': 'securityCounters',
                'source': 'cybersec:security/counters',
                'system_type': 'sosa:Sensor',
                'category': 'security_event_counts',
                'procedure_type': 'stream:discrete',
                'interaction_type': 'sosa:Observation',
                'value': str(security_values)
            })

        event_data = {
            'activity': activity,
            'timestamp': row.get('SYSCALL_timestamp', 0),
            'process_comm': row.get('PROCESS_comm', ''),
            'process_exe': row.get('PROCESS_exe', ''),
            'process_path': row.get('PROCESS_PATH', ''),
            'syscall_exit': row.get('SYSCALL_exit', ''),
            'iot_points': iot_points
        }

        if pid not in traces_data:
            traces_data[pid] = []
        traces_data[pid].append(event_data)

    return traces_data, missing_label_count


def main():
    print("=" * 70)
    print("Cybersecurity IoT DataStream XES Generator v2.1")
    print("Converting system call logs to DataStream XES format")
    print("(Following official DataStream XES Extension - Mangler et al., 2023)")
    print("=" * 70)
    print(f"Train data path: {TRAIN_PATH}")
    print(f"Test data path: {TEST_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Batch size: {BATCH_SIZE} files")
    print()
    print("Data sources included:")
    print("  PROCESS EVENTS: System calls (activity = syscall_user_success)")
    print("  STATE SENSORS: openFiles, loadedLibs, openSockets")
    print("  SECURITY SENSORS: 18 aggregated security event counters")
    print("  PROCESS CONTEXT: comm, exe, path, exit code")
    print()

    train_files = sorted([f for f in os.listdir(TRAIN_PATH) if f.endswith('.csv')])
    test_files = sorted([f for f in os.listdir(TEST_PATH) if f.endswith('.csv')])

    print(f"Found {len(train_files):,} training files")
    print(f"Found {len(test_files):,} test files")
    print(f"Total files to process: {len(train_files) + len(test_files):,}")
    print()

    main_xes_file = OUTPUT_DIR / "MainProcess.xes"
    with open(main_xes_file, 'w', encoding='utf-8') as f:
        f.write(create_main_xes_header())

    total_subprocess_files = 0
    total_traces = 0
    total_events = 0
    total_iot_points = 0
    total_missing_labels = 0

    print("=" * 50)
    print("Processing TRAINING data...")
    print("=" * 50)

    num_train_batches = (len(train_files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_train_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(train_files))
        batch_files = train_files[start_idx:end_idx]

        print(f"\nBatch {batch_idx + 1}/{num_train_batches}: Processing files {start_idx + 1}-{end_idx}...")

        batch_subprocesses = 0
        batch_traces = 0
        batch_events = 0
        batch_iot_points = 0
        batch_missing = 0

        with open(main_xes_file, 'a', encoding='utf-8') as main_f:
            for csv_file in batch_files:
                csv_path = os.path.join(TRAIN_PATH, csv_file)
                source_file = csv_file.replace('.csv', '')

                try:
                    traces_data, file_missing = process_csv_file(csv_path, 'train')

                    if traces_data:
                        subprocess_id = str(uuid.uuid4())
                        subprocess_xes = create_subprocess_xes(subprocess_id, traces_data)
                        subprocess_file = SUBPROCESS_DIR / f"{subprocess_id}.xes"
                        with open(subprocess_file, 'w', encoding='utf-8') as sp_f:
                            sp_f.write(subprocess_xes)

                        num_traces = len(traces_data)
                        num_events = sum(len(events) for events in traces_data.values())
                        num_iot = sum(
                            len(e.get('iot_points', []))
                            for events in traces_data.values()
                            for e in events
                        )

                        batch_subprocesses += 1
                        batch_traces += num_traces
                        batch_events += num_events
                        batch_iot_points += num_iot
                        batch_missing += file_missing

                        main_f.write(f'''    <trace>
        <string key="concept:name" value="{xml_escape(source_file)}"/>
        <string key="dataset" value="train"/>
        <string key="SubProcessID" value="{subprocess_id}"/>
        <int key="num_traces" value="{num_traces}"/>
        <int key="num_events" value="{num_events}"/>
    </trace>
''')
                except Exception as e:
                    print(f"  Warning: Error processing {csv_file}: {e}")

        total_subprocess_files += batch_subprocesses
        total_traces += batch_traces
        total_events += batch_events
        total_iot_points += batch_iot_points
        total_missing_labels += batch_missing

        print(f"  Created {batch_subprocesses} subprocess files")
        print(f"  Traces: {batch_traces:,}, Events: {batch_events:,}, IoT points: {batch_iot_points:,}, Missing labels: {batch_missing:,}")

        gc.collect()

    print()
    print("=" * 50)
    print("Processing TEST data...")
    print("=" * 50)

    num_test_batches = (len(test_files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_test_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(test_files))
        batch_files = test_files[start_idx:end_idx]

        print(f"\nBatch {batch_idx + 1}/{num_test_batches}: Processing files {start_idx + 1}-{end_idx}...")

        batch_subprocesses = 0
        batch_traces = 0
        batch_events = 0
        batch_iot_points = 0
        batch_missing = 0

        with open(main_xes_file, 'a', encoding='utf-8') as main_f:
            for csv_file in batch_files:
                csv_path = os.path.join(TEST_PATH, csv_file)
                source_file = csv_file.replace('.csv', '')

                try:
                    traces_data, file_missing = process_csv_file(csv_path, 'test')

                    if traces_data:
                        subprocess_id = str(uuid.uuid4())
                        subprocess_xes = create_subprocess_xes(subprocess_id, traces_data)
                        subprocess_file = SUBPROCESS_DIR / f"{subprocess_id}.xes"
                        with open(subprocess_file, 'w', encoding='utf-8') as sp_f:
                            sp_f.write(subprocess_xes)

                        num_traces = len(traces_data)
                        num_events = sum(len(events) for events in traces_data.values())
                        num_iot = sum(
                            len(e.get('iot_points', []))
                            for events in traces_data.values()
                            for e in events
                        )

                        batch_subprocesses += 1
                        batch_traces += num_traces
                        batch_events += num_events
                        batch_iot_points += num_iot
                        batch_missing += file_missing

                        main_f.write(f'''    <trace>
        <string key="concept:name" value="{xml_escape(source_file)}"/>
        <string key="dataset" value="test"/>
        <string key="SubProcessID" value="{subprocess_id}"/>
        <int key="num_traces" value="{num_traces}"/>
        <int key="num_events" value="{num_events}"/>
    </trace>
''')
                except Exception as e:
                    print(f"  Warning: Error processing {csv_file}: {e}")

        total_subprocess_files += batch_subprocesses
        total_traces += batch_traces
        total_events += batch_events
        total_iot_points += batch_iot_points
        total_missing_labels += batch_missing

        print(f"  Created {batch_subprocesses} subprocess files")
        print(f"  Traces: {batch_traces:,}, Events: {batch_events:,}, IoT points: {batch_iot_points:,}, Missing labels: {batch_missing:,}")

        gc.collect()

    with open(main_xes_file, 'a', encoding='utf-8') as f:
        f.write('</log>\n')

    print()
    print("=" * 70)
    print("Cybersecurity IoT DataStream XES Generation Complete!")
    print("=" * 70)
    print(f"Main XES file: {main_xes_file}")
    print(f"Subprocess directory: {SUBPROCESS_DIR}")
    print()
    print("Statistics:")
    print(f"  Total files processed: {len(train_files) + len(test_files):,}")
    print(f"    - Training files: {len(train_files):,}")
    print(f"    - Test files: {len(test_files):,}")
    print(f"  Total subprocess files: {total_subprocess_files:,}")
    print(f"  Total traces (processes): {total_traces:,}")
    print(f"  Total events (syscalls): {total_events:,}")
    print(f"  Total IoT points embedded: {total_iot_points:,}")
    print()
    print("Missing Activity Labels (natural, from missing SYSCALL_success):")
    print(f"  Events with missing labels: {total_missing_labels:,}")
    if total_events > 0:
        print(f"  Missing label rate: {100 * total_missing_labels / total_events:.2f}%")
    print()
    print("Data sources included:")
    print("  PROCESS EVENTS (Activities):")
    print("    - System calls with format: syscall_user_success")
    print("  STATE SENSORS (Single Activity Context):")
    print("    - openFiles: Currently open file handles")
    print("    - loadedLibs: Dynamically loaded libraries")
    print("    - openSockets: Active network connections")
    print("  SECURITY SENSORS (Aggregated Counters):")
    print("    - 18 security event counters (AUTH, CRED, ANOM, etc.)")
    print("  PROCESS CONTEXT:")
    print("    - Process name, executable, hierarchy, exit codes")


if __name__ == "__main__":
    main()
