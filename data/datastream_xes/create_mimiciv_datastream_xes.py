
import duckdb
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
import xml.etree.ElementTree as ET
from pathlib import Path


HOSP_PATH = "/mnt/d/mimiciv-v3/hosp"
ICU_PATH = "/mnt/d/mimiciv-v3/icu"
ED_PATH = "/mnt/d/mimic-iv-ed-2.2/mimic-iv-ed-2.2/ed"
OUTPUT_PATH = "/mnt/d/mimiciv-v3/datastream_xes_v4"
DB_PATH = "/mnt/d/mimiciv-v3/mimiciv_complete.duckdb"

ADMISSIONS_PER_FILE = 1000

MAX_ADMISSIONS = 20000

INCLUDE_EMAR_EVENTS = False

INCLUDE_NO_IOT_EVENTS = False

INCLUDE_WORKFLOW_MARKERS = True

INCLUDE_PROCEDURE_EVENTS = True

INCLUDE_LAB_SCORE_SNAPSHOTS = True

RESTRICT_CHARTEVENTS_TO_VITALS = True
CORE_VITAL_ITEMIDS = {
    220045,
    220210,
    220277,
    223761,
    220052,
    220181,
    220180,
    220179,
}

CORE_VITAL_CATEGORIES = set()

INCLUDE_ORDER_KIND_IOT = False

INCLUDE_CASE_CONTEXT_IOT = False


def order_kind_indicators(order_cat):
    text = (order_cat or '').lower()
    return {
        'is_continuous': 1.0 if 'continuous' in text or 'drip' in text else 0.0,
        'is_push':       1.0 if 'push' in text else 0.0,
        'is_bolus':      1.0 if 'bolus' in text else 0.0,
        'is_oral':       1.0 if 'oral' in text or 'po ' in text or 'non iv' in text else 0.0,
    }


INCLUDE_LAB_SCORE_DELTAS = False

INCLUDE_PHYSIO_AGGREGATES = False

INCLUDE_LOCATION_TEMPORAL = False

INCLUDE_LAB_SCORE_AGGREGATES = INCLUDE_LAB_SCORE_SNAPSHOTS or INCLUDE_PHYSIO_AGGREGATES


def create_database(con):

    print("=" * 80)
    print("PHASE 1: LOADING ALL DATA INTO DUCKDB DATABASE")
    print("=" * 80)
    print(f"Database path: {DB_PATH}")
    print("This is a one-time operation. Subsequent runs will skip this step.")
    print()
    sys.stdout.flush()

    con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [row[0] for row in con.fetchall()]

    if 'admissions' in existing_tables and 'chartevents' in existing_tables:
        print("Database already loaded. Verifying...")
        con.execute("SELECT COUNT(*) FROM chartevents")
        count = con.fetchone()[0]
        print(f"  chartevents: {count:,} rows")
        if count > 400000000:
            print("Database verified. Skipping data load.")
            return
        else:
            print("Database incomplete. Reloading...")

    print("\n--- Loading HOSP module ---")
    sys.stdout.flush()

    tables_hosp = [
        ('admissions', 'admissions.csv.gz'),
        ('transfers', 'transfers.csv.gz'),
        ('labevents', 'labevents.csv.gz'),
        ('emar', 'emar.csv.gz'),
        ('microbiologyevents', 'microbiologyevents.csv.gz'),
        ('d_labitems', 'd_labitems.csv.gz'),
    ]

    for table_name, filename in tables_hosp:
        filepath = f"{HOSP_PATH}/{filename}"
        print(f"  Loading {table_name}...", end=" ", flush=True)
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
        con.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = con.fetchone()[0]
        print(f"{count:,} rows")
        sys.stdout.flush()

    print("\n--- Loading ICU module ---")
    sys.stdout.flush()

    tables_icu = [
        ('icustays', 'icustays.csv.gz'),
        ('chartevents', 'chartevents.csv.gz'),
        ('inputevents', 'inputevents.csv.gz'),
        ('outputevents', 'outputevents.csv.gz'),
        ('procedureevents', 'procedureevents.csv.gz'),
        ('d_items', 'd_items.csv.gz'),
    ]

    for table_name, filename in tables_icu:
        filepath = f"{ICU_PATH}/{filename}"
        print(f"  Loading {table_name}...", end=" ", flush=True)
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
        con.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = con.fetchone()[0]
        print(f"{count:,} rows")
        sys.stdout.flush()

    print("\n--- Loading ED module ---")
    sys.stdout.flush()

    tables_ed = [
        ('edstays', 'edstays.csv.gz'),
        ('vitalsign', 'vitalsign.csv.gz'),
        ('triage', 'triage.csv.gz'),
        ('pyxis', 'pyxis.csv.gz'),
    ]

    for table_name, filename in tables_ed:
        filepath = f"{ED_PATH}/{filename}"
        print(f"  Loading {table_name}...", end=" ", flush=True)
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
        con.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = con.fetchone()[0]
        print(f"{count:,} rows")
        sys.stdout.flush()

    print("\n--- Creating indexes for fast lookups ---")
    sys.stdout.flush()

    indexes = [
        ("idx_admissions_hadm", "admissions", "hadm_id"),
        ("idx_transfers_hadm", "transfers", "hadm_id"),
        ("idx_labevents_hadm", "labevents", "hadm_id"),
        ("idx_emar_hadm", "emar", "hadm_id"),
        ("idx_micro_hadm", "microbiologyevents", "hadm_id"),
        ("idx_icustays_hadm", "icustays", "hadm_id"),
        ("idx_icustays_stay", "icustays", "stay_id"),
        ("idx_chartevents_stay", "chartevents", "stay_id"),
        ("idx_inputevents_stay", "inputevents", "stay_id"),
        ("idx_outputevents_stay", "outputevents", "stay_id"),
        ("idx_procedureevents_stay", "procedureevents", "stay_id"),
        ("idx_edstays_hadm", "edstays", "hadm_id"),
        ("idx_edstays_stay", "edstays", "stay_id"),
        ("idx_vitalsign_stay", "vitalsign", "stay_id"),
        ("idx_triage_stay", "triage", "stay_id"),
        ("idx_pyxis_stay", "pyxis", "stay_id"),
    ]

    for idx_name, table, column in indexes:
        print(f"  Creating {idx_name}...", end=" ", flush=True)
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
            print("done")
        except Exception as e:
            print(f"skipped ({e})")
        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("DATABASE LOADING COMPLETE")
    print("=" * 80)
    sys.stdout.flush()


def format_timestamp(ts):
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    return ts.strftime('%Y-%m-%dT%H:%M:%S.000+00:00')

def add_string_attr(parent, key, value):
    if value is not None:
        attr = ET.SubElement(parent, 'string')
        attr.set('key', key)
        attr.set('value', str(value))

def add_date_attr(parent, key, value):
    ts = format_timestamp(value)
    if ts:
        attr = ET.SubElement(parent, 'date')
        attr.set('key', key)
        attr.set('value', ts)

def add_float_attr(parent, key, value):
    if value is not None:
        try:
            attr = ET.SubElement(parent, 'float')
            attr.set('key', key)
            attr.set('value', str(float(value)))
        except (ValueError, TypeError):
            pass

def add_int_attr(parent, key, value):
    if value is not None:
        try:
            attr = ET.SubElement(parent, 'int')
            attr.set('key', key)
            attr.set('value', str(int(value)))
        except (ValueError, TypeError):
            pass

def attach_location_temporal(event, timestamp, unit_label):
    if not INCLUDE_LOCATION_TEMPORAL:
        return
    if not isinstance(timestamp, datetime):
        return
    is_icu = 1.0 if unit_label == 'ICU' else 0.0
    is_ed = 1.0 if unit_label == 'ED' else 0.0
    add_datastream_reading(event, 'location_monitor', 'unit_ICU', timestamp, is_icu, None, 'sensor', 'discrete')
    add_datastream_reading(event, 'location_monitor', 'unit_ED', timestamp, is_ed, None, 'sensor', 'discrete')
    add_datastream_reading(event, 'temporal_monitor', 'hour_of_day', timestamp, float(timestamp.hour), None, 'sensor', 'discrete')
    add_datastream_reading(event, 'temporal_monitor', 'day_of_week', timestamp, float(timestamp.weekday()), None, 'sensor', 'discrete')


def add_datastream_reading(parent, source, reading_type, timestamp, value, unit=None, device_type='sensor', procedure_type='discrete'):
    datastream = None
    for child in parent:
        if child.get('key') == 'stream:datastream':
            datastream = child
            break

    if datastream is None:
        datastream = ET.SubElement(parent, 'list')
        datastream.set('key', 'stream:datastream')

    point = ET.SubElement(datastream, 'list')
    point.set('key', 'stream:point')

    point.set('stream:source', source)

    point.set('stream:system_type', 'sosa:Sensor' if device_type == 'sensor' else 'sosa:Actuator')
    point.set('stream:interaction_type', 'sosa:Observation' if device_type == 'sensor' else 'sosa:Actuation')

    id_elem = ET.SubElement(point, 'string')
    id_elem.set('key', 'stream:id')
    id_elem.set('value', str(reading_type))

    ts = format_timestamp(timestamp)
    if ts:
        ts_elem = ET.SubElement(point, 'date')
        ts_elem.set('key', 'stream:timestamp')
        ts_elem.set('value', ts)

    if value is not None:
        if isinstance(value, (int, float)):
            val_elem = ET.SubElement(point, 'float')
            val_elem.set('key', 'stream:value')
            val_elem.set('value', str(float(value)))
        else:
            val_elem = ET.SubElement(point, 'string')
            val_elem.set('key', 'stream:value')
            val_elem.set('value', str(value))

    if unit or procedure_type:
        meta = ET.SubElement(point, 'list')
        meta.set('key', 'stream:meta')
        if unit:
            unit_elem = ET.SubElement(meta, 'string')
            unit_elem.set('key', 'unit')
            unit_elem.set('value', str(unit))
        if procedure_type:
            proc_elem = ET.SubElement(meta, 'string')
            proc_elem.set('key', 'procedure_type')
            proc_elem.set('value', procedure_type)

def create_xes_header(log_name):
    log = ET.Element('log')
    log.set('xes.version', '1.0')
    log.set('xmlns:stream', 'https://cpee.org/datastream/datastream.xesext')
    log.set('xmlns', 'http://code.deckfour.org/xes')
    log.set('xes.creator', 'MIMIC-IV IoT Event Log Generator')

    extensions = [
        ('Concept', 'concept', 'http://code.deckfour.org/xes/concept.xesext'),
        ('Time', 'time', 'http://code.deckfour.org/xes/time.xesext'),
        ('Organizational', 'org', 'http://code.deckfour.org/xes/org.xesext'),
        ('Sensorstream', 'stream', 'https://cpee.org/datastream/datastream.xesext')
    ]
    for name, prefix, uri in extensions:
        ext = ET.SubElement(log, 'extension')
        ext.set('name', name)
        ext.set('prefix', prefix)
        ext.set('uri', uri)

    global_trace = ET.SubElement(log, 'global')
    global_trace.set('scope', 'trace')
    attr = ET.SubElement(global_trace, 'string')
    attr.set('key', 'concept:name')
    attr.set('value', '__INVALID__')

    global_event = ET.SubElement(log, 'global')
    global_event.set('scope', 'event')
    attr = ET.SubElement(global_event, 'string')
    attr.set('key', 'concept:name')
    attr.set('value', '__INVALID__')
    attr = ET.SubElement(global_event, 'date')
    attr.set('key', 'time:timestamp')
    attr.set('value', '1970-01-01T00:00:00.000+00:00')

    classifier = ET.SubElement(log, 'classifier')
    classifier.set('name', 'Activity')
    classifier.set('keys', 'concept:name')

    add_string_attr(log, 'concept:name', log_name)

    return log

def write_xml_directly(elem, filepath):
    tree = ET.ElementTree(elem)
    with open(filepath, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)


D_ITEMS = {}
D_LABITEMS = {}

def load_dimension_caches(con):
    global D_ITEMS, D_LABITEMS

    print("Loading dimension tables into memory...")

    con.execute("SELECT itemid, label, category, unitname FROM d_items")
    for row in con.fetchall():
        D_ITEMS[row[0]] = (row[1], row[2], row[3])
    print(f"  Loaded {len(D_ITEMS):,} ICU items")

    con.execute("SELECT itemid, label FROM d_labitems")
    for row in con.fetchall():
        D_LABITEMS[row[0]] = row[1]
    print(f"  Loaded {len(D_LABITEMS):,} lab items")
    sys.stdout.flush()

def get_item_info(itemid):
    return D_ITEMS.get(itemid, (f'item_{itemid}', None, None))


SUPER_CATEGORY = {
    'Antibiotics': 'MEDICATION_ABX',
    'Pain/Sedation': 'MEDICATION_SEDATION',
    'Medications': 'MEDICATION_OTHER',
    'Blood Products/Colloids': 'MEDICATION_OTHER',
    'Ingredients': 'MEDICATION_OTHER',
    'Ingredients - general (Not In Use)': 'MEDICATION_OTHER',
    'Access Lines - Invasive': 'VASCULAR_ACCESS',
    'Access Lines - Peripheral': 'VASCULAR_ACCESS',
    'Arterial Line Insertion': 'VASCULAR_ACCESS',
    'CVL Insertion': 'VASCULAR_ACCESS',
    'PICC Line Insertion': 'VASCULAR_ACCESS',
    'PA Line Insertion': 'VASCULAR_ACCESS',
    '1-Intubation/Extubation': 'RESPIRATORY',
    '2-Ventilation': 'RESPIRATORY',
    'Bronchoscopy': 'RESPIRATORY',
    'Intubation': 'RESPIRATORY',
    'Pulmonary': 'RESPIRATORY',
    'Respiratory': 'RESPIRATORY',
    'Cardiovascular': 'CARDIO_SUPPORT',
    'Cardiovascular (Pacer Data)': 'CARDIO_SUPPORT',
    'Cardiovascular (Pulses)': 'CARDIO_SUPPORT',
    'Centrimag': 'CARDIO_SUPPORT',
    'Durable VAD': 'CARDIO_SUPPORT',
    'ECMO': 'CARDIO_SUPPORT',
    'Hemodynamics': 'CARDIO_SUPPORT',
    'Heartware': 'CARDIO_SUPPORT',
    'IABP': 'CARDIO_SUPPORT',
    'Impella': 'CARDIO_SUPPORT',
    'NICOM': 'CARDIO_SUPPORT',
    'PiCCO': 'CARDIO_SUPPORT',
    'Tandem Heart': 'CARDIO_SUPPORT',
    '4-Procedures': 'PROCEDURE',
    '5-Imaging': 'PROCEDURE',
    '6-Cultures': 'PROCEDURE',
    'Lumbar Puncture': 'PROCEDURE',
    'Paracentesis': 'PROCEDURE',
    'Thoracentesis': 'PROCEDURE',
    'Labs': 'OBSERVATION',
    'Toxicology': 'OBSERVATION',
    'Routine Vital Signs': 'OBSERVATION',
    'ApacheII Parameters': 'OBSERVATION',
    'ApacheIV Parameters': 'OBSERVATION',
    'Scores - APACHE II': 'OBSERVATION',
    'Scores - APACHE IV': 'OBSERVATION',
    'Scores - APACHE IV (2)': 'OBSERVATION',
    'Neurological': 'OBSERVATION',
    'GI/GU': 'OBSERVATION',
    'OB-GYN': 'OBSERVATION',
    'Swallow Evaluation': 'OBSERVATION',
    'RDOS': 'OBSERVATION',
    'Nutrition - Enteral': 'NUTRITION_FLUIDS',
    'Nutrition - Parenteral': 'NUTRITION_FLUIDS',
    'Nutrition - Supplements': 'NUTRITION_FLUIDS',
    'Fluids - Other (Not In Use)': 'NUTRITION_FLUIDS',
    'Fluids/Intake': 'NUTRITION_FLUIDS',
    'ZIntake': 'NUTRITION_FLUIDS',
    'Output': 'NUTRITION_FLUIDS',
    'Skin - Assessment': 'SKIN_WOUND',
    'Skin - Impairment': 'SKIN_WOUND',
    'Skin - Incisions': 'SKIN_WOUND',
    'Drains': 'SKIN_WOUND',
    '7-Communication': 'DOCUMENTATION',
    'ADT': 'DOCUMENTATION',
    'Adm History/FHPA': 'DOCUMENTATION',
    'Block Charting Note': 'DOCUMENTATION',
    'Care Plans': 'DOCUMENTATION',
    'Case Management': 'DOCUMENTATION',
    'Family Mtg Note': 'DOCUMENTATION',
    'Generic Proc Note': 'DOCUMENTATION',
    'MD Progress Note': 'DOCUMENTATION',
    'OT Notes': 'DOCUMENTATION',
    'Pastoral Care Note': 'DOCUMENTATION',
    'PatientSafetyInitialNote': 'DOCUMENTATION',
    'RNTriggerNote': 'DOCUMENTATION',
    'Research Enrollment Note': 'DOCUMENTATION',
    'SBNET': 'DOCUMENTATION',
    'Triggers Note': 'DOCUMENTATION',
    '3-Significant Events': 'DOCUMENTATION',
    'Dialysis': 'OTHER_TREATMENT',
    'Restraint/Support Systems': 'OTHER_TREATMENT',
    'Treatments': 'OTHER_TREATMENT',
    'Alarms': 'OTHER_TREATMENT',
    'General': 'OTHER_TREATMENT',
}


def super_category(category):
    if category is None:
        return 'OTHER_TREATMENT'
    return SUPER_CATEGORY.get(category, 'OTHER_TREATMENT')

def get_lab_label(itemid):
    return D_LABITEMS.get(itemid, f'lab_{itemid}')


def build_trace(con, hadm_id):

    con.execute(f"""
        SELECT subject_id, hadm_id, admittime, dischtime, deathtime,
               admission_type, admission_location, discharge_location
        FROM admissions WHERE hadm_id = {hadm_id}
    """)
    adm_info = con.fetchone()
    if not adm_info:
        return None, 0

    subject_id, hadm_id, admittime, dischtime, deathtime, adm_type, adm_loc, disch_loc = adm_info

    trace = ET.Element('trace')
    add_string_attr(trace, 'concept:name', f"hadm_{hadm_id}")
    add_string_attr(trace, 'subject_id', str(subject_id))
    add_string_attr(trace, 'hadm_id', str(hadm_id))

    events = []
    iot_count = 0

    con.execute(f"SELECT stay_id, intime, outtime, arrival_transport, disposition FROM edstays WHERE hadm_id = {hadm_id}")
    ed_stays = con.fetchall()

    for ed_stay in ed_stays:
        ed_stay_id, ed_intime, ed_outtime, transport, disposition = ed_stay

        if ed_intime and INCLUDE_WORKFLOW_MARKERS:
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'ED_Arrival')
            add_date_attr(event, 'time:timestamp', ed_intime)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'arrival_transport', transport)
            events.append((ed_intime, event))

        con.execute(f"SELECT temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint FROM triage WHERE stay_id = {ed_stay_id}")
        triage = con.fetchone()
        if triage and ed_intime:
            temp, hr, rr, o2, sbp, dbp, pain, acuity, complaint = triage
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'Triage_Assessment')
            add_date_attr(event, 'time:timestamp', ed_intime)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'acuity', str(acuity) if acuity else None)
            add_string_attr(event, 'chief_complaint', complaint)

            if temp: add_datastream_reading(event, 'triage_sensor', 'temperature', ed_intime, temp, '°F', 'sensor', 'discrete'); iot_count += 1
            if hr: add_datastream_reading(event, 'triage_sensor', 'heart_rate', ed_intime, hr, 'bpm', 'sensor', 'discrete'); iot_count += 1
            if rr: add_datastream_reading(event, 'triage_sensor', 'resp_rate', ed_intime, rr, '/min', 'sensor', 'discrete'); iot_count += 1
            if o2: add_datastream_reading(event, 'triage_sensor', 'spo2', ed_intime, o2, '%', 'sensor', 'discrete'); iot_count += 1
            if sbp: add_datastream_reading(event, 'triage_sensor', 'sbp', ed_intime, sbp, 'mmHg', 'sensor', 'discrete'); iot_count += 1
            if dbp: add_datastream_reading(event, 'triage_sensor', 'dbp', ed_intime, dbp, 'mmHg', 'sensor', 'discrete'); iot_count += 1
            if pain: add_datastream_reading(event, 'triage_sensor', 'pain_score', ed_intime, pain, None, 'sensor', 'discrete'); iot_count += 1

            attach_location_temporal(event, ed_intime, 'ED')

            triage_time = ed_intime + timedelta(minutes=1) if isinstance(ed_intime, datetime) else ed_intime
            events.append((triage_time, event))

        con.execute(f"SELECT charttime, temperature, heartrate, resprate, o2sat, sbp, dbp, rhythm, pain FROM vitalsign WHERE stay_id = {ed_stay_id} ORDER BY charttime")
        for vital in con.fetchall():
            vtime, temp, hr, rr, o2, sbp, dbp, rhythm, pain = vital
            if vtime:
                event = ET.Element('event')
                add_string_attr(event, 'concept:name', 'ED_Vital_Measurement')
                add_date_attr(event, 'time:timestamp', vtime)
                add_string_attr(event, 'lifecycle:transition', 'complete')
                add_string_attr(event, 'measurement_context', 'ED')

                if temp: add_datastream_reading(event, 'ed_vital_monitor', 'temperature', vtime, temp, '°F', 'sensor', 'discrete'); iot_count += 1
                if hr: add_datastream_reading(event, 'ed_vital_monitor', 'heart_rate', vtime, hr, 'bpm', 'sensor', 'discrete'); iot_count += 1
                if rr: add_datastream_reading(event, 'ed_vital_monitor', 'resp_rate', vtime, rr, '/min', 'sensor', 'discrete'); iot_count += 1
                if o2: add_datastream_reading(event, 'ed_vital_monitor', 'spo2', vtime, o2, '%', 'sensor', 'discrete'); iot_count += 1
                if sbp: add_datastream_reading(event, 'ed_vital_monitor', 'sbp', vtime, sbp, 'mmHg', 'sensor', 'discrete'); iot_count += 1
                if dbp: add_datastream_reading(event, 'ed_vital_monitor', 'dbp', vtime, dbp, 'mmHg', 'sensor', 'discrete'); iot_count += 1
                if rhythm: add_datastream_reading(event, 'ed_vital_monitor', 'rhythm', vtime, rhythm, None, 'sensor', 'discrete'); iot_count += 1
                if pain: add_datastream_reading(event, 'ed_vital_monitor', 'pain_score', vtime, pain, None, 'sensor', 'discrete'); iot_count += 1

                attach_location_temporal(event, vtime, 'ED')

                events.append((vtime, event))

        if INCLUDE_NO_IOT_EVENTS:
            con.execute(f"SELECT charttime, name FROM pyxis WHERE stay_id = {ed_stay_id} ORDER BY charttime")
            for px in con.fetchall():
                ptime, med_name = px
                if ptime:
                    event = ET.Element('event')
                    add_string_attr(event, 'concept:name', 'ED_Medication_Dispensed')
                    add_date_attr(event, 'time:timestamp', ptime)
                    add_string_attr(event, 'lifecycle:transition', 'complete')
                    add_string_attr(event, 'medication', med_name)
                    add_string_attr(event, 'dispenser', 'pyxis')
                    events.append((ptime, event))

        if ed_outtime and INCLUDE_WORKFLOW_MARKERS:
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'ED_Departure')
            add_date_attr(event, 'time:timestamp', ed_outtime)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'disposition', disposition)
            events.append((ed_outtime, event))

    if admittime and INCLUDE_WORKFLOW_MARKERS:
        event = ET.Element('event')
        add_string_attr(event, 'concept:name', 'Hospital_Admission')
        add_date_attr(event, 'time:timestamp', admittime)
        add_string_attr(event, 'lifecycle:transition', 'complete')
        add_string_attr(event, 'admission_type', adm_type)
        add_string_attr(event, 'admission_location', adm_loc)
        events.append((admittime, event))

    if INCLUDE_WORKFLOW_MARKERS:
        con.execute(f"SELECT transfer_id, eventtype, careunit, intime, outtime FROM transfers WHERE hadm_id = {hadm_id} ORDER BY intime")
        for transfer in con.fetchall():
            transfer_id, eventtype, careunit, intime, outtime = transfer
            if intime and eventtype:
                event = ET.Element('event')
                add_string_attr(event, 'concept:name', f'Transfer_{eventtype}')
                add_date_attr(event, 'time:timestamp', intime)
                add_string_attr(event, 'lifecycle:transition', 'complete')
                add_string_attr(event, 'careunit', careunit)
                events.append((intime, event))

    con.execute(f"SELECT stay_id, first_careunit, last_careunit, intime, outtime, los FROM icustays WHERE hadm_id = {hadm_id} ORDER BY intime")
    icu_stays = con.fetchall()

    for icu_stay in icu_stays:
        icu_stay_id, first_unit, last_unit, icu_intime, icu_outtime, los = icu_stay

        if icu_intime and INCLUDE_WORKFLOW_MARKERS:
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'ICU_Admission')
            add_date_attr(event, 'time:timestamp', icu_intime)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'careunit', first_unit)
            add_string_attr(event, 'icu_stay_id', str(icu_stay_id))
            events.append((icu_intime, event))

        con.execute(f"SELECT charttime, itemid, value, valuenum, valueuom FROM chartevents WHERE stay_id = {icu_stay_id} ORDER BY charttime")
        chart_events = con.fetchall()

        hourly_readings = defaultdict(list)
        for ce in chart_events:
            ctime, itemid, value, valuenum, unit = ce
            if ctime and valuenum is not None:
                label, category, default_unit = get_item_info(itemid)
                if RESTRICT_CHARTEVENTS_TO_VITALS:
                    if itemid not in CORE_VITAL_ITEMIDS and category not in CORE_VITAL_CATEGORIES:
                        continue
                hour_key = ctime.replace(minute=0, second=0, microsecond=0) if isinstance(ctime, datetime) else ctime
                hourly_readings[hour_key].append((ctime, label, category, valuenum, unit or default_unit))

        for hour, readings in sorted(hourly_readings.items()):
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'ICU_Vital_Measurement')
            add_date_attr(event, 'time:timestamp', hour)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'org:resource', 'ICU_Nurse')
            add_string_attr(event, 'measurement_context', 'ICU')
            add_int_attr(event, 'reading_count', len(readings))

            for rtime, rlabel, rcategory, rvalue, runit in readings:
                add_datastream_reading(event, 'icu_vital_monitor', rlabel, rtime, rvalue, runit, 'sensor', 'continuous')
                iot_count += 1

            attach_location_temporal(event, hour, 'ICU')

            events.append((hour, event))

        LAB_ITEMS = {
            50912: 'Creatinine', 50971: 'Potassium', 50983: 'Sodium',
            51006: 'BUN', 51221: 'Hematocrit', 51222: 'Hemoglobin',
            51265: 'Platelets', 51301: 'WBC', 50902: 'Chloride',
            50882: 'Bicarbonate', 50931: 'Glucose', 50813: 'Lactate',
            50893: 'Calcium', 50960: 'Magnesium', 50970: 'Phosphate',
            50861: 'ALT', 50862: 'Albumin', 50868: 'AnionGap',
        }
        SCORE_ITEMS = {
            220739: 'GCS_Eye', 223900: 'GCS_Verbal', 223901: 'GCS_Motor',
        }
        CLASS_LAB_PANEL = {
            'MEDICATION_ABX':       ['WBC', 'Lactate', 'BUN', 'Creatinine'],
            'MEDICATION_SEDATION':  [],
            'MEDICATION_OTHER':     ['Sodium', 'Potassium', 'Glucose'],
            'NUTRITION_FLUIDS':     ['Albumin', 'Glucose', 'Magnesium', 'Phosphate'],
        }
        CLASS_SCORE_PANEL = {
            'MEDICATION_SEDATION':  ['GCS_Eye', 'GCS_Verbal', 'GCS_Motor'],
        }
        LAB_ID_BY_NAME = {name: itemid for itemid, name in LAB_ITEMS.items()}
        SCORE_ID_BY_NAME = {name: itemid for itemid, name in SCORE_ITEMS.items()}

        lab_history = defaultdict(list)
        lab_ids_csv = ','.join(str(k) for k in LAB_ITEMS)
        con.execute(f"SELECT charttime, itemid, valuenum FROM labevents WHERE hadm_id = {hadm_id} AND itemid IN ({lab_ids_csv}) AND valuenum IS NOT NULL ORDER BY charttime")
        for ctime, itemid, valuenum in con.fetchall():
            lab_history[itemid].append((ctime, valuenum))
        score_history = defaultdict(list)
        score_ids_csv = ','.join(str(k) for k in SCORE_ITEMS)
        con.execute(f"SELECT charttime, itemid, valuenum FROM chartevents WHERE stay_id = {icu_stay_id} AND itemid IN ({score_ids_csv}) AND valuenum IS NOT NULL ORDER BY charttime")
        for ctime, itemid, valuenum in con.fetchall():
            score_history[itemid].append((ctime, valuenum))

        PHYSIO_AGG_ITEMS = {
            220045: 'HeartRate', 220210: 'RespiratoryRate',
            220277: 'SpO2', 223761: 'TemperatureF',
            220052: 'MAP_arterial', 220181: 'MAP_noninvasive',
        }
        physio_history = defaultdict(list)
        physio_ids_csv = ','.join(str(k) for k in PHYSIO_AGG_ITEMS)
        con.execute(f"SELECT charttime, itemid, valuenum FROM chartevents WHERE stay_id = {icu_stay_id} AND itemid IN ({physio_ids_csv}) AND valuenum IS NOT NULL ORDER BY charttime")
        for ctime, itemid, valuenum in con.fetchall():
            physio_history[itemid].append((ctime, valuenum))

        def _window_max(history, itemid, istart, hours):
            if not isinstance(istart, datetime):
                return None
            cutoff = istart - timedelta(hours=hours)
            values = [v for t, v in history.get(itemid, ()) if t and cutoff <= t <= istart]
            return max(values) if values else None

        def _window_min(history, itemid, istart, hours):
            if not isinstance(istart, datetime):
                return None
            cutoff = istart - timedelta(hours=hours)
            values = [v for t, v in history.get(itemid, ()) if t and cutoff <= t <= istart]
            return min(values) if values else None

        def _two_latest(history, itemid, istart, max_hours=24):
            if not isinstance(istart, datetime):
                return None, None
            cutoff = istart - timedelta(hours=max_hours)
            series = history.get(itemid)
            if not series:
                return None, None
            candidates = [(t, v) for t, v in series if t and cutoff <= t <= istart]
            if len(candidates) < 2:
                return None, None
            candidates.sort(key=lambda x: x[0])
            return candidates[-2][1], candidates[-1][1]

        def _nearest_history(history, item_dict, istart, max_hours=24):
            if not isinstance(istart, datetime):
                return {}
            cutoff = istart - timedelta(hours=max_hours)
            snapshot = {}
            for itemid, label in item_dict.items():
                series = history.get(itemid)
                if not series:
                    continue
                for ctime, value in reversed(series):
                    if ctime is None:
                        continue
                    if ctime > istart:
                        continue
                    if ctime < cutoff:
                        break
                    snapshot[label] = (ctime, value)
                    break
            return snapshot

        con.execute(f"SELECT starttime, endtime, itemid, amount, amountuom, rate, rateuom, ordercategorydescription FROM inputevents WHERE stay_id = {icu_stay_id} ORDER BY starttime")
        for ie in con.fetchall():
            istart, iend, itemid, amount, amountuom, rate, rateuom, order_cat = ie
            if istart:
                label, category, _ = get_item_info(itemid)
                super_cat = super_category(category)

                start_event = ET.Element('event')
                add_string_attr(start_event, 'concept:name', f'START {super_cat}')
                add_date_attr(start_event, 'time:timestamp', istart)
                add_string_attr(start_event, 'lifecycle:transition', 'start')
                add_string_attr(start_event, 'item_label', label)
                add_string_attr(start_event, 'category', category)
                add_string_attr(start_event, 'order_category', order_cat)

                if amount is not None:
                    add_datastream_reading(start_event, 'iv_pump', 'infusion_amount', istart, amount, amountuom, 'actuator', 'continuous')
                    iot_count += 1
                if rate is not None:
                    add_datastream_reading(start_event, 'iv_pump', 'infusion_rate', istart, rate, rateuom, 'actuator', 'continuous')
                    iot_count += 1

                if INCLUDE_LAB_SCORE_SNAPSHOTS:
                    lab_snapshot = _nearest_history(lab_history, LAB_ITEMS, istart, max_hours=24)
                    for lab_name, (ltime, lvalue) in lab_snapshot.items():
                        add_datastream_reading(start_event, 'lab_state', lab_name, ltime, lvalue, None, 'sensor', 'discrete')
                        iot_count += 1

                    score_snapshot = _nearest_history(score_history, SCORE_ITEMS, istart, max_hours=12)
                    for score_name, (stime, svalue) in score_snapshot.items():
                        add_datastream_reading(start_event, 'score_state', score_name, stime, svalue, None, 'sensor', 'discrete')
                        iot_count += 1

                if INCLUDE_LAB_SCORE_DELTAS:
                    for lab_itemid, lab_name in LAB_ITEMS.items():
                        old_v, new_v = _two_latest(lab_history, lab_itemid, istart, max_hours=24)
                        if old_v is not None and new_v is not None:
                            delta = float(new_v) - float(old_v)
                            add_datastream_reading(start_event, 'lab_delta', lab_name, istart, delta, None, 'sensor', 'discrete')
                            iot_count += 1
                    for score_itemid, score_name in SCORE_ITEMS.items():
                        old_v, new_v = _two_latest(score_history, score_itemid, istart, max_hours=12)
                        if old_v is not None and new_v is not None:
                            delta = float(new_v) - float(old_v)
                            add_datastream_reading(start_event, 'score_delta', score_name, istart, delta, None, 'sensor', 'discrete')
                            iot_count += 1

                if INCLUDE_PHYSIO_AGGREGATES:
                    agg_specs = [
                        ('temperature_max_4h', physio_history, 223761, 4, 'max'),
                        ('heart_rate_max_4h', physio_history, 220045, 4, 'max'),
                        ('respiratory_rate_max_4h', physio_history, 220210, 4, 'max'),
                        ('SpO2_min_4h', physio_history, 220277, 4, 'min'),
                        ('MAP_min_4h_art', physio_history, 220052, 4, 'min'),
                        ('MAP_min_4h_nibp', physio_history, 220181, 4, 'min'),
                        ('lactate_max_24h', lab_history, 50813, 24, 'max'),
                        ('WBC_max_24h', lab_history, 51301, 24, 'max'),
                        ('creatinine_max_24h', lab_history, 50912, 24, 'max'),
                    ]
                    for name, hist, iid, win_h, op in agg_specs:
                        val = _window_max(hist, iid, istart, win_h) if op == 'max' else _window_min(hist, iid, istart, win_h)
                        if val is not None:
                            add_datastream_reading(start_event, 'physio_aggregate', name, istart, val, None, 'sensor', 'discrete')
                            iot_count += 1

                attach_location_temporal(start_event, istart, 'ICU')

                if INCLUDE_ORDER_KIND_IOT:
                    for ind_name, ind_value in order_kind_indicators(order_cat).items():
                        add_datastream_reading(start_event, 'order_kind', ind_name, istart, ind_value, None, 'sensor', 'discrete')
                        iot_count += 1

                events.append((istart, start_event))

                if iend and INCLUDE_NO_IOT_EVENTS:
                    end_event = ET.Element('event')
                    add_string_attr(end_event, 'concept:name', f'END {super_cat}')
                    add_date_attr(end_event, 'time:timestamp', iend)
                    add_string_attr(end_event, 'lifecycle:transition', 'complete')
                    add_string_attr(end_event, 'item_label', label)
                    add_string_attr(end_event, 'category', category)
                    events.append((iend, end_event))

        if INCLUDE_NO_IOT_EVENTS:
            con.execute(f"SELECT charttime, itemid, value, valueuom FROM outputevents WHERE stay_id = {icu_stay_id} ORDER BY charttime")
            for oe in con.fetchall():
                otime, itemid, value, unit = oe
                if otime:
                    label, category, default_unit = get_item_info(itemid)
                    event = ET.Element('event')
                    add_string_attr(event, 'concept:name', 'ICU_Output_Measurement')
                    add_date_attr(event, 'time:timestamp', otime)
                    add_string_attr(event, 'lifecycle:transition', 'complete')
                    add_string_attr(event, 'measurement_context', 'ICU_Output')
                    add_string_attr(event, 'measurement_type', label)
                    if value is not None:
                        add_string_attr(event, 'value', str(value))
                    if unit or default_unit:
                        add_string_attr(event, 'unit', unit or default_unit)
                    events.append((otime, event))

        if INCLUDE_PROCEDURE_EVENTS:
            con.execute(f"SELECT starttime, endtime, itemid, value, valueuom FROM procedureevents WHERE stay_id = {icu_stay_id} ORDER BY starttime")
            for proc in con.fetchall():
                pstart, pend, itemid, value, unit = proc
                if pstart:
                    label, category, _ = get_item_info(itemid)
                    super_cat = super_category(category)

                    start_event = ET.Element('event')
                    add_string_attr(start_event, 'concept:name', f'START {super_cat}')
                    add_date_attr(start_event, 'time:timestamp', pstart)
                    add_string_attr(start_event, 'lifecycle:transition', 'start')
                    add_string_attr(start_event, 'item_label', label)
                    add_string_attr(start_event, 'category', category)
                    if value:
                        add_string_attr(start_event, 'value', str(value))
                    if unit:
                        add_string_attr(start_event, 'unit', unit)
                    events.append((pstart, start_event))

                    if pend:
                        end_event = ET.Element('event')
                        add_string_attr(end_event, 'concept:name', f'END {super_cat}')
                        add_date_attr(end_event, 'time:timestamp', pend)
                        add_string_attr(end_event, 'lifecycle:transition', 'complete')
                        add_string_attr(end_event, 'item_label', label)
                        add_string_attr(end_event, 'category', category)
                        events.append((pend, end_event))

        if icu_outtime and INCLUDE_WORKFLOW_MARKERS:
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'ICU_Discharge')
            add_date_attr(event, 'time:timestamp', icu_outtime)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'careunit', last_unit)
            events.append((icu_outtime, event))

    if INCLUDE_NO_IOT_EVENTS:
        con.execute(f"SELECT charttime, itemid, value, valuenum, valueuom, flag FROM labevents WHERE hadm_id = {hadm_id} ORDER BY charttime")
        lab_events = con.fetchall()

        lab_by_hour = defaultdict(list)
        for le in lab_events:
            ltime, itemid, value, valuenum, unit, flag = le
            if ltime:
                hour_key = ltime.replace(minute=0, second=0, microsecond=0) if isinstance(ltime, datetime) else ltime
                label = get_lab_label(itemid)
                lab_by_hour[hour_key].append((ltime, label, valuenum if valuenum is not None else value, unit, flag))

        for hour, results in sorted(lab_by_hour.items()):
            event = ET.Element('event')
            add_string_attr(event, 'concept:name', 'Lab_Results_Received')
            add_date_attr(event, 'time:timestamp', hour)
            add_string_attr(event, 'lifecycle:transition', 'complete')
            add_string_attr(event, 'org:resource', 'Lab')
            add_int_attr(event, 'result_count', len(results))

            labels = [rlabel for _, rlabel, _, _, _ in results if rlabel]
            flags = [rflag for _, _, _, _, rflag in results if rflag]
            if labels:
                add_string_attr(event, 'lab_tests', ';'.join(labels[:20]))
            if flags:
                add_string_attr(event, 'lab_flags', ';'.join(set(flags)))

            events.append((hour, event))

    if INCLUDE_EMAR_EVENTS:
        con.execute(f"SELECT charttime, medication, event_txt FROM emar WHERE hadm_id = {hadm_id} ORDER BY charttime")
        for em in con.fetchall():
            etime, medication, event_txt = em
            if etime and medication:
                event = ET.Element('event')
                if event_txt and event_txt.lower() in ['started', 'restarted']:
                    add_string_attr(event, 'concept:name', 'START MEDICATION_OTHER')
                    add_string_attr(event, 'lifecycle:transition', 'start')
                elif event_txt and event_txt.lower() in ['stopped', 'ended']:
                    add_string_attr(event, 'concept:name', 'END MEDICATION_OTHER')
                    add_string_attr(event, 'lifecycle:transition', 'complete')
                else:
                    add_string_attr(event, 'concept:name', 'MEDICATION_OTHER')
                    add_string_attr(event, 'lifecycle:transition', 'complete')

                add_date_attr(event, 'time:timestamp', etime)
                add_string_attr(event, 'event_type', event_txt)
                add_string_attr(event, 'medication', medication)
                add_string_attr(event, 'item_label', medication)
                events.append((etime, event))

    if INCLUDE_NO_IOT_EVENTS:
        con.execute(f"SELECT charttime, spec_type_desc, test_name, org_name, interpretation FROM microbiologyevents WHERE hadm_id = {hadm_id} ORDER BY charttime")
        for micro in con.fetchall():
            mtime, spec_type, test_name, org_name, interpretation = micro
            if mtime:
                event = ET.Element('event')
                add_string_attr(event, 'concept:name', 'Microbiology_Result')
                add_date_attr(event, 'time:timestamp', mtime)
                add_string_attr(event, 'lifecycle:transition', 'complete')
                add_string_attr(event, 'specimen_type', spec_type)
                add_string_attr(event, 'test_name', test_name)
                if org_name:
                    add_string_attr(event, 'organism', org_name)
                if interpretation:
                    add_string_attr(event, 'interpretation', interpretation)
                events.append((mtime, event))

    if dischtime and INCLUDE_WORKFLOW_MARKERS:
        event = ET.Element('event')
        add_string_attr(event, 'concept:name', 'Hospital_Discharge')
        add_date_attr(event, 'time:timestamp', dischtime)
        add_string_attr(event, 'lifecycle:transition', 'complete')
        add_string_attr(event, 'discharge_location', disch_loc)
        if deathtime:
            add_string_attr(event, 'outcome', 'deceased')
        else:
            add_string_attr(event, 'outcome', 'alive')
        events.append((dischtime, event))

    events.sort(key=lambda x: x[0] if x[0] else datetime.min)

    if INCLUDE_CASE_CONTEXT_IOT and isinstance(admittime, datetime):
        prev_ts = None
        for idx, (ts, event) in enumerate(events):
            if not isinstance(ts, datetime):
                continue
            since_admit = max(0.0, (ts - admittime).total_seconds() / 3600.0)
            since_prev = 0.0 if prev_ts is None else max(0.0, (ts - prev_ts).total_seconds() / 3600.0)
            add_datastream_reading(event, 'case_context', 'time_since_admission', ts, since_admit, 'h', 'sensor', 'discrete')
            add_datastream_reading(event, 'case_context', 'event_index', ts, float(idx), None, 'sensor', 'discrete')
            add_datastream_reading(event, 'case_context', 'time_since_last_event', ts, since_prev, 'h', 'sensor', 'discrete')
            iot_count += 3
            prev_ts = ts

    for _, event in events:
        trace.append(event)

    add_int_attr(trace, 'iot_reading_count', iot_count)
    return trace, iot_count


def main():
    print("=" * 80)
    print("MIMIC-IV DataStream XES Generator (Optimized with DuckDB)")
    print("=" * 80)
    sys.stdout.flush()

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print(f"\nConnecting to database: {DB_PATH}")
    con = duckdb.connect(DB_PATH)

    create_database(con)

    load_dimension_caches(con)

    con.execute("SELECT COUNT(DISTINCT hadm_id) FROM admissions")
    total_admissions = con.fetchone()[0]
    print(f"\nTotal hospital admissions to process: {total_admissions:,}")

    print("\n" + "=" * 80)
    print("PHASE 2: GENERATING XES FILES")
    print("=" * 80)
    sys.stdout.flush()

    if MAX_ADMISSIONS is not None and MAX_ADMISSIONS < total_admissions:
        print(f"NOTE: capping regeneration at MAX_ADMISSIONS = {MAX_ADMISSIONS:,} of "
              f"{total_admissions:,} available admissions")
        total_admissions = MAX_ADMISSIONS

    processed = 0
    file_num = 0
    total_traces = 0
    total_iot_readings = 0
    total_errors = 0

    while processed < total_admissions:
        file_num += 1
        batch_start = processed + 1
        batch_end = min(processed + ADMISSIONS_PER_FILE, total_admissions)

        output_file = os.path.join(OUTPUT_PATH, f'MIMIC_IV_DataStream_Part_{file_num:04d}.xes')

        if os.path.exists(output_file):
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"\n--- File {file_num}: Admissions {batch_start:,} - {batch_end:,} --- SKIPPED (exists, {file_size_mb:.1f} MB)")
            processed += ADMISSIONS_PER_FILE
            continue

        print(f"\n--- File {file_num}: Admissions {batch_start:,} - {batch_end:,} ---")
        sys.stdout.flush()

        con.execute(f"SELECT DISTINCT hadm_id FROM admissions ORDER BY hadm_id LIMIT {ADMISSIONS_PER_FILE} OFFSET {processed}")
        hadm_ids = [row[0] for row in con.fetchall()]

        if not hadm_ids:
            break

        log = create_xes_header(f'MIMIC-IV_IoT_EventLog_Part_{file_num}')

        traces_in_file = 0
        iot_in_file = 0
        errors_in_file = 0

        for i, hadm_id in enumerate(hadm_ids):
            try:
                trace, iot_count = build_trace(con, hadm_id)
                if trace is not None and len(trace) > 0:
                    log.append(trace)
                    traces_in_file += 1
                    iot_in_file += iot_count

                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(hadm_ids)} | {traces_in_file} traces | {iot_in_file:,} IoT readings")
                    sys.stdout.flush()

            except Exception as e:
                errors_in_file += 1
                if errors_in_file <= 3:
                    print(f"  Error on hadm_id {hadm_id}: {str(e)[:80]}")
                continue

        print(f"  Saving {output_file}...")
        sys.stdout.flush()

        write_xml_directly(log, output_file)

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Saved: {traces_in_file} traces, {iot_in_file:,} IoT readings, {file_size_mb:.1f} MB")
        if errors_in_file > 0:
            print(f"  Errors: {errors_in_file}")
        sys.stdout.flush()

        processed += len(hadm_ids)
        total_traces += traces_in_file
        total_iot_readings += iot_in_file
        total_errors += errors_in_file

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total files: {file_num}")
    print(f"Total traces: {total_traces:,}")
    print(f"Total IoT readings: {total_iot_readings:,}")
    print(f"Total errors: {total_errors}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Database: {DB_PATH}")
    print("=" * 80)

    con.close()

if __name__ == "__main__":
    main()
