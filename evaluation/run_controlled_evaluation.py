import sys
import copy
sys.path.insert(0, '.')

from data.parsers.datastream_xes_parser import DataStreamXESParser, DATASET_PATHS
from src.streaming_ml_repair.pipeline.streaming_ml_repair import StreamingMLRepairPipeline
from evaluation.injection.label_injector import MissingLabelInjector
from evaluation.metrics.accuracy_metrics import AccuracyMetrics
from evaluation.metrics.natural_missing_metrics import NaturalMissingMetrics
from config.default_config import get_config

def deep_copy_events(events):
    copied = []
    for e in events:
        from data.parsers.datastream_xes_parser import DataStreamXESEvent
        c = DataStreamXESEvent()
        c.case_id = e.case_id
        c.subprocess_id = e.subprocess_id
        c.concept_name = e.concept_name
        c.timestamp = e.timestamp
        c.lifecycle = e.lifecycle
        c.resource = e.resource
        c.attributes = dict(e.attributes)
        c.sensor_readings = dict(e.sensor_readings)
        c.file_path = e.file_path
        copied.append(c)
    return copied

def run_evaluation(dataset_name, config_key, dataset_path, injection_rates, max_events=None):
    print(f"\n{'=' * 70}")
    print(f"Controlled Evaluation: {dataset_name}")
    print(f"{'=' * 70}")

    print(f"\nParsing {dataset_name}...")
    parser = DataStreamXESParser(dataset_path)
    original_events = parser.parse_all(filter_events=False)
    sensor_vocab = parser.build_sensor_vocabulary()

    if max_events and len(original_events) > max_events:
        original_events = original_events[:max_events]

    labelled_count = sum(1 for e in original_events if e.concept_name and str(e.concept_name).strip())
    print(f"  Total events: {len(original_events)}")
    print(f"  Labelled events: {labelled_count}")
    print(f"  Sensors: {sensor_vocab}")
    print(f"  Activities: {len(parser.activity_vocabulary)}")

    config = get_config(config_key)
    config['sensor_vocabulary'] = sensor_vocab
    config['warmup_events'] = min(config['warmup_events'], len(original_events) // 2)

    for rate in injection_rates:
        print(f"\n--- Injection rate: {rate:.0%} ---")

        events = deep_copy_events(original_events)
        injector = MissingLabelInjector(seed=42)
        events = injector.inject_random(events, rate=rate)
        ground_truth = injector.get_ground_truth()

        print(f"  Injected {len(ground_truth)} missing labels")

        pipeline = StreamingMLRepairPipeline(config)
        results = []
        for result in pipeline.process_stream(iter(events)):
            results.append(result)

        metrics = AccuracyMetrics()
        metrics.evaluate(results, ground_truth, injector.injected_indices)
        summary = metrics.summary()

        print(f"  Results:")
        print(f"    Precision:     {summary['precision']:.4f}")
        print(f"    Recall:        {summary['recall']:.4f}")
        print(f"    F1:            {summary['f1']:.4f}")
        print(f"    Recovery rate: {summary['recovery_rate']:.4f}")
        print(f"    Abstain rate:  {summary['abstain_rate']:.4f}")
        print(f"    TP: {summary['true_positives']}, FP: {summary['false_positives']}, Abstained: {summary['abstained']}")

        per_act = metrics.per_activity_summary()
        top_activities = sorted(per_act.items(), key=lambda x: x[1]['tp'] + x[1]['fn'], reverse=True)[:5]
        if top_activities:
            print(f"  Top activities:")
            for act, scores in top_activities:
                print(f"    {act[:30]:30s}  P={scores['precision']:.2f} R={scores['recall']:.2f} F1={scores['f1']:.2f} (TP={scores['tp']} FP={scores['fp']} FN={scores['fn']})")

        cal = metrics.calibration_summary()
        if cal:
            print(f"  Confidence calibration:")
            for conf_bin, data in cal.items():
                print(f"    conf={conf_bin:.1f}: accuracy={data['accuracy']:.2f} (n={data['total']})")

        nat_metrics = NaturalMissingMetrics()
        nat_metrics.evaluate(
            results=results,
            events=original_events,
            injected_indices=injector.injected_indices,
            dfg=dict(pipeline.dfg) if hasattr(pipeline, 'dfg') else None,
        )
        nat_metrics.print_summary(dataset_name=dataset_name)

if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'chess'
    if len(sys.argv) > 2:
        injection_rates = [float(r) for r in sys.argv[2:]]
    else:
        injection_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    if dataset not in DATASET_PATHS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available (set the path in data/parsers/datastream_xes_parser.py): "
              f"{list(DATASET_PATHS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print("MissMend - Controlled Injection Evaluation")
    print("=" * 70)

    run_evaluation(dataset, dataset, DATASET_PATHS[dataset], injection_rates)
