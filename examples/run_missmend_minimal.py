from __future__ import annotations
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.streaming_ml_repair.pipeline.streaming_ml_repair import StreamingMLRepairPipeline
from config.default_config import get_config
from data.parsers.datastream_xes_parser import DataStreamXESEvent

def synthetic_stream(n: int = 80):
    t0 = datetime(2025, 1, 1)
    activities = ["Cut", "Weld", "Inspect", "Pack"]
    for i in range(n):
        e = DataStreamXESEvent()
        e.case_id = f"case_{i % 6}"
        e.concept_name = None if i % 7 == 0 else activities[i % len(activities)]
        e.timestamp = t0 + timedelta(minutes=i)
        e.lifecycle = "complete"
        e.resource = "line_A"
        e.attributes = {}
        e.sensor_readings = {"temperature": 20.0 + (i % 5), "vibration": float(i % 3)}
        yield e

def main() -> None:
    config = get_config("chess")
    config["warmup_events"] = 30

    pipeline = StreamingMLRepairPipeline(config)

    print(f"{'event':>5} | {'case':<8} | {'observed':<10} | {'repaired':<10} | "
          f"{'conf':>5} | provenance")
    print("-" * 66)
    for i, result in enumerate(pipeline.process_stream(synthetic_stream())):
        observed = result["original_label"] or "—(missing)"
        repaired = result["recovered_label"] or "—(abstain)"
        print(f"{i:>5} | {result['event'].case_id:<8} | {observed:<10} | "
              f"{repaired:<10} | {result['confidence']:>5.2f} | {result['provenance']}")

if __name__ == "__main__":
    main()
