# MissMend

MissMend is a streaming framework for recovering **missing activity labels** in
IoT-enriched process event streams. It processes events one at a time in arrival
order under bounded memory and bounded latency, and for each event with a missing
`concept:name` it either commits a repaired label with a calibrated confidence and
provenance flag, or abstains.

The framework matches each event's multi-perspective representation (control-flow,
event attributes, and IoT sensor readings, encoded jointly by a sparse denoising
autoencoder) against streaming reference clusters maintained per known activity,
with an optional sequence rescue head and count cache for prefix-deterministic
streams. It supports open-vocabulary discovery of unseen activities and ADWIN-based
drift adaptation.

## Installation

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Run the self-contained example (no external data required):

```bash
python examples/run_missmend_minimal.py
```

It streams a short synthetic log through the pipeline and prints the repair
decision for each event.

## Usage on a real log

```python
from src.streaming_ml_repair.pipeline.streaming_ml_repair import StreamingMLRepairPipeline
from config.default_config import get_config
from data.parsers.datastream_xes_parser import DataStreamXESParser

config = get_config("chess")                    # per-dataset config (see HYPERPARAMETERS.md)
pipeline = StreamingMLRepairPipeline(config)

events = DataStreamXESParser("/path/to/log").parse()
for result in pipeline.process_stream(events):
    # result: original_label, recovered_label, confidence, provenance, event
    ...
```

Each event exposes `case_id`, `concept_name`, `timestamp`, `lifecycle`,
`resource`, `attributes`, and `sensor_readings`. A missing label is signalled by
`concept_name` being `None` or empty.

## Repository layout

```
src/streaming_ml_repair/     the framework
  pipeline/                  streaming orchestrator (entry point)
  sdae/                      sparse denoising autoencoder + losses
  clustering/                streaming fuzzy BFR reference clusters
  undiscovered/              candidate-activity BFR (open-vocabulary discovery)
  feature_vector/            control-flow, data, and IoT feature builders
  sequence_head/             LSTM rescue head, count cache, prefix buffer
  streaming_dfg/             online directly-follows graph
  calibration/               temperature / Platt scalers
  drift/                     ADWIN drift detection
config/default_config.py     shared defaults + per-dataset settings
data/parsers/                DataStream XES, BPIC XES, and CSV loaders
evaluation/                  metrics, label injection, and a controlled-evaluation runner
examples/                    minimal runnable example
HYPERPARAMETERS.md           complete hyperparameter value sets and per-dataset settings
```

## Configuration

`config/default_config.py` holds `DEFAULT_CONFIG` (shared defaults) and
`DATASET_CONFIGS` (per-dataset overrides). `get_config(name)` returns the merged
configuration. See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for the full value
sets explored during tuning and the selected per-dataset settings.

## Reproducing the controlled-injection evaluation

`evaluation/run_controlled_evaluation.py` runs the injection protocol (uniform
random label removal at rates 0.05–0.30, three seeds) and reports precision,
recall, coverage, and F1. Point `DATASET_PATHS` in
`data/parsers/datastream_xes_parser.py` at your local copies of the logs first.

## License

See [LICENSE](LICENSE).
