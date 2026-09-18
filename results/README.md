# Experimental results

Aggregated results backing every figure and table in the paper. Accuracy values are
three-seed means at injection rate 0.10 unless a file carries its own rate/seed columns.

| File | Contents | Backs |
|------|----------|-------|
| `per_pes_metrics.csv` | Per-PES, per-method mean precision / recall / F1 (with best-seed precision/F1) for all 20 PESs and 7 methods. | Mean-precision and mean-recall bar charts, the F1 heatmap, the critical-difference diagram, the coverage/precision figure, and the per-PES precision table in the top-level README. |
| `rate_sensitivity.csv` | MissMend precision / recall / F1 per dataset across the six injection rates (0.05–0.30), three seeds. | Injection-rate sensitivity figure. |
| `hyperparameter_sensitivity/` | Five sweep files (`sensitivity_alpha`, `sensitivity_latent_dim`, `sensitivity_warmup_events`, `sensitivity_window_size`, `sensitivity_n_reliable`): metric vs swept value, per dataset, per seed. | Hyperparameter-sensitivity figures (F1 and committed precision). |
| `natural_missingness.csv` | Per-dataset natural recovery rate (mean, std) and mean composite confidence on committed vs abstained events, for the four PESs with real missing labels. | Natural-missingness figure. |
| `ablation.csv` | Per-seed precision / recall / F1 for the leave-one-out component ablation on the four PESs with full arm coverage. `component_disabled` ∈ {`full`, `no_iot_sensor`, `no_warmup_repair`, `no_abstention`}. | Component-ablation figure (ΔF1 = arm − `full`). |
| `operational_metrics.csv` | Runtime, mean and p99 per-event latency, and mean resident memory per PES, single-thread CPU. | Operational-feasibility table. |
| `scalability_cyberseciot.json` | MissMend-only scalability run on CybersecIoT prefixes of 100k / 500k / 1M events: per-event latency, bounded working-set sizes (clusters, cache prefixes), and resident memory alongside the retained-output length. | Scalability paragraph. |

Notes:
- Dataset labels match the paper (e.g. `EnvPermits` for the environmental-permit receipt-phase log).
- CybersecIoT and MIMIC-IV metrics are on the evaluated subsets described in the paper (first 100,000 CybersecIoT events; the top-18-activity subset of the first 500,000 MIMIC-IV events).
