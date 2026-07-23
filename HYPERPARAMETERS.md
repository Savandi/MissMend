# MissMend — Hyperparameter Specification

This document accompanies the paper and provides the complete hyperparameter
value sets, the selected per-dataset settings, and the architectural constants
referenced in the *Implementation details and hyperparameters* subsection.

The authoritative source is [`config/default_config.py`](config/default_config.py):
`DEFAULT_CONFIG` holds the shared defaults and `DATASET_CONFIGS` holds the
per-dataset overrides layered on top of it. `get_config(dataset_name)` returns
the merged configuration actually used at runtime. The tables below are a
human-readable rendering of that file.

---

## 1. Selection procedure

Hyperparameters were selected through a two-stage procedure:

1. **Coarse search (single seed).** A grid search over the discrete value sets
   in Section 3 was run at a single injection seed (`42`) and injection rate
   `0.10` to identify a shortlist of promising configurations per dataset. The
   already-strong axes were held fixed while each remaining axis was swept, so
   the search is a sequence of targeted grids rather than a full cross-product.
2. **Validation and lock (three seeds).** Each shortlisted configuration was
   re-evaluated across the three injection seeds `{42, 123, 2025}`, and the
   configuration attaining the highest mean F1 was fixed as the per-dataset
   operating point used for all reported results.

Every baseline observes the identical persisted injection sets under the same
three-seed schedule, so the comparison isolates model differences from input
variability.

---

## 2. Fixed architectural constants

These are held constant across all datasets (from `DEFAULT_CONFIG`).

| Parameter | Symbol | Value | Meaning |
|---|---|---|---|
| Fuzzifier | m | 2.0 | Gustafson–Kessel fuzzy-membership exponent |
| Entropy exponent | λ | 1.0 | Exponent in the entropy-weighted composite confidence |
| Covariance regulariser | ε | 1e-4 | Diagonal loading for per-cluster covariances |
| SDAE sparsity weight | — | 0.01 | KL sparsity penalty on the latent code |
| SDAE denoising noise | — | 0.1 | Std. dev. of input corruption during pretraining |
| Embedding dimension | — | 8 | Online activity-embedding width |
| Training epochs (warm-up) | — | 100 | SDAE warm-up training epochs |
| Learning rate | — | 0.001 | Adam learning rate |
| Drift severity threshold | δ | 2.0 | Gradual fine-tune vs. selective reset boundary |
| Drift decay factor | — | 0.5 | Proportional cluster-statistic decay on drift |
| Synthetic merge threshold | τ_merge | 2.0 | Symmetric Mahalanobis synthetic-to-real merge |
| Candidate new-member count | n_new | 10 | Candidate-cluster promotion accumulation |
| Candidate persistence | Δt_persist | 5 | Candidate persistence before eligibility |
| Candidate separation | τ_sep | 3.0 | Candidate separation (Mahalanobis) |
| Candidate match threshold | τ_cand | 2.0 | Candidate match threshold (Mahalanobis) |
| Candidate min members | — | 3 | Minimum members for a candidate cluster |
| Candidate max stored | — | 200 | Cap on stored candidate members |
| Retrain interval | — | 1000 | Recent-labelled buffer length |
| Confidence mode | — | entropy | Composite-confidence formulation |

---

## 3. Tuned hyperparameters — value sets explored

The following discrete value sets were explored during the coarse search. Not
every axis was swept for every dataset; the strong axes were held at values
that earlier rounds had established, and the remaining axes were swept over the
sets below.

| Hyperparameter | Symbol | Value set explored | Default |
|---|---|---|---|
| Confidence / matching threshold | α | {0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.30, 0.50} | 0.50 |
| Warm-up window size | — | {n/2 (coverage-aware), fixed per-dataset event counts} | 1000 |
| SDAE latent dimension | d | {32, 48, 64, 128} | 32 |
| SDAE hidden dimensions | — | {[128, 64], [256, 128]} | [128, 64] |
| Cluster eligibility count | n_min | {15, 20, 30} | 15 |
| Cluster reliability count | n_reliable | {40, 50, 80} | 40 |
| Prefix window length | N | {10, 20} | 10 |
| Sequence-head hidden dim | — | {64, 128} | 64 |
| Sequence-head layers | — | {1, 2} | 1 |
| Sequence-head gate | α_seq | {0.05, 0.30} | 0.50 |
| Count-cache order | — | {2, 3} | 3 |

Search ranges were chosen from preliminary tests and standard practice. The
confidence threshold α converged to the framework default (0.50) on every
dataset once the warm-up window was co-tuned.

---

## 4. Selected per-dataset settings

Merged configuration used for all reported results. A dash (—) means the key is
absent from the dataset override and the framework's internal default applies
(the sequence head and count cache are off unless a dataset enables them). IoT
features are disabled on the fourteen non-IoT benchmarks, which carry no sensor
channels.

### IoT-enriched datasets

| Dataset | α | warmup | d | hidden | n_min | n_rel | N | IoT off | seq head | seq dim | seq layers | α_seq | count cache | cc order |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chess | 0.5 | 3000 | 32 | [128,64] | 20 | 50 | 10 | N | — | — | — | — | — | — |
| cottoncandy | 0.5 | 7000 | 32 | [128,64] | 15 | 40 | 10 | N | N | — | — | — | — | — |
| cyberseciot | 0.5 | 2000 | 48 | [256,128] | 15 | 40 | 20 | N | Y | 128 | 2 | 0.3 | Y | 3 |
| mimiciv | 0.5 | 5000 | 64 | [256,128] | 15 | 40 | 10 | N | N | — | — | — | — | — |
| smartfactory | 0.5 | 5000 | 32 | [128,64] | 15 | 40 | 10 | N | N | — | — | — | — | — |
| vienna | 0.5 | 2000 | 64 | [256,128] | 15 | 40 | 20 | N | Y | 128 | 2 | 0.3 | Y | 3 |

### Non-IoT PM benchmarks

| Dataset | α | warmup | d | hidden | n_min | n_rel | N | IoT off | seq head | count cache | cc order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bpic2013_closed | 0.5 | 1000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2013_open | 0.5 | 500 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2013_incidents | 0.5 | 10000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2012 | 0.5 | 40000 | 48 | [256,128] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2017 | 0.5 | 200000 | 64 | [256,128] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2020_domestic | 0.5 | 8000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2020_international | 0.5 | 10000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2020_permit | 0.5 | 12000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2020_prepaid | 0.5 | 3000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| bpic2020_request | 0.5 | 6000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| sepsis | 0.5 | 3000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| road_traffic_fine | 0.5 | 80000 | 48 | [256,128] | 15 | 40 | 10 | Y | Y | Y | 3 |
| wabo | 0.5 | 1500 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |
| helpdesk | 0.5 | 3000 | 32 | [128,64] | 15 | 40 | 10 | Y | Y | Y | 3 |

The per-dataset `data_attribute_keys` (which event attributes enter the data
perspective) are also fixed per dataset; see `DATASET_CONFIGS` in
`config/default_config.py` for the exact attribute lists.
