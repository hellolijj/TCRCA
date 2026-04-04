# TCRCA: Topology-Guided Causal Inference for Root Cause Analysis in Microservices

Official reference implementation for the TCRCA paper — a topology-guided causal inference framework for root cause analysis in microservices.

## Overview

TCRCA is a comprehensive approach that combines multiple advanced techniques:

- **Multi-modal Service Call Graph (MSCG)** — builds rich service dependency representations
- **Graph Autoencoder (GAE)** — performs unsupervised anomaly detection
- **Multi-scale Topology-Guided Inference (MS-TGI)** — learns causal relationships guided by service topology
- **Failure Propagation Graphs (FPG)** — outputs interpretable failure propagation paths
- **Anomaly-aware Causal Path Search (A-CPS)** — ranks root causes with explainable paths

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**[Installation](#installation) · [Datasets](#datasets) · [Metrics](#evaluation-metrics) · [Quick Start](#quick-start) · [Citation](#citation)**

---

## Method Overview

The TCRCA pipeline consists of four main stages:

| Stage | Description | Paper Reference | Implementation |
|--------|-------------|-----------------|----------------|
| **MSCG Construction** | Multi-modal service call graph with GAT embeddings | Sec. IV-B, Eq. (7)–(17) | [`tcrca/mscg.py`](tcrca/mscg.py), [`tcrca/embeddings.py`](tcrca/embeddings.py), [`tcrca/gat.py`](tcrca/gat.py) |
| **Anomaly Detection** | Graph autoencoder for unsupervised detection | Sec. IV-C, Eq. (18)–(23) | [`tcrca/gae.py`](tcrca/gae.py) |
| **Causal Learning** | Multi-scale topology-guided inference | Sec. IV-D, Eq. (24)–(28) | [`tcrca/mstgi.py`](tcrca/mstgi.py) |
| **Root Cause Ranking** | FPG construction with A-CPS path search | Sec. IV-E, Eq. (29)–(37), Alg. 1 | [`tcrca/fpg.py`](tcrca/fpg.py) |

> **Architecture:** See [`img/architecutre.png`](./img/architecutre.png) for the complete system architecture.

---

## Installation

### Prerequisites

- Python **3.10+**
- PyTorch (CPU or CUDA)

### Setup

```bash
# Clone the repository
git clone https://github.com/hellolijj/TCRCA.git
cd TRCRA

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Datasets

TCRCA supports two public benchmarks for evaluation:

| Benchmark | Type | Description | Data Source |
|-----------|------|-------------|-------------|
| **Train Ticket** | Controlled Testbed | Chaos injection with OpenTelemetry traces and trace-level root labels | [FudanSELab/train-ticket](https://github.com/FudanSELab/train-ticket) |
| **AIOps Live Benchmark** | Real-world Workload | Large-scale cloud workload data | [aiops.cn — AIOps Live Benchmark](https://www.aiops.cn/aiops-live-benchmark) |

### Dataset Directory Structure

Both datasets use the same multi-modal layout. Place exports under:

- `datasets/train_ticket/` for Train Ticket
- `datasets/aiops_live/` for AIOps Live Benchmark

```
<dataset_root>/
└── data/
    ├── service_dependencies.json     # Optional; same JSON schema as Train Ticket
    └── <snapshot>/                   # Train Ticket: TT.* ; AIOps: any folder name
        ├── spans.json                # Distributed trace spans
        ├── logs.json                 # Service logs
        └── metrics/                  # Time-series metrics
            ├── ts-<service>-service.csv
            └── ...
```

### Data Loading

- **Train Ticket:** Uses `TrainTicketDataSource` with `TT.*` snapshot convention
- **AIOps Live:** Uses `AIOpsLiveBenchmarkDataSource` which automatically selects the latest subdirectory containing `spans.json` (lexicographic order)

---

## Evaluation metrics

Reporting follows **Section V** of the paper.

### Localization

- **A@k (Top-k accuracy):** fraction of fault cases where the ground-truth root service appears in the model’s top-*k* ranked candidates.
- **Avg@5:** average of **A@1 … A@5**, as in the paper (Eq. (41) with *k* = 5).
- **MRR:** mean reciprocal rank of the true root in the ranked list.

**Ranking rule (aligned with Eq. (32)–(33)):** candidates are ordered by **RScore on the FPG node set *V′*** (descending). If the true root is not in *V′*, the rank is treated as missing and contributes **0** to MRR / A@k, consistent with strict evaluation on the FPG candidate set.

The CLI summary prints **`A@1`, `A@3`, `A@5`, `Avg@5`, `MRR`**. Aliases `hit@1` / `hit@3` / `hit@5` duplicate **A@*** for backward compatibility.

### FPG / alert-style diagnostics

Per trace, the runner logs proxies comparable to Table VI:

| Symbol (paper) | Implemented quantity |
|----------------|----------------------|
| Avg. Alerts | `n_raw_alerts`: count of services with node anomaly score *sᵢ* > *τ* (fallback: argmax if none exceed *τ*) |
| Avg. Chains | `n_compact_paths`: number of **A-CPS** paths returned |
| Compression (%) | `(n_raw_alerts − n_compact_paths) / n_raw_alerts × 100` |
| Top-3 Cov. (%) | `top3_path_coverage_pct`: share of anomalous services that appear on at least one of the **top-3** paths by path score |

Batch means are reported as `mean_n_raw_alerts`, `mean_compression_pct`, `mean_top3_path_coverage_pct`, etc.

### Optional

- **Edge Jaccard / F1** vs. `ground_truth_edges` in the manifest (path reconstruction studies).
- **Trace-level alert confusion** if the manifest includes `is_faulty_trace` and `trace_anomaly_threshold` (Task 1, Eq. (4)).

### Training vs. evaluation

The paper trains on **normal** multimodal graphs with the **unsupervised** reconstruction loss (Eq. (21)); **root-cause labels are evaluation-only**. The bundled `run_evaluation.py` currently **fits per manifest row** on the corresponding trace window for convenience; for strict reproduction of the paper protocol, use **normal-trace pre-training** and **frozen inference** on fault traces (see comments in `tcrca/eval/runner.py`).

---

## Quick start

### Single run (dominant `trace_id` in tables)

```bash
python main.py
```

Uses `datasets/train_ticket` via `TrainTicketDataSource` (adjust path in `main.py` if needed).

### Batch evaluation (JSONL manifest)

**Train Ticket:**

```bash
python experiments/run_evaluation.py ^
  --manifest experiments/eval_manifest.example.jsonl ^
  --dataset train_ticket ^
  --epochs 40
```

**AIOps Live:**

```bash
python experiments/run_evaluation.py ^
  --manifest experiments/eval_manifest.example.jsonl ^
  --dataset aiops_live ^
  --dataset-root datasets/aiops_live ^
  --epochs 40
```

Summary JSON includes **A@k**, **Avg@5**, **MRR**, and Table VI–style means; per-case lines are written to `*.per_case.jsonl`.

### Python API

```python
from tcrca.data import DataSourceManager, TrainTicketDataSource
from tcrca import TCRCAConfig, run_tcrca_from_tables

manager = DataSourceManager()
manager.add_data_source(TrainTicketDataSource("datasets/train_ticket"))
data = manager.get_data()

cfg = TCRCAConfig()
out = run_tcrca_from_tables(data, cfg, train_epochs=40)
print(out.trace_anomaly, out.fpg.top_roots, out.fpg.acps_paths)
```

Hyperparameters match the paper’s notation where possible (`lambda_structure` = *λₛ*, `tau_anomaly` = *τ*, …); see `tcrca/config.py`.

---

## Project structure

```
tcrca/
  data/sources.py       # Train Ticket, AIOps Live, Hipo-style loaders
  eval/metrics.py       # A@k, Avg@k, Table VI–style stats, path metrics
  eval/runner.py        # Manifest evaluation
  pipeline.py           # End-to-end TCRCA
  ...
experiments/run_evaluation.py
tests/
```

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Citation

If you use this code, please cite the TCRCA paper (use the final venue/year from your camera-ready PDF).

```bibtex
@article{tcrca2026,
  title   = {Topology-Guided Causal Inference for Root Cause Analysis in Microservices},
  author  = {Li, Junjun and Ying, Shi},
  journal = {TODO},
  year    = {2026},
}
```

---

## Contact

- Junjun Li: [lijunjun@whu.edu.cn](mailto:lijunjun@whu.edu.cn)  
- Shi Ying (corresponding): [yingshi@whu.edu.cn](mailto:yingshi@whu.edu.cn)  
- Institution: School of Computer Science, Wuhan University, China  

Issues: GitHub issue tracker for this repository.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
