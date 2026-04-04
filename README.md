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

### Dependencies

TCRCA requires the following dependencies:

| Dependency | Version | Purpose |
|------------|---------|---------|
| numpy | >=1.21.0 | Numerical computing |
| pandas | >=1.3.0 | Data manipulation |
| networkx | >=2.6.0 | Graph operations |
| torch | >=1.9.0 | Deep learning framework |
| pytest | >=6.0.0 | Testing framework |
| matplotlib | >=3.5.0 | Visualization (optional) |

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

## Evaluation Metrics

### Localization Metrics

TCRCA reports standard root cause localization metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| **A@k** (Top-k accuracy) | Fraction of fault cases where ground-truth root service appears in top-*k* candidates | Eq. (40) |
| **Avg@5** | Average of A@1 through A@5 | Eq. (41) with *k* = 5 |
| **MRR** | Mean reciprocal rank of true root in ranked list | — |

**Ranking Rule:** Candidates are ordered by RScore on the FPG node set *V′* (descending). If the true root is not in *V′*, it contributes 0 to MRR/A@k (strict evaluation on FPG candidate set).

The CLI outputs: `A@1`, `A@3`, `A@5`, `Avg@5`, `MRR` (with `hit@1/3/5` aliases for backward compatibility).

### FPG / Alert-style Diagnostics

Per-trace diagnostics comparable to Table VI in the paper:

| Metric | Implementation | Description |
|--------|----------------|-------------|
| **Avg. Alerts** | `n_raw_alerts` | Count of services with anomaly score *sᵢ* > *τ* (fallback: argmax) |
| **Avg. Chains** | `n_compact_paths` | Number of A-CPS paths returned |
| **Compression (%)** | `(n_raw_alerts − n_compact_paths) / n_raw_alerts × 100` | Alert reduction ratio |
| **Top-3 Cov. (%)** | `top3_path_coverage_pct` | Anomalous services on top-3 paths by path score |

Batch means are reported as `mean_n_raw_alerts`, `mean_compression_pct`, `mean_top3_path_coverage_pct`, etc.

### Optional Metrics

- **Edge Jaccard / F1** — vs. `ground_truth_edges` in manifest (path reconstruction studies)
- **Trace-level alert confusion** — if manifest includes `is_faulty_trace` and `trace_anomaly_threshold` (Task 1, Eq. (4))

### Training vs Evaluation Protocol

The paper uses **unsupervised training** on normal multimodal graphs (reconstruction loss, Eq. (21)), with root-cause labels used **only for evaluation**.

The bundled `run_evaluation.py` fits per manifest row on the corresponding trace window for convenience. For strict paper protocol:
1. Pre-train on normal traces
2. Use frozen inference on fault traces
3. See comments in [`tcrca/eval/runner.py`](tcrca/eval/runner.py) for details

---

## Quick Start

### Single Run

Run the default example (dominant `trace_id` in tables):

```bash
python main.py
```

Uses `datasets/train_ticket` via `TrainTicketDataSource` (adjust path in `main.py` if needed).

### Batch Evaluation

Evaluate using a JSONL manifest:

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

Output includes:
- Summary JSON with A@k, Avg@5, MRR, and Table VI–style means
- Per-case lines written to `*.per_case.jsonl`

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

Hyperparameters match the paper's notation where possible (`lambda_structure` = *λₛ*, `tau_anomaly` = *τ*, …); see [`tcrca/config.py`](tcrca/config.py).

---

## Project Structure

```
TRCRA/
├── tcrca/               # Core implementation
│   ├── data/            # Data loading and preprocessing
│   │   ├── __init__.py  # Package initialization
│   │   └── sources.py   # Data source implementations (Train Ticket, AIOps Live)
│   ├── eval/            # Evaluation utilities
│   │   ├── __init__.py  # Package initialization
│   │   ├── metrics.py   # Evaluation metrics (A@k, Avg@k, MRR, etc.)
│   │   └── runner.py    # Manifest-based evaluation runner
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration settings and hyperparameters
│   ├── embeddings.py    # Service embedding utilities
│   ├── fpg.py           # Failure Propagation Graph construction
│   ├── gae.py           # Graph Autoencoder for anomaly detection
│   ├── gat.py           # Graph Attention Network for node embeddings
│   ├── mscg.py          # Multi-modal Service Call Graph construction
│   ├── mstgi.py         # Multi-scale Topology-Guided Inference
│   ├── pipeline.py      # End-to-end TCRCA pipeline
│   └── tensorize.py     # Data tensorization utilities
├── experiments/         # Experimental scripts
│   ├── eval_manifest.example.jsonl  # Example evaluation manifest
│   └── run_evaluation.py            # Batch evaluation script
├── figure/              # Visualization scripts
│   ├── case.py          # Case study visualization
│   ├── windows1.py      # Window analysis visualization
│   └── windows2.py      # Additional window analysis
├── img/                 # Images and diagrams
│   └── architecutre.png # System architecture diagram
├── tests/               # Test suite
│   ├── test_eval_metrics.py      # Tests for evaluation metrics
│   └── test_tcrca_pipeline.py    # Tests for TCRCA pipeline
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
├── main.py              # Main entry point
└── trace_graph.html     # Trace graph visualization
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

- **Junjun Li:** [lijunjun@whu.edu.cn](mailto:lijunjun@whu.edu.cn)
- **Shi Ying** (corresponding): [yingshi@whu.edu.cn](mailto:yingshi@whu.edu.cn)
- **Institution:** School of Computer Science, Wuhan University, China

**Issues:** GitHub issue tracker for this repository.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
