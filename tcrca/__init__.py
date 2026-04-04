"""
TCRCA — Topology-Guided Causal Inference for Root Cause Analysis in Microservices.

Implementation follows the manuscript structure (Sec. IV): MSCG, GAE, MS-TGI, FPG/A-CPS.
"""

from tcrca.config import TCRCAConfig, TraceTensorBundle
from tcrca.pipeline import TCRCAOutput, TCRCAPipeline, run_tcrca_from_tables
from tcrca.tensorize import build_trace_tensor_bundle

__all__ = [
    "TCRCAConfig",
    "TraceTensorBundle",
    "TCRCAPipeline",
    "TCRCAOutput",
    "build_trace_tensor_bundle",
    "run_tcrca_from_tables",
]
