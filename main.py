"""TCRCA entry point — runs the paper-aligned pipeline in `tcrca/`."""

from __future__ import annotations

import logging

from tcrca.data import DataSourceManager, TrainTicketDataSource
from tcrca import TCRCAConfig, run_tcrca_from_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    manager = DataSourceManager()
    train_ticket_source = TrainTicketDataSource("datasets/train_ticket")
    manager.add_data_source(train_ticket_source)

    data = manager.get_data()
    logger.info("Loaded keys: %s", list(data.keys()))

    for key in ("traces", "metrics", "logs"):
        if key not in data or data[key] is None or getattr(data[key], "empty", True):
            raise ValueError(f"Missing or empty required table: {key}")

    cfg = TCRCAConfig()
    out = run_tcrca_from_tables(data, cfg, train_epochs=40, lr=1e-3)

    id_to_name = {i: str(sid) for i, sid in enumerate(out.bundle.node_ids)}
    logger.info("Trace-level anomaly score (Eq. 23): %.6f", out.trace_anomaly)
    logger.info("Top-%s root-cause candidates (Eq. 33): %s", cfg.root_cause_top_k, out.fpg.top_roots)
    for rank, nid in enumerate(out.fpg.top_roots, start=1):
        logger.info(
            "  #%s service=%s RScore=%.6f",
            rank,
            id_to_name.get(nid, nid),
            out.fpg.root_scores.get(nid, float("nan")),
        )
    logger.info("A-CPS paths (Alg. 1): %s", [(round(sc, 4), [id_to_name.get(x, x) for x in p]) for sc, p in out.fpg.acps_paths])


if __name__ == "__main__":
    main()
