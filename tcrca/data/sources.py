"""
Dataset loaders for public microservice benchmarks (Train Ticket, Hipo-style layout).

Used by `main.py` and evaluation scripts; outputs the pandas tables expected by
`tcrca.tensorize.build_trace_tensor_bundle`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataSource:
    def __init__(self, name: str, data_path: str):
        self.name = name
        self.data_path = Path(data_path)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


class HipoOnlineSystemDataSource(DataSource):
    def __init__(self, data_path: str):
        super().__init__("HipoOnlineSystem", data_path)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        topology_path = self.data_path / "service_topology.json"
        if topology_path.exists():
            with open(topology_path, "r", encoding="utf-8") as f:
                topology = json.load(f)
            services = []
            for service in topology.get("services", []):
                services.append(
                    {
                        "service_id": service.get("id"),
                        "service_name": service.get("name"),
                        "service_type": service.get("type"),
                        "dependencies": service.get("dependencies", []),
                    }
                )
            data["topology"] = pd.DataFrame(services)

        metrics_path = self.data_path / "metrics"
        if metrics_path.exists():
            metrics_data = []
            for metric_file in metrics_path.glob("*.parquet"):
                try:
                    metrics_data.append(pd.read_parquet(metric_file))
                except Exception as e:
                    logger.warning("Error reading metrics file %s: %s", metric_file, e)
            if metrics_data:
                data["metrics"] = pd.concat(metrics_data, ignore_index=True)

        traces_path = self.data_path / "traces"
        if traces_path.exists():
            traces_data = []
            for trace_file in traces_path.glob("*.parquet"):
                try:
                    traces_data.append(pd.read_parquet(trace_file))
                except Exception as e:
                    logger.warning("Error reading traces file %s: %s", trace_file, e)
            if traces_data:
                data["traces"] = pd.concat(traces_data, ignore_index=True)

        logs_path = self.data_path / "logs"
        if logs_path.exists():
            logs_data = []
            for log_file in logs_path.glob("*.parquet"):
                try:
                    logs_data.append(pd.read_parquet(log_file))
                except Exception as e:
                    logger.warning("Error reading logs file %s: %s", log_file, e)
            if logs_data:
                data["logs"] = pd.concat(logs_data, ignore_index=True)

        return data

    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed_data: Dict[str, pd.DataFrame] = {}
        if "topology" in data and not data["topology"].empty:
            topology_df = data["topology"].copy()
            for col in ["service_id", "service_name", "service_type", "dependencies"]:
                if col not in topology_df.columns:
                    topology_df[col] = None
            processed_data["topology"] = topology_df

        if "metrics" in data and not data["metrics"].empty:
            metrics_df = data["metrics"].copy()
            if "timestamp" in metrics_df.columns:
                metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])
            else:
                metrics_df["timestamp"] = pd.to_datetime("now")
            processed_data["metrics"] = metrics_df.sort_values("timestamp")

        if "traces" in data and not data["traces"].empty:
            traces_df = data["traces"].copy()
            for col in ["trace_id", "span_id", "parent_id", "service_id", "start_time", "end_time"]:
                if col not in traces_df.columns:
                    traces_df[col] = None
            if "start_time" in traces_df.columns:
                traces_df["start_time"] = pd.to_datetime(traces_df["start_time"])
            if "end_time" in traces_df.columns:
                traces_df["end_time"] = pd.to_datetime(traces_df["end_time"])
            processed_data["traces"] = traces_df

        if "logs" in data and not data["logs"].empty:
            logs_df = data["logs"].copy()
            if "timestamp" in logs_df.columns:
                logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
            else:
                logs_df["timestamp"] = pd.to_datetime("now")
            processed_data["logs"] = logs_df.sort_values("timestamp")

        return processed_data


class TrainTicketDataSource(DataSource):
    def __init__(self, data_path: str):
        super().__init__("TrainTicket", data_path)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        data_dirs = sorted(list((self.data_path / "data").glob("TT.*")), reverse=True)
        if not data_dirs:
            logger.error("No data directories found under %s", self.data_path / "data")
            return data

        latest_dir = next((p for p in data_dirs if p.is_dir()), None)
        if latest_dir is None:
            logger.error("No valid TT.* directory under %s", self.data_path / "data")
            return data

        logger.info("Using Train Ticket data from %s", latest_dir)

        topology_path = self.data_path / "data" / "service_dependencies.json"
        if topology_path.exists():
            with open(topology_path, "r", encoding="utf-8") as f:
                topology = json.load(f)
            services = []
            for service in topology.get("services", []):
                services.append(
                    {
                        "service_id": service.get("serviceId"),
                        "service_name": service.get("serviceName"),
                        "service_type": service.get("serviceType"),
                        "dependencies": service.get("dependencies", []),
                    }
                )
            data["topology"] = pd.DataFrame(services)

        metrics_path = latest_dir / "metrics"
        if metrics_path.exists():
            metrics_data = []
            for metric_file in metrics_path.glob("*.csv"):
                try:
                    metric_df = pd.read_csv(metric_file)
                    service_name = metric_file.stem.replace("ts-", "").replace("-service", "")
                    metric_df["service_id"] = service_name
                    metrics_data.append(metric_df)
                except Exception as e:
                    logger.warning("Error reading metrics file %s: %s", metric_file, e)
            if metrics_data:
                data["metrics"] = pd.concat(metrics_data, ignore_index=True)

        traces_file = latest_dir / "spans.json"
        if traces_file.exists():
            try:
                data["traces"] = pd.read_json(traces_file)
            except Exception as e:
                logger.error("Error reading traces file %s: %s", traces_file, e)

        logs_file = latest_dir / "logs.json"
        if logs_file.exists():
            try:
                with open(logs_file, "r", encoding="utf-8") as f:
                    logs_data = json.load(f)
                rows = []
                for log_entry in logs_data:
                    rows.append(
                        {
                            "timestamp": log_entry.get("timestamp", ""),
                            "message": log_entry.get("message", ""),
                            "service_id": log_entry.get("service_id", ""),
                        }
                    )
                data["logs"] = pd.DataFrame(rows)
            except Exception as e:
                logger.error("Error reading logs file %s: %s", logs_file, e)

        return data

    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return HipoOnlineSystemDataSource.preprocess(self, data)


class AIOpsLiveBenchmarkDataSource(DataSource):
    """
    AIOps Live Benchmark (Sec. V) — multimodal layout aligned with our Train Ticket adapter.

    Expected layout (place downloaded benchmark under e.g. ``datasets/aiops_live/``)::

        data/
          <snapshot_name>/
            spans.json
            logs.json
            metrics/*.csv
        data/service_dependencies.json   (optional; same schema as Train Ticket)

    Official site: https://www.aiops.cn/aiops-live-benchmark
    """

    def __init__(self, data_path: str):
        super().__init__("AIOpsLive", data_path)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        data_root = self.data_path / "data"
        if not data_root.is_dir():
            logger.error("AIOps Live: directory not found: %s", data_root)
            return data

        candidates = sorted(
            [p for p in data_root.iterdir() if p.is_dir() and (p / "spans.json").exists()],
            key=lambda p: p.name,
            reverse=True,
        )
        latest_dir = candidates[0] if candidates else None
        if latest_dir is None:
            logger.error("AIOps Live: no snapshot folder with spans.json under %s", data_root)
            return data

        logger.info("Using AIOps Live snapshot from %s", latest_dir)

        topology_path = self.data_path / "data" / "service_dependencies.json"
        if topology_path.exists():
            with open(topology_path, "r", encoding="utf-8") as f:
                topology = json.load(f)
            services = []
            for service in topology.get("services", []):
                services.append(
                    {
                        "service_id": service.get("serviceId"),
                        "service_name": service.get("serviceName"),
                        "service_type": service.get("serviceType"),
                        "dependencies": service.get("dependencies", []),
                    }
                )
            data["topology"] = pd.DataFrame(services)

        metrics_path = latest_dir / "metrics"
        if metrics_path.exists():
            metrics_data = []
            for metric_file in metrics_path.glob("*.csv"):
                try:
                    metric_df = pd.read_csv(metric_file)
                    service_name = metric_file.stem.replace("ts-", "").replace("-service", "")
                    metric_df["service_id"] = service_name
                    metrics_data.append(metric_df)
                except Exception as e:
                    logger.warning("Error reading metrics file %s: %s", metric_file, e)
            if metrics_data:
                data["metrics"] = pd.concat(metrics_data, ignore_index=True)

        traces_file = latest_dir / "spans.json"
        if traces_file.exists():
            try:
                data["traces"] = pd.read_json(traces_file)
            except Exception as e:
                logger.error("Error reading traces file %s: %s", traces_file, e)

        logs_file = latest_dir / "logs.json"
        if logs_file.exists():
            try:
                with open(logs_file, "r", encoding="utf-8") as f:
                    logs_data = json.load(f)
                rows = []
                for log_entry in logs_data:
                    rows.append(
                        {
                            "timestamp": log_entry.get("timestamp", ""),
                            "message": log_entry.get("message", ""),
                            "service_id": log_entry.get("service_id", ""),
                        }
                    )
                data["logs"] = pd.DataFrame(rows)
            except Exception as e:
                logger.error("Error reading logs file %s: %s", logs_file, e)

        return data

    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return HipoOnlineSystemDataSource.preprocess(self, data)


class DataSourceManager:
    def __init__(self) -> None:
        self.data_sources: Dict[str, DataSource] = {}

    def add_data_source(self, data_source: DataSource) -> None:
        self.data_sources[data_source.name] = data_source

    def get_data(self, source_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        if source_name:
            if source_name not in self.data_sources:
                raise ValueError(f"Data source {source_name} not found")
            ds = self.data_sources[source_name]
            return ds.preprocess(ds.load_data())

        combined_data: Dict[str, pd.DataFrame] = {
            "metrics": pd.DataFrame(),
            "traces": pd.DataFrame(),
            "logs": pd.DataFrame(),
            "topology": pd.DataFrame(),
        }

        for ds in self.data_sources.values():
            raw = ds.load_data()
            processed = ds.preprocess(raw)
            for key in ["metrics", "traces", "logs", "topology"]:
                if key in processed and not processed[key].empty:
                    if combined_data[key].empty:
                        combined_data[key] = processed[key]
                    else:
                        combined_data[key] = pd.concat(
                            [combined_data[key], processed[key]], ignore_index=True
                        )

        return combined_data
