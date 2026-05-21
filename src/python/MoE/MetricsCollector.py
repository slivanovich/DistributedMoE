import timeit
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
import statistics
import json
import csv
import os

from MoE.Metrics import DispatchMetrics, CombineMetrics, ExpertsBlockMetrics, ForwardMetrics, Metrics, ExpertsRunMetrics
from utils.logging_config import get_logger


@dataclass
class MetricsCollectorConfig:
    """Configuration for metrics collection"""

    enable_collection: bool = True
    enable_detailed_timing: bool = True
    enable_transfer_metrics: bool = True
    enable_expert_block_metrics: bool = True
    max_stored_forwards: int = 10000  # Maximum number of forward passes to store
    auto_export_interval: Optional[int] = None  # Auto-export every N forwards (None = disabled)
    export_directory: str = "metrics_output"


class MetricsCollector:
    """
    Centralized metrics collection system for DistMoE operations.

    This class collects detailed performance metrics during MoE forward passes,
    including timing information, transfer throughput, and expert block utilization.
    """

    def __init__(self, config: MetricsCollectorConfig):
        self.config = config
        self.logger = get_logger("dist_moe.metrics")

        # Thread-safe storage for metrics
        self._lock = threading.RLock()

        # Current forward pass metrics being collected
        self._current_forward: Optional[ForwardMetrics] = None
        self._current_forward_start_time: Optional[float] = None
        self._current_setup_start_time: Optional[float] = None

        # Storage for completed metrics
        self._completed_forwards: List[ForwardMetrics] = []
        self._forward_counter = 0

        # Temporary storage for ongoing operations
        self._active_dispatches: Dict[str, float] = {}  # task_id -> start_time
        self._active_combines: Dict[str, float] = {}  # task_id -> start_time
        self._active_expert_runs: Dict[str, Dict[str, Any]] = {}  # task_id -> run_data

        # Aggregated statistics
        self._aggregated_stats: Optional[Dict[str, Any]] = None
        self._stats_dirty = True

        if self.config.enable_collection:
            os.makedirs(self.config.export_directory, exist_ok=True)
            self.logger.info(f"MetricsCollector initialized with export directory: {self.config.export_directory}")

    def start_forward_pass(self) -> None:
        """Start collecting metrics for a new forward pass"""
        if not self.config.enable_collection:
            return

        with self._lock:
            if self._current_forward is not None:
                self.logger.warning("Starting new forward pass while previous one is still active")
                self._finalize_current_forward()

            self._current_forward_start_time = timeit.default_timer()
            self._current_setup_start_time = self._current_forward_start_time
            self._current_forward = ForwardMetrics(
                e2e_latency=0.0,
                setup_latency=0.0,
                avg_dispatches_e2e_latency=0.0,
                avg_combines_e2e_latency=0.0,
                dispatches=[],
                combines=[],
                experts_blocks=[],
            )

            self.logger.debug(f"Started forward pass #{self._forward_counter + 1}")

    def end_setup_phase(self) -> None:
        """Mark the end of the setup phase in the current forward pass"""
        if not self.config.enable_collection or self._current_forward is None:
            return

        with self._lock:
            if self._current_setup_start_time is not None:
                setup_duration = (timeit.default_timer() - self._current_setup_start_time) * 1000
                self._current_forward.setup_latency = setup_duration
                self.logger.debug(f"Setup phase completed in {setup_duration:.2f} ms")

    def start_dispatch(self, task_id: str) -> None:
        """Start timing a dispatch operation"""
        if not self.config.enable_collection or not self.config.enable_detailed_timing:
            return

        with self._lock:
            self._active_dispatches[task_id] = timeit.default_timer()

    def end_dispatch(self, task_id: str) -> None:
        """End timing a dispatch operation"""
        if not self.config.enable_collection or not self.config.enable_detailed_timing:
            return

        with self._lock:
            if task_id in self._active_dispatches:
                duration = (timeit.default_timer() - self._active_dispatches[task_id]) * 1000
                dispatch_metrics = DispatchMetrics(e2e_latency=duration)

                if self._current_forward is not None:
                    self._current_forward.dispatches.append(dispatch_metrics)

                del self._active_dispatches[task_id]
                self.logger.debug(f"Dispatch {task_id[:8]} completed in {duration:.2f} ms")

    def start_combine(self, task_id: str) -> None:
        """Start timing a combine operation"""
        if not self.config.enable_collection or not self.config.enable_detailed_timing:
            return

        with self._lock:
            self._active_combines[task_id] = timeit.default_timer()

    def end_combine(self, task_id: str) -> None:
        """End timing a combine operation"""
        if not self.config.enable_collection or not self.config.enable_detailed_timing:
            return

        with self._lock:
            if task_id in self._active_combines:
                duration = (timeit.default_timer() - self._active_combines[task_id]) * 1000
                combine_metrics = CombineMetrics(e2e_latency=duration)

                if self._current_forward is not None:
                    self._current_forward.combines.append(combine_metrics)

                del self._active_combines[task_id]
                self.logger.debug(f"Combine {task_id[:8]} completed in {duration:.2f} ms")

    def start_expert_run(self, task_id: str, experts_block_index: int) -> None:
        """Start timing an expert block run"""
        if not self.config.enable_collection or not self.config.enable_expert_block_metrics:
            return

        with self._lock:
            self._active_expert_runs[task_id] = {
                "start_time": timeit.default_timer(),
                "experts_block_index": experts_block_index,
                "number_of_transfers": 0,
                "read_transfer_start": None,
                "write_transfer_start": None,
                "read_transfer_bytes": 0,
                "write_transfer_bytes": 0,
                "read_transfer_throughput": 0.0,
                "write_transfer_throughput": 0.0,
            }

    def record_transfer_start(self, task_id: str, transfer_type: str, bytes_count: int) -> None:
        """Record the start of a transfer operation"""
        if not self.config.enable_collection or not self.config.enable_transfer_metrics:
            return

        with self._lock:
            if task_id in self._active_expert_runs:
                run_data = self._active_expert_runs[task_id]
                if transfer_type == "read":
                    run_data["read_transfer_start"] = timeit.default_timer()
                    run_data["read_transfer_bytes"] = bytes_count
                elif transfer_type == "write":
                    run_data["write_transfer_start"] = timeit.default_timer()
                    run_data["write_transfer_bytes"] = bytes_count

    def record_transfer_end(self, task_id: str, transfer_type: str) -> None:
        """Record the end of a transfer operation"""
        if not self.config.enable_collection or not self.config.enable_transfer_metrics:
            return

        with self._lock:
            if task_id in self._active_expert_runs:
                run_data = self._active_expert_runs[task_id]
                current_time = timeit.default_timer()

                if transfer_type == "read" and run_data["read_transfer_start"] is not None:
                    duration = (current_time - run_data["read_transfer_start"]) * 1000
                    throughput = run_data["read_transfer_bytes"] / (1024 * 1024) / duration if duration > 0 else 0
                    run_data["read_transfer_throughput"] = throughput
                elif transfer_type == "write" and run_data["write_transfer_start"] is not None:
                    duration = (current_time - run_data["write_transfer_start"]) * 1000
                    throughput = run_data["write_transfer_bytes"] / (1024 * 1024) / duration if duration > 0 else 0
                    run_data["write_transfer_throughput"] = throughput

    def record_transfer_metrics(self, task_id: str, read_throughput: float, write_throughput: float) -> None:
        """Record transfer throughput metrics for an expert run"""
        if not self.config.enable_collection or not self.config.enable_transfer_metrics:
            return

        with self._lock:
            if task_id in self._active_expert_runs:
                run_data = self._active_expert_runs[task_id]
                run_data["read_transfer_throughput"] = read_throughput
                run_data["write_transfer_throughput"] = write_throughput

    def end_expert_run(self, task_id: str) -> None:
        """End timing an expert block run"""
        if not self.config.enable_collection or not self.config.enable_expert_block_metrics:
            return

        with self._lock:
            if task_id in self._active_expert_runs:
                run_data = self._active_expert_runs[task_id]
                duration = (timeit.default_timer() - run_data["start_time"]) * 1000

                # Create ExpertsRunMetrics
                experts_run_metrics = ExpertsRunMetrics(
                    experts_block_index=run_data["experts_block_index"],
                    e2e_latency=duration,
                    read_transfer_throughput=run_data["read_transfer_throughput"],
                    write_transfer_throughput=run_data["write_transfer_throughput"],
                )

                # Create ExpertsBlockMetrics
                expert_block_metrics = ExpertsBlockMetrics(
                    number_of_runs=1,
                    number_of_transfers=run_data["number_of_transfers"],
                    avg_e2e_latency=duration,
                    avg_transfers_throughput=(run_data["read_transfer_throughput"] + run_data["write_transfer_throughput"]) / 2,
                    experts_runs=[experts_run_metrics],
                )

                if self._current_forward is not None:
                    self._current_forward.experts_blocks.append(expert_block_metrics)

                del self._active_expert_runs[task_id]
                self.logger.debug(f"Expert run {task_id[:8]} completed in {duration:.2f} ms")

    def end_forward_pass(self) -> None:
        """Complete the current forward pass and store metrics"""
        if not self.config.enable_collection or self._current_forward is None:
            return

        with self._lock:
            self._finalize_current_forward()
            self._forward_counter += 1
            self._stats_dirty = True

            # Auto-export if configured
            if self.config.auto_export_interval is not None and self._forward_counter % self.config.auto_export_interval == 0:
                self._auto_export()

            self.logger.debug(f"Forward pass #{self._forward_counter} completed and stored")

    def _finalize_current_forward(self) -> None:
        """Finalize the current forward pass metrics (called with lock held)"""
        if self._current_forward is None or self._current_forward_start_time is None:
            return

        # Calculate total e2e latency
        self._current_forward.e2e_latency = (timeit.default_timer() - self._current_forward_start_time) * 1000

        # Calculate average dispatch latency
        if self._current_forward.dispatches:
            dispatch_latencies = [d.e2e_latency for d in self._current_forward.dispatches]
            self._current_forward.avg_dispatches_e2e_latency = statistics.mean(dispatch_latencies)

        # Calculate average combine latency
        if self._current_forward.combines:
            combine_latencies = [c.e2e_latency for c in self._current_forward.combines]
            self._current_forward.avg_combines_e2e_latency = statistics.mean(combine_latencies)

        # Store the completed forward pass
        self._completed_forwards.append(self._current_forward)

        # Limit stored forwards to prevent memory issues
        if len(self._completed_forwards) > self.config.max_stored_forwards:
            self._completed_forwards.pop(0)

        # Reset current forward
        self._current_forward = None
        self._current_forward_start_time = None
        self._current_setup_start_time = None

    def get_metrics(self) -> Metrics:
        """Get the complete metrics object"""
        with self._lock:
            return Metrics(number_of_forwards=self._forward_counter, forwards=self._completed_forwards.copy())

    def get_aggregated_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics across all collected metrics"""
        with self._lock:
            if not self._stats_dirty and self._aggregated_stats is not None:
                return self._aggregated_stats.copy()

            self._aggregated_stats = self._calculate_aggregated_statistics()
            self._stats_dirty = False
            return self._aggregated_stats.copy()

    def _calculate_aggregated_statistics(self) -> Dict[str, Any]:
        """Calculate aggregated statistics from collected metrics"""
        if not self._completed_forwards:
            return {}

        # E2E latencies
        e2e_latencies = [f.e2e_latency for f in self._completed_forwards]
        setup_latencies = [f.setup_latency for f in self._completed_forwards]

        # Dispatch and combine latencies
        all_dispatch_latencies = []
        all_combine_latencies = []
        for forward in self._completed_forwards:
            all_dispatch_latencies.extend([d.e2e_latency for d in forward.dispatches])
            all_combine_latencies.extend([c.e2e_latency for c in forward.combines])

        # Expert block latencies and throughputs
        all_expert_latencies = []
        all_read_throughputs = []
        all_write_throughputs = []
        expert_block_counts = defaultdict(int)

        for forward in self._completed_forwards:
            for eb in forward.experts_blocks:
                all_expert_latencies.append(eb.avg_e2e_latency)
                for expert_run in eb.experts_runs:
                    all_read_throughputs.append(expert_run.read_transfer_throughput)
                    all_write_throughputs.append(expert_run.write_transfer_throughput)
                expert_block_counts[eb.number_of_runs] += 1

        def safe_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "std": 0.0}
            return {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        return {
            "total_forwards": len(self._completed_forwards),
            "e2e_latency": safe_stats(e2e_latencies),
            "setup_latency": safe_stats(setup_latencies),
            "dispatch_latency": safe_stats(all_dispatch_latencies),
            "combine_latency": safe_stats(all_combine_latencies),
            "expert_block_latency": safe_stats(all_expert_latencies),
            "read_throughput": safe_stats(all_read_throughputs),
            "write_throughput": safe_stats(all_write_throughputs),
            "expert_block_utilization": dict(expert_block_counts),
            "avg_dispatches_per_forward": (
                statistics.mean([len(f.dispatches) for f in self._completed_forwards]) if self._completed_forwards else 0
            ),
            "avg_combines_per_forward": (
                statistics.mean([len(f.combines) for f in self._completed_forwards]) if self._completed_forwards else 0
            ),
            "avg_expert_blocks_per_forward": (
                statistics.mean([len(f.experts_blocks) for f in self._completed_forwards]) if self._completed_forwards else 0
            ),
        }

    def export_metrics_json_only(self, filename_prefix: str = "moe_metrics") -> str:
        """Export metrics to JSON format only (matching Metrics.py structure exactly)"""
        if not self.config.enable_collection:
            return ""

        timestamp = str(int(timeit.default_timer()))

        with self._lock:
            # Export raw metrics as JSON
            json_filename = f"{self.config.export_directory}/{filename_prefix}_{timestamp}.json"
            metrics = self.get_metrics()

            # Convert to serializable format that exactly matches Metrics.py structure
            metrics_dict = {"number_of_forwards": metrics.number_of_forwards, "forwards": []}

            for forward in metrics.forwards:
                forward_dict = {
                    "e2e_latency": forward.e2e_latency,
                    "setup_latency": forward.setup_latency,
                    "avg_dispatches_e2e_latency": forward.avg_dispatches_e2e_latency,
                    "avg_combines_e2e_latency": forward.avg_combines_e2e_latency,
                    "dispatches": [{"e2e_latency": d.e2e_latency} for d in forward.dispatches],
                    "combines": [{"e2e_latency": c.e2e_latency} for c in forward.combines],
                    "experts_blocks": [],
                }

                for eb in forward.experts_blocks:
                    eb_dict = {
                        "number_of_runs": eb.number_of_runs,
                        "number_of_transfers": eb.number_of_transfers,
                        "avg_e2e_latency": eb.avg_e2e_latency,
                        "avg_transfers_throughput": eb.avg_transfers_throughput,
                        "experts_runs": [
                            {
                                "experts_block_index": expert_run.experts_block_index,
                                "e2e_latency": expert_run.e2e_latency,
                                "read_transfer_throughput": expert_run.read_transfer_throughput,
                                "write_transfer_throughput": expert_run.write_transfer_throughput,
                            }
                            for expert_run in eb.experts_runs
                        ],
                    }
                    forward_dict["experts_blocks"].append(eb_dict)

                metrics_dict["forwards"].append(forward_dict)

            with open(json_filename, "w") as f:
                json.dump(metrics_dict, f, indent=2)

        self.logger.info(f"Metrics exported to JSON: {json_filename}")
        return json_filename

    def export_metrics(self, filename_prefix: str = "moe_metrics") -> Dict[str, str]:
        """Export metrics to various formats"""
        if not self.config.enable_collection:
            return {}

        exported_files = {}
        timestamp = str(int(timeit.default_timer()))

        with self._lock:
            # Export raw metrics as JSON
            json_filename = f"{self.config.export_directory}/{filename_prefix}_{timestamp}.json"
            metrics = self.get_metrics()

            # Convert to serializable format
            metrics_dict = {"number_of_forwards": metrics.number_of_forwards, "forwards": []}

            for forward in metrics.forwards:
                forward_dict = {
                    "e2e_latency": forward.e2e_latency,
                    "setup_latency": forward.setup_latency,
                    "avg_dispatches_e2e_latency": forward.avg_dispatches_e2e_latency,
                    "avg_combines_e2e_latency": forward.avg_combines_e2e_latency,
                    "dispatches": [{"e2e_latency": d.e2e_latency} for d in forward.dispatches],
                    "combines": [{"e2e_latency": c.e2e_latency} for c in forward.combines],
                    "experts_blocks": [],
                }

                for eb in forward.experts_blocks:
                    eb_dict = {
                        "number_of_runs": eb.number_of_runs,
                        "number_of_transfers": eb.number_of_transfers,
                        "avg_e2e_latency": eb.avg_e2e_latency,
                        "avg_transfers_throughput": eb.avg_transfers_throughput,
                        "experts_runs": [
                            {
                                "experts_block_index": expert_run.experts_block_index,
                                "e2e_latency": expert_run.e2e_latency,
                                "read_transfer_throughput": expert_run.read_transfer_throughput,
                                "write_transfer_throughput": expert_run.write_transfer_throughput,
                            }
                            for expert_run in eb.experts_runs
                        ],
                    }
                    forward_dict["experts_blocks"].append(eb_dict)

                metrics_dict["forwards"].append(forward_dict)

            with open(json_filename, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            exported_files["json"] = json_filename

            # Export aggregated statistics as JSON
            stats_filename = f"{self.config.export_directory}/{filename_prefix}_stats_{timestamp}.json"
            with open(stats_filename, "w") as f:
                json.dump(self.get_aggregated_statistics(), f, indent=2)
            exported_files["stats"] = stats_filename

            # Export summary as CSV
            csv_filename = f"{self.config.export_directory}/{filename_prefix}_summary_{timestamp}.csv"
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "forward_id",
                        "e2e_latency",
                        "setup_latency",
                        "avg_dispatch_latency",
                        "avg_combine_latency",
                        "num_dispatches",
                        "num_combines",
                        "num_expert_blocks",
                    ]
                )

                for i, forward in enumerate(metrics.forwards):
                    writer.writerow(
                        [
                            i,
                            forward.e2e_latency,
                            forward.setup_latency,
                            forward.avg_dispatches_e2e_latency,
                            forward.avg_combines_e2e_latency,
                            len(forward.dispatches),
                            len(forward.combines),
                            len(forward.experts_blocks),
                        ]
                    )
            exported_files["csv"] = csv_filename

        self.logger.info(f"Metrics exported to {len(exported_files)} files: {list(exported_files.values())}")
        return exported_files

    def _auto_export(self) -> None:
        """Automatically export metrics (called with lock held)"""
        try:
            self.export_metrics(f"auto_export_forward_{self._forward_counter}")
        except Exception as e:
            self.logger.error(f"Auto-export failed: {e}")

    def clear_metrics(self) -> None:
        """Clear all collected metrics"""
        with self._lock:
            self._completed_forwards.clear()
            self._forward_counter = 0
            self._current_forward = None
            self._current_forward_start_time = None
            self._current_setup_start_time = None
            self._active_dispatches.clear()
            self._active_combines.clear()
            self._active_expert_runs.clear()
            self._aggregated_stats = None
            self._stats_dirty = True

        self.logger.debug("All metrics cleared")

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        stats = self.get_aggregated_statistics()
        if not stats:
            return "No metrics collected yet."

        report = []
        report.append("=== DistMoE Metrics Summary ===")
        report.append(f"Total Forward Passes: {stats['total_forwards']}")
        report.append("")

        report.append("E2E Latency (ms):")
        e2e = stats["e2e_latency"]
        report.append(f"  Mean: {e2e['mean']:.2f}, Min: {e2e['min']:.2f}, Max: {e2e['max']:.2f}")
        report.append(f"  Median: {e2e['median']:.2f}, Std: {e2e['std']:.2f}")
        report.append("")

        report.append("Setup Latency (ms):")
        setup = stats["setup_latency"]
        report.append(f"  Mean: {setup['mean']:.2f}, Min: {setup['min']:.2f}, Max: {setup['max']:.2f}")
        report.append("")

        if stats["dispatch_latency"]["mean"] > 0:
            report.append("Dispatch Latency (ms):")
            dispatch = stats["dispatch_latency"]
            report.append(f"  Mean: {dispatch['mean']:.2f}, Min: {dispatch['min']:.2f}, Max: {dispatch['max']:.2f}")
            report.append("")

        if stats["combine_latency"]["mean"] > 0:
            report.append("Combine Latency (ms):")
            combine = stats["combine_latency"]
            report.append(f"  Mean: {combine['mean']:.2f}, Min: {combine['min']:.2f}, Max: {combine['max']:.2f}")
            report.append("")

        if stats["read_throughput"]["mean"] > 0:
            report.append("Transfer Throughput (MB/ms):")
            read_tp = stats["read_throughput"]
            write_tp = stats["write_throughput"]
            report.append(f"  Read - Mean: {read_tp['mean']:.2f}, Min: {read_tp['min']:.2f}, Max: {read_tp['max']:.2f}")
            report.append(f"  Write - Mean: {write_tp['mean']:.2f}, Min: {write_tp['min']:.2f}, Max: {write_tp['max']:.2f}")
            report.append("")

        report.append("Operation Counts:")
        report.append(f"  Avg Dispatches per Forward: {stats['avg_dispatches_per_forward']:.1f}")
        report.append(f"  Avg Combines per Forward: {stats['avg_combines_per_forward']:.1f}")
        report.append(f"  Avg Expert Blocks per Forward: {stats['avg_expert_blocks_per_forward']:.1f}")

        return "\n".join(report)


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector instance"""
    return _global_metrics_collector


def initialize_metrics_collector(config: MetricsCollectorConfig) -> MetricsCollector:
    """Initialize the global metrics collector"""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(config)
    return _global_metrics_collector


def shutdown_metrics_collector() -> None:
    """Shutdown the global metrics collector"""
    global _global_metrics_collector
    if _global_metrics_collector is not None:
        _global_metrics_collector.logger.info("Shutting down metrics collector")
        _global_metrics_collector = None
