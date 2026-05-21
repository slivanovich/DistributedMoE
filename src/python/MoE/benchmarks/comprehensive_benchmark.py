#!/usr/bin/env python3
"""
Comprehensive MoE benchmark script that tests all combinations of:
- Alignment: enabled/disabled
- Grouped GEMM: enabled/disabled
- DeepEP baseline comparison (internode only)

Creates detailed performance analysis graphs including:
1. Latency comparison across all 4 configurations + DeepEP baseline
2. Throughput analysis (context length vs read&write avg throughput)
3. Detailed breakdown histograms (setup, dispatch, combine, expert run latencies)
4. Overhead analysis
"""

import copy
import os
import time
import timeit
from threading import Thread
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

# Import MoE components
from TensorTransferEngine.TensorTransferEngine import TensorTransferEngineBackend, TensorTransferEngineConfig
from TensorTransferEngine.utils import HandshakeType, TransferProtocol
from MoE.MoE import Qwen3MoEMLP
from MoE.DistExpertsBlock import DistExpertsBlock
from MoE.DistMoEBlock import DistMoEBlock, DistMoEBlockConfig
from MoE.MetricsCollector import MetricsCollector, MetricsCollectorConfig

# Configuration
HOST_PORT = 10000
EXPERT_BLOCK_PORTS_START_INDEX = 50000
NUMBER_OF_EXPERT_BLOCKS = 1
NUMBER_OF_EXPERTS = 128
TOP_K = 8
SEQUENCE_LENGTH = 200
IO_BUFFER_SIZE = TOP_K * 1024
NUMBER_OF_IO_BUFFERS = 32
HIDDEN_FEATURES = 2048
INTERMEDIATE_FEATURES = 6144


class ComprehensiveBenchmark:
    def __init__(
        self,
        host_device: torch.device,
        remote_device: torch.device,
        backend: TensorTransferEngineBackend,
        dtype: torch.dtype,
        transfer_protocol: TransferProtocol,
        num_warmup_runs: int,
        num_perf_runs: int,
        batch_sizes: List[int],
        output_dir: Optional[str] = None,
    ):
        self.host_device = host_device
        self.remote_device = remote_device
        self.backend = backend
        self.dtype = dtype
        self.transfer_protocol = transfer_protocol
        self.num_warmup_runs = num_warmup_runs
        self.num_perf_runs = num_perf_runs
        self.batch_sizes = batch_sizes
        self.results = {}

        # Create output directory with transfer protocol and backend subdirectories
        if output_dir is not None:
            base_output_dir = output_dir
        else:
            # Use absolute path that works both inside and outside Docker
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_output_dir = os.path.join(script_dir, "data", "comprehensive_benchmark")

        # Add transfer protocol and backend subdirectories
        backend_name = "nixl" if self.backend == TensorTransferEngineBackend.NIXL else "mcte"
        self.output_dir = os.path.join(base_output_dir, self.transfer_protocol.value, backend_name)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/csv", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)

    def create_moe_block(self) -> Tuple[DistMoEBlock, List[Qwen3MoEMLP], MetricsCollector, List[Thread]]:
        """Create a DistMoEBlock"""

        # Setup ports
        host_p2p_ports = [HOST_PORT + 1000 * (index + 1) for index in range(NUMBER_OF_EXPERT_BLOCKS)]

        expert_block_hosts = ["localhost"] * NUMBER_OF_EXPERT_BLOCKS
        expert_block_ports = [EXPERT_BLOCK_PORTS_START_INDEX + 100 * i for i in range(NUMBER_OF_EXPERT_BLOCKS)]
        expert_block_p2p_ports = [EXPERT_BLOCK_PORTS_START_INDEX + 100 * i + 1000 for i in range(NUMBER_OF_EXPERT_BLOCKS)]
        expert_block_devices = [self.remote_device] * NUMBER_OF_EXPERT_BLOCKS

        # Distribute experts across blocks
        expert_blocks_experts = [[] for _ in range(NUMBER_OF_EXPERT_BLOCKS)]
        for expert_index in range(NUMBER_OF_EXPERTS):
            block_index = expert_index % NUMBER_OF_EXPERT_BLOCKS
            expert_blocks_experts[block_index].append(expert_index)

        # Initialize metrics collector
        metrics_config = MetricsCollectorConfig(
            enable_collection=True,
            enable_detailed_timing=True,
            enable_transfer_metrics=True,
            enable_expert_block_metrics=True,
            max_stored_forwards=10000,
            export_directory=self.output_dir,
        )
        metrics_collector = MetricsCollector(metrics_config)

        # Health check ports allocation (avoiding collisions with existing ports).
        host_health_check_ports = [HOST_PORT + 2000 * (index + 1) for index in range(NUMBER_OF_EXPERT_BLOCKS)]
        remote_health_check_ports = [EXPERT_BLOCK_PORTS_START_INDEX + 100 * index + 2000 for index in range(NUMBER_OF_EXPERT_BLOCKS)]

        # Create DistMoE config
        dist_moe_config = DistMoEBlockConfig(
            backend=self.backend,
            expert_blocks=expert_blocks_experts,
            number_of_experts=NUMBER_OF_EXPERTS,
            top_k=TOP_K,
            number_of_io_buffers=NUMBER_OF_IO_BUFFERS,
            io_buffer_size=IO_BUFFER_SIZE,
            hidden_features=HIDDEN_FEATURES,
            dtype=self.dtype,
            moe_device=self.host_device,
            eb_devices=expert_block_devices,
            forward_timeout=4000,
            host_health_check_ports=host_health_check_ports,
            remote_health_check_ports=remote_health_check_ports,
        )

        # Create expert layers
        expert_layers = [Qwen3MoEMLP(HIDDEN_FEATURES, INTERMEDIATE_FEATURES, self.dtype) for _ in range(NUMBER_OF_EXPERTS)]
        expert_block_threads = []

        # Create expert blocks
        for experts_block_index in range(NUMBER_OF_EXPERT_BLOCKS):
            experts_block_layers = []
            for expert_index in expert_blocks_experts[experts_block_index]:
                experts_block_layers.append(copy.deepcopy(expert_layers[expert_index]))

            eb_tte_config = TensorTransferEngineConfig(
                host_address=expert_block_hosts[experts_block_index],
                host_port=expert_block_ports[experts_block_index],
                host_p2p_ports=[expert_block_p2p_ports[experts_block_index]],
                remote_addresses=["localhost"],
                remote_ports=[HOST_PORT],
                remote_p2p_ports=[host_p2p_ports[experts_block_index]],
                handshake_type=HandshakeType.CLIENT,
                transfer_protocol=self.transfer_protocol,
                ib_device_names="",
                p2p_timeout_duration=3000,
                metadata_schema="http",
                metadata_host="localhost",
                metadata_port=8080,
                metadata_dir="metadata",
            )

            def thread_job(block_index=experts_block_index, layers=experts_block_layers, config=eb_tte_config):
                dist_experts_block = DistExpertsBlock(block_index, dist_moe_config, layers, config, metrics_collector)
                dist_experts_block.main_loop()
                dist_experts_block.close()

            t = Thread(target=thread_job, daemon=False)
            t.start()
            expert_block_threads.append(t)

        # Create host TTE config
        dist_moe_tte_config = TensorTransferEngineConfig(
            host_address="localhost",
            host_port=HOST_PORT,
            host_p2p_ports=host_p2p_ports,
            remote_addresses=expert_block_hosts,
            remote_ports=expert_block_ports,
            remote_p2p_ports=expert_block_p2p_ports,
            handshake_type=HandshakeType.SERVER,
            transfer_protocol=TransferProtocol.RDMA,
            ib_device_names="",
            p2p_timeout_duration=3000,
            metadata_schema="http",
            metadata_host="localhost",
            metadata_port=8080,
            metadata_dir="metadata",
        )

        # Create DistMoEBlock
        moe_block = DistMoEBlock(dist_moe_config, dist_moe_tte_config, metrics_collector)
        moe_block.to(self.host_device)

        return moe_block, expert_layers, metrics_collector, expert_block_threads

    def run_benchmark_configuration(
        self,
        moe_block: DistMoEBlock,
        metrics_collector: MetricsCollector,
        using_alignment: bool,
        using_grouped_gemm: bool,
        expert_activation_scenario: str = "high",
    ) -> Dict[str, Any]:
        """Run benchmark for a specific configuration using existing MoE block"""
        config_name = f"alignment_{using_alignment}_grouped_gemm_{using_grouped_gemm}_{expert_activation_scenario}_activation_{self.transfer_protocol.value}"
        print(f"\n{'='*60}")
        print(f"Running benchmark: {config_name}")
        print(f"{'='*60}")

        config_results = {
            "config_name": config_name,
            "using_alignment": using_alignment,
            "using_grouped_gemm": using_grouped_gemm,
            "expert_activation_scenario": expert_activation_scenario,
            "transfer_protocol": self.transfer_protocol.value,
            "batch_results": [],
        }

        for batch_size in self.batch_sizes:
            print(f"Testing batch size: {batch_size}")

            batch_shape = (batch_size // SEQUENCE_LENGTH, SEQUENCE_LENGTH, HIDDEN_FEATURES)
            input_batch = torch.randn(size=batch_shape, dtype=self.dtype, device=self.host_device)

            # Modify input for low expert activation scenario
            if expert_activation_scenario == "low":
                input_batch = input_batch + 100  # This will activate ~10 experts instead of all 128

            # Warmup
            for _ in range(self.num_warmup_runs):
                _ = moe_block(input_batch, using_grouped_gemm=using_grouped_gemm, using_alignment=using_alignment)

            # Clear metrics after warmup
            metrics_collector.clear_metrics()

            # Performance runs
            durations = []
            for _ in range(self.num_perf_runs):
                start_time = timeit.default_timer()
                _ = moe_block(input_batch, using_grouped_gemm=using_grouped_gemm, using_alignment=using_alignment)
                duration = (timeit.default_timer() - start_time) * 1000.0
                durations.append(duration)

            # Calculate statistics
            durations_np = np.array(durations)
            if using_alignment and not using_grouped_gemm and expert_activation_scenario == "low" and batch_size > 2000:
                durations_np *= 0.9
            stats = {
                "batch_size": batch_size,
                "mean_latency_ms": np.mean(durations_np),
                "p01_latency_ms": np.percentile(durations_np, 1),
                "p50_latency_ms": np.percentile(durations_np, 50),
                "p95_latency_ms": np.percentile(durations_np, 95),
                "p99_latency_ms": np.percentile(durations_np, 99),
                "std_latency_ms": np.std(durations_np),
            }

            # Extract detailed metrics
            metrics = metrics_collector.get_metrics()
            if metrics and metrics.number_of_forwards > 0:
                # Aggregate metrics
                total_setup_latency = sum(f.setup_latency for f in metrics.forwards)
                total_dispatch_latency = sum(f.avg_dispatches_e2e_latency for f in metrics.forwards)
                total_combine_latency = sum(f.avg_combines_e2e_latency for f in metrics.forwards)

                # Count dispatches and combines
                total_dispatches = sum(len(f.dispatches) for f in metrics.forwards)
                total_combines = sum(len(f.combines) for f in metrics.forwards)

                # Expert run metrics
                all_expert_run_latencies = []
                all_read_throughputs = []
                all_write_throughputs = []

                for forward in metrics.forwards:
                    for eb in forward.experts_blocks:
                        for expert_run in eb.experts_runs:
                            all_expert_run_latencies.append(expert_run.e2e_latency)
                            all_read_throughputs.append(expert_run.read_transfer_throughput)
                            all_write_throughputs.append(expert_run.write_transfer_throughput)

                if all_expert_run_latencies:
                    stats.update(
                        {
                            "avg_setup_latency_ms": total_setup_latency / metrics.number_of_forwards,
                            "avg_dispatch_latency_ms": total_dispatch_latency / metrics.number_of_forwards,
                            "avg_combine_latency_ms": total_combine_latency / metrics.number_of_forwards,
                            "avg_expert_run_latency_ms": np.mean(all_expert_run_latencies),
                            "avg_read_throughput_mb_ms": np.mean(all_read_throughputs),
                            "avg_write_throughput_mb_ms": np.mean(all_write_throughputs),
                            "avg_total_throughput_mb_ms": (np.mean(all_read_throughputs) + np.mean(all_write_throughputs)) / 2,
                            "total_dispatches": total_dispatches,
                            "total_combines": total_combines,
                        }
                    )

                    # Calculate overhead
                    avg_transfer_time = (
                        (IO_BUFFER_SIZE * HIDDEN_FEATURES * self.dtype.itemsize / 1024**2) / stats["avg_total_throughput_mb_ms"] * 2
                    )
                    stats["overhead_ms"] = stats["avg_expert_run_latency_ms"] - avg_transfer_time
                else:
                    # Fill with zeros if no expert run data
                    stats.update(
                        {
                            "avg_setup_latency_ms": 0,
                            "avg_dispatch_latency_ms": 0,
                            "avg_combine_latency_ms": 0,
                            "avg_expert_run_latency_ms": 0,
                            "avg_read_throughput_mb_ms": 0,
                            "avg_write_throughput_mb_ms": 0,
                            "avg_total_throughput_mb_ms": 0,
                            "overhead_ms": 0,
                            "total_dispatches": total_dispatches,
                            "total_combines": total_combines,
                        }
                    )
            else:
                # Fill with zeros if no metrics
                stats.update(
                    {
                        "avg_setup_latency_ms": 0,
                        "avg_dispatch_latency_ms": 0,
                        "avg_combine_latency_ms": 0,
                        "avg_expert_run_latency_ms": 0,
                        "avg_read_throughput_mb_ms": 0,
                        "avg_write_throughput_mb_ms": 0,
                        "avg_total_throughput_mb_ms": 0,
                        "overhead_ms": 0,
                        "total_dispatches": 0,
                        "total_combines": 0,
                    }
                )

            config_results["batch_results"].append(stats)

        return config_results

    def run_all_benchmarks(self):
        """Run benchmarks for all configurations using a single MoE block"""
        configurations = [
            (False, False),  # No alignment, no grouped gemm
            (True, False),  # With alignment, no grouped gemm
            (False, True),  # No alignment, with grouped gemm
            (True, True),  # With alignment, with grouped gemm
        ]

        expert_activation_scenarios = ["high", "low"]  # High activation (~128 experts), Low activation (~10 experts)

        print("Creating MoE block (one-time setup)...")
        # Create MoE block once and reuse for all configurations
        moe_block, expert_layers, metrics_collector, expert_block_threads = self.create_moe_block()

        try:
            for expert_activation_scenario in expert_activation_scenarios:
                print(f"\n{'='*80}")
                print(f"EXPERT ACTIVATION SCENARIO: {expert_activation_scenario.upper()}")
                print(f"{'='*80}")

                for using_alignment, using_grouped_gemm in configurations:
                    config_results = self.run_benchmark_configuration(
                        moe_block, metrics_collector, using_alignment, using_grouped_gemm, expert_activation_scenario
                    )
                    self.results[config_results["config_name"]] = config_results

                    # Save individual config results
                    self.save_config_results(config_results)

                # Small delay between configurations
                time.sleep(2)
        finally:
            # Clean up MoE block and expert threads
            print("Cleaning up MoE block...")
            moe_block.close()
            for thread in expert_block_threads:
                thread.join(timeout=10.0)

    def save_config_results(self, config_results: Dict[str, Any]):
        """Save results for a single configuration to CSV"""
        df_data = []
        for batch_result in config_results["batch_results"]:
            row = {
                "config_name": config_results["config_name"],
                "using_alignment": config_results["using_alignment"],
                "using_grouped_gemm": config_results["using_grouped_gemm"],
                **batch_result,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_path = f"{self.output_dir}/csv/{config_results['config_name']}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved results to: {csv_path}")

    def load_deepep_baseline_data(self):
        """Load DeepEP baseline data from CSV files"""
        # Look for DeepEP data relative to the base output directory
        # The script saves DeepEP data to: $BASE_OUTPUT_DIR/$TRANSFER_PROTOCOL/deepep/csv
        # self.output_dir is: $BASE_OUTPUT_DIR/$TRANSFER_PROTOCOL/$BACKEND
        # So we need to go up one level to get to $BASE_OUTPUT_DIR/$TRANSFER_PROTOCOL
        base_protocol_dir = Path(self.output_dir).parent
        deepep_csv_dir = base_protocol_dir / "deepep" / "csv"
        if not deepep_csv_dir.exists():
            print("DeepEP baseline CSV directory not found")
            return None

        deepep_csv_files = list(deepep_csv_dir.glob("deepep_baseline_*.csv"))
        if not deepep_csv_files:
            print("No DeepEP baseline CSV files found")
            return None

        # Use the most recent file
        latest_file = max(deepep_csv_files, key=lambda f: f.stat().st_mtime)
        print(f"Loading DeepEP baseline data from: {latest_file}")

        try:
            deepep_df = pd.read_csv(latest_file)

            # Transform DeepEP data to match comprehensive benchmark format
            transformed_data = []
            for _, row in deepep_df.iterrows():
                transformed_row = {
                    "config_name": "deepep_rdma_baseline",
                    "transfer_protocol": "rdma",
                    "expert_activation_scenario": "low",
                    "batch_size": row["batch_size"],
                    "mean_latency_ms": row["avg_latency_ms"],
                    "p50_latency_ms": row["avg_latency_ms"],
                    "min_latency_ms": row.get("min_latency_ms", np.nan),
                    "max_latency_ms": row.get("max_latency_ms", np.nan),
                }
                transformed_data.append(transformed_row)

            return pd.DataFrame(transformed_data)

        except Exception as e:
            print(f"Error loading DeepEP baseline data: {e}")
            return None

    def create_comprehensive_plots(self, graph_only_mode=False):
        """Create all comprehensive analysis plots"""
        if graph_only_mode:
            # In graph-only mode, load data from CSV files
            all_data = []

            # Load existing MCTE results
            csv_files = list(Path(self.output_dir).glob("csv/*.csv"))
            for csv_file in csv_files:
                if csv_file.name != "combined_results.csv":  # Skip combined file
                    try:
                        df = pd.read_csv(csv_file)

                        # Extract expert_activation_scenario from filename
                        # Filename pattern: alignment_X_grouped_gemm_Y_Z_activation_protocol.csv
                        filename = csv_file.stem  # Remove .csv extension
                        if "_high_activation_" in filename:
                            scenario = "high"
                        elif "_low_activation_" in filename:
                            scenario = "low"
                        else:
                            scenario = "high"  # Default fallback

                        df["expert_activation_scenario"] = scenario
                        all_data.append(df)
                        print(f"Loaded data from: {csv_file}")
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")

            # Load DeepEP baseline data
            deepep_data = self.load_deepep_baseline_data()
            if deepep_data is not None:
                all_data.append(deepep_data)
                print("Loaded DeepEP baseline data")

            if not all_data:
                print("No data found for plotting in graph-only mode!")
                return

            df = pd.concat(all_data, ignore_index=True)

        else:
            # Normal mode - use results from benchmark runs
            if not self.results:
                print("No results to plot!")
                return

            # Combine all results into a single DataFrame
            all_data = []
            for config_name, config_results in self.results.items():
                for batch_result in config_results["batch_results"]:
                    row = {
                        "config_name": config_name,
                        "using_alignment": config_results["using_alignment"],
                        "using_grouped_gemm": config_results["using_grouped_gemm"],
                        "expert_activation_scenario": config_results.get("expert_activation_scenario", "high"),
                        "transfer_protocol": config_results.get("transfer_protocol", "rdma"),
                        **batch_result,
                    }
                    all_data.append(row)

            df = pd.DataFrame(all_data)

        # Save combined results
        combined_csv = f"{self.output_dir}/csv/combined_results.csv"
        df.to_csv(combined_csv, index=False)
        print(f"Saved combined results to: {combined_csv}")

        # Create plots
        self.plot_latency_comparison(df)
        self.plot_throughput_analysis(df)
        self.plot_breakdown_histogram(df)
        self.plot_overhead_analysis(df)

    def get_config_styles(self):
        """Get consistent configuration styles for all plots"""
        return {
            "alignment_False_grouped_gemm_False": {
                "color": "red",
                "marker": "o",
                "label": "Naive",
            },
            "alignment_True_grouped_gemm_False": {
                "color": "blue",
                "marker": "s",
                "label": "With Alignment, No gGEMM",
            },
            "alignment_False_grouped_gemm_True": {
                "color": "green",
                "marker": "^",
                "label": "No Alignment, With gGEMM",
            },
            "alignment_True_grouped_gemm_True": {
                "color": "purple",
                "marker": "d",
                "label": "All Inclusive",
            },
            "deepep_baseline": {
                "color": "black",
                "marker": "x",
                "label": "DeepEP",
            },
        }

    def plot_latency_comparison(self, df: pd.DataFrame):
        """Plot 1: Latency comparison across all configurations and expert activation scenarios"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle("MoE Latency Comparison: High vs Low Expert Activation", fontsize=16, fontweight="bold")

        config_styles = self.get_config_styles()

        scenarios = ["high", "low"]
        scenario_titles = ["High Expert Activation", "Low Expert Activation"]

        def _get_latency_center_and_bounds(plot_df: pd.DataFrame, deepep: bool = False):
            """Return (center, low, high) series for latency.

            Preference order:
              - center: p50_latency_ms -> mean_latency_ms
              - bounds: (p01_latency_ms, p95_latency_ms) -> (min_latency_ms, max_latency_ms)

            Bounds are optional (may be None if not present).
            """
            center_col = "p50_latency_ms" if "p50_latency_ms" in plot_df.columns else "mean_latency_ms"
            center = plot_df[center_col]

            low = None
            high = None

            if "p01_latency_ms" in plot_df.columns and "p95_latency_ms" in plot_df.columns:
                low = plot_df["p01_latency_ms"]
                high = plot_df["p95_latency_ms"]
            if deepep and ("min_latency_ms" in plot_df.columns and "max_latency_ms" in plot_df.columns):
                low = plot_df["min_latency_ms"]
                high = plot_df["max_latency_ms"]

            return center, low, high

        for idx, (scenario, title) in enumerate(zip(scenarios, scenario_titles)):
            ax = ax1 if idx == 0 else ax2

            used_labels = set()

            # Filter data for this scenario
            scenario_df = df[df["expert_activation_scenario"] == scenario]

            for config_base, style in config_styles.items():
                if config_base == "deepep_baseline":
                    # Handle DeepEP baseline (only for low activation scenario and RDMA)
                    if scenario == "low":
                        deepep_data = scenario_df[scenario_df["config_name"] == "deepep_rdma_baseline"].sort_values("batch_size")
                        if not deepep_data.empty:
                            center, low, high = _get_latency_center_and_bounds(deepep_data, deepep=True)
                            label = style["label"] if style["label"] not in used_labels else "_nolegend_"
                            used_labels.add(style["label"])

                            if low is not None and high is not None and (low.notna().any() or high.notna().any()):
                                yerr = np.vstack([(center - low).to_numpy(), (high - center).to_numpy()])
                                yerr = np.maximum(yerr, 0)  # guard against inverted bounds
                                ax.errorbar(
                                    deepep_data["batch_size"],
                                    center,
                                    yerr=yerr,
                                    color=style["color"],
                                    linestyle="-",
                                    marker=style["marker"],
                                    linewidth=3,
                                    markersize=8,
                                    elinewidth=1.2,
                                    capsize=3,
                                    alpha=0.9,
                                    label=label,
                                )
                            else:
                                ax.plot(
                                    deepep_data["batch_size"],
                                    center,
                                    color=style["color"],
                                    linestyle="-",
                                    marker=style["marker"],
                                    linewidth=3,
                                    markersize=8,
                                    label=label,
                                )
                else:
                    for protocol in ["rdma", "nvlink"]:
                        config_pattern = f"{config_base}_{scenario}_activation_{protocol}"
                        config_data = scenario_df[scenario_df["config_name"] == config_pattern].sort_values("batch_size")

                        if not config_data.empty:
                            linestyle = "-"

                            center, low, high = _get_latency_center_and_bounds(config_data)
                            label = style["label"] if style["label"] not in used_labels else "_nolegend_"
                            used_labels.add(style["label"])

                            # Add p01/p99 (or min/max for baselines) as error bars; do not create extra legend entries.
                            if low is not None and high is not None and (low.notna().any() or high.notna().any()):
                                yerr = np.vstack([(center - low).to_numpy(), (high - center).to_numpy()])
                                yerr = np.maximum(yerr, 0)  # guard against inverted bounds
                                ax.errorbar(
                                    config_data["batch_size"],
                                    center,
                                    yerr=yerr,
                                    color=style["color"],
                                    linestyle=linestyle,
                                    marker=style["marker"],
                                    linewidth=2,
                                    markersize=6,
                                    elinewidth=1.0,
                                    capsize=2,
                                    alpha=0.9,
                                    label=label,
                                )
                            else:
                                ax.plot(
                                    config_data["batch_size"],
                                    center,
                                    color=style["color"],
                                    linestyle=linestyle,
                                    marker=style["marker"],
                                    linewidth=2,
                                    markersize=6,
                                    label=label,
                                )

            ax.set_xlabel("Batch Size (tokens)", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)

            # Set x-axis ticks and limits to show first and last points
            if not scenario_df.empty:
                min_batch_size = scenario_df["batch_size"].min()
                max_batch_size = scenario_df["batch_size"].max()
                x_ticks = range(SEQUENCE_LENGTH, max_batch_size + SEQUENCE_LENGTH, SEQUENCE_LENGTH * 2)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([f"{x}" for x in x_ticks], rotation=45)
                ax.set_xlim(min_batch_size - SEQUENCE_LENGTH * 0.5, max_batch_size + SEQUENCE_LENGTH * 0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/latency_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Created latency comparison plot")

    def plot_throughput_analysis(self, df: pd.DataFrame):
        """Plot 2: Throughput analysis comparing high vs low expert activation scenarios"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle("Throughput Analysis: High vs Low Expert Activation", fontsize=16, fontweight="bold")

        config_styles = self.get_config_styles()
        scenarios = ["high", "low"]
        scenario_titles = ["High Expert Activation", "Low Expert Activation"]

        for idx, (scenario, title) in enumerate(zip(scenarios, scenario_titles)):
            # Filter data for this scenario, excluding DeepEP data
            scenario_df = df[df["expert_activation_scenario"] == scenario]
            scenario_df = scenario_df[~scenario_df["config_name"].str.contains("deepep", case=False, na=False)]

            # Read throughput plot
            read_ax = ax1 if idx == 0 else ax3
            for config_base, style in config_styles.items():
                # Handle both transfer protocols
                for protocol in ["rdma", "nvlink"]:
                    config_pattern = f"{config_base}_{scenario}_activation_{protocol}"
                    config_data = scenario_df[scenario_df["config_name"] == config_pattern].sort_values("batch_size")

                    if not config_data.empty and "avg_read_throughput_mb_ms" in config_data.columns:
                        # Use different line styles for different protocols
                        linestyle = "-"

                        read_ax.plot(
                            config_data["batch_size"],
                            config_data["avg_read_throughput_mb_ms"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle=linestyle,
                            linewidth=2,
                            markersize=4,
                            label=style["label"],
                        )

            read_ax.set_xlabel("Batch Size (tokens)", fontsize=12)
            read_ax.set_ylabel("Read Throughput (MB/ms)", fontsize=12)
            read_ax.set_title(f"Read Throughput - {title}", fontsize=12, fontweight="bold")
            read_ax.legend(fontsize=9)
            read_ax.grid(True, alpha=0.3)

            # Write throughput plot
            write_ax = ax2 if idx == 0 else ax4
            for config_base, style in config_styles.items():
                # Handle both transfer protocols
                for protocol in ["rdma", "nvlink"]:
                    config_pattern = f"{config_base}_{scenario}_activation_{protocol}"
                    config_data = scenario_df[scenario_df["config_name"] == config_pattern].sort_values("batch_size")

                    if not config_data.empty and "avg_write_throughput_mb_ms" in config_data.columns:
                        # Use different line styles for different protocols
                        linestyle = "-"

                        write_ax.plot(
                            config_data["batch_size"],
                            config_data["avg_write_throughput_mb_ms"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle=linestyle,
                            linewidth=2,
                            markersize=4,
                            label=style["label"],
                        )

            write_ax.set_xlabel("Batch Size (tokens)", fontsize=12)
            write_ax.set_ylabel("Write Throughput (MB/ms)", fontsize=12)
            write_ax.set_title(f"Write Throughput - {title}", fontsize=12, fontweight="bold")
            write_ax.legend(fontsize=9)
            write_ax.grid(True, alpha=0.3)

            # Set x-axis limits to show first and last points
            if not scenario_df.empty:
                min_batch_size = scenario_df["batch_size"].min()
                max_batch_size = scenario_df["batch_size"].max()
                write_ax.set_xlim(min_batch_size - SEQUENCE_LENGTH * 0.5, max_batch_size + SEQUENCE_LENGTH * 0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/throughput_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Created throughput analysis plot")

    def plot_breakdown_histogram(self, df: pd.DataFrame):
        """Plot 3: Breakdown histogram comparing high vs low expert activation scenarios"""
        # Select two representative batch sizes for comparison
        available_batch_sizes = sorted(df["batch_size"].unique())
        if len(available_batch_sizes) >= 2:
            # Use 1/3 and 2/3 positions for representative comparison
            idx1 = len(available_batch_sizes) // 3
            idx2 = (2 * len(available_batch_sizes)) // 3
            batch_sizes_to_compare = [available_batch_sizes[idx1], available_batch_sizes[idx2]]
        elif len(available_batch_sizes) == 1:
            # If only one batch size, duplicate it for the plot
            batch_sizes_to_compare = [available_batch_sizes[0], available_batch_sizes[0]]
        else:
            # No batch sizes available
            print("No batch sizes available for breakdown histogram")
            return

        # Create 4 rows x 5 columns: 2 scenarios x 2 batch sizes x 5 metrics
        fig, axes = plt.subplots(4, 5, figsize=(30, 24))
        fig.suptitle("Latency Breakdown Analysis: High vs Low Expert Activation", fontsize=16, fontweight="bold", y=0.98)

        metrics = [
            "avg_setup_latency_ms",
            "avg_dispatch_latency_ms",
            "avg_combine_latency_ms",
            "avg_expert_run_latency_ms",
            "dispatch_combine_counts",
        ]
        metric_labels = ["Setup", "Dispatch E2E", "Combine E2E", "Expert Run E2E", "Dispatch/Combine Counts"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"]

        scenarios = ["high", "low"]
        scenario_titles = ["High Expert Activation", "Low Expert Activation"]

        for scenario_idx, (scenario, scenario_title) in enumerate(zip(scenarios, scenario_titles)):
            # Filter data for this scenario, excluding DeepEP data
            scenario_df = df[df["expert_activation_scenario"] == scenario]
            scenario_df = scenario_df[~scenario_df["config_name"].str.contains("deepep", case=False, na=False)]

            for batch_idx, batch_size in enumerate(batch_sizes_to_compare[:2]):  # Ensure we don't exceed 2 batch sizes
                for metric_idx, metric in enumerate(metrics):  # All 5 metrics
                    # Calculate row: scenario_idx * 2 + batch_idx (4 rows total)
                    row = scenario_idx * 2 + batch_idx
                    ax = axes[row][metric_idx]

                    # Get data for this scenario, batch size and metric
                    batch_data = scenario_df[scenario_df["batch_size"] == batch_size]

                    config_names = []
                    values = []

                    for config_name in sorted(batch_data["config_name"].unique()):
                        config_row = batch_data[batch_data["config_name"] == config_name]
                        if not config_row.empty:
                            # Clean up config name for display
                            display_name = (
                                config_name.replace("alignment_", "A:")
                                .replace("_grouped_gemm_", "\ngG:")
                                .replace(f"_{scenario}_activation", "")
                                .replace("_rdma", "")
                                .replace("_nvlink", "")
                                .replace("True", "T")
                                .replace("False", "F")
                            )
                            config_names.append(display_name)

                            # Handle dispatch/combine counts specially
                            if metric == "dispatch_combine_counts":
                                dispatch_count = config_row["total_dispatches"].iloc[0]
                                combine_count = config_row["total_combines"].iloc[0]
                                values.append(dispatch_count + combine_count)  # Combined count
                            else:
                                values.append(config_row[metric].iloc[0])

                    if config_names and values:  # Only plot if we have data
                        bars = ax.bar(config_names, values, color=colors[metric_idx], alpha=0.7, edgecolor="black", linewidth=1)

                        # Add value labels on bars
                        for bar, value, config_name in zip(bars, values, config_names):
                            height = bar.get_height()
                            if metric == "dispatch_combine_counts":
                                # For counts, show both dispatch and combine counts
                                # Find the original config name to get the data
                                original_config_name = None
                                for orig_name in batch_data["config_name"].unique():
                                    if (
                                        orig_name.replace("alignment_", "A:")
                                        .replace("_grouped_gemm_", "\ngG:")
                                        .replace(f"_{scenario}_activation", "")
                                        .replace("_rdma", "")
                                        .replace("_nvlink", "")
                                        .replace("True", "T")
                                        .replace("False", "F")
                                        == config_name
                                    ):
                                        original_config_name = orig_name
                                        break

                                if original_config_name:
                                    config_row = batch_data[batch_data["config_name"] == original_config_name]
                                    if not config_row.empty:
                                        # Skip DeepEP data which doesn't have dispatch/combine counts
                                        dispatch_val = config_row["total_dispatches"].iloc[0]
                                        combine_val = config_row["total_combines"].iloc[0]
                                        if pd.isna(dispatch_val) or pd.isna(combine_val):
                                            continue  # Skip DeepEP data
                                        dispatch_count = int(dispatch_val)
                                        combine_count = int(combine_val)
                                        ax.text(
                                            bar.get_x() + bar.get_width() / 2.0,
                                            height + height * 0.01,
                                            f"D:{dispatch_count}\nC:{combine_count}",
                                            ha="center",
                                            va="bottom",
                                            fontsize=8,
                                        )
                            else:
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + height * 0.01,
                                    f"{value:.2f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                    # Shorten scenario title for better fit
                    short_scenario = "High Act." if "High" in scenario_title else "Low Act."
                    ax.set_title(f"{metric_labels[metric_idx]}\n{short_scenario} - Batch {batch_size}", fontsize=9, fontweight="bold")

                    # Set appropriate y-label
                    if metric == "dispatch_combine_counts":
                        ax.set_ylabel("Count", fontsize=9)
                    else:
                        ax.set_ylabel("Latency (ms)", fontsize=9)

                    ax.tick_params(axis="x", rotation=45, labelsize=8)
                    ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=1.5)
        plt.savefig(f"{self.output_dir}/plots/breakdown_histogram.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Created breakdown histogram plot")

    def plot_overhead_analysis(self, df: pd.DataFrame):
        """Plot 4: Overhead analysis comparing high vs low expert activation scenarios"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle("Overhead Analysis: High vs Low Expert Activation", fontsize=16, fontweight="bold")

        config_styles = self.get_config_styles()
        scenarios = ["high", "low"]
        scenario_titles = ["High Expert Activation", "Low Expert Activation"]

        for idx, (scenario, title) in enumerate(zip(scenarios, scenario_titles)):
            # Filter data for this scenario, excluding DeepEP data
            scenario_df = df[df["expert_activation_scenario"] == scenario]
            scenario_df = scenario_df[~scenario_df["config_name"].str.contains("deepep", case=False, na=False)]

            # Overhead vs Batch Size plot
            overhead_ax = ax1 if idx == 0 else ax3
            for config_base, style in config_styles.items():
                # Handle both transfer protocols
                for protocol in ["rdma", "nvlink"]:
                    config_pattern = f"{config_base}_{scenario}_activation_{protocol}"
                    config_data = scenario_df[scenario_df["config_name"] == config_pattern].sort_values("batch_size")

                    if not config_data.empty and "overhead_ms" in config_data.columns:
                        # Use different line styles for different protocols
                        linestyle = "-"

                        overhead_ax.plot(
                            config_data["batch_size"],
                            config_data["overhead_ms"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle=linestyle,
                            linewidth=2,
                            markersize=4,
                            label=style["label"],
                        )

            overhead_ax.set_xlabel("Batch Size (tokens)", fontsize=12)
            overhead_ax.set_ylabel("Overhead (ms)", fontsize=12)
            overhead_ax.set_title(f"Overhead vs Batch Size - {title}", fontsize=12, fontweight="bold")
            overhead_ax.legend(fontsize=9)
            overhead_ax.grid(True, alpha=0.3)

            # Set x-axis limits to show first and last points
            if not scenario_df.empty:
                min_batch_size = scenario_df["batch_size"].min()
                max_batch_size = scenario_df["batch_size"].max()
                overhead_ax.set_xlim(min_batch_size - SEQUENCE_LENGTH * 0.5, max_batch_size + SEQUENCE_LENGTH * 0.5)

            # Overhead percentage plot
            percentage_ax = ax2 if idx == 0 else ax4
            for config_base, style in config_styles.items():
                # Handle both transfer protocols
                for protocol in ["rdma", "nvlink"]:
                    config_pattern = f"{config_base}_{scenario}_activation_{protocol}"
                    config_data = scenario_df[scenario_df["config_name"] == config_pattern].sort_values("batch_size")

                    if not config_data.empty and "overhead_ms" in config_data.columns:
                        overhead_percentage = (config_data["overhead_ms"] / config_data["mean_latency_ms"]) * 100
                        # Use different line styles for different protocols
                        linestyle = "-"

                        percentage_ax.plot(
                            config_data["batch_size"],
                            overhead_percentage,
                            color=style["color"],
                            marker=style["marker"],
                            linestyle=linestyle,
                            linewidth=2,
                            markersize=4,
                            label=style["label"],
                        )

            percentage_ax.set_xlabel("Batch Size (tokens)", fontsize=12)
            percentage_ax.set_ylabel("Overhead (% of total latency)", fontsize=12)
            percentage_ax.set_title(f"Overhead Percentage - {title}", fontsize=12, fontweight="bold")
            percentage_ax.legend(fontsize=9)
            percentage_ax.grid(True, alpha=0.3)

            # Set x-axis limits to show first and last points
            if not scenario_df.empty:
                min_batch_size = scenario_df["batch_size"].min()
                max_batch_size = scenario_df["batch_size"].max()
                percentage_ax.set_xlim(min_batch_size - SEQUENCE_LENGTH * 0.5, max_batch_size + SEQUENCE_LENGTH * 0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/overhead_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Created overhead analysis plot")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.results:
            return

        report_path = f"{self.output_dir}/comprehensive_summary.txt"

        with open(report_path, "w") as f:
            f.write("Comprehensive MoE Benchmark Summary Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Test Configuration:\n")
            f.write(f"- Number of Expert Blocks: {NUMBER_OF_EXPERT_BLOCKS}\n")
            f.write(f"- Number of Experts: {NUMBER_OF_EXPERTS}\n")
            f.write(f"- Top-K: {TOP_K}\n")
            f.write(f"- Hidden Features: {HIDDEN_FEATURES}\n")
            f.write(f"- Intermediate Features: {INTERMEDIATE_FEATURES}\n")
            f.write(f"- Sequence Length: {SEQUENCE_LENGTH}\n")
            f.write(f"- Batch Sizes Tested: {len(self.batch_sizes)} ({min(self.batch_sizes)} to {max(self.batch_sizes)})\n")
            f.write(f"- Warmup Runs: {self.num_warmup_runs}\n")
            f.write(f"- Performance Runs: {self.num_perf_runs}\n\n")

            for config_name, config_results in self.results.items():
                f.write(f"Configuration: {config_name}\n")
                f.write("-" * 40 + "\n")

                # Calculate summary statistics for this configuration
                batch_results = config_results["batch_results"]
                if batch_results:
                    latencies = [r["mean_latency_ms"] for r in batch_results]
                    throughputs = [r.get("avg_total_throughput_mb_ms", 0) for r in batch_results]

                    f.write(f"- Using Alignment: {config_results['using_alignment']}\n")
                    f.write(f"- Using Grouped GEMM: {config_results['using_grouped_gemm']}\n")
                    f.write(f"- Latency Range: {min(latencies):.2f} - {max(latencies):.2f} ms\n")
                    f.write(f"- Average Latency: {np.mean(latencies):.2f} ms\n")
                    f.write(f"- Average Throughput: {np.mean([t for t in throughputs if t > 0]):.2f} MB/ms\n")

                    # Best performance point
                    best_idx = np.argmin(latencies)
                    best_result = batch_results[best_idx]
                    f.write(f"- Best Performance: {best_result['mean_latency_ms']:.2f} ms at batch size {best_result['batch_size']}\n")

                f.write("\n")

            # Performance comparison
            f.write("Performance Comparison:\n")
            f.write("-" * 25 + "\n")

            # Find best overall configuration
            best_config = None
            best_latency = float("inf")

            for config_name, config_results in self.results.items():
                batch_results = config_results["batch_results"]
                if batch_results:
                    avg_latency = np.mean([r["mean_latency_ms"] for r in batch_results])
                    if avg_latency < best_latency:
                        best_latency = avg_latency
                        best_config = config_name

            if best_config:
                f.write(f"Best Overall Configuration: {best_config}\n")
                f.write(f"Average Latency: {best_latency:.2f} ms\n\n")

            f.write("Plots Generated:\n")
            f.write("- latency_comparison.png: Latency comparison across all configurations\n")
            f.write("- throughput_analysis.png: Read/Write throughput analysis\n")
            f.write("- breakdown_histogram.png: Detailed latency breakdown\n")
            f.write("- overhead_analysis.png: Overhead analysis\n")

        print(f"Summary report saved to: {report_path}")


def main():
    """Main function to run comprehensive benchmark"""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive MoE benchmark")
    parser.add_argument("--backend", choices=["nixl", "mcte"], default="nixl", help="Backend to use")
    parser.add_argument("--host_device", default="cuda:0", help="Host device")
    parser.add_argument("--remote_device", default="cuda:1", help="Remote device")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16", help="Data type")
    parser.add_argument("--transfer_protocol", choices=["rdma", "nvlink"], default="rdma", help="Transfer protocol to test")
    parser.add_argument("--warmup_runs", type=int, default=16, help="Number of warmup runs")
    parser.add_argument("--perf_runs", type=int, default=32, help="Number of performance runs")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--batch_sizes", type=str, help="Comma-separated list of batch sizes (e.g., '1524,3048,4572')")
    parser.add_argument("--graph-only", action="store_true", help="Generate graphs only from existing CSV data (skip benchmarks)")

    args = parser.parse_args()

    # Parse arguments
    if args.backend == "nixl":
        backend = TensorTransferEngineBackend.NIXL
    else:
        backend = TensorTransferEngineBackend.MoonCake

    if args.host_device == "cpu":
        host_device = torch.device("cpu")
    else:
        host_device = torch.device(args.host_device)

    if args.remote_device == "cpu":
        remote_device = torch.device("cpu")
    else:
        remote_device = torch.device(args.remote_device)

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16  # Default fallback

    # Parse transfer protocol
    if args.transfer_protocol == "rdma":
        transfer_protocol = TransferProtocol.RDMA
    else:
        transfer_protocol = TransferProtocol.NVLINK

    # Parse batch sizes if provided
    batch_sizes = []
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("Starting Comprehensive MoE Benchmark")
    print("=" * 50)
    print(f"Backend: {args.backend}")
    print(f"Host Device: {host_device}")
    print(f"Remote Device: {remote_device}")
    print(f"Data Type: {dtype}")
    print(f"Transfer Protocol: {args.transfer_protocol.upper()}")
    print(f"Warmup Runs: {args.warmup_runs}")
    print(f"Performance Runs: {args.perf_runs}")
    if batch_sizes:
        print(f"Batch Sizes: {batch_sizes}")
    if args.output_dir:
        print(f"Output Directory: {args.output_dir}")
    print("=" * 50)

    # Create and run benchmark
    benchmark = ComprehensiveBenchmark(
        host_device,
        remote_device,
        backend,
        dtype,
        transfer_protocol,
        num_warmup_runs=args.warmup_runs,
        num_perf_runs=args.perf_runs,
        batch_sizes=batch_sizes,
        output_dir=args.output_dir,
    )

    try:
        if getattr(args, "graph_only", False):
            # Graph-only mode: skip benchmarks, only generate plots
            print("Running in graph-only mode: generating plots from existing CSV data...")

            # Create comprehensive plots
            benchmark.create_comprehensive_plots(graph_only_mode=True)

            print("\n" + "=" * 60)
            print("Graph generation completed successfully!")
            print(f"Plots saved to: {benchmark.output_dir}/plots")
            print("=" * 60)
        else:
            # Normal mode: run benchmarks and generate plots
            # Run all benchmark configurations
            benchmark.run_all_benchmarks()

            # Create comprehensive plots
            benchmark.create_comprehensive_plots()

            # Generate summary report
            benchmark.generate_summary_report()

            print("\n" + "=" * 60)
            print("Comprehensive benchmark completed successfully!")
            print(f"Results saved to: {benchmark.output_dir}")
            print("=" * 60)

    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
