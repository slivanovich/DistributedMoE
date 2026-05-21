#!/usr/bin/env python3
"""
Fault-tolerance benchmark for DistMoE system.

This benchmark tests the fault tolerance capabilities of the DistMoE system by:
1. Creating a queue of input requests with configurable RPS
2. Processing requests through DistMoE instance with 2 Expert Blocks (each on unique GPU)
3. Measuring actual RPS and system performance under load
4. Simulating expert block failures and recovery scenarios
"""

import asyncio
import copy
import os
import time
import timeit
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Event
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import MoE components
from TensorTransferEngine.TensorTransferEngine import TensorTransferEngineBackend, TensorTransferEngineConfig
from TensorTransferEngine.utils import HandshakeType, TransferProtocol
from MoE.MoE import Qwen3MoEMLP
from MoE.DistExpertsBlock import DistExpertsBlock
from MoE.DistMoEBlock import DistMoEBlock, DistMoEBlockConfig
from MoE.MetricsCollector import MetricsCollector, MetricsCollectorConfig
from utils.logging_config import get_logger

# Configuration constants
HOST_PORT = 40000  # Use high port range to avoid conflicts
EXPERT_BLOCK_PORTS_START_INDEX = 45000
NUMBER_OF_EXPERT_BLOCKS = 2  # Back to original 2 Expert Blocks for debugging
NUMBER_OF_EXPERTS = 128
TOP_K = 8
CONTEXT_LENGTH = 8192  # 8k context as requested
IO_BUFFER_SIZE = TOP_K * 1024
NUMBER_OF_IO_BUFFERS = 32
HIDDEN_FEATURES = 2048
INTERMEDIATE_FEATURES = 6144


@dataclass
class RequestBatch:
    """Represents a single request batch."""

    id: int
    data: torch.Tensor
    timestamp: float
    processed_timestamp: Optional[float] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration for fault tolerance benchmark."""

    target_rps: float  # Target requests per second
    duration_seconds: int  # Benchmark duration
    failure_injection_time_seconds: int = 10
    failure_downtime_seconds: int = 10
    context_length: int = CONTEXT_LENGTH
    batch_size: int = 1  # Requests per batch
    backend: TensorTransferEngineBackend = TensorTransferEngineBackend.NIXL
    transfer_protocol: TransferProtocol = TransferProtocol.RDMA
    dtype: torch.dtype = torch.float16
    host_device: torch.device = torch.device("cuda:0")  # Host on GPU 0
    expert_block_devices: Optional[List[torch.device]] = None  # Will be set to [cuda:1, cuda:2]
    output_dir: str = "./fault_tolerance_results"

    def __post_init__(self):
        if self.expert_block_devices is None:
            # Each Expert Block on its own unique GPU
            if NUMBER_OF_EXPERT_BLOCKS == 1:
                self.expert_block_devices = [torch.device("cuda:1")]
            else:
                self.expert_block_devices = [torch.device("cuda:1"), torch.device("cuda:2")]


class RequestGenerator:
    """Generates requests at specified RPS."""

    def __init__(self, config: BenchmarkConfig, request_queue: Queue):
        self.config = config
        self.request_queue = request_queue
        self.logger = get_logger("fault_tolerance.generator")
        self.stop_event = Event()
        self.request_id = 0

    def generate_random_batch(self) -> torch.Tensor:
        """Generate a random batch with 8k context."""
        # Calculate batch shape to achieve 8k context
        sequence_length = min(self.config.context_length, 2048)  # Max sequence length per batch
        batch_size = max(1, self.config.context_length // sequence_length)

        batch_shape = (batch_size, sequence_length, HIDDEN_FEATURES)
        return torch.randn(size=batch_shape, dtype=self.config.dtype, device=self.config.host_device) + 100

    def run(self):
        """Run request generation at target RPS."""
        interval = 1.0 / self.config.target_rps
        next_request_time = time.time()

        self.logger.info(f"Starting request generation at {self.config.target_rps} RPS (interval: {interval:.4f}s)")

        while not self.stop_event.is_set():
            current_time = time.time()

            if current_time >= next_request_time:
                # Generate and queue request
                batch_data = self.generate_random_batch()
                request = RequestBatch(id=self.request_id, data=batch_data, timestamp=current_time)

                try:
                    self.request_queue.put_nowait(request)
                    self.request_id += 1
                except:
                    self.logger.warning("Request queue full, dropping request")

                next_request_time += interval
            else:
                # Sleep until next request time
                sleep_time = min(0.001, next_request_time - current_time)
                time.sleep(sleep_time)

    def stop(self):
        """Stop request generation."""
        self.stop_event.set()


class RequestProcessor:
    """Processes requests using DistMoE instance with 2 Expert Blocks on separate GPUs."""

    def __init__(self, config: BenchmarkConfig, request_queue: Queue, result_queue: Queue):
        self.config = config
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.logger = get_logger("fault_tolerance.processor")
        self.stop_event = Event()
        self.ready_event = Event()
        self.dist_moe: Optional[DistMoEBlock] = None
        self.expert_block_threads = []
        self.expert_blocks = []
        self.metrics_collector: Optional[MetricsCollector] = None
        self.dist_moe_config: Optional[DistMoEBlockConfig] = None
        self.expert_layers_per_block: List[List[Qwen3MoEMLP]] = []
        self.expert_block_tte_configs: List[TensorTransferEngineConfig] = []
        self.expert_block_runtime: Dict[int, DistExpertsBlock] = {}
        self.expert_blocks_experts: List[List[int]] = []
        self.expert_block_threads_map: Dict[int, Thread] = {}

    def start_expert_block(self, experts_block_index: int) -> None:
        assert self.dist_moe_config is not None
        assert self.metrics_collector is not None
        assert experts_block_index < len(self.expert_layers_per_block)
        assert experts_block_index < len(self.expert_block_tte_configs)

        dist_moe_config = self.dist_moe_config
        metrics_collector = self.metrics_collector

        if experts_block_index in self.expert_block_threads_map:
            thread = self.expert_block_threads_map[experts_block_index]
            if thread.is_alive():
                self.logger.warning(f"Expert Block {experts_block_index} is already running")
                return

        def thread_job(block_index=experts_block_index):
            expert_block = DistExpertsBlock(
                block_index,
                dist_moe_config,
                self.expert_layers_per_block[block_index],
                self.expert_block_tte_configs[block_index],
                metrics_collector,
            )
            self.expert_block_runtime[block_index] = expert_block
            expert_block.main_loop()
            expert_block.close()
            self.expert_block_runtime.pop(block_index, None)

        t = Thread(target=thread_job, daemon=False)
        t.start()
        self.expert_block_threads_map[experts_block_index] = t
        self.expert_block_threads.append(t)

    def stop_expert_block(self, experts_block_index: int, join_timeout: float = 10.0) -> None:
        expert_block = self.expert_block_runtime.get(experts_block_index)
        if expert_block is not None:
            self.logger.info(f"Stopping Expert Block {experts_block_index}")
            expert_block.close()

        thread = self.expert_block_threads_map.get(experts_block_index)
        if thread is not None:
            thread.join(timeout=join_timeout)
            if not thread.is_alive():
                self.expert_block_threads_map.pop(experts_block_index, None)

    def setup_dist_moe(self):
        """Setup DistMoE instance and expert blocks on separate GPUs."""
        self.logger.info("Setting up DistMoE instance with 2 Expert Blocks on separate GPUs...")

        # Setup expert blocks distribution - all experts are shared across all blocks.
        expert_blocks_experts = [list(range(NUMBER_OF_EXPERTS)) for _ in range(NUMBER_OF_EXPERT_BLOCKS)]
        self.expert_blocks_experts = expert_blocks_experts

        for i in range(NUMBER_OF_EXPERT_BLOCKS):
            self.logger.info(f"Expert Block {i}: {len(expert_blocks_experts[i])} experts")

        # Setup ports
        host_p2p_ports = [HOST_PORT + 1000 * (index + 1) for index in range(NUMBER_OF_EXPERT_BLOCKS)]
        expert_block_hosts = ["localhost"] * NUMBER_OF_EXPERT_BLOCKS
        expert_block_ports = [EXPERT_BLOCK_PORTS_START_INDEX + 100 * i for i in range(NUMBER_OF_EXPERT_BLOCKS)]
        expert_block_p2p_ports = [EXPERT_BLOCK_PORTS_START_INDEX + 100 * i + 1000 for i in range(NUMBER_OF_EXPERT_BLOCKS)]

        # Health check ports allocation - ensure unique ports for each expert block with large spacing
        host_health_check_ports = [HOST_PORT + 5000 + 2000 * index for index in range(NUMBER_OF_EXPERT_BLOCKS)]
        remote_health_check_ports = [EXPERT_BLOCK_PORTS_START_INDEX + 100 * index + 5000 for index in range(NUMBER_OF_EXPERT_BLOCKS)]

        # Initialize metrics collector
        metrics_config = MetricsCollectorConfig(
            enable_collection=True,
            enable_detailed_timing=True,
            enable_transfer_metrics=True,
            enable_expert_block_metrics=True,
            max_stored_forwards=10000,
            export_directory=self.config.output_dir,
        )
        metrics_collector = MetricsCollector(metrics_config)
        self.metrics_collector = metrics_collector

        # Ensure expert_block_devices is not None
        assert self.config.expert_block_devices is not None, "expert_block_devices must be set"

        # Create DistMoE config
        dist_moe_config = DistMoEBlockConfig(
            backend=self.config.backend,
            expert_blocks=expert_blocks_experts,
            number_of_experts=NUMBER_OF_EXPERTS,
            top_k=TOP_K,
            number_of_io_buffers=NUMBER_OF_IO_BUFFERS,
            io_buffer_size=IO_BUFFER_SIZE,
            hidden_features=HIDDEN_FEATURES,
            dtype=self.config.dtype,
            moe_device=self.config.host_device,
            eb_devices=self.config.expert_block_devices,
            forward_timeout=3000,
            host_health_check_ports=host_health_check_ports,
            remote_health_check_ports=remote_health_check_ports,
        )
        self.dist_moe_config = dist_moe_config

        # Create expert layers for each expert block
        expert_layers_per_block = []
        for experts_block_index in range(NUMBER_OF_EXPERT_BLOCKS):
            expert_layers = []
            for expert_index in expert_blocks_experts[experts_block_index]:
                expert_layer = Qwen3MoEMLP(HIDDEN_FEATURES, INTERMEDIATE_FEATURES, self.config.dtype)
                expert_layers.append(expert_layer)
            expert_layers_per_block.append(expert_layers)
        self.expert_layers_per_block = expert_layers_per_block

        self.expert_block_tte_configs = []
        # Start expert blocks on their respective GPUs FIRST
        for experts_block_index in range(NUMBER_OF_EXPERT_BLOCKS):
            device = self.config.expert_block_devices[experts_block_index]
            self.logger.info(f"Starting Expert Block {experts_block_index} on {device}")

            eb_tte_config = TensorTransferEngineConfig(
                host_address=expert_block_hosts[experts_block_index],
                host_port=expert_block_ports[experts_block_index],
                host_p2p_ports=[expert_block_p2p_ports[experts_block_index]],
                remote_addresses=["localhost"],
                remote_ports=[HOST_PORT],
                remote_p2p_ports=[host_p2p_ports[experts_block_index]],
                handshake_type=HandshakeType.CLIENT,
                transfer_protocol=self.config.transfer_protocol,
                ib_device_names="",
                p2p_timeout_duration=1000,
                metadata_schema="",
                metadata_host="",
                metadata_port=0,
                metadata_dir="",
            )
            self.expert_block_tte_configs.append(eb_tte_config)

            self.start_expert_block(experts_block_index)

        # Create TensorTransferEngine config for DistMoE
        dist_moe_tte_config = TensorTransferEngineConfig(
            host_address="localhost",
            host_port=HOST_PORT,
            host_p2p_ports=host_p2p_ports,
            remote_addresses=expert_block_hosts,
            remote_ports=expert_block_ports,
            remote_p2p_ports=expert_block_p2p_ports,
            handshake_type=HandshakeType.SERVER,
            transfer_protocol=self.config.transfer_protocol,
            ib_device_names="",
            p2p_timeout_duration=1000,
            metadata_schema="",
            metadata_host="",
            metadata_port=0,
            metadata_dir="",
        )

        # Create DistMoEBlock (this will block until handshake completes)
        self.dist_moe = DistMoEBlock(dist_moe_config, dist_moe_tte_config, metrics_collector)

        self.logger.info("DistMoE setup completed with 2 Expert Blocks on separate GPUs")

    def process_request(self, request: RequestBatch) -> RequestBatch:
        """Process a single request through DistMoE."""
        try:
            start_time = time.time()

            # Process through DistMoE
            if self.dist_moe is not None:
                output = self.dist_moe.forward(request.data)
            else:
                raise RuntimeError("DistMoE instance is not initialized")

            request.processed_timestamp = time.time()
            request.success = True

            latency_ms = (request.processed_timestamp - start_time) * 1000
            self.logger.info(f"Processed request {request.id} in {latency_ms:.2f} ms")

        except Exception as e:
            request.processed_timestamp = time.time()
            request.success = False
            request.error = str(e)
            self.logger.error(f"Failed to process request {request.id}: {e}")

        return request

    def run(self):
        """Run request processing."""
        self.setup_dist_moe()
        self.ready_event.set()

        self.logger.info("Starting request processing...")

        while not self.stop_event.is_set():
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=0.1)

                # Process request
                processed_request = self.process_request(request)

                # Put result in result queue
                self.result_queue.put(processed_request)

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in request processing loop: {e}")

    def stop(self):
        """Stop request processing."""
        self.stop_event.set()
        if self.dist_moe:
            self.dist_moe.close()

        # Wait for expert block threads to finish
        for thread in self.expert_block_threads:
            thread.join(timeout=10.0)


class FaultToleranceBenchmark:
    """Main fault tolerance benchmark class."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = get_logger("fault_tolerance.benchmark")
        self.request_queue = Queue(maxsize=100000)  # Limit queue size
        self.result_queue = Queue()

        # Components
        self.generator = RequestGenerator(config, self.request_queue)
        self.processor = RequestProcessor(config, self.request_queue, self.result_queue)

        # Threads
        self.generator_thread = None
        self.processor_thread = None

        # Results
        self.results = []
        self.rps_timeline: List[Dict[str, float]] = []
        self.failure_triggered = False
        self.restore_triggered = False

    def run(self):
        """Run the fault tolerance benchmark."""
        self.logger.info(
            f"Starting fault tolerance benchmark - Target RPS: {self.config.target_rps}, Duration: {self.config.duration_seconds}s"
        )
        self.logger.info(f"Host device: {self.config.host_device}")
        self.logger.info(f"Expert Block devices: {self.config.expert_block_devices}")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Start processor thread
        self.processor_thread = Thread(target=self.processor.run, daemon=True)
        self.processor_thread.start()

        # Wait until the processor has fully initialized DistMoE.
        if not self.processor.ready_event.wait(timeout=60.0):
            raise RuntimeError("Processor failed to initialize DistMoE within timeout")

        # Start generator thread
        self.generator_thread = Thread(target=self.generator.run, daemon=True)
        self.generator_thread.start()

        # Monitor and collect results
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds
        completed_per_second: Dict[int, int] = {}

        while time.time() < end_time:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if (not self.failure_triggered) and elapsed_time >= self.config.failure_injection_time_seconds:
                self.logger.warning("Injecting failure: disabling Expert Block 1")
                if self.processor.dist_moe is not None:
                    self.processor.dist_moe.disable_experts_block(1)
                self.failure_triggered = True

            if (
                self.failure_triggered
                and (not self.restore_triggered)
                and elapsed_time >= self.config.failure_injection_time_seconds + self.config.failure_downtime_seconds
            ):
                self.logger.warning("Restoring Expert Block 1")
                if self.processor.dist_moe is not None:
                    self.processor.dist_moe.enable_experts_block(1)
                self.restore_triggered = True

            try:
                # Collect processed results
                result = self.result_queue.get(timeout=0.1)
                self.results.append(result)

                if result.processed_timestamp is not None:
                    bucket = int(result.processed_timestamp - start_time)
                    completed_per_second[bucket] = completed_per_second.get(bucket, 0) + 1

                if len(self.results) % 100 == 0:
                    self.logger.info(f"Processed {len(self.results)} requests")

                # Add timeout check for forward pass
                if current_time - start_time > self.config.duration_seconds:
                    self.logger.warning(
                        f"Benchmark duration exceeded ({current_time - start_time:.2f}s > {self.config.duration_seconds}s), stopping..."
                    )
                    break

            except Empty:
                continue

        # Stop components
        self.generator.stop()
        if self.generator_thread is not None:
            self.generator_thread.join(timeout=5.0)

        # Allow the processor to drain already generated requests before shutdown
        drain_deadline = time.time() + 10.0
        while (not self.request_queue.empty()) and time.time() < drain_deadline:
            time.sleep(0.05)

        self.processor.stop()
        if self.processor_thread is not None:
            self.processor_thread.join(timeout=10.0)

        # Collect remaining results
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                self.results.append(result)
                if result.processed_timestamp is not None:
                    bucket = int(result.processed_timestamp - start_time)
                    completed_per_second[bucket] = completed_per_second.get(bucket, 0) + 1
            except Empty:
                break

        self.rps_timeline = [
            {"time_s": second, "rps": float(completed_per_second.get(second, 0))}
            for second in range(max(self.config.duration_seconds, max(completed_per_second.keys(), default=-1) + 1))
        ]

        self.logger.info(f"Benchmark completed. Processed {len(self.results)} requests")

        # Analyze results
        self.analyze_results()

    def analyze_results(self):
        """Analyze benchmark results and calculate metrics."""
        if not self.results:
            self.logger.warning("No results to analyze")
            return

        # Calculate metrics
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]

        total_requests = len(self.results)
        success_rate = len(successful_requests) / total_requests if total_requests > 0 else 0

        # Calculate actual RPS
        if successful_requests:
            first_timestamp = min(r.timestamp for r in successful_requests)
            last_timestamp = max(r.processed_timestamp for r in successful_requests if r.processed_timestamp)
            duration = last_timestamp - first_timestamp
            actual_rps = len(successful_requests) / duration if duration > 0 else 0
        else:
            actual_rps = 0

        # Calculate latencies
        latencies = []
        for r in successful_requests:
            if r.processed_timestamp:
                latency = r.processed_timestamp - r.timestamp
                latencies.append(latency)

        avg_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0

        avg_latency_ms = avg_latency * 1000
        p95_latency_ms = p95_latency * 1000
        p99_latency_ms = p99_latency * 1000

        # Log results
        self.logger.info("=== Benchmark Results ===")
        self.logger.info(f"Target RPS: {self.config.target_rps}")
        self.logger.info(f"Actual RPS: {actual_rps:.2f}")
        self.logger.info(f"Total Requests: {total_requests}")
        self.logger.info(f"Successful Requests: {len(successful_requests)}")
        self.logger.info(f"Failed Requests: {len(failed_requests)}")
        self.logger.info(f"Success Rate: {success_rate:.2%}")
        self.logger.info(f"Average Latency: {avg_latency_ms:.2f} ms")
        self.logger.info(f"P95 Latency: {p95_latency_ms:.2f} ms")
        self.logger.info(f"P99 Latency: {p99_latency_ms:.2f} ms")

        # Save results to CSV
        results_data = []
        for r in self.results:
            results_data.append(
                {
                    "request_id": r.id,
                    "timestamp": r.timestamp,
                    "processed_timestamp": r.processed_timestamp,
                    "success": r.success,
                    "error": r.error,
                    "latency": (r.processed_timestamp - r.timestamp) if r.processed_timestamp else None,
                }
            )

        df = pd.DataFrame(results_data)
        csv_path = os.path.join(self.config.output_dir, f"fault_tolerance_results_rps_{self.config.target_rps}.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")

        if self.rps_timeline:
            timeline_df = pd.DataFrame(self.rps_timeline)
            timeline_csv_path = os.path.join(self.config.output_dir, f"fault_tolerance_rps_timeline_rps_{self.config.target_rps}.csv")
            timeline_df.to_csv(timeline_csv_path, index=False)

            fig, ax = plt.subplots(1, 1, figsize=(20, 8))
            fig.suptitle("Fault Tolerance Benchmark: RPS Over Time", fontsize=16, fontweight="bold", x=0.5, ha="center")

            ax.plot(
                timeline_df["time_s"],
                timeline_df["rps"],
                color="#1f77b4",
                marker="o",
                linewidth=3,
                markersize=6,
                label="Observed RPS",
            )
            ax.axhline(
                y=self.config.target_rps,
                color="#9467bd",
                linestyle=":",
                linewidth=2,
                label="Target RPS",
            )
            ax.axvline(
                x=self.config.failure_injection_time_seconds,
                color="#d62728",
                linestyle="--",
                linewidth=2,
                label="Expert block failure",
            )
            ax.axvline(
                x=self.config.failure_injection_time_seconds + self.config.failure_downtime_seconds,
                color="#2ca02c",
                linestyle="--",
                linewidth=2,
                label="Expert block restore",
            )
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("RPS", fontsize=12)
            ax.set_title("Low Expert Activation", fontsize=14, fontweight="bold", loc="center")
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(self.config.output_dir, f"fault_tolerance_rps_timeline_rps_{self.config.target_rps}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"RPS timeline saved to {timeline_csv_path}")
            self.logger.info(f"RPS plot saved to {plot_path}")


def main():
    """Main function to run fault tolerance benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Run fault tolerance MoE benchmark")
    parser.add_argument("--backend", choices=["nixl", "mcte"], default="nixl", help="Backend to use")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16", help="Data type")
    parser.add_argument("--host_device", default="cuda:0", help="Host device")
    parser.add_argument("--expert_device1", default="cuda:1", help="First expert block device")
    parser.add_argument("--expert_device2", default="cuda:2", help="Second expert block device")
    parser.add_argument("--transfer_protocol", choices=["rdma", "nvlink"], default="rdma", help="Transfer protocol")
    parser.add_argument("--target_rps", type=float, default=10.0, help="Target requests per second")
    parser.add_argument("--duration", type=int, default=40, help="Benchmark duration in seconds")
    parser.add_argument("--failure_after", type=int, default=10, help="Seconds before expert-block failure injection")
    parser.add_argument("--downtime", type=int, default=10, help="Seconds before expert-block restore")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Parse backend
    if args.backend == "nixl":
        backend = TensorTransferEngineBackend.NIXL
    else:
        backend = TensorTransferEngineBackend.MoonCake

    # Parse dtype
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Parse devices
    host_device = torch.device(args.host_device)
    expert_device1 = torch.device(args.expert_device1)
    expert_device2 = torch.device(args.expert_device2)

    # Parse transfer protocol
    if args.transfer_protocol == "rdma":
        transfer_protocol = TransferProtocol.RDMA
    else:
        transfer_protocol = TransferProtocol.NVLINK

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./fault_tolerance_results"

    print("Starting Fault Tolerance MoE Benchmark")
    print("=" * 50)
    print(f"Backend: {args.backend}")
    print(f"Host Device: {host_device}")
    print(f"Experts Block #0 Device: {expert_device1}")
    print(f"Experts Block #1 Device: {expert_device2}")
    print(f"Data Type: {dtype}")
    print(f"Transfer Protocol: {args.transfer_protocol.upper()}")
    print(f"Target RPS: {args.target_rps}")
    print(f"Duration: {args.duration}s")
    print(f"Failure Injection After: {args.failure_after}s")
    print(f"Failure Downtime: {args.downtime}s")
    print(f"Output Directory: {output_dir}")
    print("=" * 50)

    # Create configuration
    config = BenchmarkConfig(
        target_rps=args.target_rps,
        duration_seconds=args.duration,
        failure_injection_time_seconds=args.failure_after,
        failure_downtime_seconds=args.downtime,
        backend=backend,
        transfer_protocol=transfer_protocol,
        dtype=dtype,
        host_device=host_device,
        expert_block_devices=[expert_device1, expert_device2],
        output_dir=output_dir,
    )

    benchmark = FaultToleranceBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()
