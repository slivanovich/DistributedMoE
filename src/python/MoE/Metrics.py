from dataclasses import dataclass
from typing import List, Any


@dataclass
class DispatchMetrics:
    e2e_latency: float  # ms


@dataclass
class CombineMetrics:
    e2e_latency: float  # ms


@dataclass
class ExpertsRunMetrics:
    experts_block_index: int
    e2e_latency: float  # ms
    read_transfer_throughput: float  # MB/ms
    write_transfer_throughput: float  # MB/ms


@dataclass
class ExpertsBlockMetrics:
    number_of_runs: int
    number_of_transfers: int  # 2 * number_of_runs
    avg_e2e_latency: float  # ms
    avg_transfers_throughput: float  # MB/ms
    experts_runs: List[ExpertsRunMetrics]


@dataclass
class ForwardMetrics:
    e2e_latency: float  # ms
    setup_latency: float  # ms
    avg_dispatches_e2e_latency: float  # ms
    avg_combines_e2e_latency: float  # ms
    dispatches: List[DispatchMetrics]
    combines: List[CombineMetrics]
    experts_blocks: List[ExpertsBlockMetrics]


@dataclass
class Metrics:
    number_of_forwards: int
    forwards: List[ForwardMetrics]
