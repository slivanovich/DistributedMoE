import bisect
from dataclasses import dataclass
import logging
import math
import struct
from typing import Any, Dict, List, Tuple, Optional
from queue import Queue

import asyncio
import threading
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import zmq
import zmq.asyncio

from TensorTransferEngine.TensorTransferEngine import (
    TensorTransferEngine,
    TensorTransferEngineBackend,
    TensorTransferEngineConfig,
)
from TensorTransferEngine.MCTETensorTransferEngine import MCTETensorTransferEngine
from TensorTransferEngine.NIXLTensorTransferEngine import NIXLTensorTransferEngine
from TensorTransferEngine.utils import P2PPair, join_host_port

from MoE.DistMoE import ExpertMetaData, P2PHeaders
from MoE.MetricsCollector import MetricsCollector
from MoE.PipelineTask import HostPipelineTask
from MoE.pools.BuffersPool import BuffersPool
from MoE.pools.ExpertBlocksPool import ExpertBlocksPool
from utils.logging_config import get_logger


@dataclass
class DistMoEBlockConfig:
    # TensorTransferEngine backend.
    backend: TensorTransferEngineBackend

    # Total number of unique experts (shared ones count as one expert).
    number_of_experts: int

    # List of the expert indexes in each of the expert blocks.
    expert_blocks: List[List[int]]

    # Total number (N) of the I/O buffers in buffer pools (N for input and N for output).
    number_of_io_buffers: int

    # I/O buffers size in tokens.
    io_buffer_size: int

    # Model's parameters.
    top_k: int
    hidden_features: int

    # DType of the input/output batches.
    dtype: torch.dtype

    # DistMoE block device.
    moe_device: torch.device

    # DistExpertsBlocks' devices.
    eb_devices: List[torch.device]

    # Forward pass timeout in milliseconds (SLA guarantee).
    forward_timeout: int

    # Health check ports configuration for separate P2P channels.
    host_health_check_ports: List[int]
    remote_health_check_ports: List[int]


class DistMoEBlock(nn.Module):

    def __init__(
        self, config: DistMoEBlockConfig, tte_config: TensorTransferEngineConfig, metrics_collector: Optional[MetricsCollector] = None
    ):
        super().__init__()

        # Initialize logger.
        self.logger: logging.Logger | logging.LoggerAdapter = get_logger("dist_moe.moe.block")

        # Track close state to prevent double closing.
        self._is_closed: bool = False

        # Metrics collector for performance monitoring
        self.metrics_collector: Optional[MetricsCollector] = metrics_collector

        # MoE host device.
        self.device: torch.device = config.moe_device

        # Expert blocks parameters.
        self.number_of_expert_blocks: int = len(config.expert_blocks)
        self.number_of_experts: int = config.number_of_experts
        self.top_k: int = config.top_k

        # List of hit experts.
        self.experts_hit: List[int] = []

        # Expert layers parameters.
        self.hidden_features: int = config.hidden_features
        self.dtype: torch.dtype = config.dtype

        # Forward pass timeout configuration.
        self.forward_timeout: int = config.forward_timeout

        # Store health check configuration
        self.host_health_check_ports: List[int] = config.host_health_check_ports
        self.remote_health_check_ports: List[int] = config.remote_health_check_ports
        self.health_check_timeout: int = tte_config.p2p_timeout_duration

        # Create an event loop for async tasks and register it for the current thread
        # so that library code calling asyncio.get_event_loop() sees the correct loop.
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Event that is set once the event loop has fully stopped (replaces busy-wait).
        self._loop_stopped = threading.Event()

        # Initialize TensorTransferEngine instance.
        self.tensor_transfer_engine: TensorTransferEngine
        if config.backend == TensorTransferEngineBackend.NIXL:
            self.tensor_transfer_engine = NIXLTensorTransferEngine(tte_config)
        elif config.backend == TensorTransferEngineBackend.MoonCake:
            self.tensor_transfer_engine = MCTETensorTransferEngine(tte_config)

        # Setup separate health check P2P channels
        self.health_check_context = zmq.asyncio.Context()
        self.health_check_p2p_pairs: Dict[str, P2PPair] = {}
        self.setup_health_check(tte_config)

        # Router layer definition (can be replaced like the layers in the expert block class).
        self.gate: nn.Module = nn.Linear(self.hidden_features, self.number_of_experts, bias=False, dtype=self.dtype, device=self.device)

        # Allocate output batch tensor.
        self.output_batch: torch.Tensor = torch.zeros(1, dtype=self.dtype, device=self.device)

        # Initialize experts and expert blocks metadata.
        self.expert_blocks: List[List[int]] = config.expert_blocks
        self.expert_blocks_pool: ExpertBlocksPool = ExpertBlocksPool(self.expert_blocks)
        self.expert_streams: Dict[int, torch.cuda.Stream] = {}
        self.expert_events: Dict[int, torch.cuda.Event] = {}
        self.initialize_metadata()

        # initialize I/O buffers.
        self.number_of_io_buffers: int = config.number_of_io_buffers
        self.io_buffer_size: int = config.io_buffer_size
        self.activations_batch_size_candidates: List[int] = []
        self.initialize_io_buffers()

        # Async locks.
        self.async_lock: asyncio.Lock = asyncio.Lock()

        # Async event is set if there is no dispatched left (every dispatch is paired with its combine).
        self.combining: asyncio.Event = asyncio.Event()

        # Container for the pipeline tasks.
        self.pipeline_tasks: Dict[bytes, HostPipelineTask] = {}

        # Containers for the concurrent dispatch/combine execution.
        self.dispatch_tasks: Dict[bytes, asyncio.Task] = {}
        self.combine_tasks: Dict[bytes, asyncio.Task] = {}

        # Mapping expert block indexes to host pipeline tasks.
        self.expert_block_tasks: Dict[int, List[bytes]] = {}

        # Active pipeline tasks counter (must be mutable).
        self.active_pipeline_tasks: List[int] = [0]

        # Transferred activations counter per one forward pass (must be mutable).
        self.transferred_activations: List[int] = [0]

        # Total activations number for the current input batch.
        self.total_activations = 0

        # Monitoring tasks (P2P, health check) for some kind of fault tolerance.
        self.monitoring_flag: bool = True
        self.monitoring_tasks: List[asyncio.Task] = []
        self.disabled_expert_blocks: set[int] = set()
        self.setup_monitoring()

        self.logger.info(
            f"DistMoEBlock initialized - Experts: {self.number_of_experts}, Blocks: {self.number_of_expert_blocks}, Device: {self.device}"
        )

    def disable_experts_block(self, experts_block_index: int) -> None:
        self.disabled_expert_blocks.add(experts_block_index)
        self.expert_blocks_pool.mark_dead(experts_block_index)
        self.logger.info(f"Expert block {experts_block_index} disabled")

    def enable_experts_block(self, experts_block_index: int) -> None:
        self.disabled_expert_blocks.discard(experts_block_index)
        self.expert_blocks_pool.mark_alive(experts_block_index)
        self.logger.info(f"Expert block {experts_block_index} enabled")

    def setup_health_check(self, tte_config: TensorTransferEngineConfig) -> None:
        # Validate health check ports configuration.
        assert len(self.host_health_check_ports) == len(tte_config.remote_addresses)
        assert len(self.remote_health_check_ports) == len(tte_config.remote_addresses)

        for remote_index in range(len(tte_config.remote_addresses)):
            host_health_port = self.host_health_check_ports[remote_index]
            remote_name = join_host_port(tte_config.remote_addresses[remote_index], tte_config.remote_ports[remote_index])
            remote_health_address = tte_config.remote_addresses[remote_index]
            remote_health_port = self.remote_health_check_ports[remote_index]

            # Create separate P2P pair for health checker.
            self.health_check_p2p_pairs[remote_name] = P2PPair(
                self.health_check_context.socket(zmq.PUSH), self.health_check_context.socket(zmq.PULL)
            )

            # Setup health check pusher (for sending pings).
            self.health_check_p2p_pairs[remote_name].pusher.bind(join_host_port(tte_config.host_address, host_health_port, "tcp"))
            self.health_check_p2p_pairs[remote_name].pusher.setsockopt(zmq.RCVTIMEO, self.health_check_timeout)

            # Setup health check puller (for receiving pongs).
            self.health_check_p2p_pairs[remote_name].puller.connect(join_host_port(remote_health_address, remote_health_port, "tcp"))
            self.health_check_p2p_pairs[remote_name].puller.setsockopt(zmq.RCVTIMEO, self.health_check_timeout)

            self.logger.debug(
                f"Health check P2P PAIR: PUSH {tte_config.host_address}:{host_health_port}; PULL {remote_health_address}:{remote_health_port}"
            )

    def initialize_io_buffers(self) -> None:
        assert self.tensor_transfer_engine is not None

        self.input_buffers_pool: BuffersPool = BuffersPool(
            self.number_of_io_buffers,
            self.tensor_transfer_engine,
            self.io_buffer_size,
            self.hidden_features,
            self.dtype,
            self.device,
        )

        self.output_buffers_pool: BuffersPool = BuffersPool(
            self.number_of_io_buffers,
            self.tensor_transfer_engine,
            self.io_buffer_size,
            self.hidden_features,
            self.dtype,
            self.device,
        )

        # Syncronisation point.
        self.tensor_transfer_engine.handshake()

        packet = self._build_io_descriptors_packet()

        # Share the descriptors eagerly with all currently connected expert blocks.
        async def send_io_descriptors():
            for _, remote_name in enumerate(self.tensor_transfer_engine.remote_names):
                await self.tensor_transfer_engine.async_send_metadata_to(
                    remote_name,
                    P2PHeaders.IOTransferDescriptorsSubmit.value,
                    packet,
                )

        self.event_loop.run_until_complete(send_io_descriptors())

        step = self.io_buffer_size // 1024
        for candidate in range(step, self.io_buffer_size, step):
            self.activations_batch_size_candidates.append(candidate)
        self.activations_batch_size_candidates.append(self.io_buffer_size)

        self.logger.info(f"Experts activations aliignment candidates: {self.activations_batch_size_candidates}")

    def _build_io_descriptors_packet(self) -> List[Tuple[int, bytes]]:
        # Mapping from the I/O buffers transfer descs to the 2 bytes ids.
        # Packet structure:
        #   1) Number of the I/O buffers (the number of input and output buffers is always the same by design).
        #   2) [(2 bytes per buffer's ID, N bytes per serialized buffer's transfer descriptors) x 2 * (Number of the I/O buffers)] (input buffers' ids first).
        packet: List[Tuple[int, bytes]] = [(2, struct.pack("!H", self.number_of_io_buffers))]

        # Assign unique 2-byte IDs to input buffers (0..N-1) and output buffers (N..2N-1).
        for index in range(self.number_of_io_buffers):
            input_buffer = self.input_buffers_pool.buffers[index]
            input_buffer.id = index.to_bytes(length=2, byteorder="big")
            packet.append((2, input_buffer.id))
            packet.append(input_buffer.serialized_transfer_descs)

        for index in range(self.number_of_io_buffers):
            output_buffer = self.output_buffers_pool.buffers[index]
            output_buffer.id = (self.number_of_io_buffers + index).to_bytes(length=2, byteorder="big")
            packet.append((2, output_buffer.id))
            packet.append(output_buffer.serialized_transfer_descs)

        return packet

    def initialize_metadata(self) -> None:
        # Setup experts metadata.
        self.experts_metadata: Dict[int, ExpertMetaData] = {}
        for expert_index in range(self.number_of_experts):
            self.experts_metadata[expert_index] = ExpertMetaData(
                block_indexes=[],
                expert_shared_rate=0,
                expert_weights=torch.empty(1, dtype=self.dtype, device=self.device),
                expert_token_indexes=torch.empty(1, dtype=torch.int, device=self.device),
                expert_total_activations=0,
                expert_offsets=Queue(),
            )
            self.expert_streams[expert_index] = torch.cuda.Stream(device=self.device)
            self.expert_events[expert_index] = torch.cuda.Event()

        # Setup expert blocks metadata.
        for experts_block_index in range(self.number_of_expert_blocks):
            for expert_index in self.expert_blocks[experts_block_index]:
                self.experts_metadata[expert_index].block_indexes.append(experts_block_index)
                self.experts_metadata[expert_index].expert_shared_rate += 1

    async def _abort_all_dispatched_to(self, experts_block_index: int) -> None:
        # Hold async_lock while mutating the shared counters so that concurrent
        # dispatch() / combine() coroutines (which also hold the lock when they
        # update the same counters) cannot interleave with this abort path.
        async with self.async_lock:
            pipeline_task_names_to_be_aborted = self.expert_block_tasks.get(experts_block_index, [])
            self.expert_block_tasks[experts_block_index] = []

            for pipeline_task_name in pipeline_task_names_to_be_aborted:
                pipeline_task = self.pipeline_tasks.get(pipeline_task_name)
                if pipeline_task is not None:
                    pipeline_task.abort()
                    if pipeline_task.dispatch_done:
                        self.transferred_activations[0] -= pipeline_task.used_activations
                        if not pipeline_task.combine_done:
                            self.active_pipeline_tasks[0] -= 1

                dispatch_task = self.dispatch_tasks.get(pipeline_task_name)
                if dispatch_task is not None:
                    dispatch_task.cancel()

                combine_task = self.combine_tasks.get(pipeline_task_name)
                if combine_task is not None:
                    combine_task.cancel()

                self.dispatch_tasks.pop(pipeline_task_name, None)
                self.combine_tasks.pop(pipeline_task_name, None)
                self.pipeline_tasks.pop(pipeline_task_name, None)

            if len(pipeline_task_names_to_be_aborted) > 0:
                self.combining.set()

    async def _expert_block_health_checker(self, experts_block_index: int, remote_name: str) -> None:
        try:
            while self.monitoring_flag:
                try:
                    await asyncio.sleep(0.02)  # 20 ms

                    if experts_block_index in self.disabled_expert_blocks:
                        continue

                    # Get dedicated health check P2P pair.
                    health_check_pair = self.health_check_p2p_pairs.get(remote_name)

                    if not health_check_pair:
                        self.logger.warning(f"There is no p2p setup for the EB #{experts_block_index} ({remote_name}) health check")
                        continue

                    # Send ping using dedicated health check channel.
                    try:
                        ping_packet = self.tensor_transfer_engine.serialize_p2p_packet(P2PHeaders.Ping.value, [])
                        await health_check_pair.pusher.send(ping_packet)
                        self.logger.debug(f"Health checker: sent ping to EB #{experts_block_index}")
                    except Exception as ping_error:
                        self.logger.error(f"Health checker: failed to send ping to EB #{experts_block_index}: {ping_error}")
                        continue

                    # Wait for pong response using dedicated health check channel.
                    try:
                        packet = await asyncio.wait_for(health_check_pair.puller.recv(), timeout=1.0)
                        header, _ = self.tensor_transfer_engine.deserialize_p2p_packet(packet)

                        if header == P2PHeaders.Pong.value:
                            self.logger.debug(f"Health checker: received pong from EB #{experts_block_index}")
                            # Mark expert block as alive.
                            if (
                                experts_block_index not in self.disabled_expert_blocks
                                and not self.expert_blocks_pool.expert_blocks_metadata[experts_block_index].is_alive
                            ):
                                self.expert_blocks_pool.mark_alive(experts_block_index)
                                self.logger.info(f"EB #{experts_block_index} recovered and marked as alive")
                        else:
                            self.logger.warning(f"Unexpected header {header} in health check response from {remote_name}")
                            raise Exception(f"Unexpected health check response header: {header}")

                    except Exception as pong_error:
                        # Health check failed - mark as dead and abort tasks.
                        self.logger.debug(f"Health checker: failed to get pong from EB #{experts_block_index}: {pong_error}")
                        if self.expert_blocks_pool.expert_blocks_metadata[experts_block_index].is_alive:
                            self.expert_blocks_pool.mark_dead(experts_block_index)
                            self.logger.warning(f"EB #{experts_block_index} marked as dead due to health check failure")
                            await self._abort_all_dispatched_to(experts_block_index)

                    # If we are stuck (failed at dispatching) and there is no active tasks - trying to dispatch more.
                    if self.active_pipeline_tasks[0] == 0:
                        self.combining.set()

                except asyncio.CancelledError:
                    self.logger.debug(f"Health checker for block {experts_block_index} cancelled")
                    break
                except Exception as e:
                    self.expert_blocks_pool.mark_dead(experts_block_index)
                    self.logger.warning(f"Expert block {experts_block_index} marked as dead in health checker")
                    await self._abort_all_dispatched_to(experts_block_index)
                    self.logger.error(f"Health check failure for block {experts_block_index}: {e}")
                finally:
                    if self.monitoring_flag:
                        status = (
                            "alive" if self.expert_blocks_pool.expert_blocks_metadata[experts_block_index].alive_event.is_set() else "dead"
                        )
                        self.logger.debug(f"Health check report: {remote_name} (expert block #{experts_block_index}) is {status}")
        except asyncio.CancelledError:
            self.logger.debug(f"Health checker for block {experts_block_index} cancelled")

    async def _expert_block_p2p_handler(self, experts_block_index: int, remote_name: str):
        try:
            while self.monitoring_flag:
                # If this expert block is not alive we can just wait for the alive-event to be set.
                # await self.expert_blocks_pool.expert_blocks_metadata[experts_block_index].alive_event.wait()

                header = -1
                metadata = []

                try:
                    header, metadata = await self.tensor_transfer_engine.async_get_metadata_from(remote_name)
                except asyncio.CancelledError:
                    self.logger.warning(f"P2P handler for block {experts_block_index} cancelled")
                    break
                except Exception as e:
                    if not self.expert_blocks_pool.expert_blocks_metadata[experts_block_index].is_alive:
                        self.logger.info(f"P2P handler: expert block #{experts_block_index} already marked as dead, continuing...")
                        continue

                if header != -1:
                    self.logger.debug(
                        f"P2P handler: got '{P2PHeaders(header).name}' header from block #{experts_block_index}; {self.transferred_activations[0]}/{self.total_activations} activations passed"
                    )

                    # Not really necessary due to P2PHeaders.Pong handling but still.
                    if (
                        experts_block_index not in self.disabled_expert_blocks
                        and not self.expert_blocks_pool.expert_blocks_metadata[experts_block_index].is_alive
                    ):
                        self.expert_blocks_pool.mark_alive(experts_block_index)

                    if header == P2PHeaders.HostOutputBufferUnlocked.value:
                        pipeline_task_name: bytes = metadata[-1][1]
                        pipeline_task = self.pipeline_tasks.get(pipeline_task_name)

                        if pipeline_task is not None:

                            async def combine(pipeline_task: HostPipelineTask | None):
                                try:
                                    await pipeline_task.combine(self.active_pipeline_tasks, self.output_batch, self.combining)  # type: ignore
                                finally:
                                    self.combine_tasks.pop(pipeline_task_name, None)
                                    self.dispatch_tasks.pop(pipeline_task_name, None)
                                    self.pipeline_tasks.pop(pipeline_task_name, None)
                                    pipeline_task = None

                            combine_task = asyncio.create_task(combine(pipeline_task))
                            self.combine_tasks[pipeline_task_name] = combine_task
                        else:
                            self.combine_tasks.pop(pipeline_task_name, None)
                            self.dispatch_tasks.pop(pipeline_task_name, None)
                            in_dispatch_tasks = pipeline_task_name in self.dispatch_tasks
                            in_combine_tasks = pipeline_task_name in self.combine_tasks
                            owning_blocks = [
                                block_index
                                for block_index, task_names in self.expert_block_tasks.items()
                                if pipeline_task_name in task_names
                            ]
                            self.logger.debug(
                                f"P2P handler: stale combine ack for cleaned task '{pipeline_task_name.hex()[:8]}' "
                                f"(in_dispatch_tasks={in_dispatch_tasks}, in_combine_tasks={in_combine_tasks}, owning_blocks={owning_blocks})"
                            )
                    elif header == P2PHeaders.HostInputBufferUnlocked.value:
                        pipeline_task_name: bytes = metadata[-1][1]
                        pipeline_task = self.pipeline_tasks.get(pipeline_task_name)

                        if pipeline_task is not None:
                            pipeline_task.release_input_buffer()
                        else:
                            self.logger.warning(
                                f"P2P monitor: pipeline task mismatch while releasing input buffer - got task with name '{pipeline_task_name.hex()[:8]}'"
                            )
                    elif header == P2PHeaders.IOTransferDescriptorsRequest.value:
                        packet = self._build_io_descriptors_packet()
                        await self.tensor_transfer_engine.async_send_metadata_to(
                            remote_name,
                            P2PHeaders.IOTransferDescriptorsSubmit.value,
                            packet,
                        )
                    elif header == P2PHeaders.Pong.value:
                        self.expert_blocks_pool.mark_alive(experts_block_index)
                    else:
                        self.logger.warning(f"P2P monitor: unknown P2P header '{header}'")
        except asyncio.CancelledError:
            self.logger.debug(f"P2P handler for block {experts_block_index} cancelled")

    def setup_monitoring(self):
        for experts_block_index, remote_name in enumerate(self.tensor_transfer_engine.remote_names):
            self.monitoring_tasks.append(self.event_loop.create_task(self._expert_block_health_checker(experts_block_index, remote_name)))
            self.monitoring_tasks.append(self.event_loop.create_task(self._expert_block_p2p_handler(experts_block_index, remote_name)))

    def _setup_activations_batch_size_candidates(self, expert_activations_number: List[int]):
        n = 32
        sorted_activations = sorted(expert_activations_number)

        quantile_candidates = []
        for k in range(n + 1):
            percentile = k / n
            if percentile == 0:
                quantile_val = sorted_activations[0]
            elif percentile == 1:
                quantile_val = sorted_activations[-1]
            else:
                # Use the standard quantile formula: Q(p) = x[floor(p*(n-1))] + (p*(n-1) - floor(p*(n-1))) * (x[ceil(p*(n-1))] - x[floor(p*(n-1))]).
                index = percentile * (len(sorted_activations) - 1)
                lower_idx = int(index)
                upper_idx = min(lower_idx + 1, len(sorted_activations) - 1)
                weight = index - lower_idx
                quantile_val = int(sorted_activations[lower_idx] + weight * (sorted_activations[upper_idx] - sorted_activations[lower_idx]))

            quantile_candidates.append(quantile_val)

        # Ensure minimum distance between consecutive values.
        min_distance = 2
        filtered_candidates = []
        for candidate in quantile_candidates:
            if not filtered_candidates or candidate - filtered_candidates[-1] >= min_distance:
                filtered_candidates.append(candidate)
            else:
                # Take the larger one.
                filtered_candidates[-1] = max(filtered_candidates[-1], candidate)

        self.activations_batch_size_candidates = filtered_candidates

    def _setup_forward_pass(self, input_batch: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        assert input_batch.dtype == self.dtype
        assert input_batch.device == self.device

        if len(input_batch.shape) == 3:
            assert input_batch.shape[2] == self.hidden_features
            input_batch = input_batch.view(-1, self.hidden_features)
        elif len(input_batch.shape) == 2:
            assert input_batch.shape[1] == self.hidden_features
        else:
            self.logger.error(f"Wrong input batch shape: {input_batch.shape} with hidden size: {self.hidden_features}")
            raise RuntimeError(f"Wrong input batch shape: {input_batch.shape} with hidden size: {self.hidden_features}")

        self.output_batch = torch.zeros(input_batch.shape, dtype=input_batch.dtype, device=input_batch.device)

        # router_logits: (batch_size * sequence_length, number_of_experts)
        router_logits = self.gate(input_batch)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # router_weights/selected_experts: (batch_size * sequence_length, top_k) (router_weights: weight, selected_experts: index)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # We cast back to the input dtype.
        routing_weights = routing_weights.to(self.dtype)

        # One hot encode the selected experts to create an expert mask.
        # This will be used to easily index which expert is going to be sollicitated.
        # Expert_mask before permute: (batch_size * sequence_length, top_k, number_of_experts) -- for each token there are top_k one hot masks
        # .permute: (number_of_experts, top_k, batch_size * sequence_length) -- for each expert we have top_k ways being selected (one hotted)
        expert_mask = F.one_hot(selected_experts, num_classes=self.number_of_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert.
        # .sum: get all activations for each of experts; expert_hit: (number_of_experts)
        # .greater: boolean tensor with true label is >0 activations; expert_hit: (number_of_experts)
        # .nonzero: tensor with experts index if its activations is >0; expert_hit: (number_of_experts, 1)
        # experts_hit: List[int] = torch.greater(expert_mask.sum(dim=(-1, -2)).to("cpu"), 0).nonzero().squeeze().tolist() -- less allocks but not really do anything.
        experts_hit_tensor = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze()
        if experts_hit_tensor.dim() == 0:
            experts_hit: List[int] = [int(experts_hit_tensor.item())]
        else:
            experts_hit: List[int] = experts_hit_tensor.tolist()

        expert_activations_number = []

        # Vectorized approach: process all experts in parallel using batch operations.
        if self.top_k != 1:
            # For multi-expert case, process all experts at once.
            for expert_index in experts_hit:
                # Find all entries with not 0 activations.
                # Returns 2 tensors: experts priority (from top_k-1 to 0 - the hottest is 0 priority), token_index (from 0 to batch_size * sequence_length - index of the token for the expert)
                # .squeeze(0) has effect only is top_k=1 ((top_k, batch_size * sequence_length) -> (batch_size * sequence_length) and expert priority is None)
                expert_priority, self.experts_metadata[expert_index].expert_token_indexes = torch.where(
                    expert_mask[expert_index].squeeze(0)
                )
                self.experts_metadata[expert_index].expert_total_activations = self.experts_metadata[
                    expert_index
                ].expert_token_indexes.shape[0]

                self.experts_metadata[expert_index].expert_offsets = Queue()
                self.experts_metadata[expert_index].expert_offsets.put((0, self.experts_metadata[expert_index].expert_total_activations))

                # routing_weights[tokens_indexes, expert_priorities]: (number of tokens_indexes) - 1d tensor, but we are multiplicating 2d expert_output_batch => None in the end
                self.experts_metadata[expert_index].expert_weights = routing_weights[
                    self.experts_metadata[expert_index].expert_token_indexes, expert_priority, None
                ]

                expert_activations_number.append(self.experts_metadata[expert_index].expert_total_activations)
        else:
            # For top_k=1 case, we can vectorize the torch.where operations.
            # Create a batch tensor for all hit experts.
            experts_hit_tensor = torch.tensor(experts_hit, device=self.device)

            # Batch process: get all expert masks at once.
            hit_expert_masks = expert_mask[experts_hit_tensor].squeeze(1)  # [num_experts_hit, batch_size]

            # Single batched torch.where call for all experts.
            expert_priorities, all_token_indexes = torch.where(hit_expert_masks)

            # Split results back to individual experts.
            for i, expert_index in enumerate(experts_hit):
                # Find tokens for this specific expert.
                expert_tokens_mask = expert_priorities == i
                expert_token_indexes = all_token_indexes[expert_tokens_mask]

                self.experts_metadata[expert_index].expert_token_indexes = expert_token_indexes
                self.experts_metadata[expert_index].expert_total_activations = expert_token_indexes.shape[0]

                self.experts_metadata[expert_index].expert_offsets = Queue()
                self.experts_metadata[expert_index].expert_offsets.put((0, self.experts_metadata[expert_index].expert_total_activations))

                # Get routing weights for this expert's tokens.
                self.experts_metadata[expert_index].expert_weights = routing_weights[expert_token_indexes, 0, None]

                expert_activations_number.append(self.experts_metadata[expert_index].expert_total_activations)

        # self._setup_activations_batch_size_candidates(expert_activations_number)

        return experts_hit, input_batch

    def dispatch(
        self, input_batch: torch.Tensor, number_of_dispatches: int, using_grouped_gemm: bool = True, using_alignment: bool = False
    ):
        dispatch_submits = 0

        while self.transferred_activations[0] < self.total_activations and dispatch_submits < number_of_dispatches:
            pipeline_task = HostPipelineTask(
                self.tensor_transfer_engine,
                self.input_buffers_pool,
                self.output_buffers_pool,
                self.experts_hit,
                self.expert_blocks_pool,
                self.experts_metadata,
                self.async_lock,
                self.metrics_collector,
            )
            self.pipeline_tasks[pipeline_task.name] = pipeline_task

            dispatch_task = asyncio.create_task(
                pipeline_task.dispatch(
                    input_batch,
                    self.active_pipeline_tasks,
                    self.transferred_activations,
                    self.expert_block_tasks,
                    self.activations_batch_size_candidates if using_alignment else None,
                    using_grouped_gemm,
                )
            )
            self.dispatch_tasks[pipeline_task.name] = dispatch_task

            dispatch_submits += 1

    # Forward pass for MoE layer.
    async def async_forward(
        self, input_batch: torch.Tensor, using_grouped_gemm: bool = True, using_alignment: bool = False
    ) -> torch.Tensor:
        assert self.tensor_transfer_engine is not None

        if self.metrics_collector is not None:
            self.metrics_collector.start_forward_pass()

        # Wait for all pending combine tasks to complete before resetting counters
        if self.combine_tasks:
            pending_combines = list(self.combine_tasks.values())
            try:
                await asyncio.gather(*pending_combines, return_exceptions=True)
            except Exception as e:
                self.logger.warning(f"Some combine tasks failed during cleanup: {e}")

        self.dispatch_tasks.clear()
        self.combine_tasks.clear()
        self.pipeline_tasks.clear()
        self.expert_block_tasks.clear()

        input_batch_original_shape = input_batch.shape
        self.experts_hit, input_batch = self._setup_forward_pass(input_batch)

        if not self.experts_hit:
            if self.metrics_collector is not None:
                self.metrics_collector.end_setup_phase()
                self.metrics_collector.end_forward_pass()
            return self.output_batch.reshape(input_batch_original_shape)

        self.total_activations = self.top_k * input_batch.shape[0]
        self.active_pipeline_tasks[0] = 0
        self.transferred_activations[0] = 0

        if self.metrics_collector is not None:
            self.metrics_collector.end_setup_phase()

        start_time = timeit.default_timer()
        while self.transferred_activations[0] < self.total_activations:
            self.combining.clear()
            number_of_dispatches: int = max(math.ceil((self.total_activations - self.transferred_activations[0]) / self.io_buffer_size), 1)
            self.dispatch(input_batch, number_of_dispatches, using_grouped_gemm, using_alignment)

            try:
                elapsed_time = timeit.default_timer() - start_time
                await asyncio.wait_for(self.combining.wait(), timeout=(self.forward_timeout / 1000 - elapsed_time))
            except asyncio.TimeoutError:
                break

        for pipeline_task_name, dispatch_task in self.dispatch_tasks.items():
            dispatch_task.cancel()
            self.pipeline_tasks[pipeline_task_name].close()
        for pipeline_task_name, combine_task in self.combine_tasks.items():
            combine_task.cancel()
            self.pipeline_tasks[pipeline_task_name].close()

        # End metrics collection for this forward pass
        if self.metrics_collector is not None:
            self.metrics_collector.end_forward_pass()

        return self.output_batch.reshape(input_batch_original_shape)

    def forward(self, input_batch: torch.Tensor, using_grouped_gemm: bool = True, using_alignment: bool = False) -> torch.Tensor:
        return self.event_loop.run_until_complete(self.async_forward(input_batch, using_grouped_gemm, using_alignment))

    def close(self) -> None:
        if self._is_closed:
            self.logger.warning("DistMoEBlock already closed, skipping cleanup")
            return

        assert self.tensor_transfer_engine is not None

        self.logger.info("Sending stop signals to all expert blocks...")

        async def send_stop_signals():
            for experts_block_index, remote_name in enumerate(self.tensor_transfer_engine.remote_names):
                self.logger.info(f"Sending stop signal to expert block {experts_block_index} at {remote_name}")
                try:
                    await asyncio.wait_for(
                        self.tensor_transfer_engine.async_send_metadata_to(remote_name, P2PHeaders.ExpertsBlockStop.value, []),
                        timeout=5.0,
                    )
                    self.logger.info(f"Stop signal sent successfully to {remote_name}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout sending stop signal to {remote_name}")
                except Exception as e:
                    self.logger.error(f"Failed to send stop signal to {remote_name}: {e}")

        # Run the async function in the event loop.
        if not self.event_loop.is_closed():
            try:
                if self.event_loop.is_running():
                    stop_future = asyncio.run_coroutine_threadsafe(send_stop_signals(), self.event_loop)
                    stop_future.result(timeout=10.0)
                else:
                    self.event_loop.run_until_complete(send_stop_signals())
            except Exception as e:
                self.logger.error(f"Error sending stop signals: {e}")

        self.logger.info("Stop signal sending completed")

        # Stop the event loop.
        if not self.event_loop.is_closed():
            self.monitoring_flag = False

            # Cancel monitoring tasks.
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()

            async def wait_for_monitors_cancellation():
                try:
                    await asyncio.wait_for(asyncio.gather(*self.monitoring_tasks, return_exceptions=True), timeout=2.0)
                    self.logger.warning("All monitoring tasks cancelled successfully")
                except asyncio.TimeoutError:
                    self.logger.warning("Some monitoring tasks did not cancel within timeout")
                except Exception as e:
                    self.logger.error(f"Expected cancellation exceptions: {e}")
                finally:
                    self._loop_stopped.set()

            # If the event loop is still running, try to cancel all monitors.
            if self.event_loop.is_running():
                self.event_loop.call_soon_threadsafe(lambda: asyncio.create_task(wait_for_monitors_cancellation()))
            else:
                try:
                    self.event_loop.run_until_complete(wait_for_monitors_cancellation())
                except Exception as e:
                    self.logger.error(f"Error waiting for monitor cancellation on stopped loop: {e}")

            if not self._loop_stopped.wait(timeout=5.0):
                self.logger.warning("Timeout waiting for event loop to stop, forcing close")

            if self.event_loop.is_running():
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                if not self._loop_stopped.wait(timeout=5.0):
                    self.logger.error("Timeout waiting for running event loop to stop before close")

            # If the event loop is still running, we will try to force close it.
            if not self.event_loop.is_running() and not self.event_loop.is_closed():
                self.event_loop.close()

        # Close all buffer pools.
        self.input_buffers_pool.close()
        self.output_buffers_pool.close()

        # Close health check P2P channels.
        for remote_name, health_check_pair in self.health_check_p2p_pairs.items():
            try:
                health_check_pair.pusher.close()
                health_check_pair.puller.close()
            except Exception as e:
                self.logger.error(f"Error closing health check P2P pair for {remote_name}: {e}")

        # Terminate health check ZMQ context.
        try:
            self.health_check_context.term()
        except Exception as e:
            self.logger.error(f"Error terminating health check ZMQ context: {e}")

        # Close TensorTransfer instance (only after deregistration).
        self.tensor_transfer_engine.close()

        # Mark as closed to prevent double closing.
        self._is_closed = True
        self.logger.debug("DistMoEBlock closed successfully")
