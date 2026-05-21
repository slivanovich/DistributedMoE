from ctypes import c_bool
from dataclasses import dataclass
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
import struct
import timeit
import threading
from torch import nn
from typing import Any, Dict, List, Tuple, Optional

import asyncio
import torch
import zmq
import zmq.asyncio

from TensorTransferEngine.TensorTransferEngine import TensorTransferEngineBackend, TensorTransferEngineConfig
from TensorTransferEngine.MCTETensorTransferEngine import MCTETensorTransferEngine
from TensorTransferEngine.NIXLTensorTransferEngine import NIXLTensorTransferEngine

from MoE.DistMoE import P2PHeaders
from MoE.DistMoEBlock import DistMoEBlockConfig
from MoE.pools.BuffersPool import BuffersPool
from MoE.PipelineTask import ExpertsBlockPipelineTask
from MoE.MetricsCollector import MetricsCollector
from TensorTransferEngine.utils import P2PPair, join_host_port
from utils.logging_config import get_logger


class DistExpertsBlock(nn.Module):
    def __init__(
        self,
        experts_block_index: int,
        config: DistMoEBlockConfig,
        expert_layers: List,
        tte_config: TensorTransferEngineConfig,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        super().__init__()

        # Experts block global index (should be unqiue but it is not guaranteed).
        self.experts_block_index: int = experts_block_index

        # Initialize logger with expert block context.
        self.logger = get_logger(
            "dist_moe.moe.expert_block",
            {"block_index": str(self.experts_block_index), "num_experts": str(len(config.expert_blocks[self.experts_block_index]))},
        )

        # Track close state to prevent double closing.
        self._is_closed = False

        # Experts block's device.
        self.device: torch.device = config.eb_devices[self.experts_block_index]

        # Metrics collector for performance monitoring
        self.metrics_collector = metrics_collector

        # Create an event loop for async tasks and register it for the current thread
        # so that library code calling asyncio.get_event_loop() sees the correct loop.
        self.event_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Event that is set once the event loop has fully stopped (replaces busy-wait).
        self._loop_stopped = threading.Event()

        # Expert weights initializing.
        assert len(config.expert_blocks[self.experts_block_index]) == len(expert_layers)
        self.expert_layers: Dict[int, Any] = {}
        self.expert_streams: Dict[int, torch.cuda.Stream] = {}
        self.initialize_expert_layers(config.expert_blocks, expert_layers)

        # Expert layers parameters.
        self.hidden_features = config.hidden_features
        self.dtype = config.dtype

        # Setup TensorTransfer instance.
        self.tensor_transfer_engine = None
        if config.backend == TensorTransferEngineBackend.NIXL:
            self.tensor_transfer_engine = NIXLTensorTransferEngine(tte_config)
        elif config.backend == TensorTransferEngineBackend.MoonCake:
            self.tensor_transfer_engine = MCTETensorTransferEngine(tte_config)
        assert self.tensor_transfer_engine is not None
        self.host_address = self.tensor_transfer_engine.remote_names[0]

        # Setup separate health check P2P channels.
        self.health_check_context = zmq.asyncio.Context()
        self.health_check_p2p_pair: P2PPair | None = None
        self.setup_health_check(config, tte_config)

        # Flag for managing experts block runtime.
        self.EOR: Synchronized[bool] = Value(c_bool, False)

        # Async lock for all experts block pipeline tasks.
        self.async_lock: asyncio.Lock = asyncio.Lock()

        # initialize I/O buffers.
        self.io_buffer_size: int = config.io_buffer_size
        self.number_of_io_buffers: int = config.number_of_io_buffers

        # Mapping from host I/O buffers IDs to their transfer descriptors, that are already deserialized.
        self.host_input_buffers_transfer_descs: Dict[bytes, Any] = {}
        self.host_output_buffers_transfer_descs: Dict[bytes, Any] = {}

        self.initialize_io_buffers()

        self.logger.info(f"DistExpertsBlock initialized - Device: {self.device}, Experts: {list(self.expert_layers.keys())}")

    def setup_health_check(self, config: DistMoEBlockConfig, tte_config: TensorTransferEngineConfig) -> None:
        if self.experts_block_index >= len(config.remote_health_check_ports):
            self.logger.warning(f"No health check port configured for expert block {self.experts_block_index}")
            return

        eb_health_check_port = config.remote_health_check_ports[self.experts_block_index]
        host_health_check_port = config.host_health_check_ports[self.experts_block_index]

        # Create separate P2P pair for health checker.
        self.health_check_p2p_pair = P2PPair(self.health_check_context.socket(zmq.PUSH), self.health_check_context.socket(zmq.PULL))

        # Setup health check puller (for receiving pings).
        puller_address = join_host_port(tte_config.remote_addresses[0], host_health_check_port, "tcp")
        self.health_check_p2p_pair.puller.connect(puller_address)

        # Setup health check pusher (for sending pongs).
        pusher_address = join_host_port(tte_config.host_address, eb_health_check_port, "tcp")
        self.health_check_p2p_pair.pusher.bind(pusher_address)

        self.logger.debug(f"Health check P2P PAIR: PULL {puller_address}; PUSH {pusher_address}")

    async def _health_check_handler(self, tasks: List[asyncio.Task]) -> None:
        assert self.health_check_p2p_pair is not None
        assert self.tensor_transfer_engine is not None

        while not self.EOR.value:
            try:
                packet = await asyncio.wait_for(self.health_check_p2p_pair.puller.recv(), timeout=2.0)
                header, _ = self.tensor_transfer_engine.deserialize_p2p_packet(packet)

                if header == P2PHeaders.Ping.value:
                    tasks[:] = [t for t in tasks if not t.done()]
                    pong_packet = self.tensor_transfer_engine.serialize_p2p_packet(P2PHeaders.Pong.value, [])
                    await self.health_check_p2p_pair.pusher.send(pong_packet)
                    self.logger.debug(f"Health check: received ping, sent pong")
                else:
                    self.logger.warning(f"Health check: unexpected header {header}")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self.logger.debug("Health check handler cancelled")
                break
            except Exception as e:
                if not self.EOR.value:
                    self.logger.warning(f"Health check handler error: {e}")
                break

    def initialize_expert_layers(self, expert_blocks: List[List[int]], expert_layers: List):
        experts = []
        for expert_index, expert_layer in zip(expert_blocks[self.experts_block_index], expert_layers):
            layer = expert_layer.to(self.device)
            experts.append((expert_index, layer))
            self.expert_layers[expert_index] = layer
            self.expert_streams[expert_index] = torch.cuda.Stream(device=self.device)

        # Stack expert weights for grouped_mm: shape [num_experts, out_features, in_features].
        # experts_combine_layer holds (gate_proj_w, up_proj_w, down_proj_w) stacked tensors
        # and the ordered list of expert indices matching the stack order.
        # Pre-transpose the weights to avoid repeated transpose operations during inference.
        expert_indexes = [index for index, _ in experts]
        gate_w = torch.stack([layer.gate_proj.weight.data for _, layer in experts])  # [E, H, I]
        up_w = torch.stack([layer.up_proj.weight.data for _, layer in experts])  # [E, H, I]
        down_w = torch.stack([layer.down_proj.weight.data for _, layer in experts])  # [E, H, I]

        # Pre-transpose gate and up weights for grouped_mm optimization.
        gate_w_transposed = gate_w.transpose(1, 2)  # [E, I, H]
        up_w_transposed = up_w.transpose(1, 2)  # [E, I, H]
        down_w_transposed = down_w.transpose(1, 2)  # [E, I, H]

        self.experts_combine_layer = (expert_indexes, gate_w_transposed, up_w_transposed, down_w_transposed)

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

        assert self.tensor_transfer_engine.handshake()

    def parse_host_io_buffer_descriptors(self, metadata: List[Tuple[int, bytes]]):
        assert len(metadata) > 1
        assert self.tensor_transfer_engine is not None

        number_of_host_io_buffers: int = struct.unpack("!H", metadata[0][1])[0]

        # Number of host I/O buffers + 2*[(buffer's id, buffer's serialized descs)].
        assert len(metadata) == 1 + 2 * 2 * number_of_host_io_buffers

        offset = 1
        for index in range(2 * number_of_host_io_buffers):
            buffer_id: bytes = metadata[offset][1]
            buffer_transfer_descs = self.tensor_transfer_engine.deserialize_descs(metadata[offset + 1])

            if index < number_of_host_io_buffers:
                self.host_input_buffers_transfer_descs[buffer_id] = buffer_transfer_descs
            else:
                self.host_output_buffers_transfer_descs[buffer_id] = buffer_transfer_descs

            offset += 2

    async def async_main_loop(self) -> None:
        assert self.tensor_transfer_engine is not None

        tasks: List[asyncio.Task] = []

        # Start health checker routine.
        if self.health_check_p2p_pair is not None:
            health_check_task = asyncio.create_task(self._health_check_handler(tasks))
            tasks.append(health_check_task)

        while not self.EOR.value:
            try:
                header, metadata = await self.tensor_transfer_engine.async_get_metadata_from(self.host_address)
            except asyncio.TimeoutError:
                if self.EOR.value:
                    self.logger.warning("Timeout during shutdown, exiting main loop")
                    break
                self.logger.warning(f"Timeout waiting for metadata from {self.host_address}, retrying... (EOR={self.EOR.value})")
                continue
            except Exception as e:
                if self.EOR.value:
                    self.logger.warning(f"Exception during shutdown: {e}, exiting main loop")
                    break
                self.logger.error(f"Unexpected error in main loop: {e}")
                continue

            if header != -1:
                if header == P2PHeaders.ExpertsBlockStop.value:
                    self.logger.info("*** RECEIVED STOP SIGNAL *** - shutting down expert block")
                    self.EOR.value = True
                    break
                elif header == P2PHeaders.IOTransferDescriptorsSubmit.value:
                    self.parse_host_io_buffer_descriptors(metadata)
                elif header == P2PHeaders.DispatchSubmit.value:
                    expert_block_pipeline_task = ExpertsBlockPipelineTask(
                        self.tensor_transfer_engine,
                        self.input_buffers_pool,
                        self.output_buffers_pool,
                        self.expert_layers,
                        self.experts_combine_layer,
                        self.expert_streams,
                        self.host_address,
                        metadata,
                        self.async_lock,
                        self.metrics_collector,
                    )

                    tasks.append(
                        asyncio.create_task(
                            expert_block_pipeline_task.run(
                                self.experts_block_index, self.host_input_buffers_transfer_descs, self.host_output_buffers_transfer_descs
                            )
                        )
                    )
                else:
                    self.logger.warning(f"Unknown header received: {header}")

        # Wait for any remaining tasks to complete before exiting
        if tasks:
            self.logger.debug(f"Waiting for {len(tasks)} remaining tasks to complete")
            pending_tasks = [t for t in tasks if not t.done()]
            if pending_tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*pending_tasks, return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Some tasks did not complete within timeout, cancelling them")
                    for task in pending_tasks:
                        if not task.done():
                            task.cancel()

    def main_loop(self) -> None:
        try:
            self.event_loop.run_until_complete(self.async_main_loop())
        except asyncio.CancelledError:
            self.logger.debug("Expert block main loop cancelled during shutdown")
        except RuntimeError as e:
            if str(e) == "Event loop stopped before Future completed.":
                self.logger.debug("Expert block event loop stopped during shutdown")
            else:
                raise
        finally:
            self.close()

    # TODO: Forward pass for all of experts that are locating in this DistExperts block (agg on the DistExpertsBlock.device).
    def forward(self, input_batch: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None:
        if self._is_closed:
            self.logger.warning("DistExpertsBlock already closed, skipping cleanup")
            return

        assert self.tensor_transfer_engine is not None

        # Stop the main loop and clear all unfinished tasks.
        self.EOR.value = True
        self.logger.debug("EOR flag set, initiating shutdown")

        # Give the event loop time to process the EOR signal and exit gracefully.
        if not self.event_loop.is_closed():
            if self.event_loop.is_running():
                self.logger.debug("Stopping running expert block event loop...")
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                if not self._loop_stopped.wait(timeout=5.0):
                    self.logger.warning("Event loop did not stop within timeout, this is expected during shutdown")
                else:
                    self.logger.debug("Event loop stopped naturally")

            # Close the event loop.
            try:
                if not self.event_loop.is_closed() and not self.event_loop.is_running():
                    self.event_loop.close()
            except Exception as e:
                self.logger.warning(f"Error closing event loop: {e}")

        # Clear GPU memory.
        try:
            self.input_buffers_pool.close()
            self.output_buffers_pool.close()
        except Exception as e:
            self.logger.warning(f"Error closing buffer pools: {e}")

        # Close TensorTransfer instance (only after deregistration).
        try:
            self.tensor_transfer_engine.close()
        except Exception as e:
            self.logger.warning(f"Error closing tensor transfer engine: {e}")

        # Mark as closed to prevent double closing.
        self._is_closed = True
        self.logger.debug("DistExpertsBlock closed successfully")
