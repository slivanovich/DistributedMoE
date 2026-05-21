import bisect
import uuid

from torch import nn
from torch.nn.functional import grouped_mm  # type: ignore[attr-defined]  # added in PyTorch 2.5
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import struct
import torch

from MoE.DistMoE import P2PHeaders, ExpertMetaData
from MoE.MetricsCollector import MetricsCollector
from MoE.pools.BuffersPool import Buffer, BuffersPool
from MoE.pools.ExpertBlocksPool import ExpertBlocksPool
from TensorTransferEngine.TensorTransferEngine import TensorTransferEngine
from TensorTransferEngine.utils import _next_random_name
from utils.logging_config import get_logger


class PipelineTask:
    def __init__(
        self,
        tensor_transfer_engine: TensorTransferEngine,
        input_buffers_pool: BuffersPool,
        output_buffers_pool: BuffersPool,
        async_lock: asyncio.Lock,
    ) -> None:
        # Unique name of the pipeline task.
        self.name: bytes = _next_random_name()

        # Initialize logger with task context.
        self.logger = get_logger("dist_moe.moe.pipeline", {"task_id": self.name.hex()})

        # Tensor transfer instance for serializing/deserializing data descriptors and its transfers.
        self.tensor_transfer_engine: TensorTransferEngine = tensor_transfer_engine

        # Pools of I/O buffers.
        self.input_buffers_pool: BuffersPool = input_buffers_pool
        self.output_buffers_pool: BuffersPool = output_buffers_pool

        # Aquired pair of I/O buffers.
        self.input_buffer: Optional[Buffer] = None
        self.input_buffer_index: int = -1
        self.output_buffer: Optional[Buffer] = None
        self.output_buffer_index: int = -1

        # Async lock for changing mutable variables.
        self.async_lock: asyncio.Lock = async_lock

    async def _synchronize_cuda_stream(self, stream: torch.cuda.Stream) -> None:
        # Если именно asyncio.to_thread(...), чтобы прям асинх - весь пайплайн ломается)
        stream.synchronize()

    def release_input_buffer(self):
        if self.input_buffer_index != -1:
            self.input_buffers_pool.release(self.input_buffer_index)
            self.input_buffer = None
            self.input_buffer_index = -1

    def release_output_buffer(self):
        if self.output_buffer_index != -1:
            self.output_buffers_pool.release(self.output_buffer_index)
            self.output_buffer = None
            self.output_buffer_index = -1

    def close(self) -> None:
        self.release_input_buffer()
        self.release_output_buffer()


class HostPipelineTask(PipelineTask):
    def __init__(
        self,
        tensor_transfer_engine: TensorTransferEngine,
        input_buffers_pool: BuffersPool,
        output_buffers_pool: BuffersPool,
        experts_hit: List[int],
        experts_block_pool: ExpertBlocksPool,
        experts_metadata: Dict[int, ExpertMetaData],
        async_lock: asyncio.Lock,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        super().__init__(tensor_transfer_engine, input_buffers_pool, output_buffers_pool, async_lock)

        # Pool of expert blocks.
        self.experts_block_pool: ExpertBlocksPool = experts_block_pool

        # Metadata.
        self.experts_block_index: int = -1
        self.experts_hit: List[int] = experts_hit
        self.experts_metadata: Dict[int, ExpertMetaData] = experts_metadata

        # Number of dispatched.combined activations.
        self.used_activations: int = 0

        # Markup for the future combine operation, can be received via P2P, but it is more effective way.
        # Tuple format: (expert_index, expert_offset, padded_size, actual_size)
        self.experts_metadata_snapshot: List[Tuple[int, int, int, int]] = []

        # Flags for the status acknowlegment.
        self.dispatch_done: bool = False
        self.combine_done: bool = False

        # Metrics collector for performance monitoring.
        self.metrics_collector = metrics_collector
        # self.metrics_collector = None

        self.logger.debug("HostPipelineTask initialized")

    async def dispatch(
        self,
        input_batch: torch.Tensor,
        active_pipeline_tasks: List[int],
        transferred_activations: List[int],
        experts_block_tasks: Dict[int, List[bytes]],
        activations_batch_size_candidates: Optional[List[int]],
        using_grouped_gemm: bool = True,
    ) -> None:
        task_id = self.name.hex()
        if self.metrics_collector is not None:
            self.metrics_collector.start_dispatch(task_id)

        try:
            # Packet structure:
            # 0) 8 bytes for the host pipeline task name.
            # 1) 1 byte for using_grouped_gemm flag.
            # 2) 2 bytes for number of proccessed experts in expert block (N).
            # 3) [2 bytes for expert index; 2 bytes for expert's data offset in the input buffer; 2 bytes for number of expert's activations] x N.
            # 4) X bytes for input buffer serialized transfer descriptors.
            # 5) X bytes for output buffer serialized transfer descriptors.
            sizeof_ul = 2
            packet: List[Tuple[int, bytes]] = []
            num_experts = 0

            # Aquire an expert block index for sending metadata.
            self.experts_block_index, experts_block_metadata = await self.experts_block_pool.aquire()

            # Initialize the list if it doesnt exist.
            if self.experts_block_index not in experts_block_tasks:
                experts_block_tasks[self.experts_block_index] = []
            experts_block_tasks[self.experts_block_index].append(self.name)

            self.experts_block_pool.release(self.experts_block_index)
            self.experts_block_index = -1

            # Reserve space for packet header.
            packet.append((len(self.name), self.name))
            packet.append((1, struct.pack("!B", int(using_grouped_gemm))))
            packet.append((sizeof_ul, b""))

            # For deferred copying to the input buffer with rights offsets.
            packet_metadata: List[Tuple[int, int, int]] = []

            # Pre-collect data for batched copying optimization
            all_token_indices = []
            buffer_offsets = []

            used_with_padded_activations = 0

            # Form a data packet to send.
            for expert_index in experts_block_metadata.expert_indexes:
                if not expert_index in self.experts_hit:
                    continue
                if used_with_padded_activations >= self.input_buffers_pool.buffer_size:
                    break

                # Expert metadata by current expert serial number (in this experts block).
                expert_metadata: ExpertMetaData = self.experts_metadata[expert_index]

                if expert_metadata.expert_offsets.empty():
                    continue

                # Get the first not procceed expert activations segment.
                expert_offset_l, expert_offset_r = expert_metadata.expert_offsets.get_nowait()

                # Calculate actual expert size and remaining capacity.
                remaining_capacity = self.input_buffers_pool.buffer_size - used_with_padded_activations
                actual_expert_size = min(expert_offset_r - expert_offset_l, remaining_capacity)
                expert_size = actual_expert_size

                if expert_size <= 0:
                    expert_metadata.expert_offsets.put_nowait((expert_offset_l, expert_offset_r))
                    continue

                if activations_batch_size_candidates is not None and len(activations_batch_size_candidates) > 0:
                    idx = bisect.bisect_left(activations_batch_size_candidates, actual_expert_size)
                    if idx < len(activations_batch_size_candidates):
                        optimal_size = activations_batch_size_candidates[idx]
                        if optimal_size > remaining_capacity:
                            expert_metadata.expert_offsets.put_nowait((expert_offset_l, expert_offset_r))
                            continue
                        expert_size = optimal_size
                    else:
                        expert_metadata.expert_offsets.put_nowait((expert_offset_l, expert_offset_r))
                        continue

                # Save the snapshot of the passing activations.
                packet_metadata.append((expert_index, used_with_padded_activations, actual_expert_size))
                self.experts_metadata_snapshot.append((expert_index, expert_offset_l, expert_size, actual_expert_size))

                # Collect data for batched copying optimization.
                token_indices = expert_metadata.expert_token_indexes[expert_offset_l : expert_offset_l + actual_expert_size]
                all_token_indices.append(token_indices)
                buffer_offsets.append((used_with_padded_activations, actual_expert_size))

                # Build packet data directly.
                packet.append((sizeof_ul, struct.pack("!H", expert_index)))
                packet.append((sizeof_ul, struct.pack("!H", used_with_padded_activations)))
                packet.append((sizeof_ul, struct.pack("!H", expert_size)))

                # Update counters.
                self.used_activations += actual_expert_size
                used_with_padded_activations += expert_size
                num_experts += 1

                # Put remaining to the expert's queue of activation offsets.
                expert_metadata.expert_offsets.put_nowait((expert_offset_l + actual_expert_size, expert_offset_r))

            if num_experts > 0:
                # Fill in the expert count in the packet header.
                packet[2] = (sizeof_ul, struct.pack("!H", num_experts))

                # Aquire an input buffer for data transfers.
                if self.input_buffer_index == -1:
                    self.input_buffer_index, self.input_buffer = await self.input_buffers_pool.acquire()
                if self.input_buffer is None:
                    self.logger.error(f"While dispatching {self.name}, input buffer acquisition error has been occured.")
                    raise RuntimeError(f"While dispatching {self.name}, input buffer acquisition error has been occured.")

                with self.input_buffer.cuda_stream:
                    with torch.no_grad():
                        if len(all_token_indices) > 1:
                            concatenated_indices = torch.cat(all_token_indices, dim=0)
                            concatenated_data = input_batch[concatenated_indices]

                            expert_global_offset = 0
                            for expert_local_offset, actual_expert_size in buffer_offsets:
                                self.input_buffer.data[expert_local_offset : expert_local_offset + actual_expert_size, :].copy_(
                                    concatenated_data[expert_global_offset : expert_global_offset + actual_expert_size],
                                    non_blocking=True,
                                )
                                expert_global_offset += actual_expert_size
                        elif len(all_token_indices) == 1:
                            expert_local_offset, actual_expert_size = buffer_offsets[0]
                            self.input_buffer.data[expert_local_offset : expert_local_offset + actual_expert_size, :].copy_(
                                input_batch[all_token_indices[0]],
                                non_blocking=True,
                            )

                # Wait for all async copys before submiting the dispatch task to the expert block.
                await self._synchronize_cuda_stream(self.input_buffer.cuda_stream)

                # Aquire an output buffer for data transfers.
                if self.output_buffer_index == -1:
                    self.output_buffer_index, self.output_buffer = await self.output_buffers_pool.acquire()
                if self.output_buffer is None:
                    self.logger.error(f"While dispatching {self.name}, output buffer acquisition error has been occured.")
                    raise RuntimeError(f"While dispatching {self.name}, output buffer acquisition error has been occured.")

                # Append to the matadata packet current I/O buffers IDs.
                packet.append((len(self.input_buffer.id), self.input_buffer.id))
                packet.append((len(self.output_buffer.id), self.output_buffer.id))

                # Send a dispatch request to an expert block.
                await self.tensor_transfer_engine.async_send_metadata_to(
                    self.tensor_transfer_engine.remote_names[experts_block_metadata.experts_block_index],
                    header=P2PHeaders.DispatchSubmit.value,
                    body=packet,
                )

                # Record the dispatch results under the lock so that
                async with self.async_lock:
                    transferred_activations[0] += self.used_activations
                    active_pipeline_tasks[0] += 1
                    task_id_short = self.name.hex()[:8]
            else:
                task_id_short = self.name.hex()[:8]
                self.logger.debug(f"[task_id={task_id_short}] Dispatch skipped - no experts to dispatch (redundant dispatch)")
        except asyncio.TimeoutError as e:
            task_id_short = self.name.hex()[:8]
            self.logger.error(f"[task_id={task_id_short}] Dispatch timeout error: {str(e)}")
            self.close()
        except RuntimeError as e:
            task_id_short = self.name.hex()[:8]
            self.logger.error(f"[task_id={task_id_short}] Dispatch runtime error: {str(e)}")
            self.close()
        except KeyError as e:
            task_id_short = self.name.hex()[:8]
            self.logger.error(f"[task_id={task_id_short}] Dispatch key error: {str(e)}")
            self.close()
        except Exception as e:
            task_id_short = self.name.hex()[:8]
            self.logger.error(f"[task_id={task_id_short}] Dispatch unknown error: {str(e)}")
            self.close()
        finally:
            # Release an experts block index in the end of dispatch no matter what happens.
            if self.experts_block_index != -1:
                self.experts_block_pool.release(self.experts_block_index)
                self.experts_block_index = -1
            self.dispatch_done = True

            if self.metrics_collector is not None:
                self.metrics_collector.end_dispatch(task_id)

    async def combine(
        self,
        active_pipeline_tasks: List[int],
        output_batch: torch.Tensor,
        combining: asyncio.Event,
    ) -> None:
        task_id = self.name.hex()
        if self.metrics_collector is not None:
            self.metrics_collector.start_combine(task_id)

        should_signal = False
        try:
            if self.output_buffer is None:
                self.logger.error(f"While combining {self.name}, there is no acquired output buffer.")
                raise RuntimeError(f"While combining {self.name}, there is no acquired output buffer.")

            # Aggregate expert block compute results into an output batch.
            output_buffer_offset = 0
            with torch.cuda.stream(self.output_buffer.cuda_stream):
                with torch.no_grad():
                    for expert_metadata_snapshot in self.experts_metadata_snapshot:
                        expert_index, expert_offset, expert_size, actual_expert_size = expert_metadata_snapshot
                        expert_tokens_indexes = self.experts_metadata[expert_index].expert_token_indexes[
                            expert_offset : expert_offset + actual_expert_size
                        ]
                        weighted_expert_output_batch = (
                            self.output_buffer.data[output_buffer_offset : output_buffer_offset + actual_expert_size]
                            * self.experts_metadata[expert_index].expert_weights[expert_offset : expert_offset + actual_expert_size]
                        )

                        # С этим методом оно (при присутствующей проблеме с compute-ом) работает существенно быстрее.. (3.3s vs 4.5s batch_size=8k/top_k=8/io_size=2k/EB=1)
                        output_batch.scatter_add_(
                            dim=0,
                            index=expert_tokens_indexes.unsqueeze(-1).expand(-1, output_batch.size(-1)),
                            src=weighted_expert_output_batch,
                        )
                        # output_batch.index_add_(dim=0, index=expert_tokens_indexes, source=weighted_expert_output_batch)

                        # Use padded expert_size for buffer offset calculation to maintain alignment.
                        # This ensures we skip over the padded portion in the output buffer.
                        output_buffer_offset += expert_size

            # Имперически выяснили, что тут прям важно делать await, чтобы переключиться на другие таски.
            # Несмотря на то, что сам combine начинает работать на 0.1ms больше, все становиться чууучуть побыстрее.
            # Synchronize the CUDA stream to store all activations into output batch.
            await self._synchronize_cuda_stream(self.output_buffer.cuda_stream)

        except Exception as e:
            # Release I/O buffers.
            self.close()
            self.logger.error(f"Combine failed for task {self.name.hex()[:8]}: {e}")
        finally:
            async with self.async_lock:
                active_pipeline_tasks[0] -= 1
                should_signal = active_pipeline_tasks[0] <= 0
            
            self.release_input_buffer()
            self.release_output_buffer()

            if should_signal:
                combining.set()
            self.combine_done = True

            if self.metrics_collector is not None:
                self.metrics_collector.end_combine(task_id)

    def abort(self):
        for expert_index, expert_offset, expert_size, actual_expert_size in self.experts_metadata_snapshot:
            self.experts_metadata[expert_index].expert_offsets.put((expert_offset, expert_offset + actual_expert_size))
        self.close()

    def close(self):
        if self.experts_block_index != -1:
            self.experts_block_pool.release(self.experts_block_index)
            self.experts_block_index = -1

        super().close()


class ExpertsBlockPipelineTask(PipelineTask):
    def __init__(
        self,
        tensor_transfer_engine: TensorTransferEngine,
        input_buffers_pool: BuffersPool,
        output_buffers_pool: BuffersPool,
        expert_layers: Dict[int, nn.Module],
        experts_combine_layer: Optional[Any],
        expert_streams: Dict[int, torch.cuda.Stream],
        host_address: str,
        metadata: List[Tuple[int, bytes]],
        async_lock: asyncio.Lock,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        super().__init__(tensor_transfer_engine, input_buffers_pool, output_buffers_pool, async_lock)

        # Address of the origin model.
        self.host_address: str = host_address

        # Metadata from the P2P chanel (expert markup).
        self.metadata: List[Tuple[int, bytes]] = metadata

        # Experts layers for compute.
        self.expert_layers: Dict[int, nn.Module] = expert_layers

        # Per-expert CUDA streams — one per expert so all experts in this block
        # can execute concurrently on the GPU instead of sequentially.
        self.expert_streams: Dict[int, torch.cuda.Stream] = expert_streams

        # Stacked weight tensors for grouped_mm fast path.
        # Tuple of (expert_indices_ordered, gate_w, up_w, down_w) or None.
        self.experts_combine_layer: Optional[Any] = experts_combine_layer

        # Metrics collector for performance monitoring
        self.metrics_collector = metrics_collector
        # self.metrics_collector = None

        self.logger.debug("ExpertsBlockPipelineTask initialized")

    async def run(
        self,
        experts_block_index: int,
        host_input_buffers_transfer_descs: Dict[bytes, Any],
        host_output_buffers_transfer_descs: Dict[bytes, Any],
    ) -> None:
        task_id = self.name.hex()
        if self.metrics_collector is not None:
            self.metrics_collector.start_expert_run(task_id, experts_block_index)

        try:
            # Packet integrity check.
            host_pipeline_task_name: bytes = self.metadata[0][1]
            using_grouped_gemm: bool = bool(struct.unpack("!B", self.metadata[1][1])[0])
            number_of_experts = struct.unpack("!H", self.metadata[2][1])[0]
            if not (len(self.metadata) == 3 + number_of_experts * 3 + 2):
                self.logger.error(f"While running {self.name}, metadata packet structure is broken.")
                raise RuntimeError(f"While running {self.name}, metadata packet structure is broken.")

            # Host pipeline task transfer descriptors.
            host_input_buffer_id: bytes = self.metadata[-2][1]
            host_output_buffer_id: bytes = self.metadata[-1][1]

            # READ:
            if self.input_buffer_index == -1:
                self.input_buffer_index, self.input_buffer = await self.input_buffers_pool.acquire()
            if self.input_buffer is None:
                self.logger.error(f"While running {self.name}, input buffer acquisition error has been occured.")
                raise RuntimeError(f"While running {self.name}, input buffer acquisition error has been occured.")

            # Record start of transfer for metrics.
            if self.metrics_collector is not None:
                transfer_size = self.input_buffer.data.numel() * self.input_buffer.data.element_size()
                self.metrics_collector.record_transfer_start(task_id, "read", transfer_size)

            await self.input_buffer.read_from(self.host_address, host_input_buffers_transfer_descs[host_input_buffer_id])

            # Record end of transfer for metrics.
            if self.metrics_collector is not None:
                self.metrics_collector.record_transfer_end(task_id, "read")

            # Send an ack that host input buffer is free now (should be released on host).
            # await self.tensor_transfer_engine.async_send_metadata_to(
            #     self.host_address, P2PHeaders.HostInputBufferUnlocked.value, [(len(host_pipeline_task_name), host_pipeline_task_name)]
            # )

            # COMPUTE:
            if self.output_buffer_index == -1:
                self.output_buffer_index, self.output_buffer = await self.output_buffers_pool.acquire()
            if self.output_buffer is None:
                self.logger.error(f"While running {self.name}, output buffer acquisition error has been occured.")
                raise RuntimeError(f"While running {self.name}, output buffer acquisition error has been occured.")

            # COMPUTE: use grouped_mm when it is available and requested, otherwise fall back to the per-expert loop.
            if self.experts_combine_layer is not None and using_grouped_gemm:
                with torch.no_grad():
                    expert_indexes, gate_w_transposed, up_w_transposed, down_w_transposed = self.experts_combine_layer

                    # Build a mapping from expert_index -> position in the stacked weight tensors.
                    active_expert_positions = []
                    offsets_list = []

                    # Collect expert data with their positions for sorting
                    for index in range(number_of_experts):
                        expert_index = struct.unpack("!H", self.metadata[3 + index * 3 + 0][1])[0]
                        expert_offset = struct.unpack("!H", self.metadata[3 + index * 3 + 1][1])[0]
                        expert_size = struct.unpack("!H", self.metadata[3 + index * 3 + 2][1])[0]

                        # Map expert_index to its position in the stacked weight tensors
                        expert_position = expert_indexes.index(expert_index)
                        active_expert_positions.append(expert_position)
                        offsets_list.append(expert_offset + expert_size)

                    # Offsets for grouped_mm kernel. We multiply the whole input buffer for better caching.
                    # If there's a gap at the end, add buffer size and reuse the first expert position.
                    # if len(offsets_list) > 0:
                    #     if offsets_list[-1] != self.input_buffer.size:
                    #         offsets_list.append(self.input_buffer.size)
                    #         active_expert_positions.append(active_expert_positions[-1])
                    # else:
                    #     self.logger.error("No experts data found in metadata, using dummy data for grouped_mm")
                    #     offsets_list = [self.input_buffer.size]
                    #     active_expert_positions = [0]

                    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=self.input_buffer.device)

                    # Select stacked weight slices for active experts in dispatch order.
                    active_gate_w_transposed = gate_w_transposed[active_expert_positions]  # [E_active, H, I]
                    active_up_w_transposed = up_w_transposed[active_expert_positions]  # [E_active, H, I]
                    active_down_w_transposed = down_w_transposed[active_expert_positions]  # [E_active, I, H]

                    # with torch.cuda.stream(self.input_buffer.cuda_stream):
                    gate_out = grouped_mm(self.input_buffer.data[: offsets_list[-1]], active_gate_w_transposed, offs=offsets)  # [T, I]
                    up_out = grouped_mm(self.input_buffer.data[: offsets_list[-1]], active_up_w_transposed, offs=offsets)  # [T, I]
                    hidden = torch.nn.functional.relu(gate_out) * up_out
                    result = grouped_mm(hidden, active_down_w_transposed, offs=offsets)  # [T, H]
                    self.output_buffer.data[: offsets_list[-1]].copy_(result, non_blocking=False)

                    # Проблема в том что все вычисления настолько быстрые, что лишний оверхед на синхронизацию > чем просто все в блок режиме делать)
                    # Unblock I/O stream for the p2p queue handling.
                    # await self._synchronize_cuda_stream(self.input_buffer.cuda_stream)
            else:
                # Actual copute-bound part.
                with self.input_buffer.cuda_stream:
                    with torch.no_grad():
                        for index in range(number_of_experts):
                            expert_index = struct.unpack("!H", self.metadata[3 + index * 3 + 0][1])[0]
                            expert_offset = struct.unpack("!H", self.metadata[3 + index * 3 + 1][1])[0]
                            expert_size = struct.unpack("!H", self.metadata[3 + index * 3 + 2][1])[0]

                            sharded_input_batch = self.input_buffer.data[expert_offset : expert_offset + expert_size, :]
                            sharded_output_buffer = self.output_buffer.data[expert_offset : expert_offset + expert_size, :]

                            sharded_output_batch: torch.Tensor = self.expert_layers[expert_index](sharded_input_batch)
                            sharded_output_buffer.copy_(sharded_output_batch, non_blocking=True)

                # Тоже самое: просто лишний await который не сильно но портит e2e статистики.
                # Unblock I/O for the p2p queue handling.
                self.input_buffer.cuda_stream.synchronize()
                # await self._synchronize_cuda_stream(self.input_buffer.cuda_stream)

            # Release the input buffer after all compute done.
            self.input_buffers_pool.release(self.input_buffer_index)
            self.input_buffer = None
            self.input_buffer_index = -1

            # WRITE:
            if self.metrics_collector is not None:
                transfer_size = self.output_buffer.data.numel() * self.output_buffer.data.element_size()
                self.metrics_collector.record_transfer_start(task_id, "write", transfer_size)

            await self.output_buffer.write_to(self.host_address, host_output_buffers_transfer_descs[host_output_buffer_id])

            if self.metrics_collector is not None:
                self.metrics_collector.record_transfer_end(task_id, "write")

            # Send an ack that host have all computed data in its output buffer now.
            await self.tensor_transfer_engine.async_send_metadata_to(
                self.host_address,
                P2PHeaders.HostOutputBufferUnlocked.value,
                [(len(host_pipeline_task_name), host_pipeline_task_name)],
            )
        except Exception as e:
            self.logger.error(f"Experts block runtime error: {e}")
        finally:
            # End of expert run tracking.
            if self.metrics_collector is not None:
                self.metrics_collector.end_expert_run(task_id)

            # Release I/O buffers if something went wrong during running the expert block pipeline task.
            self.close()
