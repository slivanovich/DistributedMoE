from typing import List, Tuple

import asyncio
import torch

from TensorTransferEngine.TensorTransferEngine import TensorTransferEngine
from TensorTransferEngine.utils import _next_random_name
from utils.logging_config import get_logger

class Buffer:
    def __init__(
        self,
        tensor_transfer_engine: TensorTransferEngine,
        size: int,
        hidden_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assert tensor_transfer_engine is not None

        # Buffer 2-bytes ID (will be set later, when this buffer will be sent to the other TTE instance).
        self.id: bytes = b""

        # Initialize logger.
        self.logger = get_logger("dist_moe.buffer")

        # Tensor transfer instance for data registration and transfers.
        self._tensor_transfer_engine = tensor_transfer_engine

        # Buffer parameters.
        self.size = size  # in token
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.size_in_bytes = self.size * self.hidden_features * self.dtype.itemsize  # in bytes
        self.device = device

        # Buffer data.
        self.data = torch.empty(
            self.size,
            self.hidden_features,
            dtype=self.dtype,
            device=self.device,
        )

        # Create a dedicated CUDA stream for this buffer to avoid per-task stream creation overhead
        self.cuda_stream = torch.cuda.Stream(device=self.device)

        # Register and transfer descriptors.
        self.transfer_descs, self.register_descs = self._tensor_transfer_engine.register_batch(
            [self.data], "VRAM" if self.device.type == "cuda" else "DRAM", self.device.index
        )
        self.serialized_transfer_descs = self._tensor_transfer_engine.serialize_descs(self.transfer_descs)

    # Wrapper over the TTE.
    async def read_from(self, remote_name: str, src_buffer_transfer_descs):
        read_transfer_name = _next_random_name()
        self._tensor_transfer_engine.read_batch_transfer(
            remote_name,
            read_transfer_name,
            src_buffer_transfer_descs,
            self.transfer_descs,
            [self.size_in_bytes],
        )

        timeout = 1000  # in ms
        success = await asyncio.to_thread(self._tensor_transfer_engine.wait_for_transfer_efficient, read_transfer_name, timeout)
        if not success:
            self.logger.error(f"Read transfer {read_transfer_name.hex()} timed out after {timeout} ms")
            raise TimeoutError(f"Read transfer timed out")
        else:
            self.logger.debug(f"Read transfer throughput: {(success.size / 1024**2 / success.duration):.2f} MB/ms")

    # Wrapper over the TTE.
    async def write_to(self, remote_name: str, dst_buffer_transfer_descs):
        write_transfer_name = _next_random_name()
        self._tensor_transfer_engine.write_batch_transfer(
            remote_name,
            write_transfer_name,
            self.transfer_descs,
            dst_buffer_transfer_descs,
            [self.size_in_bytes],
        )

        timeout = 1000  # in ms
        success = await asyncio.to_thread(self._tensor_transfer_engine.wait_for_transfer_efficient, write_transfer_name, timeout)
        if not success:
            self.logger.error(f"Write transfer {write_transfer_name.hex()} timed out after {timeout} ms")
            raise TimeoutError(f"Write transfer timed out")
        else:
            self.logger.debug(f"Write transfer throughput: {(success.size / 1024**2 / success.duration):.2f} MB/ms")

    def close(self) -> None:
        try:
            self._tensor_transfer_engine.deregister_batch(self.register_descs)
        except Exception as e:
            error_msg = str(e)
            if "NIXL_ERR_NOT_FOUND" in error_msg:
                self.logger.warning(f"Buffer memory was already deregistered or not found: {error_msg}")
            else:
                self.logger.error(f"Failed to deregister buffer memory: {error_msg}")


class BuffersPool:
    def __init__(
        self,
        pool_size: int,
        tensor_transfer_engine: TensorTransferEngine,
        buffer_size: int,
        hidden_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Initialize logger.
        self.logger = get_logger("dist_moe.buffer_pool", {"pool_size": str(pool_size)})

        # Buffer pool parameters.
        self.pool_size = pool_size  # in units
        self.buffer_size = buffer_size  # in activations number
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.device = device

        # Buffers and pool of their indexes.
        self.buffers: List[Buffer] = []
        self.buffer_index_pool: asyncio.Queue = asyncio.Queue()
        for buffer_index in range(self.pool_size):
            self.buffers.append(Buffer(tensor_transfer_engine, self.buffer_size, self.hidden_features, self.dtype, self.device))
            self.buffer_index_pool.put_nowait(buffer_index)

        self.logger.info(f"BuffersPool initialized with {pool_size} buffers")

    async def acquire(self) -> Tuple[int, Buffer]:
        try:
            buffer_index: int = await self.buffer_index_pool.get()
            return buffer_index, self.buffers[buffer_index]
        except (asyncio.QueueEmpty, AttributeError):
            raise RuntimeError("Buffer pool queue is shut down.")

    def release(self, buffer_index: int) -> None:
        self.buffer_index_pool.put_nowait(buffer_index)

    def close(self) -> None:
        self.logger.info(f"Closing BuffersPool with {len(self.buffers)} buffers")
        buffers_cleaned = 0
        buffers_failed = 0

        for buffer_index, buffer in enumerate(self.buffers):
            try:
                self.logger.debug(f"Closing buffer {buffer_index}")
                buffer.close()
                buffers_cleaned += 1
            except Exception as e:
                buffers_failed += 1
                self.logger.error(f"Failed to close buffer {buffer_index}: {e}")

        if buffers_failed > 0:
            self.logger.warning(f"BuffersPool cleanup: {buffers_cleaned} cleaned, {buffers_failed} failed out of {len(self.buffers)}")
