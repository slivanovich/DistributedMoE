from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import struct
import timeit
from typing import Dict, List, Optional, Tuple

import asyncio
import time
import threading
import torch
import zmq
import zmq.asyncio

import nixl._bindings as nixlBind  # type: ignore

from TensorTransferEngine.utils import HandshakeType, P2PPair, Transfer, TransferStatus, TransferProtocol, join_host_port
from utils.logging_config import get_logger


class TensorTransferEngineBackend(Enum):
    MoonCake = "MoonCake"
    NIXL = "NIXL"


@dataclass
class TensorTransferEngineConfig:
    # Host part (DistMoEBlock): host, port (for TTE initialization), P2P ports for the P2P communication.
    host_address: str
    host_port: int
    host_p2p_ports: List[int]

    # Remote parts (DistExpertBlock x N): host, port (for TTE initialization), P2P ports for the P2P communication.
    remote_addresses: List[str]
    remote_ports: List[int]
    remote_p2p_ports: List[int]

    # P2P send/receive timeout duration in milliseconds.
    p2p_timeout_duration: int

    # Type of interaction between host and remote while handshaking.
    handshake_type: HandshakeType

    # Transfer protocol (via RDMA/NVLink)
    transfer_protocol: TransferProtocol

    # If there is RDMA transfer protocol, IB device names can be set.
    ib_device_names: str

    # Additional metadata* fields for the MoonCake backend runtime.
    metadata_schema: str | None
    metadata_host: str | None
    metadata_port: int | None
    metadata_dir: str | None


# Tensor transfer engine class.
class TensorTransferEngine(ABC):
    def __init__(self, config: TensorTransferEngineConfig) -> None:
        # Initialize logger.
        self.logger = get_logger("dist_moe.tte.engine")

        # Check the P2P setup correctness.
        assert len(config.remote_addresses) == len(config.remote_p2p_ports)
        assert len(config.remote_addresses) == len(config.remote_ports)
        assert len(config.host_p2p_ports) == len(config.remote_p2p_ports)

        # Set the transfer protocol for the TTE.
        self.transfer_protocol = config.transfer_protocol
        self.ib_device_names = config.ib_device_names  # If the transfer protocol is 'RDMA', data transfers are performing via IB devices.

        # Set the host and remotes names (format: f"{address}:{port}")
        self.host_name = join_host_port(config.host_address, config.host_port)
        self.remote_names = [join_host_port(dst_host, dst_port) for dst_host, dst_port in zip(config.remote_addresses, config.remote_ports)]

        # Setup P2P sockets.
        self.handshake_type = config.handshake_type
        self.p2p_timeout_duration = config.p2p_timeout_duration
        self.metadata_p2p_pairs: Dict[str, P2PPair] = {}
        self.setup_p2p_connections(config)

        # Transfer engine instance.
        self.transfer_engine = None

        # Collect metrics for done data transfers.
        self.done_transfers: Dict[bytes, Transfer] = {}

        # Background monitoring for the async data transfers handling.
        self._stop_event = threading.Event()
        self._transfers: List[Transfer] = []
        self._transfers_lock = threading.Lock()

        # Condition variable for efficient transfer waiting.
        self._transfer_condition = threading.Condition(self._transfers_lock)

        self._transfers_thread = threading.Thread(target=self._monitor_transfers, daemon=True)
        self._transfers_thread.start()

        # Size of one serialized element in bytes.
        self._serialized_size = 2

        # Flag for avoiding double-closing issue.
        self.is_closed = False

        self.logger.info(f"TensorTransferEngine initialized - Host: {self.host_name}, Remotes: {len(self.remote_names)}")

    def setup_p2p_connections(self, config: TensorTransferEngineConfig) -> None:
        self.context = zmq.asyncio.Context()

        for remote_index in range(len(self.remote_names)):
            host_p2p_port = config.host_p2p_ports[remote_index]
            remote_name = self.remote_names[remote_index]
            remote_p2p_address = config.remote_addresses[remote_index]
            remote_p2p_port = config.remote_p2p_ports[remote_index]

            self.metadata_p2p_pairs[remote_name] = P2PPair(self.context.socket(zmq.constants.PUSH), self.context.socket(zmq.constants.PULL))  # type: ignore

            self.metadata_p2p_pairs[remote_name].pusher.bind(join_host_port(config.host_address, host_p2p_port, "tcp"))
            self.metadata_p2p_pairs[remote_name].pusher.setsockopt(zmq.RCVTIMEO, self.p2p_timeout_duration)
            self.metadata_p2p_pairs[remote_name].puller.connect(join_host_port(remote_p2p_address, remote_p2p_port, "tcp"))
            self.metadata_p2p_pairs[remote_name].puller.setsockopt(zmq.RCVTIMEO, self.p2p_timeout_duration)

            self.logger.debug(f"P2P PAIR: PUSH {config.host_address}:{host_p2p_port}; PULL {remote_p2p_address}:{remote_p2p_port}")

    @abstractmethod
    def _init_transfer_engine(self) -> None:
        self.transfer_engine = None

    @abstractmethod
    def handshake_with_remote(self, remote_name: str) -> bool:
        raise NotImplementedError

    def handshake(self) -> bool:
        if self.handshake_type == HandshakeType.NONE:
            return True

        try:
            for remote_name in self.remote_names:
                assert self.handshake_with_remote(remote_name)
        except Exception as e:
            self.logger.error(f"While handshake process an exception has been occured: '{e}'")
            return False

        return True

    @abstractmethod
    def register_batch(
        self,
        data_ptrs: List[torch.Tensor | int],
        data_location: Optional[str],  # or "VRAM"/"DRAM"
        device_id: int,  # GPU id, 0 for CPU located data, fd for "files"
        data_sizes: Optional[List[int]] = None,
    ) -> Tuple[List[int] | nixlBind.nixlXferDList, List[int] | nixlBind.nixlRegDList]:
        raise NotImplementedError

    @abstractmethod
    def deregister_batch(
        self,
        data_registration_descs: List[int] | nixlBind.nixlRegDList,
        data_sizes: Optional[List[int]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_batch_transfer(
        self,
        remote_name: str,
        transfer_name: bytes,
        src_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        dst_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        src_data_sizes: Optional[List[int]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_batch_transfer(
        self,
        remote_name: str,
        transfer_name: bytes,
        src_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        dst_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        src_data_sizes: Optional[List[int]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def serialize_descs(self, data_descs: List[int] | nixlBind.nixlXferDList | nixlBind.nixlRegDList) -> Tuple[int, bytes]:
        raise NotImplementedError

    @abstractmethod
    def deserialize_descs(self, data_serialized_descs: Tuple[int, bytes]) -> List[int] | nixlBind.nixlXferDList | nixlBind.nixlRegDList:
        raise NotImplementedError

    def serialize_p2p_packet(self, header: int, body: List[Tuple[int, bytes]] | Tuple[int, bytes]) -> bytes:
        parts = [struct.pack("!H", header)]

        if isinstance(body, List):
            parts.append(struct.pack("!H", len(body)))
            for offset, metadata in body:
                parts.append(struct.pack("!H", offset))
                parts.append(metadata)
        else:
            parts.append(struct.pack("!H", 1))
            parts.append(struct.pack("!H", body[0]))
            parts.append(body[1])

        return b"".join(parts)

    def deserialize_p2p_packet(self, packet: bytes) -> Tuple[int, List[Tuple[int, bytes]]]:
        offset = 0

        header = struct.unpack("!H", packet[offset : offset + self._serialized_size])[0]
        body = []
        offset += self._serialized_size

        body_length = struct.unpack("!H", packet[offset : offset + self._serialized_size])[0]
        offset += self._serialized_size

        for _ in range(body_length):
            size = struct.unpack("!H", packet[offset : offset + self._serialized_size])[0]
            offset += self._serialized_size

            meta = packet[offset : offset + size]
            offset += size

            body.append((size, meta))

        return header, body

    async def async_send_metadata_to(self, remote_name: str, header: int, body: List[Tuple[int, bytes]] | Tuple[int, bytes]) -> None:
        try:
            await asyncio.wait_for(
                self.metadata_p2p_pairs[remote_name].pusher.send(self.serialize_p2p_packet(header, body)),
                self.p2p_timeout_duration / 1000.0,
            )
        except zmq.Again:
            raise asyncio.TimeoutError()

    async def async_get_metadata_from(self, remote_name: str) -> Tuple[int, List[Tuple[int, bytes]]]:
        try:
            packet = await asyncio.wait_for(self.metadata_p2p_pairs[remote_name].puller.recv(), self.p2p_timeout_duration / 1000.0)
        except zmq.Again:
            raise asyncio.TimeoutError()
        return self.deserialize_p2p_packet(packet)

    # Synchronous wrapper for async_send_metadata_to.
    # Prefer async_send_metadata_to in async context.
    #
    # When called from a thread that is NOT running the event loop (the normal
    # case for handshake / close), asyncio.run() spins up a temporary loop.
    # When called from a *different* thread while the loop IS running on another
    # thread, run_coroutine_threadsafe() submits the coroutine to that loop and
    # blocks the calling thread until it completes — no deadlock.
    # Calling this method from *within* the running loop's own thread is still
    # unsafe (would deadlock); callers in that context must use
    # async_send_metadata_to directly.
    # Retries indefinitely on timeout (zmq.Again / asyncio.TimeoutError), logging a warning each time.
    def send_metadata_to(self, remote_name: str, header: int, body: List[Tuple[int, bytes]] | Tuple[int, bytes]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        while True:
            try:
                if loop is None:
                    # No event loop running in this thread — safe to use asyncio.run().
                    asyncio.run(self.async_send_metadata_to(remote_name, header, body))
                else:
                    # A loop is running on another thread; submit and block until done.
                    future = asyncio.run_coroutine_threadsafe(self.async_send_metadata_to(remote_name, header, body), loop)
                    future.result()
                return
            except (asyncio.TimeoutError, zmq.Again):
                self.logger.warning(f"Waiting to send metadata to {remote_name}, retrying...")

    # Synchronous wrapper for async_get_metadata_from.
    # Prefer async_get_metadata_from in async context.
    # Same threading contract as send_metadata_to above.
    # Retries indefinitely on timeout (zmq.Again / asyncio.TimeoutError), logging a warning each time.
    def get_metadata_from(self, remote_name: str) -> Tuple[int, List[Tuple[int, bytes]]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        while True:
            try:
                if loop is None:
                    return asyncio.run(self.async_get_metadata_from(remote_name))
                else:
                    future = asyncio.run_coroutine_threadsafe(self.async_get_metadata_from(remote_name), loop)
                    return future.result()
            except (asyncio.TimeoutError, zmq.Again):
                self.logger.warning(f"Waiting for metadata from {remote_name}, retrying...")

    # Legacy busy-polling wait. Prefer wait_for_transfer_efficient() which uses condition variables.
    def wait_for_transfer(self, transfer_name: bytes) -> Optional[Transfer]:
        while True:
            with self._transfers_lock:
                if len(transfer_name) == 0 or transfer_name in self.done_transfers:
                    return self.done_transfers[transfer_name]
            time.sleep(0.001)

    def wait_for_transfer_efficient(self, transfer_name: bytes, timeout: Optional[int] = None) -> Optional[Transfer]:
        deadline = time.monotonic() + (timeout / 1000) if timeout is not None else None
        with self._transfer_condition:
            while len(transfer_name) > 0 and transfer_name not in self.done_transfers:
                remaining = (deadline - time.monotonic()) if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    return None
                if not self._transfer_condition.wait(remaining):
                    return None
            return self.done_transfers[transfer_name]

    @abstractmethod
    def _check_handler_status(self, transfer: Transfer) -> bool:
        raise NotImplementedError

    def _monitor_transfers(self) -> None:
        while not self._stop_event.is_set():
            with self._transfers_lock:
                if not self._transfers:
                    pass
                else:
                    # Process transfers efficiently without list recreation.
                    finished_transfers: List[int] = []
                    for index, transfer in enumerate(self._transfers):
                        if self._check_handler_status(transfer):
                            transfer.status = TransferStatus.FINISHED
                            transfer.duration = (timeit.default_timer() - transfer.duration) * 1000
                            self.done_transfers[transfer.name] = transfer
                            finished_transfers.append(index)

                    # Remove finished transfers in reverse order to maintain indices.
                    for index in reversed(finished_transfers):
                        del self._transfers[index]

                    # Notify waiting threads only if transfers completed.
                    if finished_transfers:
                        self._transfer_condition.notify_all()

            # Avoid busy waiting.
            time.sleep(0.001)

    def close(self) -> None:
        if not self.is_closed:
            self._stop_event.set()

            # Wait for transfers thread to finish with timeout.
            self._transfers_thread.join(timeout=5.0)
            if self._transfers_thread.is_alive():
                self.logger.warning("Transfers thread did not stop within timeout")

            # Close ZMQ sockets with error handling.
            for remote_name, metadata_p2p_pair in self.metadata_p2p_pairs.items():
                try:
                    metadata_p2p_pair.pusher.close()
                    metadata_p2p_pair.puller.close()
                except Exception as e:
                    self.logger.warning(f"Error closing P2P pair for {remote_name}: {e}")

            # Terminate ZMQ context with timeout.
            try:
                self.context.term()
            except Exception as e:
                self.logger.warning(f"Error terminating ZMQ context: {e}")

            self.is_closed = True
            self.logger.debug("TensorTransferEngine closed successfully")
