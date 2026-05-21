from typing import List, Optional, Tuple

import os
import struct
import timeit
import torch

import nixl._bindings as nixlBind  # type: ignore
from mooncake.engine import TransferEngine  # type: ignore

from TensorTransferEngine import TensorTransferEngine, TensorTransferEngineConfig
from TensorTransferEngine.utils import HandshakeType, Transfer, TransferStatus, TransferProtocol, join_host_port
from utils.logging_config import get_logger


class MCTETensorTransferEngine(TensorTransferEngine):
    def __init__(
        self,
        config: TensorTransferEngineConfig,
    ) -> None:
        super().__init__(config)

        # Initialize MCTE-specific logger.
        self.mcte_logger = get_logger("dist_moe.tte.mcte")

        assert config.metadata_host is not None
        assert config.metadata_port is not None
        assert config.metadata_dir is not None
        assert config.metadata_schema is not None

        self.metadata_host = config.metadata_host
        self.metadata_port = config.metadata_port
        self.metadata_dir = config.metadata_dir
        self.metadata_schema = config.metadata_schema

        # Has backend specifics, so parsing it here, not in the TensorTransferEngine class.
        self.ib_device_names = config.ib_device_names

        self._init_transfer_engine()
        assert self.handshake()

        self.mcte_logger.info(f"MCTETensorTransferEngine initialized with protocol: {self.transfer_protocol}")

    def _init_transfer_engine(self) -> None:
        assert self.transfer_protocol == TransferProtocol.RDMA or self.transfer_protocol == TransferProtocol.NVLINK

        if self.transfer_protocol == TransferProtocol.NVLINK:
            os.environ["MC_FORCE_MNNVL"] = "True"  # )) (можно ставить и False и "" и че угодно == поставить True)
            self.ib_device_names = ""  # никакого эффекта не имеет на фрупут и конечный е2е
        else:
            if "MC_FORCE_MNNVL" in os.environ:
                del os.environ["MC_FORCE_MNNVL"]

        self.transfer_engine = TransferEngine()
        transfer_engine_status = self.transfer_engine.initialize(
            self.host_name,
            join_host_port(
                self.metadata_host,
                self.metadata_port,
                self.metadata_schema,
                self.metadata_dir,
            ),
            self.transfer_protocol.value,
            self.ib_device_names,
        )
        if transfer_engine_status != 0:
            self.mcte_logger.error(
                f"While creating the transfer engine instance that is working on MoonCake backend and with the transfer protocol {self.transfer_protocol} an error has been occured, aborting"
            )
            raise RuntimeError(
                f"While creating the transfer engine instance that is working on MoonCake backend and with the transfer protocol {self.transfer_protocol} an error has been occured, aborting"
            )

    def handshake_with_remote(self, remote_name: str) -> bool:
        if self.handshake_type == HandshakeType.CLIENT:
            self.send_metadata_to(remote_name, 0, [])
            header, _ = self.get_metadata_from(remote_name)
            assert header == 0
        elif self.handshake_type == HandshakeType.SERVER:
            header, _ = self.get_metadata_from(remote_name)
            assert header == 0
            self.send_metadata_to(remote_name, 0, [])
        else:
            return True

        self.mcte_logger.debug(f"Connected with {remote_name} from {self.host_name}")
        return True

    def register_batch(
        self,
        data_ptrs: List[torch.Tensor | int],
        data_location: Optional[str],  # or "VRAM"/"DRAM"
        device_id: int,  # GPU id, 0 for CPU located data, fd for "files"
        data_sizes: Optional[List[int]] = None,
        data_offsets: Optional[List[int]] = None,
    ) -> Tuple[List[int] | nixlBind.nixlXferDList, List[int] | nixlBind.nixlRegDList]:
        if data_offsets is not None:
            assert len(data_ptrs) == len(data_offsets)
        if data_sizes is not None:
            assert len(data_ptrs) == len(data_sizes)
        else:
            data_sizes = [0 for _ in range(len(data_ptrs))]

        for index in range(len(data_ptrs)):
            data_ptr = data_ptrs[index]
            if isinstance(data_ptr, torch.Tensor):
                data_sizes[index] = data_ptr.shape.numel() * data_ptr.dtype.itemsize
                data_ptrs[index] = data_ptr.data_ptr()

            assert data_sizes is not None and data_sizes[index] > 0

        status = self.transfer_engine.batch_register_memory(data_ptrs, data_sizes)
        assert status >= 0

        return data_ptrs, data_ptrs

    def deregister_batch(
        self,
        data_registration_descs: List[int] | nixlBind.nixlRegDList,
        data_sizes: Optional[List[int]] = None,
    ) -> None:
        assert not isinstance(data_registration_descs, nixlBind.nixlRegDList)

        if data_sizes is not None:
            assert len(data_registration_descs) == len(data_sizes)

        status = self.transfer_engine.batch_unregister_memory(data_registration_descs)
        assert status == 0

    def write_batch_transfer(
        self,
        remote_name: str,
        transfer_name: bytes,
        src_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        dst_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        src_data_sizes: Optional[List[int]] = None,
    ) -> None:
        assert not isinstance(src_data_transfer_descs, nixlBind.nixlXferDList)
        assert not isinstance(dst_data_transfer_descs, nixlBind.nixlXferDList)
        assert len(src_data_transfer_descs) == len(dst_data_transfer_descs)
        assert src_data_sizes is not None and len(src_data_sizes) == len(src_data_transfer_descs)

        handler = self.transfer_engine.batch_transfer_async_write(
            remote_name,
            src_data_transfer_descs,
            dst_data_transfer_descs,
            src_data_sizes,
        )

        assert handler >= 0
        with self._transfers_lock:
            self._transfers.append(
                Transfer(
                    status=TransferStatus.CREATED,
                    name=transfer_name,
                    handler=handler,
                    remote_name=remote_name,
                    size=sum(src_data_sizes),
                    duration=timeit.default_timer(),
                )
            )

    def read_batch_transfer(
        self,
        remote_name: str,
        transfer_name: bytes,
        src_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        dst_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        src_data_sizes: Optional[List[int]] = None,
    ) -> None:
        assert not isinstance(src_data_transfer_descs, nixlBind.nixlXferDList)
        assert not isinstance(dst_data_transfer_descs, nixlBind.nixlXferDList)
        assert len(src_data_transfer_descs) == len(dst_data_transfer_descs)
        assert src_data_sizes is not None and len(src_data_sizes) == len(src_data_transfer_descs)

        handler = self.transfer_engine.batch_transfer_async_read(
            remote_name,
            dst_data_transfer_descs,
            src_data_transfer_descs,
            src_data_sizes,
        )

        assert handler >= 0
        with self._transfers_lock:
            self._transfers.append(
                Transfer(
                    status=TransferStatus.CREATED,
                    name=transfer_name,
                    handler=handler,
                    remote_name=remote_name,
                    size=sum(src_data_sizes),
                    duration=timeit.default_timer(),
                )
            )

    def serialize_descs(self, data_descs: List[int] | nixlBind.nixlXferDList | nixlBind.nixlRegDList) -> Tuple[int, bytes]:
        assert not isinstance(data_descs, nixlBind.nixlXferDList) and not isinstance(data_descs, nixlBind.nixlRegDList)

        sizeof_ull = 8
        data_serialized_descs = []
        for data_desc in data_descs:
            data_serialized_descs.append(struct.pack("!Q", data_desc))
        return (sizeof_ull * len(data_descs), b"".join(data_serialized_descs))

    def deserialize_descs(self, data_serialized_descs: Tuple[int, bytes]) -> List[int] | nixlBind.nixlXferDList | nixlBind.nixlRegDList:
        assert isinstance(data_serialized_descs, Tuple)

        data_descs: List[int] = []
        size = data_serialized_descs[0]
        packet = data_serialized_descs[1]
        sizeof_ull = 8

        offset = 0
        for _ in range(size // sizeof_ull):
            data_descs.append(struct.unpack("!Q", packet[offset : offset + sizeof_ull])[0])
            offset += sizeof_ull

        return data_descs

    def _check_handler_status(self, transfer: Transfer) -> bool:
        status = self.transfer_engine.get_batch_transfer_status([transfer.handler])

        if status == 0:
            return True
        elif status == -1:
            self.mcte_logger.error(f"Transfer with name: {transfer.name} has been timed out; working on MoonCake backend")
        else:
            self.mcte_logger.error(
                f"Transfer with name: {transfer.name} has been failed due to unknown reason; working on MoonCake backend"
            )
        return False

    def close(self) -> None:
        if "MC_FORCE_MNNVL" in os.environ:
            del os.environ["MC_FORCE_MNNVL"]

        if self.is_closed:
            return

        super().close()
