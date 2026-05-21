import time
from typing import List, Optional, Tuple

import os
import timeit
import traceback
import torch

from nixl._api import nixl_agent, nixl_agent_config  # type: ignore
import nixl._bindings as nixlBind  # type: ignore
import nixl._utils as nixl_utils  # type: ignore

from TensorTransferEngine import TensorTransferEngine, TensorTransferEngineConfig
from TensorTransferEngine.utils import HandshakeType, Transfer, TransferStatus, TransferProtocol
from utils.logging_config import get_logger


class NIXLTensorTransferEngine(TensorTransferEngine):
    def __init__(self, config: TensorTransferEngineConfig) -> None:
        super().__init__(config)

        # Initialize NIXL-specific logger.
        self.nixl_logger = get_logger("dist_moe.tte.nixl")

        # Has backend specifics, so parsing it here, not in the TensorTransferEngine class.
        if isinstance(self.ib_device_names, str) and self.ib_device_names != "":
            self.ib_device_names = self.ib_device_names.split(",")

        self._init_transfer_engine()
        assert self.handshake()

        self.nixl_logger.info(f"NIXLTensorTransferEngine initialized with protocol: {self.transfer_protocol}")

    def _init_transfer_engine(self) -> None:
        if self.transfer_protocol == TransferProtocol.RDMA:
            os.environ["UCX_TLS"] = "rc,cuda_copy"

            if len(self.ib_device_names) == 0:
                self.ib_device_names = [f"mlx5_{index}" for index in range(8)]  # TODO: propose that we are always working with 8 GPUs.
            devices = ""
            for index in range(len(self.ib_device_names)):
                devices += self.ib_device_names[index] + (":1," if index < len(self.ib_device_names) - 1 else ":1")
            os.environ["UCX_NET_DEVICES"] = devices
        elif self.transfer_protocol == TransferProtocol.NVLINK:
            os.environ["UCX_TLS"] = "cuda_ipc,cuda_copy,tcp"  # == "cuda,tcp"
            if "UCX_NET_DEVICES" in os.environ:
                del os.environ["UCX_NET_DEVICES"]

        backends = []
        backends.append(self.transfer_protocol.value)
        agent_config = nixl_agent_config(backends=["UCX"])  # backends=["UCX_MO"]
        self.transfer_engine = nixl_agent(self.host_name, agent_config)

    def handshake(self) -> bool:
        if self.handshake_type == HandshakeType.NONE:
            self.nixl_logger.error(f"NIXL TTE can not be correctly instaciated with handshake_type=HandshakeType.NONE, aborting...")
            return False

        try:
            for remote_name in self.remote_names:
                if self.handshake_type == HandshakeType.CLIENT:
                    transfer_engine_metadata: bytes = self.transfer_engine.get_agent_metadata()
                    self.send_metadata_to(remote_name, 0, (len(transfer_engine_metadata), transfer_engine_metadata))

                    header, body = self.get_metadata_from(remote_name)
                    assert header == 0

                    self.transfer_engine.add_remote_agent(body[-1][1])
                elif self.handshake_type == HandshakeType.SERVER:
                    header, body = self.get_metadata_from(remote_name)
                    assert header == 0

                    self.transfer_engine.add_remote_agent(body[-1][1])

                    transfer_engine_metadata: bytes = self.transfer_engine.get_agent_metadata()
                    self.send_metadata_to(remote_name, 0, (len(transfer_engine_metadata), transfer_engine_metadata))
                else:
                    continue
                self.nixl_logger.debug(f"Connected with {remote_name} from {self.host_name}")
        except Exception as e:
            self.nixl_logger.error(f"While handshake process an exception has been occured: '{e}'")
            return False

        return True

    def handshake_with_remote(self, remote_name: str) -> bool:
        try:
            if self.handshake_type == HandshakeType.CLIENT:
                transfer_engine_metadata: bytes = self.transfer_engine.get_agent_metadata()
                self.send_metadata_to(remote_name, 0, (len(transfer_engine_metadata), transfer_engine_metadata))

                header, body = self.get_metadata_from(remote_name)
                assert header == 0

                self.transfer_engine.add_remote_agent(body[-1][1])
            elif self.handshake_type == HandshakeType.SERVER:
                header, body = self.get_metadata_from(remote_name)
                assert header == 0

                self.transfer_engine.add_remote_agent(body[-1][1])

                transfer_engine_metadata: bytes = self.transfer_engine.get_agent_metadata()
                self.send_metadata_to(remote_name, 0, (len(transfer_engine_metadata), transfer_engine_metadata))
            else:
                return True

            self.nixl_logger.debug(f"Connected with {remote_name} from {self.host_name}")
            return True
        except Exception as e:
            self.nixl_logger.error(f"While runtime handshake with {remote_name} an exception has been occured: '{e}'")
            return False

    def register_batch(
        self,
        data_ptrs: List[torch.Tensor | int],
        data_location: Optional[str],  # or "VRAM"/"DRAM"
        device_id: int,  # GPU id, 0 for CPU located data, fd for "files"
        data_sizes: Optional[List[int]] = None,
    ) -> Tuple[List[int] | nixlBind.nixlXferDList, List[int] | nixlBind.nixlRegDList]:
        if data_sizes is not None:
            assert len(data_ptrs) == len(data_sizes)

        transfer_tuples = []
        registration_tuples = []
        for index in range(len(data_ptrs)):
            data_size = 0
            data_ptr = data_ptrs[index]
            if isinstance(data_ptr, torch.Tensor):
                data_size = data_ptr.element_size() * data_ptr.numel()
                data_ptr = data_ptr.data_ptr()
            elif data_sizes is not None:
                data_size = data_sizes[index]

            assert data_size > 0

            registration_tuples.append((data_ptr, data_size, device_id, ""))
            transfer_tuples.append((data_ptr, data_size, device_id))

        data_registration_descs = self.transfer_engine.get_reg_descs(registration_tuples, mem_type=data_location)
        data_transfer_descs = self.transfer_engine.get_xfer_descs(transfer_tuples, mem_type=data_location)

        assert data_transfer_descs is not None and isinstance(data_transfer_descs, nixlBind.nixlXferDList)
        assert data_registration_descs is not None and isinstance(data_registration_descs, nixlBind.nixlRegDList)

        register_status = self.transfer_engine.register_memory(data_registration_descs)
        assert register_status is not None

        return data_transfer_descs, data_registration_descs

    def deregister_batch(
        self,
        data_registration_descs: List[int] | nixlBind.nixlRegDList,
        data_sizes: Optional[List[int]] = None,
    ) -> None:
        assert isinstance(data_registration_descs, nixlBind.nixlRegDList) or isinstance(data_registration_descs, Tuple)

        self.transfer_engine.deregister_memory(data_registration_descs)

    def write_batch_transfer(
        self,
        remote_name: str,
        transfer_name: bytes,
        src_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        dst_data_transfer_descs: List[int] | nixlBind.nixlXferDList,
        src_data_sizes: Optional[List[int]] = None,
    ) -> None:
        assert isinstance(src_data_transfer_descs, nixlBind.nixlXferDList)
        assert isinstance(dst_data_transfer_descs, nixlBind.nixlXferDList)

        if src_data_sizes is not None:
            for index in range(len(src_data_sizes)):
                size = src_data_sizes[index]
                assert size <= max(src_data_transfer_descs[index][1], dst_data_transfer_descs[index][1])  # type: ignore
                src_data_transfer_descs[index] = (src_data_transfer_descs[index][0], size, src_data_transfer_descs[index][2])  # type: ignore
                dst_data_transfer_descs[index] = (dst_data_transfer_descs[index][0], size, dst_data_transfer_descs[index][2])  # type: ignore
        else:
            src_data_sizes = []
            for index in range(len(src_data_sizes)):
                src_data_sizes.append(src_data_transfer_descs[index][1])  # type: ignore

        handler = self.transfer_engine.initialize_xfer(
            "WRITE",
            src_data_transfer_descs,
            dst_data_transfer_descs,
            remote_name,
            transfer_name,
        )

        assert handler is not None
        state = self.transfer_engine.transfer(handler)
        assert state != "ERR"

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
        assert isinstance(src_data_transfer_descs, nixlBind.nixlXferDList)
        assert isinstance(dst_data_transfer_descs, nixlBind.nixlXferDList)

        if src_data_sizes is not None:
            for index in range(len(src_data_sizes)):
                size = src_data_sizes[index]
                assert size <= max(src_data_transfer_descs[index][1], dst_data_transfer_descs[index][1])  # type: ignore
                src_data_transfer_descs[index] = (src_data_transfer_descs[index][0], size, src_data_transfer_descs[index][2])  # type: ignore
                dst_data_transfer_descs[index] = (dst_data_transfer_descs[index][0], size, dst_data_transfer_descs[index][2])  # type: ignore
        else:
            src_data_sizes = []
            for index in range(len(src_data_sizes)):
                src_data_sizes.append(src_data_transfer_descs[index][1])  # type: ignore

        handler = self.transfer_engine.initialize_xfer(
            "READ",
            dst_data_transfer_descs,
            src_data_transfer_descs,
            remote_name,
            transfer_name,
        )

        assert handler is not None
        state = self.transfer_engine.transfer(handler)
        assert state != "ERR"

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
        data_serialized_descs = self.transfer_engine.get_serialized_descs(data_descs)
        data_serialized_descs_size = len(data_serialized_descs)

        return data_serialized_descs_size, data_serialized_descs

    def deserialize_descs(self, data_serialized_descs: Tuple[int, bytes]) -> List[int] | nixlBind.nixlXferDList | nixlBind.nixlRegDList:
        assert isinstance(data_serialized_descs, Tuple)

        return self.transfer_engine.deserialize_descs(data_serialized_descs[1])

    def _check_handler_status(self, transfer: Transfer) -> bool:
        try:
            transfer_state = self.transfer_engine.check_xfer_state(transfer.handler)
            if transfer_state == "DONE":
                return True
            if transfer_state == "ERR":
                self.nixl_logger.error(f"Transfer with name: {transfer.name} has been failed; working on NIXL backend")
                self.transfer_engine.release_xfer_handle(transfer.handler)
                transfer.status = TransferStatus.RELEASED
        except Exception as e:
            self.nixl_logger.error(f"Failed to execute transfer with name: {transfer.name}; working on NIXL backend")
            self.nixl_logger.info(f"Traceback: {traceback.format_exc()}")
            self.transfer_engine.release_xfer_handle(transfer.handler)
        return False

    def close(self) -> None:
        if "UCX_TLS" in os.environ:
            del os.environ["UCX_TLS"]
        if "UCX_NET_DEVICES" in os.environ:
            del os.environ["UCX_NET_DEVICES"]

        if self.is_closed:
            return

        super().close()

        with self._transfers_lock:
            for transfer in self._transfers + list(self.done_transfers.values()):
                if transfer.status != TransferStatus.RELEASED:
                    self.transfer_engine.release_xfer_handle(transfer.handler)
                    transfer.status = TransferStatus.RELEASED

        for remote_name in self.remote_names:
            self.transfer_engine.remove_remote_agent(remote_name)
