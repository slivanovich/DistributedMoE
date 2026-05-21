import pytest
import sys
import torch

from TensorTransferEngine.TensorTransferEngine import TensorTransferEngine, TensorTransferEngineConfig
from TensorTransferEngine.MCTETensorTransferEngine import MCTETensorTransferEngine
from TensorTransferEngine.utils import HandshakeType, TransferProtocol


@pytest.fixture
def host_tte_instance():
    config = TensorTransferEngineConfig(
        host_address="localhost",
        host_port=22345,
        host_p2p_ports=[22346],
        remote_addresses=["localhost"],
        remote_ports=[22389],
        remote_p2p_ports=[22382],
        handshake_type=HandshakeType.NONE,
        transfer_protocol=TransferProtocol.RDMA,
        ib_device_names="mlx5_2",
        p2p_timeout_duration=3000,
        metadata_schema="http",
        metadata_host="localhost",
        metadata_port=8080,
        metadata_dir="metadata",
    )

    tte = MCTETensorTransferEngine(config)

    yield tte

    tte.close()


@pytest.fixture
def remote_tte_instance():
    config = TensorTransferEngineConfig(
        host_address="localhost",
        host_port=22389,
        host_p2p_ports=[22382],
        remote_addresses=["localhost"],
        remote_ports=[22345],
        remote_p2p_ports=[22346],
        handshake_type=HandshakeType.NONE,
        transfer_protocol=TransferProtocol.RDMA,
        ib_device_names="mlx5_3",
        p2p_timeout_duration=3000,
        metadata_schema="http",
        metadata_host="localhost",
        metadata_port=8080,
        metadata_dir="metadata",
    )
    tte = MCTETensorTransferEngine(config)

    yield tte

    tte.close()


def test_tensor_registration(host_tte_instance, host_device, dtype):
    assert isinstance(host_tte_instance, TensorTransferEngine)

    tensor = torch.randn(128, dtype=dtype).to(host_device)

    tensor_transfer_descs, tensor_registration_descs = host_tte_instance.register_batch(
        [tensor], "VRAM" if host_device.type == "cuda" else "DRAM", host_device.index
    )
    assert tensor_transfer_descs is not None and tensor_registration_descs is not None

    host_tte_instance.deregister_batch(tensor_registration_descs)


def test_batch_registration(host_tte_instance, host_device, dtype):
    assert isinstance(host_tte_instance, TensorTransferEngine)

    batch_size = 64
    tensor_size = 128
    batch_1 = torch.randn((batch_size, tensor_size), dtype=dtype).to(host_device)
    batch_2 = torch.randn((batch_size, tensor_size), dtype=dtype).to(host_device)

    batch_transfer_descs, batch_registration_descs = host_tte_instance.register_batch(
        [batch_1.data_ptr(), batch_2.data_ptr()],
        "VRAM" if host_device.type == "cuda" else "DRAM",
        host_device.index,
        [batch_size * tensor_size * dtype.itemsize, batch_size * tensor_size * dtype.itemsize],
    )
    assert batch_transfer_descs is not None and batch_registration_descs is not None

    host_tte_instance.deregister_batch(batch_registration_descs)


def test_serialization(host_tte_instance, host_device, dtype):
    assert isinstance(host_tte_instance, TensorTransferEngine)

    tensor = torch.randn(128, dtype=dtype).to(host_device)

    tensor_transfer_descs, tensor_registration_descs = host_tte_instance.register_batch(
        [tensor], "VRAM" if host_device.type == "cuda" else "DRAM", host_device.index
    )
    assert tensor_transfer_descs is not None
    assert tensor_registration_descs is not None

    tensor_serialized_data = host_tte_instance.serialize_descs(tensor_transfer_descs)
    tensor_deserialized_transfer_descs = host_tte_instance.deserialize_descs(tensor_serialized_data)  # type: ignore
    assert tensor_transfer_descs == tensor_deserialized_transfer_descs

    tensor_serialized_data = host_tte_instance.serialize_descs(tensor_registration_descs)
    tensor_deserialized_register_descs = host_tte_instance.deserialize_descs(tensor_serialized_data)  # type: ignore
    assert tensor_registration_descs == tensor_deserialized_register_descs


def test_write_transfer(host_tte_instance, remote_tte_instance, host_device, remote_device, dtype):
    assert isinstance(host_tte_instance, TensorTransferEngine)
    assert isinstance(remote_tte_instance, TensorTransferEngine)

    src_tensor = torch.randn(128, dtype=dtype).to(host_device)
    dst_tensor = torch.empty(128, dtype=dtype).to(remote_device)

    src_tensor *= 100
    src_tensor_transfer_descs, src_tensor_registration_descs = host_tte_instance.register_batch(
        [src_tensor], "VRAM" if host_device.type == "cuda" else "DRAM", host_device.index
    )
    dst_tensor_transfer_descs, dst_tensor_registration_descs = remote_tte_instance.register_batch(
        [dst_tensor], "VRAM" if remote_device.type == "cuda" else "DRAM", remote_device.index
    )

    host_tte_instance.write_batch_transfer(
        remote_name=remote_tte_instance.host_name,
        transfer_name="write_test".encode(),
        src_data_transfer_descs=src_tensor_transfer_descs,
        dst_data_transfer_descs=dst_tensor_transfer_descs,
        src_data_sizes=[128 * dtype.itemsize],
    )
    host_tte_instance.wait_for_transfer_efficient("write_test".encode(), 3000)
    assert torch.allclose(src_tensor, dst_tensor.to(host_device))
    assert dst_tensor.device == remote_device

    host_tte_instance.deregister_batch(src_tensor_registration_descs)
    remote_tte_instance.deregister_batch(dst_tensor_registration_descs)


def test_read_transfer(host_tte_instance, remote_tte_instance, host_device, remote_device, dtype):
    assert isinstance(host_tte_instance, TensorTransferEngine)
    assert isinstance(remote_tte_instance, TensorTransferEngine)

    src_tensor = torch.randn(128, dtype=dtype).to(host_device)
    dst_tensor = torch.empty(128, dtype=dtype).to(remote_device)

    src_tensor_transfer_descs, src_tensor_registration_descs = host_tte_instance.register_batch(
        [src_tensor], "VRAM" if host_device.type == "cuda" else "DRAM", host_device.index
    )
    dst_tensor_transfer_descs, dst_tensor_registration_descs = remote_tte_instance.register_batch(
        [dst_tensor], "VRAM" if remote_device.type == "cuda" else "DRAM", remote_device.index
    )

    remote_tte_instance.read_batch_transfer(
        remote_name=host_tte_instance.host_name,
        transfer_name="read_test".encode(),
        src_data_transfer_descs=src_tensor_transfer_descs,
        dst_data_transfer_descs=dst_tensor_transfer_descs,
        src_data_sizes=[128 * dtype.itemsize],
    )
    remote_tte_instance.wait_for_transfer_efficient("read_test".encode(), 3000)
    assert torch.allclose(src_tensor, dst_tensor.to(host_device))
    assert dst_tensor.device == remote_device

    host_tte_instance.deregister_batch(src_tensor_registration_descs)
    remote_tte_instance.deregister_batch(dst_tensor_registration_descs)


@pytest.fixture(scope="session")
def host_device(request):
    device_id = request.config.getoption("--gpu_id_host")
    if device_id == "cpu":
        return torch.device(f"cpu")
    device_id = int(device_id)
    return torch.device(f"cuda:{device_id}")


@pytest.fixture(scope="session")
def remote_device(request):
    device_id = request.config.getoption("--gpu_id_remote")
    if device_id == "cpu":
        return torch.device(f"cpu")
    device_id = int(device_id)
    return torch.device(f"cuda:{device_id}")


@pytest.fixture(scope="session")
def dtype(request):
    precision = request.config.getoption("--precision")
    if precision == "fp16":
        return torch.float16
    elif precision == "fp32":
        return torch.float32
    elif precision == "int8":
        return torch.int8
    elif precision == "bf16":
        return torch.bfloat16


if __name__ == "__main__":
    pytest.main(sys.argv)
