from typing import Dict, List, Union

import os
import shutil
import subprocess
import torch
import torch._utils


def get_p2p_ports(ports: int | List[int], dst_ports: List[int]):  # f(ports) -> List[int] with len(dst_ports)
    if isinstance(ports, int):
        ports = [ports]
    assert isinstance(ports, List)

    if len(ports) == 1:
        ports = [ports[0] for _ in range(len(dst_ports))]

    port_offset = 1000  # TODO
    p2p_ports = [ports[index] + port_offset + 1 + index for index in range(len(dst_ports))]

    return p2p_ports


_DEVICE = Union[str, int, torch.device]


def _get_gpu_id(device_id: int) -> str:
    """Get the unmasked real GPU IDs."""

    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")

    return cuda_visible_devices[device_id].strip()


def get_nvidia_gpu_stats(device: _DEVICE) -> Dict[str, float]:
    """Get GPU stats including memory, fan speed, and temperature from nvidia-smi.

    Args:
        device: GPU device for which to get stats

    Returns:
        A dictionary mapping the metrics to their values.

    Raises:
        FileNotFoundError:
            If nvidia-smi installation not found

    """

    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found")

    gpu_stat_metrics = [
        ("utilization.gpu", "%"),
        ("memory.used", "MB"),
        ("memory.free", "MB"),
        ("utilization.memory", "%"),
        ("fan.speed", "%"),
        ("temperature.gpu", "°C"),
        ("temperature.memory", "°C"),
    ]
    gpu_stat_keys = [k for k, _ in gpu_stat_metrics]
    gpu_query = ",".join(gpu_stat_keys)

    index = torch._utils._get_device_index(device)
    gpu_id = _get_gpu_id(index)
    result = subprocess.run(
        [nvidia_smi_path, f"--query-gpu={gpu_query}", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.0

    s = result.stdout.strip()
    stats = [_to_float(x) for x in s.split(", ")]
    return {f"{x} ({unit})": stat for (x, unit), stat in zip(gpu_stat_metrics, stats)}
