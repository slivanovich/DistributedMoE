from dataclasses import dataclass
from enum import Enum
import itertools
from typing import List, Tuple
import uuid

import zmq.asyncio

from nixl._api import nixl_xfer_handle  # type: ignore


# Supported transfer protocols.
class TransferProtocol(Enum):
    RDMA = "rdma"
    NVLINK = "nvlink"


# Define endpoint role in handshake process.
class HandshakeType(Enum):
    NONE = 0
    SERVER = 1
    CLIENT = 2


# Transfer status enum.
class TransferStatus(Enum):
    CREATED = 0
    FINISHED = 1
    RELEASED = 2


# Transfer dataclass.
@dataclass
class Transfer:
    name: bytes
    status: TransferStatus
    remote_name: str
    handler: nixl_xfer_handle | int
    size: int  # bytes
    duration: float  # ms


# P2P pair of sockets (pusher/puller) encapsulation.
@dataclass
class P2PPair:
    pusher: zmq.asyncio.Socket
    puller: zmq.asyncio.Socket


def join_host_port(host: str, port: int, scheme: str = "", dir: str = "") -> str:
    if len(scheme) > 0:
        if len(dir) > 0:
            return f"{scheme}://{host}:{port}/{dir}"
        return f"{scheme}://{host}:{port}"
    return f"{host}:{port}"


# Atomic counter for generating unique transfer names
_transfer_counter = itertools.count()


def _next_random_name() -> bytes:
    # counter_value = next(_transfer_counter) % (2**33)
    # return struct.pack("!Q", counter_value)
    counter_value = uuid.uuid4().bytes[-8:]
    return counter_value
