from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from queue import Queue

import torch


class P2PHeaders(Enum):  # [1000; 1999]
    # !!! 4738 - is reserved by the TT !!!
    Ping = 1001
    Pong = 1002
    DispatchSubmit = 1038  # from moe block
    CombineSubmit = 1039  # from moe block
    CombineRequest = 1139  # from experts block
    IOTransferDescriptorsRequest = 1200  # from experts block
    IOTransferDescriptorsSubmit = 1201  # from moe block
    HostOutputBufferUnlocked = 1432  # from experts block
    HostInputBufferUnlocked = 1587  # from experts block
    ExpertsBlockStop = 1999  # from moe block


@dataclass
class ExpertMetaData:
    block_indexes: List[int]  # List of indexes of expert blocks in which this expert is located.
    expert_shared_rate: int  # Shared rate (how many blocks is this expert located in).
    expert_weights: torch.Tensor  # List of expert weights in this block.
    expert_token_indexes: torch.Tensor  # List of expert activations (as tensor) in this block.
    expert_total_activations: int  # Number of total activations on current input batch.
    expert_offsets: Queue[
        Tuple[int, int]
    ]  # There is a list of not handled activations (on the start there is only one segment [0, expert_total_activations]).
