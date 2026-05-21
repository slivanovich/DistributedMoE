#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch

import nixl._utils as nixl_utils  # type: ignore
from nixl._api import nixl_agent, nixl_agent_config  # type: ignore
from nixl.logging import get_logger  # type: ignore

# Configure logging
logger = get_logger(__name__)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
        remote_device = torch.device(f"cuda:1")
    else:
        device = torch.device("cpu")

    buf_size = 256
    # Allocate memory and register with NIXL

    logger.info("Using NIXL Plugins from:\n%s", os.environ["NIXL_PLUGIN_DIR"])

    # Example using nixl_agent_config
    agent_config = nixl_agent_config(backends=["UCX"])
    nixl_agent1 = nixl_agent("target", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    assert "UCX" in plugin_list

    logger.info(
        "Plugin parameters:\n%s\n%s",
        nixl_agent1.get_plugin_mem_types("UCX"),
        nixl_agent1.get_plugin_params("UCX"),
    )

    logger.info(
        "Backend parameters:\n%s\n%s",
        nixl_agent1.get_backend_mem_types("UCX"),
        nixl_agent1.get_backend_params("UCX"),
    )

    # Just for tensor test
    tensors_1 = [torch.randn(10, dtype=torch.float32).to(device) for _ in range(2)]
    agent1_tensor_reg_descs = nixl_agent1.get_reg_descs(tensors_1, is_sorted=False)
    agent1_tensor_xfer_descs = nixl_agent1.get_xfer_descs(tensors_1, is_sorted=False)

    assert (
        nixl_agent1.register_memory(agent1_tensor_reg_descs, is_sorted=False)
        is not None
    )

    # Example using default configs, which is UCX backend only
    nixl_agent2 = nixl_agent("initiator", agent_config)
    addr3 = nixl_utils.malloc_passthru(buf_size * 2)
    addr4 = addr3 + buf_size

    tensors_2 = [
        torch.zeros(10, dtype=torch.float32).to(remote_device) for _ in range(2)  # type: ignore
    ]
    agent2_tensor_reg_descs = nixl_agent2.get_reg_descs(tensors_2, is_sorted=False)
    agent2_tensor_xfer_descs = nixl_agent2.get_xfer_descs(tensors_2, is_sorted=False)

    assert nixl_agent2.register_memory(tensors_2, is_sorted=False) is not None

    # Exchange metadata
    meta = nixl_agent2.get_agent_metadata()
    remote_name = nixl_agent1.add_remote_agent(meta)
    logger.info("Loaded name from metadata: %s", remote_name)
    nixl_agent1.remove_remote_agent(remote_name)

    meta = nixl_agent1.get_agent_metadata()
    remote_name = nixl_agent2.add_remote_agent(meta)
    logger.info("Loaded name from metadata: %s", remote_name)

    serdes = nixl_agent1.get_serialized_descs(agent1_tensor_xfer_descs)
    src_descs_recvd = nixl_agent2.deserialize_descs(serdes)
    assert src_descs_recvd == agent1_tensor_xfer_descs

    # initialize transfer mode
    xfer_handle_1 = nixl_agent2.initialize_xfer(
        "READ",
        agent2_tensor_xfer_descs,
        src_descs_recvd,
        remote_name,
        b"UUID1",
    )
    if not xfer_handle_1:
        logger.error("Creating transfer failed.")
        exit()

    # test multiple postings
    for _ in range(2):
        state = nixl_agent2.transfer(xfer_handle_1)
        assert state != "ERR"

        target_done = False
        init_done = False

        while (not init_done) or (not target_done):
            if not init_done:
                state = nixl_agent2.check_xfer_state(xfer_handle_1)
                if state == "ERR":
                    logger.error("Transfer got to Error state.")
                    exit()
                elif state == "DONE":
                    init_done = True
                    print("----------------------")
                    print(tensors_2)
                    print("----------------------")
                    logger.info("Initiator done")

            if not target_done:
                if nixl_agent1.check_remote_xfer_done("initiator", b"UUID1"):
                    target_done = True
                    print("----------------------")
                    print(tensors_1)
                    print("----------------------")
                    logger.info("Target done")

    nixl_agent1.deregister_memory(agent1_tensor_reg_descs)
    nixl_agent2.deregister_memory(agent2_tensor_reg_descs)

    logger.info("Test Complete.")
