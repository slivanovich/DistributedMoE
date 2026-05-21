def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        help="specify the backend of tensor transfer engine (nixl/mcte)",
    )
    parser.addoption(
        "--gpu_id_host",
        help="specify the host's gpu id (from 0 to 7)",
        default="0",
    )
    parser.addoption(
        "--gpu_id_remote",
        help="specify the remote's gpu id (from 0 to 7)",
        default="1",
    )
    parser.addoption(
        "--precision",
        help="specify the tensors presision (bf16/fp16/fp32/int8)",
        default="fp16",
    )
    # parser.addoption(
    #     "--schema",
    #     help="specify the schema of data transfers (push-push/pull-push)",
    #     default="pull-push",
    # )
