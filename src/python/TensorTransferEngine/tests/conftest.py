def pytest_addoption(parser):
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
