import optparse
from utils.Statistic import *


def parse_arguments():
    parser = optparse.OptionParser()
    parser.add_option(
        "--bench_name",
        dest="bench_name",
        help="specify the benchmark name (throughput/register/deregister)",
        nargs=1,
        type="string",
    )
    parser.add_option(
        "--transfer_protocol",
        dest="transfer_protocol",
        help="specify the transfer protocol (rdma/nvlink)",
        nargs=1,
        type="string",
    )

    (options, _) = parser.parse_args()

    return options


def plot(benchmark_name: str, transfer_protocol: str):
    g = Graph(
        10,
        6,
        f"TensorTransfer {benchmark_name} benchmark",
        "tensor size (MB)",
        ("throughput (GB/s)" if benchmark_name == "throughput" else f"{benchmark_name} (sec)"),
    )

    for transfer_way, color in zip(
        [
            "mcte_writing",
            "mcte_reading",
            "nixl_writing",
            "nixl_reading",
        ],
        [
            "#880000",
            "#440000",
            "#008800",
            "#004400",
        ],
    ):
        data_filename = (
            "/Users/skuralenok/arcadia/junk/skuralenok/MCTE/src/python/TensorTransfer/benchmarks/data/csv/"
            + f"TT_{transfer_protocol}_{transfer_way}_{benchmark_name}_benchmark.csv"
        )
        loaded_data = pd.read_csv(data_filename)
        data_x = loaded_data["data_size"].to_numpy() / 1024**2
        if benchmark_name == "throughput":
            data_y = loaded_data.iloc[:, 2].to_numpy()
        else:
            data_y = loaded_data.iloc[:, 0].to_numpy()
        statistic = Statistic(
            None,
            data_x,
            data_y,
            f"{transfer_way}",
        )
        statistic.data_color = color
        g.add_statistic(statistic)
    g.show(benchmark_name, specifier=1)


if __name__ == "__main__":
    options = parse_arguments()
    assert options.bench_name is not None
    benchmark_name = options.bench_name

    plot(benchmark_name, options.transfer_protocol)
