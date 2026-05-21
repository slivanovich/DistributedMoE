import optparse
from typing import List
from utils.Statistic import *


def parse_arguments():
    parser = optparse.OptionParser()
    parser.add_option(
        "--bench_name",
        dest="bench_name",
        help="specify the benchmark name (throughput/e2e_experts/e2e_moe)",
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
    parser.add_option(
        "--number_of_experts_list",
        dest="number_of_experts_list",
        help="specify the list of numbers of experts",
        nargs=1,
        type="str",
        default="1,2,4,8",
    )

    (options, _) = parser.parse_args()

    return options


def plot(benchmark_name: str, number_of_experts_list: List[int], transfer_protocol: str):
    g = Graph(
        10,
        6,
        f"DistMoE {benchmark_name} benchmark via {transfer_protocol}",
        "batch size",
        (
            "throughput (GB/s)"
            if benchmark_name == "throughput"
            else ("GPU memory usage (%)" if benchmark_name == "gpu_memory" else f"{benchmark_name} (ms)")
        ),
    )

    linestyle_list = ["-", "--", "-.", ":"]
    assert len(linestyle_list) >= len(number_of_experts_list)
    for number_of_experts, linestyle in zip(number_of_experts_list, linestyle_list[: len(number_of_experts_list)]):
        for backend, color in zip(
            ["mcte", "nixl", "none"],
            ["#880000", "#008800", "#000088"],
        ):
            if backend == "none" and (benchmark_name == "throughput" or benchmark_name == "gpu_memory"):
                continue

            data_filename = (
                "/Users/skuralenok/arcadia/junk/skuralenok/MCTE/src/python/MoE/benchmarks/data/csv/"
                + f"DistMoE[{number_of_experts}]_{transfer_protocol}_{backend}_{benchmark_name}_benchmark.csv"
            )
            loaded_data = pd.read_csv(data_filename)
            data_x = loaded_data["batch_size"].to_numpy() / 1024**2
            if benchmark_name == "throughput" or benchmark_name == "gpu_memory":
                data_y = loaded_data.iloc[:, 2].to_numpy()
            else:
                data_y = loaded_data.iloc[:, 0].to_numpy()
            statistic = Statistic(None, data_x, data_y, f"{transfer_protocol}_{backend}_{number_of_experts}_experts", linestyle=linestyle)
            statistic.data_color = color
            g.add_statistic(statistic)

    g.show(f"{transfer_protocol}_{benchmark_name}", specifier=1)


if __name__ == "__main__":
    options = parse_arguments()
    assert options.bench_name is not None
    benchmark_name = options.bench_name
    number_of_experts = list(map(int, options.number_of_experts_list.split(",")))

    plot(benchmark_name, number_of_experts, options.transfer_protocol)
