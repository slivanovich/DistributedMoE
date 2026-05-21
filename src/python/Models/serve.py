import optparse
import torch
import torch.nn as nn
from tokenizers import Tokenizer, Encoding


def parse_arguments():
    parser = optparse.OptionParser()
    parser.add_option(
        "--model",
        dest="model",
        help="Specify the model name: Qwen3-Next-30B-A3B/Qwen3-30B-A3B-FP8/... (which is installed in the 'models' directory)",
        nargs=1,
        type="str",
        default="Qwen3-Next-30B-A3B",
    )
    parser.add_option(
        "--schema",
        dest="schema",
        help="Schema of the model's serving: local/p2p-moe/deepep-moe",
        nargs=1,
        type="str",
        default="local",
    )
    parser.add_option(
        "--serving",
        dest="serving",
        help="The model's serving type: online/offline",
        nargs=1,
        type="str",
        default="online",
    )
    parser.add_option(
        "--dataset_path",
        dest="dataset_path",
        help="Specify the path to the dataset",
        nargs=1,
        type="str",
        default="",
    )

    (options, _) = parser.parse_args()

    return options


def online_serving(model: nn.Module, tokenizer: Tokenizer):
    while True:
        request: str = input()
        tokenized_request: Encoding = tokenizer.encode(request)
        output_tokens: torch.Tensor = model(tokenized_request.ids)
        response: str = tokenizer.decode(output_tokens)
        print(f"\nRequest: '{request}';\nResponse: {response}\n")


def offline_serving(model: nn.Module, tokenizer: Tokenizer, dataset_path: str): ...


if __name__ == "__main__":
    ...
