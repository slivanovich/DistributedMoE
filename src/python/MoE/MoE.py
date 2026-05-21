import timeit
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from utils.Benchmark import Benchmark


class Qwen3MoEMLP(nn.Module):
    def __init__(
        self,
        hidden_features: int,
        intermediate_features: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.intermediate_features = intermediate_features

        self.gate_proj = nn.Linear(self.hidden_features, self.intermediate_features, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_features, self.intermediate_features, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_features, self.hidden_features, bias=False, dtype=dtype)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class Qwen3SparseMoEBlock(nn.Module):
    def __init__(
        self,
        number_of_experts: int,
        top_k: int,
        hidden_features: int,
        intermediate_features: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.number_of_experts = number_of_experts
        self.top_k = top_k
        self.hidden_features = hidden_features
        self.intermediate_features = intermediate_features

        self.gate = nn.Linear(self.hidden_features, self.number_of_experts, bias=False, dtype=dtype)
        self.experts = [Qwen3MoEMLP(hidden_features, intermediate_features) for expert_index in range(self.number_of_experts)]

    def forward(self, input_batch: torch.Tensor, experts_latency_benchmark: Optional[Benchmark] = None) -> torch.Tensor:
        batch_size, sequence_length, hidden_features = input_batch.shape
        input_batch = input_batch.view(-1, hidden_features)

        # router_logits: (batch_size * sequence_length, number_of_experts)
        router_logits = self.gate(input_batch)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # router_weights/selected_experts: (batch_size * sequence_length, top_k) (router_weights: weight, selected_experts: index)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(input_batch.dtype)

        output_batch = torch.zeros((batch_size * sequence_length, hidden_features), dtype=input_batch.dtype, device=input_batch.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # expert_mask before permute: (batch_size * sequence_length, top_k, number_of_experts) -- for each token there are top_k one hot masks
        # .permute: (number_of_experts, top_k, batch_size * sequence_length) -- for each expert we have top_k ways being selected (one hotted)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.number_of_experts).permute(2, 1, 0)
        duration = 0
        is_first_expert = True

        # Loop over all available experts in the model and perform the computation on each expert
        # .sum: get all activations for each of experts; expert_hit: (number_of_experts)
        # .greater: boolean tensor with true label is >0 activations; expert_hit: (number_of_experts)
        # .nonzero: tensor with experts index if its activations is >0; expert_hit: (number_of_experts, 1)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_index in expert_hit:
            expert_layer = self.experts[expert_index]  # type: ignore

            # Find all entries with not 0 activations
            # Returns 2 tensors: experts priority (from top_k-1 to 0 - hottest is 0 priority), token_index (from 0 to batch_size * sequence_length - index of the token for the expert)
            # .squeeze(0) has effect only is top_k=1 ((top_k, batch_size * sequence_length) -> (batch_size * sequence_length) and expert priority is None)
            expert_priorities, tokens_indexes = torch.where(expert_mask[expert_index].squeeze(0))

            expert_input_batch = input_batch[tokens_indexes]

            if experts_latency_benchmark is not None:
                duration = timeit.default_timer()
                if is_first_expert:
                    experts_latency_benchmark.duration.append(0)
                    is_first_expert = False

            expert_output_batch = expert_layer(expert_input_batch)

            if experts_latency_benchmark is not None:
                duration = (timeit.default_timer() - duration) * 1000.0
                experts_latency_benchmark.duration[-1] += duration

            # routing_weights[tokens_indexes, expert_priorities]: (number of tokens_indexes) - 1d tensor, but we are multiplicating 2d expert_output_batch => None in the end
            weighted_expert_output_batch = expert_output_batch * routing_weights[tokens_indexes, expert_priorities, None]
            output_batch.index_add_(0, tokens_indexes, weighted_expert_output_batch.to(output_batch.dtype))

        output_batch = output_batch.reshape(batch_size, sequence_length, hidden_features)
        return output_batch
