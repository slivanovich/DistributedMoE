from ctypes import Union
from threading import Thread
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple

from MoE.DistMoEBlock import DistMoEBlock, DistMoEBlockConfig
from MoE.DistExpertsBlock import DistExpertsBlock
from python.TensorTransferEngine.TensorTransferEngine import TensorTransferEngineConfig
from utils.Benchmark import Benchmark


@dataclass
class Qwen3MoEConfig:
    architectures: List[str]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    decoder_sparse_step: int
    eos_token_id: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    mlp_only_layers: List
    model_type: str
    moe_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    output_router_logits: bool
    rms_norm_eps: float
    rope_scaling: Dict | None
    rope_theta: float
    router_aux_loss_coef: float
    sliding_window: int | None  # ?
    tie_word_embeddings: bool
    torch_dtype: str
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    # For DistMoE enabling
    use_dist_moe: bool


class ApplyRotaryEmb:
    def __init__(
        self,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute

        self.apply_rotary_emb_flash_attn = None

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size (optional), seq_len, num_heads, head_size]
            cos: [seq_len, head_size // 2]
            sin: [seq_len, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style.
            enable_fp32_compute: Temporarily convert x, cos, sin to FP32 dtype
                                 for higher accuracy.
        """

        origin_dtype = x.dtype
        if enable_fp32_compute:
            x = x.float()

        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        if is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        if enable_fp32_compute:
            output = output.to(origin_dtype)
        return output


class RotaryEmbeddingBase(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        dtype: torch.dtype,
    ) -> None:
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self.apply_rotary_emb = ApplyRotaryEmb()

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""

        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""

        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class RotaryEmbedding(RotaryEmbeddingBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()

        super().__init__(head_size, rotary_dim, max_position_embeddings, base, dtype)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_static(query_rot, cos, sin)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = self.apply_rotary_emb.forward_static(
                key_rot,
                cos,
                sin,
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


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


def dist_moe_builder(
    dist_moe_config: DistMoEBlockConfig, dist_moe_tte_config: TensorTransferEngineConfig, eb_tte_config: TensorTransferEngineConfig
) -> Tuple[DistMoEBlock, List[DistExpertsBlock]]:
    expert_blocks: List[DistExpertsBlock] = []
    for experts_block_index in range(len(dist_moe_config.expert_blocks)):
        # TODO:
        expert_blocks.append(DistExpertsBlock(experts_block_index, dist_moe_config, [], eb_tte_config))

        t = Thread(target=expert_blocks[-1].main_loop, daemon=True)
        t.start()

    moe_block = DistMoEBlock(dist_moe_config, dist_moe_tte_config)

    return moe_block, expert_blocks


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


class Qwen3MoEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dtype: torch.dtype,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.dtype = dtype

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=qkv_bias, dtype=self.dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=qkv_bias, dtype=self.dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=qkv_bias, dtype=self.dtype)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps, dtype=self.dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps, dtype=self.dtype)

        self.o_proj = nn.Linear(self.total_num_heads * self.head_dim, hidden_size, bias=False, dtype=self.dtype)

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            dtype=self.dtype,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]

        q: torch.Tensor = self.q_proj(hidden_states)
        k: torch.Tensor = self.k_proj(hidden_states)
        v: torch.Tensor = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q_flat: torch.Tensor = q.view(batch_size * seq_len, self.num_heads, self.head_dim)
        k_flat: torch.Tensor = k.view(batch_size * seq_len, self.num_kv_heads, self.head_dim)

        q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)

        q = q_flat.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_flat.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        dist_moe_config: Optional[DistMoEBlockConfig],
        dist_moe_tte_config: Optional[TensorTransferEngineConfig],
        eb_tte_config: Optional[TensorTransferEngineConfig],
    ) -> None:
        super().__init__()

        self.dtype = torch.bfloat16
        if config.torch_dtype == "float16":
            self.dtype = torch.float16
        elif config.torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif config.torch_dtype == "float32":
            self.dtype = torch.float32
        elif config.torch_dtype == "int8":
            self.dtype = torch.int8

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen3MoEAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            dtype=self.dtype,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
        )

        if not config.use_dist_moe:
            self.mlp = Qwen3SparseMoEBlock(
                config.num_experts,
                config.num_experts_per_tok,
                self.hidden_size,
                config.intermediate_size,
                self.dtype,
            )
        else:
            assert dist_moe_config is not None
            assert dist_moe_tte_config is not None
            assert eb_tte_config is not None
            self.mlp, _ = dist_moe_builder(dist_moe_config, dist_moe_tte_config, eb_tte_config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=self.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states.to(self.dtype)
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states.to(self.dtype)
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        dist_moe_config: Optional[DistMoEBlockConfig],
        dist_moe_tte_config: Optional[TensorTransferEngineConfig],
        eb_tte_config: Optional[TensorTransferEngineConfig],
    ):
        super().__init__()

        # self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3MoeDecoderLayer(config, dist_moe_config, dist_moe_tte_config, eb_tte_config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        dist_moe_config: Optional[DistMoEBlockConfig],
        dist_moe_tte_config: Optional[TensorTransferEngineConfig],
        eb_tte_config: Optional[TensorTransferEngineConfig],
    ):
        super().__init__()

        self.config = config

        self.dtype = torch.bfloat16
        if config.torch_dtype == "float16":
            self.dtype = torch.float16
        elif config.torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif config.torch_dtype == "float32":
            self.dtype = torch.float32
        elif config.torch_dtype == "int8":
            self.dtype = torch.int8

        self.model = Qwen3MoeModel(self.config, dist_moe_config, dist_moe_tte_config, eb_tte_config)
        self.lm_head = nn.Linear(config.vocab_size, config.hidden_size, dtype=self.dtype)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits
