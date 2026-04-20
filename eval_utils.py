from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillation import freeze_module


class LayerAssemblyOperator(str, Enum):
    SUBSTITUTE = "sub"
    MERGE = "merge"
    CONCAT = "concat"
    ENSEMBLE = "ens"

    @classmethod
    def from_id(cls, operator_id: int) -> "LayerAssemblyOperator":
        values = list(cls)
        return values[int(operator_id) % len(values)]

    @classmethod
    def ids(cls) -> list[int]:
        return list(range(len(cls)))


def extract_hidden_tensor(value):
    if torch.is_tensor(value):
        return value

    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            raise ValueError("Received empty tuple/list when extracting hidden states.")
        return extract_hidden_tensor(value[0])

    if hasattr(value, "last_hidden_state"):
        return extract_hidden_tensor(value.last_hidden_state)

    if hasattr(value, "to_tuple"):
        value_tuple = value.to_tuple()
        if len(value_tuple) == 0:
            raise ValueError("Received empty ModelOutput when extracting hidden states.")
        return extract_hidden_tensor(value_tuple[0])

    raise TypeError(f"Unsupported hidden state container type: {type(value)}")


def normalize_layer_output(value):
    if torch.is_tensor(value):
        return value, ()

    if isinstance(value, tuple):
        if len(value) == 0:
            raise ValueError("Empty tuple returned by layer.")
        hidden = extract_hidden_tensor(value[0])
        return hidden, tuple(value[1:])

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Empty list returned by layer.")
        hidden = extract_hidden_tensor(value[0])
        return hidden, tuple(value[1:])

    if hasattr(value, "to_tuple"):
        value_tuple = value.to_tuple()
        if len(value_tuple) == 0:
            raise ValueError("Empty ModelOutput returned by layer.")
        hidden = extract_hidden_tensor(value_tuple[0])
        return hidden, tuple(value_tuple[1:])

    hidden = extract_hidden_tensor(value)
    return hidden, ()


def repack_layer_output(hidden: torch.Tensor, trailing: tuple, *, return_tuple: bool):
    hidden = extract_hidden_tensor(hidden)
    if return_tuple:
        return (hidden, *trailing)
    return hidden


@dataclass(frozen=True)
class ParticipantSpec:
    model_idx: int
    layer_idx: int
    active: bool = True


@dataclass(frozen=True)
class LayerGene:
    primary: ParticipantSpec
    secondary: Optional[ParticipantSpec]
    operator: LayerAssemblyOperator


@dataclass(frozen=True)
class Chromosome:
    vertical: tuple[ParticipantSpec, ...]
    horizontal: tuple[LayerGene, ...]
    beta: float


class ResidualGlueLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = extract_hidden_tensor(hidden_states)
        return hidden_states + self.fc2(self.act(self.fc1(hidden_states)))


class BranchAdapter(nn.Module):
    """
    Wrap a source decoder layer with:
      1) pre-adapter: current hidden size -> source hidden size
      2) source layer forward
      3) post-adapter: source hidden size -> target hidden size

    Also reconstruct rotary position embeddings when a single decoder layer
    is called outside the full LlamaModel forward path.
    """

    def __init__(
        self,
        layer: nn.Module,
        rotary_emb: Optional[nn.Module],
        input_dim: int,
        layer_hidden_dim: int,
        target_dim: int,
    ):
        super().__init__()
        self.layer = copy.deepcopy(layer)
        freeze_module(self.layer)

        self.rotary_emb = copy.deepcopy(rotary_emb) if rotary_emb is not None else None
        if self.rotary_emb is not None:
            freeze_module(self.rotary_emb)

        self.pre_adapter = (
            nn.Identity() if input_dim == layer_hidden_dim else nn.Linear(input_dim, layer_hidden_dim)
        )
        self.post_adapter = (
            nn.Identity() if layer_hidden_dim == target_dim else nn.Linear(layer_hidden_dim, target_dim)
        )

    def _build_position_ids(self, hidden_states: torch.Tensor, position_ids=None, cache_position=None):
        if position_ids is not None:
            return position_ids

        bsz, seq_len = hidden_states.shape[:2]
        device = hidden_states.device

        if cache_position is not None:
            if torch.is_tensor(cache_position):
                if cache_position.dim() == 1:
                    return cache_position.unsqueeze(0).expand(bsz, -1)
                return cache_position
            cache_position = torch.tensor(cache_position, device=device)
            if cache_position.dim() == 0:
                cache_position = cache_position.unsqueeze(0)
            if cache_position.dim() == 1:
                return cache_position.unsqueeze(0).expand(bsz, -1)
            return cache_position

        return torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

    def _build_position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids=None,
        position_embeddings=None,
        cache_position=None,
    ):
        if position_embeddings is not None:
            return position_embeddings

        if self.rotary_emb is None:
            return None

        built_position_ids = self._build_position_ids(
            hidden_states,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        try:
            return self.rotary_emb(hidden_states, built_position_ids)
        except TypeError:
            return self.rotary_emb(hidden_states, built_position_ids)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        hidden_states = extract_hidden_tensor(hidden_states)
        layer_input = self.pre_adapter(hidden_states)

        kwargs = dict(kwargs)
        position_embeddings = kwargs.pop("position_embeddings", None)
        cache_position = kwargs.get("cache_position", None)

        position_embeddings = self._build_position_embeddings(
            layer_input,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )

        outputs = self.layer(
            layer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden, trailing = normalize_layer_output(outputs)
        adapted = self.post_adapter(hidden)

        return_tuple = bool(kwargs.get("output_attentions", False) or kwargs.get("use_cache", False))
        return repack_layer_output(adapted, trailing, return_tuple=return_tuple)


class CompositeDecoderLayer(nn.Module):
    """
    A layer in the reassembled model.
    Each layer may contain one or two branches and combine them using one operator.
    """

    def __init__(
        self,
        branches: Sequence[BranchAdapter],
        operator: LayerAssemblyOperator,
        hidden_dim: int,
        primary_spec: ParticipantSpec,
        secondary_spec: Optional[ParticipantSpec],
    ):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.operator = operator
        self.hidden_dim = hidden_dim
        self.primary_spec = primary_spec
        self.secondary_spec = secondary_spec

        self.output_glue = ResidualGlueLayer(hidden_dim)
        self.branch_weights = nn.Parameter(torch.zeros(len(branches)))
        self.concat_fuser = nn.Linear(hidden_dim * max(1, len(branches)), hidden_dim)

    def _combine(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(states) == 1 or self.operator == LayerAssemblyOperator.SUBSTITUTE:
            return states[0]

        if self.operator == LayerAssemblyOperator.CONCAT:
            return self.concat_fuser(torch.cat(states, dim=-1))

        if self.operator == LayerAssemblyOperator.ENSEMBLE:
            return torch.stack(states, dim=0).mean(dim=0)

        weights = torch.softmax(self.branch_weights[: len(states)], dim=0)
        stacked = torch.stack(states, dim=0)
        return torch.sum(weights.view(-1, 1, 1, 1) * stacked, dim=0)

    def forward_without_glue(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        hidden_states = extract_hidden_tensor(hidden_states)

        branch_outputs = [
            branch(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            for branch in self.branches
        ]

        normalized_outputs = [normalize_layer_output(output) for output in branch_outputs]
        states = [hidden for hidden, _ in normalized_outputs]
        combined = self._combine(states)

        trailing = normalized_outputs[0][1] if normalized_outputs else ()
        return_tuple = bool(kwargs.get("output_attentions", False) or kwargs.get("use_cache", False))
        return repack_layer_output(combined, trailing, return_tuple=return_tuple)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        outputs = self.forward_without_glue(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden, trailing = normalize_layer_output(outputs)
        hidden = self.output_glue(hidden)

        return_tuple = bool(kwargs.get("output_attentions", False) or kwargs.get("use_cache", False))
        return repack_layer_output(hidden, trailing, return_tuple=return_tuple)


class PreGlueBlockView(nn.Module):
    """
    View of a composite block before its output glue layer.
    Used for distillation.
    """

    def __init__(self, layer: CompositeDecoderLayer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        return self.layer.forward_without_glue(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


def resolve_torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, None)


def extract_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Unsupported model structure: expected `model.layers`.")


def extract_rotary_embedding(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    return None


def infer_hidden_dim(layer: nn.Module) -> int:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
        return int(layer.self_attn.q_proj.in_features)
    if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
        return int(layer.input_layernorm.weight.shape[0])
    raise AttributeError("Unable to infer hidden dimension for the decoder layer.")


def load_source_models(model_paths: Sequence[str], dtype_name: str) -> tuple[list[nn.Module], AutoTokenizer]:
    models: list[nn.Module] = []
    torch_dtype = resolve_torch_dtype(dtype_name)

    for model_path in model_paths:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cpu",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load source model `{}`. Please pass valid local paths or "
                "HF model names through --model_paths / MOEA_MODEL_PATHS. "
                "Original error: {}".format(model_path, exc)
            ) from exc

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer from `{}`. Original error: {}".format(model_paths[0], exc)
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return models, tokenizer


def build_branch_adapter(
    source_models: Sequence[nn.Module],
    spec: ParticipantSpec,
    target_hidden: int,
) -> BranchAdapter:
    source_model = source_models[spec.model_idx]
    source_layer = extract_decoder_layers(source_model)[spec.layer_idx]
    source_hidden = infer_hidden_dim(source_layer)
    rotary_emb = extract_rotary_embedding(source_model)

    return BranchAdapter(
        layer=source_layer,
        rotary_emb=rotary_emb,
        input_dim=target_hidden,
        layer_hidden_dim=source_hidden,
        target_dim=target_hidden,
    )


def build_composite_layers(
    chromosome: Chromosome,
    source_models: Sequence[nn.Module],
    target_hidden: int,
) -> list[CompositeDecoderLayer]:
    layers: list[CompositeDecoderLayer] = []

    for gene in chromosome.horizontal:
        branches = [build_branch_adapter(source_models, gene.primary, target_hidden)]
        if gene.secondary is not None and gene.secondary.active:
            branches.append(build_branch_adapter(source_models, gene.secondary, target_hidden))

        operator = gene.operator if len(branches) > 1 else LayerAssemblyOperator.SUBSTITUTE

        layers.append(
            CompositeDecoderLayer(
                branches=branches,
                operator=operator,
                hidden_dim=target_hidden,
                primary_spec=gene.primary,
                secondary_spec=gene.secondary,
            )
        )

    return layers


def build_reassembled_model(
    chromosome: Chromosome,
    source_models: Sequence[nn.Module],
    base_model_path: str,
    dtype_name: str,
) -> nn.Module:
    torch_dtype = resolve_torch_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    target_hidden = int(model.config.hidden_size)
    model.model.layers = nn.ModuleList(build_composite_layers(chromosome, source_models, target_hidden))
    model.config.num_hidden_layers = len(model.model.layers)
    model.eval()
    return model
