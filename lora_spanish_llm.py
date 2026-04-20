from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    for buf in module.buffers():
        return buf.dtype
    return torch.float32


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


@dataclass
class DistillationConfig:
    seq_length: int = 128
    total_samples: int = 1000
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-3
    beta: float = 0.5
    alpha_context: float = 0.5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    noise_std: float = 1.0
    device: Optional[str] = None


class DualGlueDistiller:
    """
    Approximate implementation of data-free glue-layer alignment.

    Student path:
        x -> prev_block -> glue_layer -> next_block

    Teacher-style supervision:
        1) next_source_blocks          : anchor student-next behavior
        2) current_successor_blocks    : context anchor for inter-layer alignment
        3) next_source_prev_blocks     : expected interface anchor for cross-layer alignment
    """

    def __init__(self, config: DistillationConfig):
        self.config = config

    def _resolve_device(self, glue_layer: nn.Module) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return _module_device(glue_layer)

    @staticmethod
    def _make_position_ids(batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _forward_hidden(
        module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = module(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden, _ = normalize_layer_output(outputs)
        return hidden

    def _aggregate_teacher_outputs(
        self,
        blocks: Sequence[nn.Module],
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if not blocks:
            return None

        outputs = []
        for block in blocks:
            out = self._forward_hidden(
                block,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            outputs.append(out)

        if len(outputs) == 1:
            return outputs[0]
        return torch.stack(outputs, dim=0).mean(dim=0)

    def fit_transition(
        self,
        glue_layer: nn.Module,
        prev_block: nn.Module,
        next_block: nn.Module,
        next_source_blocks: Sequence[nn.Module],
        next_source_prev_blocks: Sequence[nn.Module],
        current_successor_blocks: Sequence[nn.Module],
        hidden_dim: int,
    ) -> Tuple[nn.Module, List[float]]:
        """
        Optimize glue_layer only. Other modules are frozen.

        Returns:
            trained glue_layer, history(list of average epoch losses)
        """
        beta = float(self.config.beta)
        alpha_context = float(self.config.alpha_context)

        device = self._resolve_device(glue_layer)

        modules = [glue_layer, prev_block, next_block]
        modules.extend(list(next_source_blocks))
        modules.extend(list(next_source_prev_blocks))
        modules.extend(list(current_successor_blocks))

        # Capture original device and training state to restore later
        original_devices = {id(m): _module_device(m) for m in modules}
        original_training = {id(m): m.training for m in modules}

        for module in modules:
            module.to(device)

        # Freeze all non-glue modules
        freeze_module(prev_block)
        freeze_module(next_block)
        for module in next_source_blocks:
            freeze_module(module)
        for module in next_source_prev_blocks:
            freeze_module(module)
        for module in current_successor_blocks:
            freeze_module(module)

        glue_layer.train()
        for param in glue_layer.parameters():
            param.requires_grad = True

        dtype = _module_dtype(glue_layer)
        optimizer = torch.optim.Adam(
            glue_layer.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        num_batches = math.ceil(self.config.total_samples / self.config.batch_size)
        history: List[float] = []

        try:
            for _epoch in range(self.config.epochs):
                running_loss = 0.0
                seen = 0

                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.config.batch_size
                    batch_size = min(self.config.batch_size, self.config.total_samples - batch_start)
                    if batch_size <= 0:
                        continue

                    optimizer.zero_grad(set_to_none=True)

                    x = (
                        torch.randn(
                            batch_size,
                            self.config.seq_length,
                            hidden_dim,
                            device=device,
                            dtype=dtype,
                        )
                        * self.config.noise_std
                    )
                    position_ids = self._make_position_ids(batch_size, self.config.seq_length, device)

                    # Current block output before glue
                    with torch.no_grad():
                        prev_hidden = self._forward_hidden(
                            prev_block,
                            x,
                            attention_mask=None,
                            position_ids=position_ids,
                        ).detach()

                    # Student path
                    glued_hidden = glue_layer(prev_hidden)
                    student_next_hidden = self._forward_hidden(
                        next_block,
                        glued_hidden,
                        attention_mask=None,
                        position_ids=position_ids,
                    )

                    # Teacher anchors
                    with torch.no_grad():
                        teacher_next_hidden = self._aggregate_teacher_outputs(
                            next_source_blocks,
                            glued_hidden,
                            attention_mask=None,
                            position_ids=position_ids,
                        )
                        teacher_prev_interface = self._aggregate_teacher_outputs(
                            next_source_prev_blocks,
                            x,
                            attention_mask=None,
                            position_ids=position_ids,
                        )
                        teacher_context_hidden = self._aggregate_teacher_outputs(
                            current_successor_blocks,
                            prev_hidden,
                            attention_mask=None,
                            position_ids=position_ids,
                        )

                    # L_inter
                    inter_main = (
                        F.mse_loss(student_next_hidden, teacher_next_hidden)
                        if teacher_next_hidden is not None
                        else torch.zeros((), device=device, dtype=dtype)
                    )
                    inter_context = (
                        F.mse_loss(student_next_hidden, teacher_context_hidden)
                        if teacher_context_hidden is not None
                        else torch.zeros((), device=device, dtype=dtype)
                    )
                    l_inter = inter_main + alpha_context * inter_context

                    # L_cross
                    l_cross = (
                        F.mse_loss(glued_hidden, teacher_prev_interface)
                        if teacher_prev_interface is not None
                        else torch.zeros((), device=device, dtype=dtype)
                    )

                    loss = beta * l_inter + (1.0 - beta) * l_cross
                    loss.backward()

                    if self.config.grad_clip is not None and self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(glue_layer.parameters(), self.config.grad_clip)

                    optimizer.step()

                    running_loss += float(loss.detach().item()) * batch_size
                    seen += batch_size

                history.append(running_loss / max(seen, 1))

        finally:
            # Restore training states
            for module in modules:
                module.train(original_training[id(module)])

            # Move all modules back to their original device
            for module in modules:
                module.to(original_devices[id(module)])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return glue_layer, history