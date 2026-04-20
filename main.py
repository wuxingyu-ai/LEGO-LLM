from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from distillation import DistillationConfig, DualGlueDistiller

from .assembly import (
    Chromosome,
    CompositeDecoderLayer,
    ParticipantSpec,
    PreGlueBlockView,
    build_branch_adapter,
    extract_decoder_layers,
)
from .logging_utils import log_event, maybe_log


def collect_current_successor_blocks(
    current_idx: int,
    chromosome: Chromosome,
    source_models: Sequence[nn.Module],
    hidden_dim: int,
) -> list[nn.Module]:
    blocks = []
    gene = chromosome.horizontal[current_idx]
    participants = [gene.primary]
    if gene.secondary is not None and gene.secondary.active:
        participants.append(gene.secondary)

    for p in participants:
        source_layers = extract_decoder_layers(source_models[p.model_idx])
        if p.layer_idx + 1 < len(source_layers):
            blocks.append(
                build_branch_adapter(
                    source_models,
                    ParticipantSpec(p.model_idx, p.layer_idx + 1, True),
                    hidden_dim,
                )
            )
    return blocks


def collect_next_source_blocks(
    next_idx: int,
    chromosome: Chromosome,
    source_models: Sequence[nn.Module],
    hidden_dim: int,
) -> tuple[list[nn.Module], list[nn.Module]]:
    next_blocks = []
    next_prev_blocks = []

    gene = chromosome.horizontal[next_idx]
    participants = [gene.primary]
    if gene.secondary is not None and gene.secondary.active:
        participants.append(gene.secondary)

    for p in participants:
        next_blocks.append(build_branch_adapter(source_models, p, hidden_dim))
        if p.layer_idx > 0:
            next_prev_blocks.append(
                build_branch_adapter(
                    source_models,
                    ParticipantSpec(p.model_idx, p.layer_idx - 1, True),
                    hidden_dim,
                )
            )

    return next_blocks, next_prev_blocks


def train_glue_layers(
    chromosome: Chromosome,
    reassembled_model: nn.Module,
    source_models: Sequence[nn.Module],
    args,
) -> None:
    maybe_log(args, "[Distillation] Start glue-layer distillation...")
    maybe_log(args, f"[Distillation] beta = {chromosome.beta:.4f}")
    maybe_log(args, f"[Distillation] total layers = {len(reassembled_model.model.layers)}")

    log_event(
        args,
        stage="distill",
        event="start",
        beta=f"{chromosome.beta:.6f}",
        num_layers=len(reassembled_model.model.layers),
    )

    distiller = DualGlueDistiller(
        DistillationConfig(
            seq_length=args.seq_len,
            total_samples=args.total_samples,
            batch_size=args.distillation_batch_size,
            epochs=args.distillation_epochs,
            learning_rate=args.distillation_lr,
            beta=chromosome.beta,
            alpha_context=args.distillation_alpha_context,
        )
    )

    layers = list(reassembled_model.model.layers)
    hidden_dim = int(reassembled_model.config.hidden_size)

    if len(layers) <= 1:
        maybe_log(args, "[Distillation] Skip: model has <= 1 layer, no glue layer to train.")
        log_event(
            args,
            stage="distill",
            event="skip",
            beta=f"{chromosome.beta:.6f}",
            num_layers=len(layers),
            note="model has <= 1 layer",
        )
        return

    for idx in range(len(layers) - 1):
        current_layer = layers[idx]
        next_layer = layers[idx + 1]

        assert isinstance(current_layer, CompositeDecoderLayer)
        assert isinstance(next_layer, CompositeDecoderLayer)

        maybe_log(args, f"[Distillation] Training glue layer between layer {idx} -> {idx + 1} ...")
        log_event(
            args,
            stage="distill",
            event="layer_start",
            beta=f"{chromosome.beta:.6f}",
            num_layers=len(layers),
            layer_idx=f"{idx}->{idx + 1}",
        )

        next_source_blocks, next_source_prev_blocks = collect_next_source_blocks(
            idx + 1,
            chromosome,
            source_models,
            hidden_dim,
        )
        current_successor_blocks = collect_current_successor_blocks(
            idx,
            chromosome,
            source_models,
            hidden_dim,
        )

        trained_glue, history = distiller.fit_transition(
            glue_layer=current_layer.output_glue,
            prev_block=PreGlueBlockView(current_layer),
            next_block=PreGlueBlockView(next_layer),
            next_source_blocks=next_source_blocks,
            next_source_prev_blocks=next_source_prev_blocks,
            current_successor_blocks=current_successor_blocks,
            hidden_dim=hidden_dim,
        )

        current_layer.output_glue = trained_glue

        if history:
            maybe_log(
                args,
                f"[Distillation] Finished layer {idx} -> {idx + 1}, final loss = {history[-1]:.6f}",
            )
            log_event(
                args,
                stage="distill",
                event="layer_end",
                beta=f"{chromosome.beta:.6f}",
                num_layers=len(layers),
                layer_idx=f"{idx}->{idx + 1}",
                final_loss=f"{history[-1]:.6f}",
            )
        else:
            maybe_log(args, f"[Distillation] Finished layer {idx} -> {idx + 1}, no history returned.")
            log_event(
                args,
                stage="distill",
                event="layer_end",
                beta=f"{chromosome.beta:.6f}",
                num_layers=len(layers),
                layer_idx=f"{idx}->{idx + 1}",
                note="no history returned",
            )

    maybe_log(args, "[Distillation] All glue layers finished.")
    log_event(
        args,
        stage="distill",
        event="end",
        beta=f"{chromosome.beta:.6f}",
        num_layers=len(layers),
    )
