from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import lm_eval
import lm_eval.models.huggingface
import numpy as np
import torch
import torch.nn as nn

from semantic_entropy import compute_entropy

from .assembly import Chromosome, build_reassembled_model
from .distillation_runner import train_glue_layers
from .logging_utils import log_event, maybe_log


def evaluate_lm_results(results: dict, tasks: Sequence[str], metric: str) -> tuple[float, float]:
    scores = []
    fairness_values = []

    for task in tasks:
        task_results = results.get("results", {})
        if task in task_results and metric in task_results[task]:
            scores.append(float(task_results[task][metric]))
        fairness_values.append(float(compute_entropy(results, task, metric)))

    mean_score = float(np.mean(scores)) if scores else 0.0
    mean_fairness = float(np.mean(fairness_values)) if fairness_values else 0.0
    return mean_score, mean_fairness


def evaluate_chromosome(
    chromosome: Chromosome,
    source_models: Sequence[nn.Module],
    tokenizer,
    args,
    cache: Dict[Chromosome, tuple[float, float]],
) -> tuple[float, float]:
    if chromosome in cache:
        cached_score, cached_fairness = cache[chromosome]
        log_event(
            args,
            stage="evaluate",
            event="cache_hit",
            score=f"{cached_score:.6f}",
            fairness=f"{cached_fairness:.6f}",
            beta=f"{chromosome.beta:.6f}",
            num_layers=len(chromosome.vertical),
        )
        return cache[chromosome]

    model = build_reassembled_model(chromosome, source_models, args.model_paths[0], args.dtype)

    if args.enable_distillation:
        maybe_log(args, "[Evaluate] Distillation enabled for candidate evaluation.")
        train_glue_layers(chromosome, model, source_models, args)

    try:
        evaluator_model = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        results = lm_eval.evaluator.simple_evaluate(
            model=evaluator_model,
            tasks=tasks,
            num_fewshot=0,
            limit=args.limit,
            batch_size=args.batch_size,
            verbosity="WARNING",
        )

        score, fairness = evaluate_lm_results(results, tasks, args.metric)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache[chromosome] = (score, fairness)

    log_event(
        args,
        stage="evaluate",
        event="candidate_evaluated",
        score=f"{score:.6f}",
        fairness=f"{fairness:.6f}",
        beta=f"{chromosome.beta:.6f}",
        num_layers=len(chromosome.vertical),
    )

    return score, fairness


def save_final_outputs(
    chromosome: Chromosome,
    source_models: Sequence[nn.Module],
    tokenizer,
    args,
) -> None:
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_reassembled_model(chromosome, source_models, args.model_paths[0], args.dtype)
    if args.enable_distillation:
        maybe_log(args, "[Save] Distillation enabled for final best model.")
        train_glue_layers(chromosome, model, source_models, args)

    torch.save({"state_dict": model.state_dict()}, save_dir / "reassembled_model_state.pt")
    tokenizer.save_pretrained(save_dir)

    with open(save_dir / "chromosome.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "beta": chromosome.beta,
                "base_model_path": args.model_paths[0],
                "layers": [
                    {
                        "primary": gene.primary.__dict__,
                        "secondary": None if gene.secondary is None else gene.secondary.__dict__,
                        "operator": gene.operator.value,
                    }
                    for gene in chromosome.horizontal
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log_event(
        args,
        stage="save",
        event="final_model_saved",
        beta=f"{chromosome.beta:.6f}",
        num_layers=len(chromosome.vertical),
        note=f"saved to {save_dir}",
    )
