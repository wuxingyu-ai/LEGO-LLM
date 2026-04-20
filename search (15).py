from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

DEFAULT_MODEL_PATHS = [
    "model A",
    "model B",
    "model C",
]


def _split_model_path_string(value: str) -> list[str]:
    normalized = value.replace(os.pathsep, ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def resolve_default_model_paths() -> list[str]:
    env_value = os.getenv("MOEA_MODEL_PATHS")
    if env_value:
        env_paths = _split_model_path_string(env_value)
        if env_paths:
            return env_paths
    return list(DEFAULT_MODEL_PATHS)


class ArgsNamespace(argparse.Namespace):
    tasks: str


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def args_parse() -> ArgsNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_paths",
        nargs="+",
        default=None,
        help=(
            "Paths or HF names of source models. If omitted, the script checks "
            "MOEA_MODEL_PATHS first and then falls back to DEFAULT_MODEL_PATHS."
        ),
    )
    parser.add_argument("--new_model_layers", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="./Output")

    parser.add_argument("--population_size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=50)

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--total_samples", type=int, default=1000)
    parser.add_argument("--distillation_batch_size", type=int, default=32)
    parser.add_argument("--distillation_epochs", type=int, default=3)
    parser.add_argument("--distillation_lr", type=float, default=1e-3)
    parser.add_argument("--distillation_alpha_context", type=float, default=0.5)
    parser.add_argument("--enable_distillation", type=str2bool, default=True)

    parser.add_argument("--beta_candidates", type=int, default=5)

    parser.add_argument("--tasks", type=str, default="xnli")
    parser.add_argument("--metric", type=str, default="acc,none")
    parser.add_argument("--limit", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--quiet", action="store_true", help="Disable runtime progress logs.")
    parser.add_argument("--csv_log_path", type=str, default=None, help="Path to realtime CSV log file.")

    args = parser.parse_args(namespace=ArgsNamespace())

    if not args.model_paths:
        args.model_paths = resolve_default_model_paths()
        print(
            "[moea_llm_merge_single_level] --model_paths not provided; using default model set: "
            + ", ".join(args.model_paths)
        )

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
