from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path


def maybe_log(args, message: str) -> None:
    if not getattr(args, "quiet", False):
        print(message, flush=True)


class CsvLogger:
    """
    Realtime CSV logger.
    Each write is flushed to disk so you can inspect progress while the run is ongoing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.columns = [
            "timestamp",
            "stage",
            "event",
            "outer_gen",
            "inner_gen",
            "score",
            "fairness",
            "inner_best_score",
            "beta",
            "num_layers",
            "layer_idx",
            "final_loss",
            "note",
        ]

        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.fp = open(self.path, "a", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.fp, fieldnames=self.columns)

        if not file_exists:
            self.writer.writeheader()
            self.fp.flush()
            os.fsync(self.fp.fileno())

    def write(self, **kwargs):
        row = {key: "" for key in self.columns}
        row.update(kwargs)
        self.writer.writerow(row)
        self.fp.flush()
        os.fsync(self.fp.fileno())

    def close(self):
        try:
            self.fp.flush()
            os.fsync(self.fp.fileno())
        except Exception:
            pass
        try:
            self.fp.close()
        except Exception:
            pass


def setup_csv_logger(args) -> None:
    csv_path = getattr(args, "csv_log_path", None)
    if not csv_path:
        csv_path = str(Path(args.save_path) / "progress_log.csv")
    args.csv_log_path = csv_path
    args.csv_logger = CsvLogger(csv_path)


def log_event(
    args,
    *,
    stage: str,
    event: str,
    outer_gen="",
    inner_gen="",
    score="",
    fairness="",
    inner_best_score="",
    beta="",
    num_layers="",
    layer_idx="",
    final_loss="",
    note="",
):
    logger = getattr(args, "csv_logger", None)
    if logger is None:
        return

    logger.write(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        stage=stage,
        event=event,
        outer_gen=outer_gen,
        inner_gen=inner_gen,
        score=score,
        fairness=fairness,
        inner_best_score=inner_best_score,
        beta=beta,
        num_layers=num_layers,
        layer_idx=layer_idx,
        final_loss=final_loss,
        note=note,
    )
