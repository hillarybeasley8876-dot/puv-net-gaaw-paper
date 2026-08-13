"""Audit the archived 200-sample slice against the training split.

The training pipeline creates the validation subset with
``default_rng(12345).permutation``.  Older diagnostic scripts instead sampled
from the final 5% of array indices.  This script reconstructs the training
split, identifies the overlap, and summarizes the archived rows that really
belong to the held-out validation subset.  It never reruns a model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MEASURE = ROOT / "docs" / "_cv_nn_measure.json"
DIAG = ROOT / "docs" / "_ch3_diag.json"
BASE_CONFIG = ROOT / "configs" / "b002_baseline150.yaml"
TRAIN_SCRIPT = ROOT / "scripts" / "train_pu.py"
OUT = ROOT / "docs" / "_split_audit_and_holdout_subset.json"


def extract_split_parameters() -> tuple[int, float, int]:
    diag = json.loads(DIAG.read_text(encoding="utf-8"))
    n_all = int(diag["source"]["val_range"][1])

    cfg = BASE_CONFIG.read_text(encoding="utf-8")
    ratio_match = re.search(r"^\s*val_ratio:\s*([0-9.]+)\s*$", cfg, re.MULTILINE)
    if not ratio_match:
        raise RuntimeError("val_ratio not found in baseline config")
    val_ratio = float(ratio_match.group(1))

    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    seed_match = re.search(r"default_rng\((\d+)\)\.permutation\(n_all\)", source)
    if not seed_match:
        raise RuntimeError("split seed not found in train_pu.py")
    split_seed = int(seed_match.group(1))
    return n_all, val_ratio, split_seed


def summarize(rows: list[dict]) -> dict:
    result: dict[str, object] = {"n": len(rows)}
    keys = {
        "cv_nn": "nn_pred_cv",
        "cv_nn_gt": "nn_gt_cv",
        "cd": "cd",
        "hd": "hd",
        "bwd_share": "bwd_share",
    }
    for out_key, row_key in keys.items():
        values = np.asarray([float(row[row_key]) for row in rows], dtype=float)
        result[out_key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sample_se": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None,
        }

    strata = np.asarray([row["bwd_by_sparsity"] for row in rows], dtype=float)
    means = strata.mean(axis=0)
    result["bwd_by_sparsity_mean"] = [float(x) for x in means]
    result["q4_over_q1"] = float(means[3] / means[0])
    return result


def relative_change(control: dict, experiment: dict, key: str) -> float:
    a = float(control[key]["mean"])
    b = float(experiment[key]["mean"])
    return 100.0 * (b - a) / a


def main() -> int:
    data = json.loads(MEASURE.read_text(encoding="utf-8"))
    n_all, val_ratio, split_seed = extract_split_parameters()
    n_val = max(1, int(n_all * val_ratio))
    permutation = np.random.default_rng(split_seed).permutation(n_all)
    val_indices = set(int(x) for x in permutation[:n_val])

    run_indices = {
        name: [int(row["idx"]) for row in rec["per_sample"]]
        for name, rec in data["runs"].items()
    }
    first_indices = next(iter(run_indices.values()))
    if any(indices != first_indices for indices in run_indices.values()):
        raise RuntimeError("archived runs do not share identical sample indices")

    heldout_indices = sorted(idx for idx in first_indices if idx in val_indices)
    holdin_indices = sorted(idx for idx in first_indices if idx not in val_indices)
    summaries: dict[str, dict] = {}
    for name, rec in data["runs"].items():
        rows = [row for row in rec["per_sample"] if int(row["idx"]) in val_indices]
        summaries[name] = summarize(rows)
        summaries[name]["architecture_reconstruction_valid"] = name != "ABL_D1_scale_qk"
        if name == "ABL_D1_scale_qk":
            summaries[name]["invalid_reason"] = (
                "measure_cv_nn.py instantiated the default scale_qk=False model "
                "instead of the run configuration scale_qk=True"
            )

    comparisons = {}
    pairs = {
        "B0_to_B1": ("B002_baseline150_5090", "ABL_B1_adv_fixed"),
        "B1_to_B2": ("ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"),
        "B0_to_B2": ("B002_baseline150_5090", "ABL_B2_adv_adaptive"),
        "C0_to_C1": ("B002_baseline150", "ABL_C1_uniform"),
    }
    for label, (control_name, experiment_name) in pairs.items():
        control = summaries[control_name]
        experiment = summaries[experiment_name]
        comparisons[label] = {
            "control": control_name,
            "experiment": experiment_name,
            "relative_change_pct": {
                key: relative_change(control, experiment, key)
                for key in ("cv_nn", "cd", "hd")
            },
            "q4_over_q1_change_pct": 100.0 * (
                float(experiment["q4_over_q1"]) - float(control["q4_over_q1"])
            ) / float(control["q4_over_q1"]),
        }

    b1_rows = {
        int(row["idx"]): row
        for row in data["runs"]["ABL_B1_adv_fixed"]["per_sample"]
        if int(row["idx"]) in val_indices
    }
    b2_rows = {
        int(row["idx"]): row
        for row in data["runs"]["ABL_B2_adv_adaptive"]["per_sample"]
        if int(row["idx"]) in val_indices
    }
    paired = {}
    for out_key, row_key in {"cv_nn": "nn_pred_cv", "cd": "cd", "hd": "hd"}.items():
        diff = np.asarray([
            float(b2_rows[idx][row_key]) - float(b1_rows[idx][row_key])
            for idx in heldout_indices
        ])
        paired[out_key] = {
            "n": len(diff),
            "mean_difference": float(diff.mean()),
            "fraction_improved": float(np.mean(diff < 0)),
        }

    report = {
        "source": str(MEASURE.relative_to(ROOT)).replace("\\", "/"),
        "split_definition": {
            "n_all": n_all,
            "val_ratio": val_ratio,
            "n_validation": n_val,
            "split_seed": split_seed,
            "source_code": "scripts/train_pu.py",
        },
        "archived_slice": {
            "n": len(first_indices),
            "heldout_validation_n": len(heldout_indices),
            "training_holdin_n": len(holdin_indices),
            "heldout_fraction": len(heldout_indices) / len(first_indices),
            "heldout_indices": heldout_indices,
            "warning": (
                "The archived 200-row slice is dominated by training indices and must not "
                "be called a validation sample."
            ),
        },
        "heldout_subset": {
            "scope_note": (
                "Post-hoc intersection of the archived slice with the true validation split; "
                "n=11 is descriptive only and cannot support strong generalization claims."
            ),
            "runs": summaries,
            "comparisons": comparisons,
            "paired_B1_to_B2": paired,
        },
        "protocol_mismatch": {
            "measurement": "shared ground-truth centering and scale",
            "paper_previous_label": "independent prediction/ground-truth normalization",
            "required_label": "shared-ground-truth-frame diagnostic CD/HD",
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Archived slice: held-out={len(heldout_indices)}, hold-in={len(holdin_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
