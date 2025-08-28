#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train one-vs-rest L2 logistic probes *per layer* using ONLY an NPZ file.

Expected NPZ contents (any of these forms):
- Per-layer 2D arrays keyed "0","1",... (each (N, D))  <-- common case
- One 3D array 'X' with shape (N, L, D) or (L, N, D)
- 'X_layers' object/list of length L with each (N, D)
- Multiple 2D arrays named like layer_0, resid_17, etc.

Required metadata in NPZ:
- labels_text : (N,)       - target labels (strings)
Recommended:
- dialog_id   : (N,)       - group IDs for group-aware split
- turn_idx    : (N,)       - turn index (enables “inertia” metric)

Outputs (per layer) in --out_dir:
- probe_layer{idx}.joblib             (dict: scaler, classifier, label_encoder)
- metrics_layer{idx}.json
- confusion_layer{idx}.npy / .png
- predictions_layer{idx}.csv
- predictions_layer{idx}_extra.json   (debug indices/groups/turns)
"""

import argparse
import csv
import json
import os
import re
import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------- I/O utilities ----------------------------- #

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_layers_only_npz(path: str) -> List[np.ndarray]:
    """
    Robust loader for per-layer features from an NPZ.

    Accepts:
      - Keys "0","1",... each (N,D)
      - 'X' 3D with (N,L,D) or (L,N,D)
      - 'X_layers' object/list of (N,D)
      - Any single 3D array under some key
      - Many 2D arrays named like layer_0/resid_12/etc.
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    def is_2d(x): return isinstance(x, np.ndarray) and x.ndim == 2
    def is_3d(x): return isinstance(x, np.ndarray) and x.ndim == 3

    # Common case: numeric keys "0","1",...
    numeric_2d = [(k, npz[k]) for k in keys if k.isdigit() and is_2d(npz[k])]
    if numeric_2d:
        numeric_2d.sort(key=lambda kv: int(kv[0]))
        layers = [arr for _, arr in numeric_2d]
        _check_same_N(layers)
        return layers

    # 3D 'X'
    if "X" in keys and is_3d(npz["X"]):
        X = npz["X"]
        # Heuristic: fix (L,N,D) -> (N,L,D)
        if X.shape[0] <= 256 and X.shape[1] > X.shape[0]:
            X = np.transpose(X, (1, 0, 2))
        return [X[:, l, :] for l in range(X.shape[1])]

    # 'X_layers'
    if "X_layers" in keys:
        arr = npz["X_layers"]
        layers = list(arr) if isinstance(arr, (list, tuple)) else list(arr)
        _check_layers_2d(layers)
        _check_same_N(layers)
        return layers

    # Any single 3D array
    three_d_keys = [k for k in keys if is_3d(npz[k])]
    if len(three_d_keys) == 1:
        X = npz[three_d_keys[0]]
        if X.shape[0] <= 256 and X.shape[1] > X.shape[0]:
            X = np.transpose(X, (1, 0, 2))
        return [X[:, l, :] for l in range(X.shape[1])]

    # Many 2D arrays with varied names
    candidate_2d = [(k, npz[k]) for k in keys if is_2d(npz[k])]
    if candidate_2d:
        def key_num(k):
            m = re.search(r'(\d+)$', k) or re.search(r'_(\d+)$', k)
            return int(m.group(1)) if m else 10**9
        candidate_2d.sort(key=lambda kv: (key_num(kv[0]), kv[0]))
        layers = [arr for _, arr in candidate_2d]
        _check_same_N(layers)
        return layers

    raise ValueError("Could not find per-layer features in NPZ.")

def _check_layers_2d(layers: List[np.ndarray]):
    for i, Li in enumerate(layers):
        if not isinstance(Li, np.ndarray) or Li.ndim != 2:
            raise ValueError(f"Layer {i} is not 2D (N,D). Got shape {getattr(Li, 'shape', None)}")

def _check_same_N(layers: List[np.ndarray]):
    N = layers[0].shape[0]
    for i, Li in enumerate(layers):
        if Li.shape[0] != N:
            raise ValueError(f"Layer {i} has N={Li.shape[0]} but layer 0 has N={N}.")

def load_npz_labels_groups_turns(path: str, labels_key: str = "labels_text"):
    npz = np.load(path, allow_pickle=True)
    if labels_key not in npz.files:
        raise ValueError(
            f"Labels array '{labels_key}' not found in NPZ. "
            f"Available keys: {list(npz.files)}"
        )
    y = np.array(npz[labels_key], dtype=object)
    groups = np.array(npz["dialog_id"], dtype=object) if "dialog_id" in npz.files else None
    turns = np.array(npz["turn_idx"], dtype=int) if "turn_idx" in npz.files else None
    return y, groups, turns


# ------------------------------ Metrics & plots ------------------------------ #

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, float]:
    accs = {}
    for idx, c in enumerate(classes):
        mask = (y_true == idx)
        denom = mask.sum()
        accs[c] = float((y_pred[mask] == idx).sum() / denom) if denom > 0 else float("nan")
    return accs

def plot_and_save_confusion(cm: np.ndarray, classes: List[str], out_png: str, normalize: bool = True):
    fig = plt.figure(figsize=(max(6, 0.6*len(classes)), max(5, 0.6*len(classes))))
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_disp = cm / np.maximum(cm_sum, 1)
    else:
        cm_disp = cm
    im = plt.imshow(cm_disp, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    thresh = (float(np.nanmax(cm_disp)) + float(np.nanmin(cm_disp))) / 2.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            val = cm_disp[i, j]
            txt = f"{val:.2f}" if normalize else str(int(val))
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def compute_inertia_metrics_if_possible(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    groups_test: Optional[np.ndarray],
    turns_test: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    'Inertia' = after a ground-truth label change within a conversation,
    how often does the model predict the previous label at the next point?
    """
    if groups_test is None or turns_test is None:
        return {}
    # order by (group, turn)
    order = np.lexsort((turns_test, groups_test))
    y_true = y_true_test[order]
    y_pred = y_pred_test[order]
    g = groups_test[order]

    total_change_pts = 0
    inertia_errors = 0
    for i in range(1, len(y_true)):
        if g[i] == g[i-1] and y_true[i] != y_true[i-1]:
            total_change_pts += 1
            if y_pred[i] == y_true[i-1]:
                inertia_errors += 1
    if total_change_pts == 0:
        return {"inertia_error_rate": float("nan"), "change_points": 0}
    return {
        "inertia_error_rate": inertia_errors / total_change_pts,
        "change_points": int(total_change_pts)
    }


# ------------------------------ Train one layer ------------------------------ #

def group_aware_split_indices(y_enc: np.ndarray, groups: Optional[np.ndarray], test_size: float, seed: int):
    """
    If groups are provided, try up to 50 random seeds to ensure that
    *all* classes appear in the training fold. Fall back to the last split
    if that isn't possible.
    """
    idx = np.arange(len(y_enc))
    if groups is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(idx, y_enc))
        return train_idx, test_idx

    rng = np.random.RandomState(seed)
    fallback = None
    for _ in range(50):
        s = rng.randint(1_000_000)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        tr, te = next(splitter.split(idx, y_enc, groups))
        if set(np.unique(y_enc[tr])) == set(np.unique(y_enc)):
            return tr, te
        fallback = (tr, te)
    # fallback: warn if some classes missing from train
    if fallback is not None:
        tr, te = fallback
        missing = set(np.unique(y_enc)) - set(np.unique(y_enc[tr]))
        if missing:
            print(f"[warn] Train fold missing classes (fallback used): {sorted(map(int, missing))}")
        return tr, te
    raise RuntimeError("Failed to produce a valid group-aware split.")

def train_layer_probe(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    groups: Optional[np.ndarray],
    turns: Optional[np.ndarray],
    args
) -> Dict[str, object]:
    y_enc = label_encoder.transform(y)

    train_idx, test_idx = group_aware_split_indices(y_enc, groups, args.test_size, args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    groups_test = groups[test_idx] if groups is not None else None
    turns_test  = turns[test_idx] if turns is not None else None

    scaler = StandardScaler()
    base_lr = LogisticRegression(
        penalty="l2",
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        multi_class="ovr",
        class_weight=(None if args.class_weight == "none" else "balanced"),
        fit_intercept=True,
        random_state=args.seed,  # reproducible
    )
    clf = OneVsRestClassifier(base_lr, n_jobs=args.n_jobs)

    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    # Probabilities
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test_scaled)
    else:
        scores = clf.decision_function(X_test_scaled)
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        y_proba = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-9, None)

    classes_str = list(label_encoder.classes_)
    overall_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes_str)))
    per_class_acc = per_class_accuracy(y_test, y_pred, classes_str)
    inertia = compute_inertia_metrics_if_possible(y_test, y_pred, groups_test, turns_test)

    report = classification_report(
        y_test, y_pred, labels=np.arange(len(classes_str)), target_names=classes_str, output_dict=True, zero_division=0
    )

    metrics = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        **inertia
    }

    pipeline = {"scaler": scaler, "classifier": clf, "label_encoder": label_encoder}

    top1_prob = y_proba.max(axis=1)
    preds_table = {
        "y_true": [classes_str[i] for i in y_test],
        "y_pred": [classes_str[i] for i in y_pred],
        "top1_prob": top1_prob.tolist(),
    }
    preds_extra = {
        "test_indices": test_idx.tolist(),
        "groups_test": groups_test.tolist() if groups_test is not None else None,
        "turns_test": turns_test.tolist() if turns_test is not None else None,
    }

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "cm": cm,
        "preds_table": preds_table,
        "preds_extra": preds_extra
    }


# ----------------------------------- Main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train per-layer OvR L2 logistic probes (NPZ-only).")
    parser.add_argument("--data", type=str, required=True, help="Path to NPZ with layers + labels_text [+ dialog_id, turn_idx].")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layer indices (e.g. '0,1,2') or 'all'.")
    parser.add_argument("--labels_key", type=str, default="labels_text",
                        help="NPZ key to use for labels (default: labels_text).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default 0.2).")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength.")
    parser.add_argument("--max_iter", type=int, default=2000, help="Max iterations for LogisticRegression.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Optimization tolerance.")
    parser.add_argument("--solver", type=str, default="saga",
                        choices=["saga", "lbfgs", "liblinear", "newton-cg", "sag"],
                        help="Solver; 'saga' is robust for high-D.")
    parser.add_argument("--class_weight", type=str, default="none", choices=["none", "balanced"],
                        help="Use 'balanced' if classes are imbalanced.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs across OvR classes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Bucketing / label remap
    parser.add_argument("--label_map_json", type=str, default=None,
                        help="Optional JSON mapping from fine labels -> bucket labels (case-insensitive keys).")
    parser.add_argument("--skip_unmapped", action="store_true",
                        help="If set, drop examples whose labels are not found in the map.")

    args = parser.parse_args()

    # Global seeding for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.out_dir)

    # Load features & metadata
    X_layers = load_layers_only_npz(args.data)
    y, groups, turns = load_npz_labels_groups_turns(args.data, labels_key=args.labels_key)

    # --- Bucketing / remap ---
    if args.label_map_json:
        with open(args.label_map_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        # case-insensitive keys
        label_map = {str(k).casefold(): str(v) for k, v in raw_map.items()}

        keep_idx = []
        mapped_labels = []
        n_mapped = n_unmapped = n_skipped = 0

        for i, lab in enumerate(y):
            key = str(lab).casefold()
            if key in label_map:
                mapped_labels.append(label_map[key])
                keep_idx.append(i)
                n_mapped += 1
            else:
                if args.skip_unmapped:
                    n_skipped += 1
                else:
                    mapped_labels.append(str(lab))  # keep original if not skipping
                    keep_idx.append(i)
                    n_unmapped += 1

        if args.skip_unmapped:
            if len(keep_idx) == 0:
                raise ValueError("After skipping unmapped, no examples remain. Check your mapping.")
            keep_idx = np.asarray(keep_idx, dtype=int)
            # Filter labels and all feature layers + optional metadata
            y = np.array(mapped_labels, dtype=object)
            X_layers = [X[keep_idx] for X in X_layers]
            if groups is not None: groups = groups[keep_idx]
            if turns is not None:  turns  = turns[keep_idx]
        else:
            y = np.array(mapped_labels, dtype=object)

        print(f"[label_map] mapped={n_mapped} unmapped_kept={n_unmapped} skipped={n_skipped}")

    # Clean label dtypes
    y = np.array([str(v) for v in y], dtype=object)

    # Label encoding
    le = LabelEncoder()
    le.fit(y)

    # Select layers
    if args.layers.strip().lower() == "all":
        layer_idxs = list(range(len(X_layers)))
    else:
        layer_idxs = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
        for li in layer_idxs:
            if li < 0 or li >= len(X_layers):
                raise ValueError(f"Requested layer {li}, but dataset has {len(X_layers)} layers.")

    print(f"Found {len(X_layers)} layers. Training layers: {layer_idxs}")
    print(f"Examples: {X_layers[0].shape[0]}   Hidden size: {X_layers[0].shape[1]}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    if groups is not None:
        print("Group-aware split enabled (using dialog_id).")
    else:
        print("No dialog_id found; falling back to stratified split.")
    if turns is None:
        print("No turn_idx found; inertia metric will be skipped.")

    # Train per layer
    for li in layer_idxs:
        print(f"\n=== Training layer {li} ===")
        artifacts = train_layer_probe(
            X=X_layers[li],
            y=y,
            label_encoder=le,
            groups=groups,
            turns=turns,
            args=args
        )

        # Save model pipeline
        model_path = os.path.join(args.out_dir, f"probe_layer{li}.joblib")
        joblib.dump(artifacts["pipeline"], model_path)

        # Save metrics JSON
        metrics_path = os.path.join(args.out_dir, f"metrics_layer{li}.json")
        with open(metrics_path, "w") as f:
            json.dump(artifacts["metrics"], f, indent=2)

        # Save confusion matrices
        cm_path = os.path.join(args.out_dir, f"confusion_layer{li}.npy")
        np.save(cm_path, artifacts["cm"])
        cm_png = os.path.join(args.out_dir, f"confusion_layer{li}.png")
        plot_and_save_confusion(artifacts["cm"], list(le.classes_), cm_png, normalize=True)

        # Save predictions CSV
        pred_csv = os.path.join(args.out_dir, f"predictions_layer{li}.csv")
        with open(pred_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "y_pred", "top1_prob"])
            for yt, yp, p in zip(
                artifacts["preds_table"]["y_true"],
                artifacts["preds_table"]["y_pred"],
                artifacts["preds_table"]["top1_prob"]
            ):
                w.writerow([yt, yp, f"{p:.6f}"])

        # Save extra indices/groups (debug)
        extra_path = os.path.join(args.out_dir, f"predictions_layer{li}_extra.json")
        with open(extra_path, "w") as f:
            json.dump(artifacts["preds_extra"], f, indent=2)

        print(f"Saved: {model_path}")
        print(f"Saved: {metrics_path}")
        print(f"Saved: {cm_png}")
        print(f"Saved: {pred_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
