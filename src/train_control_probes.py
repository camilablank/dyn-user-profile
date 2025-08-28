#!/usr/bin/env python3
import os, glob, json, argparse
import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# ------------------------ IO ------------------------

def load_meta(train_dir):
    with open(os.path.join(train_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return meta["buckets"], meta["sigma_per_layer"]

def load_layer(train_dir, layer_idx):
    path = os.path.join(train_dir, "by_layer", f"layer_{layer_idx:02d}.npz")
    Z = np.load(path, allow_pickle=False)
    X_l = Z["X_l"].astype("float32")     # [N, d]
    y   = Z["y"].astype("int32")         # [N]
    conv_id = Z["conv_id"]               # [N] (str or int)
    turn_id = Z["turn_id"]               # [N] (int)
    return X_l, y, conv_id, turn_id

# -------------------- Group helpers --------------------

def rebuild_groups_from_turn_ids(turn_ids: np.ndarray) -> np.ndarray:
    """Create integer group ids by starting a new group whenever turn_id resets or decreases."""
    t = turn_ids.astype(int)
    gid = 0
    groups = np.empty_like(t)
    groups[0] = gid
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:  # new conversation starts
            gid += 1
        groups[i] = gid
    return groups

def prepare_groups(conv_ids: np.ndarray, turn_ids: np.ndarray, target_max_splits: int = 5):
    """
    Returns (groups, n_splits, info_str)
    - groups: integer group ids for GroupKFold
    - n_splits: min(max_splits, n_groups) but at least 2
    Tries conv_id first; if too few groups, rebuilds from turn_id resets.
    """
    # try conv_ids
    conv_ids = np.asarray(conv_ids)
    uniq = np.unique(conv_ids)
    if uniq.size >= 5:
        # factorize to integers for safety
        _, groups = np.unique(conv_ids, return_inverse=True)
        n_groups = uniq.size
        k = max(2, min(target_max_splits, n_groups))
        return groups, k, f"groups={n_groups} from conv_id, n_splits={k}"
    # fallback
    groups = rebuild_groups_from_turn_ids(np.asarray(turn_ids))
    n_groups = int(groups.max() + 1)
    k = max(2, min(target_max_splits, n_groups))
    return groups, k, f"groups={n_groups} from turn_id (rebuilt), n_splits={k}"

# ------------------- Training core -------------------

def choose_C_with_groupcv(X, y_bin, groups, max_splits: int, Cs, max_iter=400):
    """Return best C and fold metrics (ROC-AUC / AP), adapting n_splits to #groups."""
    n_groups = np.unique(groups).size
    k = max(2, min(max_splits, n_groups))
    gkf = GroupKFold(n_splits=k)

    best = {"C": None, "roc_mean": -1.0, "ap_mean": -1.0}
    metrics_per_C = {}
    for C in Cs:
        rocs, aps = [], []
        for tr, va in gkf.split(X, y_bin, groups):
            y_va = y_bin[va]
            if len(np.unique(y_va)) < 2:
                continue  # skip folds with only one class
            clf = LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                solver="liblinear",
                max_iter=max_iter,
                C=C,
            )
            clf.fit(X[tr], y_bin[tr])
            s = clf.decision_function(X[va])
            rocs.append(roc_auc_score(y_va, s))
            aps.append(average_precision_score(y_va, s))
        if not rocs:
            continue
        roc_m, ap_m = float(np.mean(rocs)), float(np.mean(aps))
        metrics_per_C[C] = {"roc_mean": roc_m, "ap_mean": ap_m, "n_splits_used": k}
        if roc_m > best["roc_mean"] or (abs(roc_m - best["roc_mean"]) < 1e-6 and ap_m > best["ap_mean"]):
            best = {"C": C, "roc_mean": roc_m, "ap_mean": ap_m, "n_splits_used": k}
    return best, metrics_per_C

def train_full_and_normalize(X, y_bin, C, max_iter=600):
    clf = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=max_iter,
        C=C,
    )
    clf.fit(X, y_bin)
    w = clf.coef_.ravel().astype("float32")
    v = w / (np.linalg.norm(w) + 1e-12)
    return v.astype("float32")

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Output of step 1 (e.g., /workspace/a1_train)")
    ap.add_argument("--out_dir", required=True, help="Where to save vectors & reports")
    ap.add_argument("--layers", default="all", help="Comma list (e.g., 16,18,20) or 'all'")
    ap.add_argument("--C_grid", default="0.25,0.5,1,2,4")
    ap.add_argument("--max_splits", type=int, default=5, help="Upper bound for GroupKFold splits")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "vectors"), exist_ok=True)

    buckets, sigma_per_layer = load_meta(args.train_dir)

    layer_files = sorted(glob.glob(os.path.join(args.train_dir, "by_layer", "layer_*.npz")))
    L = len(layer_files)
    layer_list = list(range(L)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    Cs = [float(x) for x in args.C_grid.split(",")]

    # Load labels/groups once (from any layer file)
    sample = np.load(layer_files[0])
    y_all = sample["y"].astype("int32")
    conv_id_all = sample["conv_id"]
    turn_id_all = sample["turn_id"].astype("int32")

    # Prepare groups (conv_id or rebuilt from turn_id)
    groups_all, n_splits, info = prepare_groups(conv_id_all, turn_id_all, target_max_splits=args.max_splits)
    print(f"[grouping] {info}")

    id2bucket = {i: b for i, b in enumerate(buckets)}
    report_rows = []

    for l in layer_list:
        X_l, y_ids, conv_ids, turn_ids = load_layer(args.train_dir, l)
        # sanity checks
        assert np.array_equal(y_all, y_ids), "Label mismatch across layers."
        # use the precomputed groups (they correspond 1:1 to rows)
        groups = groups_all

        N, d = X_l.shape
        cnt = Counter(y_ids.tolist())
        counts_str = ", ".join(f"{id2bucket[i]}: {cnt[i]}" for i in sorted(cnt))
        print(f"[layer {l:02d}] N={N}, d={d}, counts={{ {counts_str} }}")

        for bid in sorted(cnt.keys()):
            bname = id2bucket[bid]
            y_bin = (y_ids == bid).astype("int32")
            pos, neg = int(y_bin.sum()), int((1 - y_bin).sum())
            if pos == 0 or neg == 0:
                print(f"  - skip {bname}: pos={pos}, neg={neg}")
                continue

            best, metrics_per_C = choose_C_with_groupcv(X_l, y_bin, groups, args.max_splits, Cs)
            if best["C"] is None:
                print(f"  - no valid CV for {bname}; skipping.")
                continue

            v = train_full_and_normalize(X_l, y_bin, best["C"])
            np.save(os.path.join(args.out_dir, "vectors", f"v_layer{l:02d}_{bname}.npy"), v)

            row = {
                "layer": l,
                "bucket": bname,
                "N_pos": pos,
                "N_neg": neg,
                "C": best["C"],
                "roc_mean": best["roc_mean"],
                "ap_mean": best["ap_mean"],
                "n_splits_used": best.get("n_splits_used", n_splits),
            }
            report_rows.append(row)

            with open(os.path.join(args.out_dir, f"cv_layer{l:02d}_{bname}.json"), "w") as f:
                json.dump(metrics_per_C, f, indent=2)

    # write a compact CSV report
    import csv
    with open(os.path.join(args.out_dir, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer","bucket","N_pos","N_neg","C","roc_mean","ap_mean","n_splits_used"])
        w.writeheader()
        for r in sorted(report_rows, key=lambda x: (x["layer"], x["bucket"])):
            w.writerow(r)

    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump({
            "buckets": buckets,
            "sigma_per_layer": sigma_per_layer,
            "vectors_dir": "vectors",
            "C_grid": Cs
        }, f, indent=2)

    print(f"Saved vectors to {os.path.join(args.out_dir,'vectors')}")
    print(f"Wrote report.csv and index.json to {args.out_dir}")

if __name__ == "__main__":
    main()
