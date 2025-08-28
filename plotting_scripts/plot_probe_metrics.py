#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate probe artifacts across layers and make summary plots.

Inputs expected in --out_dir (produced by your training script):
- metrics_layer{idx}.json
- predictions_layer{idx}.csv           (columns: y_true, y_pred, top1_prob)
- predictions_layer{idx}_extra.json    (test_indices, groups_test, turns_test)

Optional inputs (for "greatest hits"):
- --data_npz: the NPZ with per-layer features (to recompute predict_proba)
- --attribute_jsonl: the {attribute}.jsonl emitted by the extractor (to print text)

Outputs (written to <out_dir>/plots/):
- overall_accuracy.png
- per_bucket_accuracy.png
- per_bucket_accuracy_present_only.png
- inertia_error_rate.png
- accuracy_by_turn_best_layer.png
- change_point_accuracy_k0_vs_layer.png
- tta_ecdf_layer{best}.png
- transition_confusion_layer{best}.json
- transition_easiest_k0.png / transition_hardest_k0.png / transition_highest_inertia.png
- greatest_hits_layer{best}.md (+ greatest_hits_layer{best}.json)
"""

import os, json, csv, re, argparse
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import joblib   # for loading probe pipelines


# ---------------------------- utils & loaders ---------------------------- #

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_layer_ids(out_dir):
    layers = []
    for name in os.listdir(out_dir):
        m = re.match(r"metrics_layer(\d+)\.json$", name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)

def load_metrics_for_layer(out_dir, layer):
    with open(os.path.join(out_dir, f"metrics_layer{layer}.json"), "r") as f:
        return json.load(f)

def load_predictions_for_layer(out_dir, layer):
    """returns [(y_true, y_pred, top1_prob)], groups_test, turns_test"""
    preds_csv = os.path.join(out_dir, f"predictions_layer{layer}.csv")
    extra_json = os.path.join(out_dir, f"predictions_layer{layer}_extra.json")

    rows = []
    with open(preds_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["y_true"], row["y_pred"], float(row["top1_prob"])))

    groups = turns = None
    if os.path.exists(extra_json):
        with open(extra_json, "r") as f:
            extra = json.load(f)
        groups = np.array(extra.get("groups_test", [None]*len(rows)), dtype=object) if extra.get("groups_test") is not None else None
        turns  = np.array(extra.get("turns_test",  [None]*len(rows)), dtype=object) if extra.get("turns_test")  is not None else None

    return rows, groups, turns

def load_test_rows(out_dir, layer):
    """Return sorted list of rows: (group, turn, y_true, y_pred)."""
    extra = json.load(open(os.path.join(out_dir, f"predictions_layer{layer}_extra.json")))
    groups = extra.get("groups_test")
    turns  = extra.get("turns_test")
    rows = []
    with open(os.path.join(out_dir, f"predictions_layer{layer}.csv")) as f:
        r = csv.DictReader(f)
        for (row, g, t) in zip(r, groups, turns):
            rows.append((g, int(t), row["y_true"], row["y_pred"]))
    rows.sort(key=lambda x: (str(x[0]), int(x[1])))
    return rows


# ------------------------------- base plots ------------------------------ #

def plot_overall_accuracy(layer_to_overall, out_path):
    xs = sorted(layer_to_overall.keys())
    ys = [layer_to_overall[x] for x in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title("Overall Accuracy vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_per_bucket_accuracy(layer_to_perclass, out_path, drop_nan=False):
    classes = set()
    for d in layer_to_perclass.values():
        classes.update(d.keys())
    classes = sorted(classes)

    plt.figure()
    for cls in classes:
        xs, ys = [], []
        for layer in sorted(layer_to_perclass.keys()):
            v = layer_to_perclass[layer].get(cls, float("nan"))
            if drop_nan and (v != v):  # NaN
                continue
            xs.append(layer); ys.append(v)
        if xs:
            plt.plot(xs, ys, marker="o", label=cls)
    plt.title("Per-Bucket Accuracy vs Layer" + (" (test-present only)" if drop_nan else ""))
    plt.xlabel("Layer"); plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_inertia(layer_to_inertia, out_path):
    xs = sorted(layer_to_inertia.keys())
    ys = [layer_to_inertia[x] for x in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title("Inertia Error Rate vs Layer")
    plt.xlabel("Layer"); plt.ylabel("Inertia Error Rate")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_accuracy_by_turn_for_layer(out_dir, layer, out_path):
    rows, groups, turns = load_predictions_for_layer(out_dir, layer)
    if turns is None:
        print(f"[warn] No turn indices for layer {layer}; skipping accuracy-by-turn plot.")
        return

    y_true = np.array([r[0] for r in rows], dtype=object)
    y_pred = np.array([r[1] for r in rows], dtype=object)
    turns  = np.array(turns)

    acc_by_turn = defaultdict(lambda: [0,0])  # turn -> [correct, total]
    for t, yt, yp in zip(turns, y_true, y_pred):
        if t is None:
            continue
        t = int(t)
        acc_by_turn[t][1] += 1
        if yt == yp:
            acc_by_turn[t][0] += 1

    if not acc_by_turn:
        print(f"[warn] Turn info empty for layer {layer}.")
        return

    ts = sorted(acc_by_turn.keys())
    acc = [ (acc_by_turn[t][0] / max(1, acc_by_turn[t][1])) for t in ts ]

    plt.figure()
    plt.plot(ts, acc, marker="o")
    plt.title(f"Accuracy by Turn Index (Layer {layer})")
    plt.xlabel("Turn index"); plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ----------------------- switch-aware computations ----------------------- #

def iter_change_events(rows):
    """
    rows: sorted (group, turn, y_true, y_pred).
    Yields events dicts at each test-visible change point within a dialog.
    """
    by_g = defaultdict(list)
    for g,t,yt,yp in rows:
        by_g[g].append((t, yt, yp))
    for g, seq in by_g.items():
        for i in range(1, len(seq)):
            t,  yt,  yp  = seq[i]
            t1, yt1, yp1 = seq[i-1]
            if yt != yt1:
                yield {
                    "group": g, "t": t,
                    "prev_t": t1, "prev_true": yt1, "true": yt,
                    "pred_at_t": yp
                }

def change_point_accuracy_curve(rows, max_k=0):
    """
    For each change point at t0, check correctness at t0+k (k=0..max_k).
    Returns: ks, acc_at_k, counts_at_k
    """
    pred = {(g,t): yp for g,t,_,yp in rows}
    truth = {(g,t): yt for g,t,yt,_ in rows}
    totals = Counter()
    correct = Counter()
    for ev in iter_change_events(rows):
        g, t0, new = ev["group"], ev["t"], ev["true"]
        for k in range(max_k+1):
            key = (g, t0 + k)
            if key in pred and key in truth:
                totals[k] += 1
                if pred[key] == new:
                    correct[k] += 1
    ks = sorted(totals.keys())
    acc = [correct[k] / totals[k] if totals[k] else float("nan") for k in ks]
    cnt = [totals[k] for k in ks]
    return ks, acc, cnt

def time_to_adapt_distribution(rows, max_k=5):
    """
    For each change point, TTA = smallest k in [0..max_k] with pred==new label.
    If none within window, mark as None.
    """
    pred = {(g,t): yp for g,t,_,yp in rows}
    tta = []
    for ev in iter_change_events(rows):
        g, t0, new = ev["group"], ev["t"], ev["true"]
        hit = None
        for k in range(max_k+1):
            yp = pred.get((g, t0 + k))
            if yp is not None and yp == new:
                hit = k; break
        tta.append(hit)
    return tta

def change_detection_prf(rows):
    """
    Binary detection: Did truth change at t vs t-1? Did prediction change?
    Compute precision/recall/F1 on test-visible consecutive pairs per dialog.
    """
    by_g = defaultdict(list)
    for g,t,yt,yp in rows:
        by_g[g].append((t, yt, yp))
    TP=FP=FN=0
    for g, seq in by_g.items():
        for i in range(1, len(seq)):
            _, yt, yp   = seq[i]
            _, yt1, yp1 = seq[i-1]
            gt_change = (yt != yt1)
            pr_change = (yp != yp1)
            if pr_change and gt_change: TP += 1
            elif pr_change and not gt_change: FP += 1
            elif (not pr_change) and gt_change: FN += 1
    prec = TP / (TP+FP) if (TP+FP)>0 else float("nan")
    rec  = TP / (TP+FN) if (TP+FN)>0 else float("nan")
    f1   = 2*prec*rec/(prec+rec) if prec==prec and rec==rec and (prec+rec)>0 else float("nan")
    return {"precision":prec, "recall":rec, "f1":f1, "tp":TP, "fp":FP, "fn":FN}

def transition_confusion_at_changes(rows):
    """
    Count predictions exactly at t0 for each true transition A->B.
    Returns dict[(A,B)][pred] = count.
    """
    M = defaultdict(Counter)
    for ev in iter_change_events(rows):
        A, B, p = ev["prev_true"], ev["true"], ev["pred_at_t"]
        M[(A,B)][p] += 1
    return M


# --------------------------- switch-aware plots --------------------------- #

def plot_change_point_accuracy_k0_across_layers(out_dir, layers, save_path=None):
    """Plot only k=0 (immediate adaptation) vs layer."""
    vals = []
    for L in layers:
        rows = load_test_rows(out_dir, L)
        ks, acc, cnt = change_point_accuracy_curve(rows, max_k=0)  # k=0 only
        a0 = acc[0] if acc else float("nan")
        vals.append((L, a0))
    vals.sort()
    plt.figure()
    plt.plot([x for x,_ in vals], [y for _,y in vals], marker="o", label="k=0")
    plt.title("Change-point Accuracy (k=0) vs Layer")
    plt.xlabel("Layer"); plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=180); plt.close()

def plot_tta_ecdf_for_best_layer(out_dir, best_layer, max_k=5, save_path=None):
    rows = load_test_rows(out_dir, best_layer)
    tta = time_to_adapt_distribution(rows, max_k=max_k)
    vals = [v for v in tta if v is not None]
    N = len(tta)
    if len(vals) == 0 or N == 0:
        print("[warn] No change events found for TTA ECDF.")
        return
    xs = sorted(set(vals))
    ecdf_y = [sum(1 for v in vals if v <= x)/N for x in xs]
    plt.figure()
    plt.step(xs, ecdf_y, where="post")
    plt.title(f"TTA ECDF (Layer {best_layer})")
    plt.xlabel("k turns after change"); plt.ylabel("Frac. adapted (≤k)")
    plt.ylim(0,1); plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=180); plt.close()

def save_transition_confusion_table(out_dir, best_layer, save_json):
    rows = load_test_rows(out_dir, best_layer)
    M = transition_confusion_at_changes(rows)
    out = {f"{a}->{b}": dict(cnt) for (a,b), cnt in M.items()}
    with open(save_json, "w") as f:
        json.dump(out, f, indent=2)

# -------- transition ranking + bars (k=0 acc & inertia) -------- #

def transition_stats_k0_and_inertia(rows):
    pred = {(g,t): yp for g,t,_,yp in rows}
    stats = {}  # (A,B) -> {"n":n0, "hits":h0, "inertia":inertia_hits}
    for ev in iter_change_events(rows):
        g, t0, A, B = ev["group"], ev["t"], ev["prev_true"], ev["true"]
        d = stats.setdefault((A,B), {"n":0, "hits":0, "inertia":0})
        p = pred.get((g,t0))
        if p is None: 
            continue
        d["n"] += 1
        if p == B: d["hits"] += 1
        if p == A: d["inertia"] += 1
    # convert
    table = []
    for (A,B), d in stats.items():
        acc = d["hits"]/d["n"] if d["n"] else float("nan")
        inertia_rate = d["inertia"]/d["n"] if d["n"] else float("nan")
        table.append({"transition": f"{A}->{B}", "n": d["n"], "acc_k0": acc, "inertia_rate": inertia_rate})
    return table

def _plot_transition_bar(table, key, title, save_path, top_n=15, min_n=5, reverse=False):
    # filter by support
    tbl = [r for r in table if r["n"] >= min_n and r[key] == r[key]]  # drop NaN
    if not tbl:
        print(f"[warn] no transitions meet min_n={min_n} for {key}")
        return
    tbl.sort(key=lambda r: r[key], reverse=reverse)
    if top_n: tbl = tbl[:top_n]
    labels = [r["transition"] for r in tbl]
    vals   = [r[key] for r in tbl]
    counts = [r["n"] for r in tbl]
    plt.figure(figsize=(8, 0.45*len(tbl)+1))
    plt.barh(labels, vals, alpha=0.85)
    for i,(v,n) in enumerate(zip(vals,counts)):
        plt.text(v + 0.01, i, f"n={n}", va="center")
    plt.xlim(0, 1 if key!="inertia_rate" else 1)
    plt.xlabel(key); plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

# ----------------------- greatest hits (examples) ----------------------- #

def load_rows_with_proba(out_dir, data_npz, layer):
    """Return sorted rows: (group, turn, y_true, y_pred, proba_dict)"""
    extra = json.load(open(os.path.join(out_dir, f"predictions_layer{layer}_extra.json")))
    groups = np.array(extra["groups_test"])
    turns  = np.array(extra["turns_test"])
    y_true = []; y_pred = []
    with open(os.path.join(out_dir, f"predictions_layer{layer}.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            y_true.append(row["y_true"]); y_pred.append(row["y_pred"])
    # rebuild probabilities using saved pipeline
    pipe = joblib.load(os.path.join(out_dir, f"probe_layer{layer}.joblib"))
    scaler, clf, le = pipe["scaler"], pipe["classifier"], pipe["label_encoder"]
    classes = list(le.classes_)
    test_idx = np.array(json.load(open(os.path.join(out_dir, f"predictions_layer{layer}_extra.json")))["test_indices"])
    X_layer = np.load(data_npz, allow_pickle=True)[str(layer)]
    X_test = scaler.transform(X_layer[test_idx])
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
    else:
        scores = clf.decision_function(X_test)
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        proba = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-9, None)
    rows = []
    for g,t,yt,yp,p in zip(groups, turns, y_true, y_pred, proba):
        rows.append((str(g), int(t), yt, yp, dict(zip(classes, p.tolist()))))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows

def change_events_with_proba(rows):
    by_g = defaultdict(list)
    for g,t,yt,yp,p in rows: by_g[g].append((t,yt,yp,p))
    for g, seq in by_g.items():
        for i in range(1, len(seq)):
            t,yt,yp,p = seq[i]
            t1,y1,_,_ = seq[i-1]
            if yt != y1:
                yield {"group":g,"t":t,"prev_true":y1,"true":yt,"pred":yp,"proba":p}

def pick_examples(rows, hits_per_category=5, neighbor_pairs=(("calm_steady","positive_low"),
                                                             ("positive_low","calm_steady"))):
    idx = {(g,t): (yt,yp,p) for g,t,yt,yp,p in rows}
    picks = {"clean_snap":[], "sticky":[], "slow_adapter":[],
             "flip_flop":[], "neighbor_confusion":[], "bold_wrong":[]}
    for ev in change_events_with_proba(rows):
        g,t,A,B,yp,p = ev["group"], ev["t"], ev["prev_true"], ev["true"], ev["pred"], ev["proba"]
        conf_pred = p.get(yp, 0.0)

        # clean snap: correct at t with high conf
        if yp==B and conf_pred>=0.8:
            picks["clean_snap"].append((g,t,A,B,conf_pred))
        # sticky: predicts A at t with decent conf
        if yp==A and conf_pred>=0.7:
            picks["sticky"].append((g,t,A,B,conf_pred))
        # slow adapter: wrong at t, right by t+2
        yp1 = idx.get((g,t+1), (None,None,None))[1]
        yp2 = idx.get((g,t+2), (None,None,None))[1]
        if yp!=B and (yp1==B or yp2==B):
            picks["slow_adapter"].append((g,t,A,B,conf_pred))
        # flip-flop: A->(not B)->A around the change
        yp_prev = idx.get((g,t-1), (None,None,None))[1]
        yp_next = idx.get((g,t+1), (None,None,None))[1]
        if yp_prev is not None and yp_next is not None and yp_prev != yp and yp_next == yp_prev:
            picks["flip_flop"].append((g,t,A,B,conf_pred))
        # neighbor confusion
        if (A,B) in neighbor_pairs and yp!=B:
            picks["neighbor_confusion"].append((g,t,A,B,conf_pred))
        # bold but wrong (not sticky)
        if yp not in (A,B) and conf_pred>=0.8:
            picks["bold_wrong"].append((g,t,A,B,conf_pred))

    # sort & trim
    for k in picks:
        picks[k] = sorted(picks[k], key=lambda x: -x[-1])[:hits_per_category]
    return picks

def fetch_text_context(attribute_jsonl, group, turn, window=2):
    if not attribute_jsonl or not os.path.exists(attribute_jsonl):
        return []
    by_g = defaultdict(dict)
    with open(attribute_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_g[str(d["dialog_id"])][int(d["turn_idx"])] = (d.get("label"), d.get("user_text"))
    ctx=[]
    for k in range(turn-window, turn+window+1):
        if k in by_g.get(str(group), {}):
            lbl, txt = by_g[str(group)][k]
            ctx.append((k, lbl, txt))
    return ctx

def write_greatest_hits_md(out_dir, best_layer, picks, attribute_jsonl=None):
    plot_dir = os.path.join(out_dir, "plots"); ensure_dir(plot_dir)
    md_path = os.path.join(plot_dir, f"greatest_hits_layer{best_layer}.md")
    json_path = os.path.join(plot_dir, f"greatest_hits_layer{best_layer}.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Greatest Hits (Layer {best_layer})\n\n")
        for section, items in picks.items():
            f.write(f"## {section.replace('_',' ').title()}\n\n")
            if not items:
                f.write("_none_\n\n"); continue
            for (g,t,A,B,conf) in items:
                f.write(f"- **{A} → {B}** | dialog={g} turn={t} | conf={conf:.2f}\n")
                ctx = fetch_text_context(attribute_jsonl, g, t, window=2)
                if ctx:
                    for (k,lbl,txt) in ctx:
                        mark = ">>" if k==t else "  "
                        snippet = (txt[:180] + "…") if (txt and len(txt)>180) else (txt or "")
                        f.write(f"    - {mark} turn {k}: [{lbl}] {snippet}\n")
                f.write("\n")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({k: [{"group":g,"turn":t,"from":A,"to":B,"confidence":conf} for (g,t,A,B,conf) in v]
                   for k,v in picks.items()}, jf, indent=2)
    print(f"[hits] wrote {md_path} and {json_path}")


# ----------------------------------- main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory with metrics_layer*.json & predictions_layer*.csv/_extra.json")
    ap.add_argument("--data_npz", type=str, default=None, help="NPZ with per-layer features (to recompute probabilities).")
    ap.add_argument("--attribute_jsonl", type=str, default=None, help="Optional extractor JSONL to print user text for examples.")
    ap.add_argument("--min_transition_support", type=int, default=5, help="Min #events to include a transition in bar charts.")
    ap.add_argument("--top_n_transitions", type=int, default=15, help="How many transitions to show in each bar chart.")
    ap.add_argument("--hits_per_category", type=int, default=5, help="How many cherry-picked examples per category.")
    args = ap.parse_args()

    out_dir = args.out_dir
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    layers = find_layer_ids(out_dir)
    if not layers:
        raise SystemExit(f"No metrics_layer*.json files found in {out_dir}")

    layer_to_overall = {}
    layer_to_perclass = {}
    layer_to_inertia = {}

    for L in layers:
        metrics = load_metrics_for_layer(out_dir, L)
        layer_to_overall[L] = metrics.get("overall_accuracy", float("nan"))
        perclass = metrics.get("per_class_accuracy", {})
        layer_to_perclass[L] = {k: (float(v) if v is not None else float("nan")) for k, v in perclass.items()}
        inertia = metrics.get("inertia_error_rate", float("nan"))
        layer_to_inertia[L] = float(inertia) if inertia is not None else float("nan")

    # 1) Overall accuracy across layers
    plot_overall_accuracy(layer_to_overall, os.path.join(plot_dir, "overall_accuracy.png"))

    # 2) Per-bucket accuracy across layers
    plot_per_bucket_accuracy(layer_to_perclass, os.path.join(plot_dir, "per_bucket_accuracy.png"), drop_nan=False)
    plot_per_bucket_accuracy(layer_to_perclass, os.path.join(plot_dir, "per_bucket_accuracy_present_only.png"), drop_nan=True)

    # 3) Inertia error rate across layers
    plot_inertia(layer_to_inertia, os.path.join(plot_dir, "inertia_error_rate.png"))

    # 4) Accuracy across turns for the best layer (by overall acc)
    best_layer = max(layer_to_overall.items(), key=lambda kv: kv[1])[0]
    plot_accuracy_by_turn_for_layer(out_dir, best_layer, os.path.join(plot_dir, "accuracy_by_turn_best_layer.png"))

    # 5) Switch-aware plots (k=0 only + TTA ECDF + transition table)
    #    5a. k=0 change-point accuracy vs layer
    plot_change_point_accuracy_k0_across_layers(
        out_dir, layers,
        save_path=os.path.join(plot_dir, "change_point_accuracy_k0_vs_layer.png")
    )
    #    5b. TTA ECDF for best layer
    plot_tta_ecdf_for_best_layer(out_dir, best_layer, max_k=5,
        save_path=os.path.join(plot_dir, f"tta_ecdf_layer{best_layer}.png"))
    #    5c. Transition confusion JSON (at change points)
    save_transition_confusion_table(out_dir, best_layer,
        os.path.join(plot_dir, f"transition_confusion_layer{best_layer}.json"))

    # 6) Transition ranking + bar charts (k=0 acc & inertia) for best layer
    rows_best = load_test_rows(out_dir, best_layer)
    trans_table = transition_stats_k0_and_inertia(rows_best)
    # easiest: highest acc@k0
    _plot_transition_bar(
        trans_table, key="acc_k0",
        title="Transition Accuracy @k=0 (Easiest — top by accuracy)",
        save_path=os.path.join(plot_dir, "transition_easiest_k0.png"),
        top_n=args.top_n_transitions, min_n=args.min_transition_support, reverse=True
    )
    # hardest: lowest acc@k0
    _plot_transition_bar(
        trans_table, key="acc_k0",
        title="Transition Accuracy @k=0 (Hardest — low accuracy)",
        save_path=os.path.join(plot_dir, "transition_hardest_k0.png"),
        top_n=args.top_n_transitions, min_n=args.min_transition_support, reverse=False
    )
    # highest inertia
    _plot_transition_bar(
        trans_table, key="inertia_rate",
        title="Transitions with Highest Inertia (prediction sticks to old label)",
        save_path=os.path.join(plot_dir, "transition_highest_inertia.png"),
        top_n=args.top_n_transitions, min_n=args.min_transition_support, reverse=True
    )

    # 7) Greatest hits (optional; needs NPZ to compute probs)
    if args.data_npz:
        try:
            rows_proba = load_rows_with_proba(out_dir, args.data_npz, best_layer)
            picks = pick_examples(rows_proba, hits_per_category=args.hits_per_category)
            write_greatest_hits_md(out_dir, best_layer, picks, attribute_jsonl=args.attribute_jsonl)
        except Exception as e:
            print(f"[warn] greatest hits generation failed: {e}")
    else:
        print("[info] --data_npz not provided; skipping greatest hits.")

    print(f"Done. Wrote plots to: {plot_dir}  (best layer={best_layer})")

if __name__ == "__main__":
    main()
