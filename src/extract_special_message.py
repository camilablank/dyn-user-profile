# extract_llama3_hidden_states.py
# PyTorch/Transformers hidden-state extractor for LLaMA-3 with bf16 + attention backend options.
# Saves per-attribute activations: {attribute}.npz (per-layer matrices) + {attribute}.jsonl (metadata).
# Tested with PyTorch 2.2+/CUDA 12.x and Transformers 4.41+.

import os, json, argparse, time
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTRIBUTES = ["emotion", "confidence", "trust"]
LABEL_KEYS = {  # edit if other dims use different label fields
    "emotion": "user_attribute_value",
    "confidence": "user_attribute_value",
    "trust": "user_attribute_value",
}

def load_dataset(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

def setup_model(model_name: str, attn_impl: str, use_bf16: bool, enable_tf32: bool):
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32

    kw = dict(
        revision=None,
        output_hidden_states=False,
        torch_dtype=dtype,
    )

    # Attention backend selection
    # valid: "flash_attention_2", "sdpa", "eager"
    if attn_impl:
        kw["attn_implementation"] = attn_impl

    # device placement
    if torch.cuda.is_available():
        kw["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    return model, tokenizer, device, dtype

def register_residual_hooks(model, layer_whitelist=None):
    """
    Register forward hooks on transformer blocks (model.model.layers[i]).
    If layer_whitelist is provided (list of ints), only hook those layers.
    """
    captured = {}
    hooks = []

    # If whitelist is None -> hook all
    all_layers = list(range(len(model.model.layers)))
    target_layers = set(layer_whitelist) if layer_whitelist is not None else set(all_layers)

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out shape: [B, S, H]
            captured[layer_idx] = out.detach().to("cpu")
        return hook

    for idx, block in enumerate(model.model.layers):
        if idx in target_layers:
            hooks.append(block.register_forward_hook(make_hook(idx)))

    return captured, hooks, sorted(target_layers)

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

def tokenize_with_truncation(tokenizer, text: str, max_len: int, device: str):
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in toks.items()}

def build_prompt(prefix_text: str, user_text: str, attribute: str) -> str:
    special_prompt = f"I think the {attribute} of this user is"
    full = ""
    if prefix_text:
        full += prefix_text.rstrip() + "\n"
    full += f"User: {user_text}\nAssistant: {special_prompt}"
    return full

@torch.no_grad()
def encode_and_capture(model, tokenizer, device, full_text: str, max_len: int, layer_whitelist=None):
    inputs = tokenize_with_truncation(tokenizer, full_text, max_len, device)
    captured, hooks, hooked_layers = register_residual_hooks(model, layer_whitelist)
    _ = model(**inputs)
    remove_hooks(hooks)

    last_idx = inputs["input_ids"].shape[1] - 1
    hidden_by_layer = {}
    for layer_idx in hooked_layers:
        tensor = captured[layer_idx]  # [1, S, H]
        vec = tensor[0, last_idx, :].to(torch.float32).numpy()
        hidden_by_layer[layer_idx] = vec
    return hidden_by_layer

def process_attribute(
    data, model, tokenizer, device,
    attribute: str, out_dir: str,
    max_seq_len: int,
    layer_whitelist=None,
    max_conversations: int = None,
):
    os.makedirs(out_dir, exist_ok=True)
    label_key = LABEL_KEYS[attribute]
    jsonl_path = os.path.join(out_dir, f"{attribute}.jsonl")
    npz_path = os.path.join(out_dir, f"{attribute}.npz")

    per_layer_vectors = defaultdict(list)
    labels, dialog_ids, turn_indices = [], [], []

    n_dialogs = 0
    n_examples = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for d_idx, dialog in enumerate(data):
            meta = dialog.get("meta", {})
            if meta.get("dimension") != attribute:
                continue

            n_dialogs += 1
            if max_conversations is not None and n_dialogs > max_conversations:
                break

            dialog_id = meta.get("dialog_id", f"dlg_{d_idx:06d}")
            turns = dialog.get("dialog", [])
            prefix = ""

            for t_idx, turn in enumerate(turns):
                speaker = (turn.get("speaker") or "").lower()
                text = turn.get("text") or ""

                if speaker == "user":
                    gt_label = turn.get(label_key)
                    if gt_label is None:
                        prefix += f"{speaker.capitalize()}: {text}\n"
                        continue

                    full_text = build_prompt(prefix, text, attribute)
                    hidden_by_layer = encode_and_capture(
                        model, tokenizer, device,
                        full_text=full_text,
                        max_len=max_seq_len,
                        layer_whitelist=layer_whitelist,
                    )

                    for layer_idx, vec in hidden_by_layer.items():
                        per_layer_vectors[layer_idx].append(vec)

                    labels.append(gt_label)
                    dialog_ids.append(dialog_id)
                    turn_indices.append(t_idx)

                    meta_out = {
                        "dialog_id": dialog_id,
                        "turn_idx": t_idx,
                        "attribute": attribute,
                        "label": gt_label,
                        "user_text": text,
                        "meta": {
                            "style": meta.get("style"),
                            "timestamp": meta.get("timestamp"),
                            "model": meta.get("model"),
                            "turn_count": meta.get("turn_count"),
                            "change_schedule": meta.get("change_schedule"),
                        }
                    }
                    jf.write(json.dumps(meta_out, ensure_ascii=False) + "\n")
                    n_examples += 1

                prefix += f"{speaker.capitalize()}: {text}\n"

    if n_examples == 0:
        print(f"[{attribute}] No examples found. Check meta.dimension and label keys.")
        return {"attribute": attribute, "n_user_turns": 0, "layers": []}

    all_layers = sorted(per_layer_vectors.keys())
    stacked = {}
    for layer_idx in all_layers:
        stacked[str(layer_idx)] = np.stack(per_layer_vectors[layer_idx], axis=0)  # [N, H]
    stacked["labels_text"] = np.array(labels, dtype=object)
    stacked["dialog_id"] = np.array(dialog_ids, dtype=object)
    stacked["turn_idx"] = np.array(turn_indices, dtype=np.int32)
    np.savez_compressed(npz_path, **stacked)

    print(f"[{attribute}] dialogs_used={n_dialogs} examples={n_examples} layers={len(all_layers)} -> {npz_path}")
    return {"attribute": attribute, "n_user_turns": n_examples, "layers": all_layers}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to conversations JSON (list of dialogs).")
    ap.add_argument("--out_dir", default="activations_out", help="Output directory.")
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--only_attributes", nargs="*", default=ATTRIBUTES,
                    help="Subset of attributes to process.")
    ap.add_argument("--max_seq_len", type=int, default=4096, help="Truncate context to this many tokens.")
    ap.add_argument("--layer_whitelist", nargs="*", type=int, default=None,
                    help="Optional list of layer indices to capture (e.g., --layer_whitelist 20 24 28).")
    ap.add_argument("--attn_impl", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"],
                    help="Attention backend.")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 when CUDA is available.")
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 on matmul/cudnn.")
    ap.add_argument("--max_conversations", type=int, default=None,
                    help="Optional cap on number of dialogs per attribute for pilot runs.")
    args = ap.parse_args()

    # Setup model/tokenizer
    model, tokenizer, device, dtype = setup_model(
        args.model_name, args.attn_impl, use_bf16=args.bf16, enable_tf32=args.tf32
    )

    print(f"Model: {args.model_name} | dtype={dtype} | attn_impl={args.attn_impl} | device={device}")
    if args.layer_whitelist is None:
        print("Capturing ALL layers")
    else:
        print(f"Capturing layers: {sorted(set(args.layer_whitelist))}")
    print(f"Max seq len: {args.max_seq_len}")

    totals = []
    t0 = time.time()
    for attr in args.only_attributes:
        if attr not in ATTRIBUTES:
            print(f"Skipping unknown attribute '{attr}'. Known: {ATTRIBUTES}")
            continue
        t_attr0 = time.time()
        stats = process_attribute(
            data=load_dataset(args.input_json),
            model=model,
            tokenizer=tokenizer,
            device=device,
            attribute=attr,
            out_dir=args.out_dir,
            max_seq_len=args.max_seq_len,
            layer_whitelist=args.layer_whitelist,
            max_conversations=args.max_conversations,
        )
        t_attr1 = time.time()
        n = stats["n_user_turns"]
        per_turn = (t_attr1 - t_attr0) / max(1, n)
        print(f"[{attr}] user_turns={n} | elapsed={t_attr1 - t_attr0:.2f}s | per_turn={per_turn:.4f}s")
        totals.append(n)
    t1 = time.time()
    print(f"Done. Total user_turns={sum(totals)} | total_elapsed={t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
