#!/usr/bin/env python3
"""
extract_last_user_token_states_jsonl.py  (emotion-filtering, robust last-token index)

- Multi-turn JSONL example:
  {
    "id": "conv_001",
    "meta": {"dimension": "emotion"},
    "dialog": [
      {"speaker":"user","text":"hi","user_attribute_name":"emotion","user_attribute_value":"joyful"},
      {"speaker":"assistant","text":"hello"},
      {"speaker":"user","text":"ok","user_attribute_name":"emotion","user_attribute_value":"content"}
    ]
  }

- Single-turn:
  - TXT: one prompt per line
  - JSONL: {"id":"...","prompt":"..."}

Outputs sharded arrays with last-user-token states for each layer.
Only multi-turn user turns with **emotion** labels are saved when --require_emotion is true (default).
"""

import argparse, json, sys, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Please ensure that your responses are socially unbiased and positive in nature. "
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something"
)

# --- YOUR TAXONOMY ---
GRANULAR_TO_BUCKET = {
    "joyful": "positive_high",
    "grateful": "positive_high",

    "hopeful": "positive_low",
    "optimistic": "positive_low",
    "relieved": "positive_low",
    "content": "positive_low",
    "curious": "positive_low",
    "thoughtful": "positive_low",
    "determined": "positive_low",
    "resolute": "positive_low",
    "planned": "positive_low",

    "calm": "calm_steady",
    "stable": "calm_steady",
    "steady": "calm_steady",
    "neutral": "calm_steady",

    "anxious": "worried",
    "nervous": "worried",
    "concerned": "worried",
    "uncertain": "worried",
    "conflicted": "worried",
    "overwhelmed": "worried",

    "angry": "neg_high",
    "frustrated": "neg_high",

    "sad": "neg_low",
    "resigned": "neg_low",
}

BUCKETS = {
    "positive_high",
    "positive_low",
    "calm_steady",
    "worried",
    "neg_high",
    "neg_low",
}


def load_tokenizer_and_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "auto" else None
    )
    if device != "auto":
        model = model.to(device)
    model.eval()
    return tok, model


def normalize_msg(m: Dict) -> Dict:
    role = m.get("role", m.get("speaker"))
    content = m.get("content", m.get("text"))
    if role not in ("user", "assistant", "system") or content is None:
        raise ValueError("Bad message record; need role/speaker and text/content.")
    out = {"role": role, "content": content}
    for k, v in m.items():
        if k not in ("role", "speaker", "content", "text"):
            out[k] = v
    return out


def iter_jsonl(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_single_txt(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            p = line.strip()
            if p:
                yield {"id": f"single_{i:06d}", "prompt": p}


def apply_chat_and_locate_last_user(tokenizer: AutoTokenizer, messages: List[Dict]):
    """
    Render with chat template; compute last user-token index WITHOUT relying on offset_mapping.
    We tokenize the substring up to the end of the user's text to count tokens directly.
    Returns: (input_ids[1,T], attention_mask[1,T], last_user_tok_idx:int)
    """
    if messages[-1]["role"] != "user":
        raise ValueError("Last message must be from 'user' for last-user-token extraction.")

    # Render with chat template (includes assistant prefix when add_generation_prompt=True)
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        sys.stderr.write(f"[WARN] apply_chat_template failed ({e}); using fallback template.\n")
        rendered = ""
        for m in messages:
            tag = m["role"]
            rendered += f"<|{tag}|>\n{m['content']}\n"
        rendered += "<|assistant|>\n"

    user_text = messages[-1]["content"]
    # Find user text span (favor exact match; fallback to stripped)
    start_char = rendered.rfind(user_text)
    if start_char == -1:
        start_char = rendered.rfind(user_text.strip())

    # Tokenize full rendered prompt (with assistant prefix)
    enc_full = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = enc_full["input_ids"]
    attn_mask = enc_full["attention_mask"]

    if start_char != -1:
        end_char = start_char + len(user_text)
        # Tokenize substring up to end_char to count tokens
        enc_up_to_user = tokenizer(rendered[:end_char], return_tensors="pt", add_special_tokens=False)
        last_idx = int(enc_up_to_user["input_ids"].shape[1]) - 1
    else:
        # Fallback: last non-pad token (before assistant generates)
        last_idx = int(attn_mask[0].sum().item()) - 1

    return input_ids, attn_mask, int(last_idx)


@torch.no_grad()
def extract_one(model, tokenizer, messages: List[Dict]):
    input_ids, attn_mask, last_idx = apply_chat_and_locate_last_user(tokenizer, messages)
    input_ids = input_ids.to(model.device)
    attn_mask = attn_mask.to(model.device)
    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    hs = out.hidden_states  # (embeddings + L blocks)
    # Collect post-block states 1..L at last_idx
    layers = [hs[l][0, last_idx, :].to(torch.float32).cpu().numpy() for l in range(1, len(hs))]
    hidden_layers = np.stack(layers, axis=0)  # [L, d]
    return hidden_layers, last_idx, int(input_ids.shape[1])


def save_shard(out_dir: Path, shard_idx: int,
               hidden_list: List[np.ndarray], last_idx_list: List[int],
               conv_id_list: List[str], turn_id_list: List[int], label_list: List[str]):
    np.savez_compressed(
        out_dir / f"shard_{shard_idx:05d}.npz",
        hidden=np.stack(hidden_list, 0).astype(np.float32),
        last_idx=np.asarray(last_idx_list, np.int32),
        turn_id=np.asarray(turn_id_list, np.int32)
    )
    with open(out_dir / f"shard_{shard_idx:05d}.meta.json", "w", encoding="utf-8") as f:
        json.dump({"conv_id": conv_id_list, "label": label_list}, f, ensure_ascii=False)


def median_layer_norms_from_shards(out_dir: Path):
    all_norms = None
    L = None
    for npz_path in sorted(out_dir.glob("shard_*.npz")):
        Z = np.load(npz_path)
        H = Z["hidden"]  # [N, L, d]
        if L is None:
            L = H.shape[1]
            all_norms = [[] for _ in range(L)]
        for l in range(L):
            all_norms[l].extend(np.linalg.norm(H[:, l, :], axis=1).tolist())
    return [float(np.median(np.asarray(all_norms[l], np.float64))) for l in range(L)]


# --------- emotion filtering helpers ---------

def to_bucket(label_str: Optional[str]) -> str:
    """Map any per-turn label to one of the allowed bucket names, or '' if unknown."""
    if not label_str:
        return ""
    lab = re.sub(r"[^\w\- ]+", "", str(label_str).lower()).strip()  # strip punctuation
    if lab in BUCKETS:
        return lab
    return GRANULAR_TO_BUCKET.get(lab, "")


def is_emotion_turn(rec: Dict, raw_msg: Dict, target_attr: str) -> bool:
    # Per-turn attribute-name flags
    name_keys = ("user_attribute_name", "attribute_name", "attr_name", "dimension")
    for k in name_keys:
        v = raw_msg.get(k)
        if v and str(v).lower().strip() == target_attr:
            return True
    # Record-level dimension (applies to entire dialog)
    meta = rec.get("meta") or {}
    dim = (meta.get("dimension") or meta.get("attr") or meta.get("label_type"))
    if dim and str(dim).lower().strip() == target_attr:
        return True
    # Fallback: treat as emotion if the value itself maps to a known bucket
    val = raw_msg.get("bucket") or raw_msg.get("user_attribute_value") or raw_msg.get("label")
    if to_bucket(val):
        return True
    return False


def get_emotion_bucket(rec: Dict, raw_msg: Dict) -> str:
    # Prefer explicit bucket; else map granular value
    val = raw_msg.get("bucket") or raw_msg.get("user_attribute_value") or raw_msg.get("label")
    return to_bucket(val)


# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--multi_jsonl", default=None)
    ap.add_argument("--single_jsonl", default=None)
    ap.add_argument("--single_txt", default=None)
    ap.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--require_emotion", type=lambda s: s.lower() != "false", default=True,
                    help="If true, only save multi-turn user turns labeled as the target attribute (emotion).")
    ap.add_argument("--target_attribute", default="emotion",
                    help="Name of the attribute to keep (default: 'emotion').")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_tokenizer_and_model(args.model, args.device)

    shard_idx = 0
    buf_hidden: List[np.ndarray] = []
    buf_last: List[int] = []
    buf_conv: List[str] = []
    buf_turn: List[int] = []
    buf_label: List[str] = []

    def flush():
        nonlocal shard_idx, buf_hidden, buf_last, buf_conv, buf_turn, buf_label
        if not buf_hidden:
            return
        save_shard(out_dir, shard_idx, buf_hidden, buf_last, buf_conv, buf_turn, buf_label)
        shard_idx += 1
        buf_hidden.clear(); buf_last.clear(); buf_conv.clear(); buf_turn.clear(); buf_label.clear()

    # ---- MULTI-TURN ----
    if args.multi_jsonl:
        for rec in tqdm(iter_jsonl(args.multi_jsonl), desc="multi-turn"):
            conv_id = str(rec.get("id", f"conv_{shard_idx:05d}"))
            dialog_raw = rec.get("dialog") or []
            dialog = [normalize_msg(m) for m in dialog_raw]
            for t, (m_norm, m_raw) in enumerate(zip(dialog, dialog_raw)):
                if m_norm["role"] != "user":
                    continue
                # Filter: keep only EMOTION-labeled user turns if required
                if args.require_emotion and not is_emotion_turn(rec, m_raw, args.target_attribute):
                    continue
                bucket = get_emotion_bucket(rec, m_raw) if is_emotion_turn(rec, m_raw, args.target_attribute) else ""
                if args.require_emotion and not bucket:
                    # labeled as emotion but no usable bucket/value → skip
                    continue

                msgs = []
                if args.system:
                    msgs.append({"role": "system", "content": args.system})
                msgs.extend(dialog[: t + 1])  # up to & including this user turn

                try:
                    H, last_idx, _T = extract_one(model, tokenizer, msgs)
                except Exception as e:
                    sys.stderr.write(f"[WARN] skip {conv_id} turn {t}: {e}\n")
                    continue

                buf_hidden.append(H.astype(np.float32))
                buf_last.append(last_idx)
                buf_conv.append(conv_id)
                buf_turn.append(t)
                buf_label.append(bucket)

                if len(buf_hidden) >= args.shard_size:
                    flush()
        flush()

    # ---- SINGLE-TURN ----
    singles = []
    if args.single_jsonl:
        singles.extend(iter_jsonl(args.single_jsonl))
    if args.single_txt:
        singles.extend(iter_single_txt(args.single_txt))
    if singles:
        for rec in tqdm(singles, desc="single-turn"):
            conv_id = str(rec.get("id", rec.get("name", f"single_{len(buf_hidden):06d}")))
            prompt = rec.get("prompt", rec.get("text"))
            if not prompt:
                sys.stderr.write(f"[WARN] single-turn {conv_id} missing 'prompt'/'text'\n")
                continue
            msgs = []
            if args.system:
                msgs.append({"role": "system", "content": args.system})
            msgs.append({"role": "user", "content": prompt})
            try:
                H, last_idx, _T = extract_one(model, tokenizer, msgs)
            except Exception as e:
                sys.stderr.write(f"[WARN] skip single {conv_id}: {e}\n")
                continue
            buf_hidden.append(H.astype(np.float32))
            buf_last.append(last_idx)
            buf_conv.append(conv_id)
            buf_turn.append(0)
            buf_label.append("")  # single-turn has no label
            if len(buf_hidden) >= args.shard_size:
                flush()
        flush()

    # ---- SIGMA ----
    sigmas = median_layer_norms_from_shards(out_dir)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "sigma_per_layer": sigmas,
            "model": args.model,
            "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer)),
            "system_prompt": args.system,
            "require_emotion": args.require_emotion,
            "target_attribute": args.target_attribute
        }, f, indent=2)
    print(f"Done. Wrote {len(list(out_dir.glob('shard_*.npz')))} shards to {out_dir}. "
          f"Per-layer sigma saved to meta.json.")


if __name__ == "__main__":
    main()
