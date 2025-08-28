#!/usr/bin/env python3
# steer_utils.py
import json, torch, numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Please ensure that your responses are socially unbiased and positive in nature. "
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something"
)

# ---------------- model loader (CPU/GPU safe) ----------------
def load_model_and_tokenizer(model_id: str, device: str = "auto", local_files_only: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_files_only)

    kwargs = dict(local_files_only=local_files_only,
                  attn_implementation="eager",     # safest everywhere
                  low_cpu_mem_usage=True)

    device = (device or "auto").lower()
    if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
        kwargs.update(dict(torch_dtype=torch.float32, device_map={"": "cpu"}))
    else:
        kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tok, model

# --------------- render chat & find last user token ---------------
def render_chat_and_last_user_idx(tokenizer, messages: List[dict]) -> Tuple[str, int, int]:
    """
    Returns: rendered_text, last_user_token_index, n_input_tokens
    """
    try:
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        rendered = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"

    # locate last user substring; fall back to last prompt token
    user_text = messages[-1]["content"]
    start = rendered.rfind(user_text)
    end = start + len(user_text) if start != -1 else None

    enc = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()

    if end is not None:
        last_idx = -1
        for i, (_s, e) in enumerate(offsets):
            if e <= end: last_idx = i
            else: break
        if last_idx < 0: last_idx = len(offsets) - 1
    else:
        last_idx = len(offsets) - 1

    return rendered, int(last_idx), int(input_ids.shape[1])


def get_eos_ids(tok):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    # Many LLaMA chat templates use <|eot_id|> to end a turn
    for s in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid != -1:
                ids.add(tid)
        except Exception:
            pass
    return list(ids)


# --------------- find a transformer block (LLaMA/Mistral/NeoX/OPT) ---------------
def get_block(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):        # LLaMA/Mistral
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"): # OPT/GPT-J style
        return model.transformer.h[layer_idx]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):  # GPT-NeoX
        return model.gpt_neox.layers[layer_idx]
    raise AttributeError("Could not locate transformer blocks on this model.")

# --------------- constant-L2 control steering ---------------
class ConstantL2Steerer:
    """
    Translate hidden state by a fixed L2 amount along a unit control vector at a given layer.

    • Prompt pass: edit LAST USER TOKEN only.
    • Generation: edit the newest token at each step (up to max_steps).
    • step_scale:
        - 'sigma':  Δ = α * σ_layer * v̂      (σ_layer from your extractor's meta)
        - 'rms'  :  Δ = α * rms(h)  * v̂      (per-token RMS magnitude)
    """
    def __init__(self, model, layer_idx: int, v_unit_np: np.ndarray,
                 sigma_layer: float, alpha: float, last_user_idx: int,
                 step_scale: str = "sigma", max_steps: int | None = None):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.v_unit = torch.from_numpy(v_unit_np.astype("float32"))
        self.v_unit /= (torch.linalg.vector_norm(self.v_unit) + 1e-12)
        self.v_unit = self.v_unit.to(model.device, dtype=getattr(model, "dtype", torch.float32))

        self.sigma = float(sigma_layer)
        self.alpha = float(alpha)
        self.last_user_idx = int(last_user_idx)
        self.step_scale = step_scale
        self.max_steps = max_steps

        self._used_prompt_pass = False
        self._gen_steps = 0
        self._handle = None

    def _delta(self, hs_slice: torch.Tensor) -> torch.Tensor:
        # hs_slice: [B, d] for the position being edited
        if self.step_scale == "rms":
            scale = hs_slice.pow(2).mean(dim=-1, keepdim=True).sqrt()  # [B,1]
            return (self.alpha * scale) * self.v_unit                  # broadcast over B
        else:
            return (self.alpha * self.sigma) * self.v_unit

    def _hook(self, module, inputs, output):
        # output can be Tensor or (Tensor, ...)
        if isinstance(output, tuple):
            hs, *rest = output
        else:
            hs, rest = output, []

        hs = hs.clone()  # avoid in-place on shared tensor
        B, T, D = hs.shape

        if not self._used_prompt_pass:
            pos = max(0, min(self.last_user_idx, T - 1))
            hs[:, pos, :] = hs[:, pos, :] + self._delta(hs[:, pos, :])
            self._used_prompt_pass = True
        elif (self.max_steps is None) or (self._gen_steps < self.max_steps):
            hs[:, -1, :] = hs[:, -1, :] + self._delta(hs[:, -1, :])
            self._gen_steps += 1

        return (hs, *rest) if rest else hs

    def __enter__(self):
        block = get_block(self.model, self.layer_idx)
        self._handle = block.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

# --------------- load control vectors ---------------
def load_control_vector_and_sigma(index_path: str, vectors_dir: str, layer: int, bucket: str):
    meta = json.load(open(index_path, "r"))
    sigma = float(meta["sigma_per_layer"][layer])
    v = np.load(f"{vectors_dir}/v_layer{layer:02d}_{bucket}.npy").astype("float32")
    v /= (np.linalg.norm(v) + 1e-12)  # unit
    return v, sigma
