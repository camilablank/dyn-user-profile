#!/usr/bin/env python3
# steer_alpha_layers.py
import argparse, csv
from steer_utils import (
    load_model_and_tokenizer,
    render_chat_and_last_user_idx,
    load_control_vector_and_sigma,
    ConstantL2Steerer,
    get_eos_ids,
    DEFAULT_SYSTEM
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--vectors_index", required=True, help="CONTROL probes index.json")
    ap.add_argument("--vectors_dir", required=True, help="CONTROL probes vectors/ dir")
    ap.add_argument("--bucket", required=True, help="e.g., worried")

    ap.add_argument("--layers", required=True, help="e.g., 13,17,20")
    ap.add_argument("--alphas", required=True, help="e.g., 0.05,0.1,0.2")
    ap.add_argument("--step_scale", default="sigma", choices=["sigma","rms"])
    ap.add_argument("--max_steps", type=int, default=40, help="max generated tokens to steer")

    ap.add_argument("--single_txt", required=True, help="prompts.txt (one user question per line)")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    tok, model = load_model_and_tokenizer(args.model, device=args.device, local_files_only=args.local_files_only)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    prompts = [ln.strip() for ln in open(args.single_txt, "r", encoding="utf-8") if ln.strip()]

    FIELDNAMES = [
        "prompt_id","prompt","layer","alpha","bucket",
        "step_scale","max_steps","max_new_tokens",
        "n_new_tokens","stopped_by_eos",   # <-- add these
        "completion"
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",          # ignores any unexpected keys
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
            escapechar="\\",
        )
        w.writeheader()

        for pid, prompt in enumerate(prompts):
            messages = [
                {"role":"system","content":DEFAULT_SYSTEM},
                {"role":"user","content":prompt},
            ]
            rendered, last_user_idx, n_input_tokens = render_chat_and_last_user_idx(tok, messages)
            inputs = tok(rendered, return_tensors="pt").to(model.device)

            for L in layers:
                v_unit, sigma = load_control_vector_and_sigma(args.vectors_index, args.vectors_dir, L, args.bucket)

                for a in alphas:
                    steerer = ConstantL2Steerer(
                        model, layer_idx=L, v_unit_np=v_unit, sigma_layer=sigma, alpha=a,
                        last_user_idx=last_user_idx, step_scale=args.step_scale, max_steps=args.max_steps
                    )
                    with steerer:
                        eos_ids = get_eos_ids(tok)
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,  # set this high enough (e.g., 160)
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=eos_ids,                # <-- accept EOS OR EOT
                            pad_token_id=(eos_ids[0] if eos_ids else tok.eos_token_id),
                        )
                    # decode only the newly generated tokens (assistant continuation)
                    new_tokens = out[0, n_input_tokens:]
                    n_in = inputs["input_ids"].shape[1]
                    gen = out[0, n_in:]
                    stopped_by_eos = any(t in set(eos_ids) for t in gen.tolist()) if eos_ids else False
                    completion = tok.decode(new_tokens, skip_special_tokens=True)

                    w.writerow({
                        "prompt_id": pid,
                        "prompt": prompt,
                        "layer": L,
                        "alpha": a,
                        "bucket": args.bucket,
                        "step_scale": args.step_scale,
                        "max_steps": args.max_steps,
                        "max_new_tokens": args.max_new_tokens,
                        "completion": completion.strip(),
                        "n_new_tokens": int(gen.shape[0]),
                        "stopped_by_eos": int(stopped_by_eos),
                    })

if __name__ == "__main__":
    main()
