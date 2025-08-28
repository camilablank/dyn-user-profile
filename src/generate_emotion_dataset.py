#!/usr/bin/env python3
"""
Generates a user-assistant multiturn conversation dataset with annotated user emotions.

API/Model: OpenAI Chat Completions (gpt-4o by default)
Output: JSONL where each line is a single dialog sample with rich metadata.

1) export OPENAI_API_KEY=sk-...
2) export OPENAI_PROJECT=proj-...
3) to run, python generate_emotion_dataset.py --out emotion_dataset.jsonl
   (use --n_samples, --styles, --min_turns/--max_turns to adjust)
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any

# ---- OpenAI client (Chat Completions) ----
def _try_set_utf8_stdio():
    """Best-effort: set UTF-8 env hints; fall back gracefully."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "en_US.UTF-8")
    os.environ.setdefault("LC_ALL", "en_US.UTF-8")
    try:
        import sys
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("Please install the OpenAI Python SDK (pip install openai>=1.0.0).") from e

# ----------------------- Config dataclasses -----------------------

@dataclass
class GenConfig:
    model: str = "gpt-4o"
    temperature: float = 0.9
    seed: int = 42
    min_turns: int = 10
    max_turns: int = 16
    min_change_span: int = 4
    max_change_span: int = 6
    system_preamble: str = (
        "You are simulating dialogues between a USER and an ASSISTANT.\n"
        "Output a single JSON object (no markdown) with keys:\n"
        "  - 'meta': run metadata\n"
        "  - 'persona': concise USER persona\n"
        "  - 'narrative': 1-2 paragraph premise\n"
        "  - 'dialog': list of turn objects, each:\n"
        "        { 'speaker': 'user'|'assistant', 'text': str,\n"
        "          'user_attribute_value': str|None }\n"
        "Notes:\n"
        "  • For emotion: fill 'user_attribute_value' every turn with the USER's emotion at that moment.\n"
        "Format strictly as JSON (no markdown fences)."
    )

# ----------------------- Prompt builders --------------------------

EMOTION_VALUES = [
    "joyful", "grateful", "hopeful", "optimistic", "relieved", "content", 
    "curious", "thoughtful", "determined", "resolute", "planned", "calm", 
    "stable", "steady", "neutral", "anxious", "nervous", "concerned", 
    "uncertain", "conflicted", "overwhelmed", "angry", "frustrated", 
    "sad", "resigned"
]

DEFAULT_STYLES = ["neutral", "casual", "formal", "supportive", "clinical", "humorous"]

def build_style_instruction(style: str) -> str:
    return (
        f"Writing style: {style}. Keep the assistant tone consistent across the dialog."
    )

def build_change_schedule(values: List[str], span_len: int) -> List[str]:
    """Create a plausible mini-trajectory of attribute values with internal change."""
    start = random.choice(values)
    traj = [start]
    for _ in range(span_len - 1):
        nxt = random.choice([v for v in values if v != traj[-1]])
        traj.append(nxt)
    return traj

def build_emotion_instruction(cfg: GenConfig) -> Dict[str, Any]:
    """Return (meta, user_msg) parts specific to the emotion dimension."""
    span_len = random.randint(cfg.min_change_span, cfg.max_change_span)
    schedule = build_change_schedule(EMOTION_VALUES, span_len)

    meta = {
        "dimension": "emotion",
        "change_span_length": span_len,
        "change_schedule": schedule,
    }
    instruction = (
        f"TARGET ATTRIBUTE: emotion.\n"
        f"Include a contiguous span of {span_len} USER turns where the USER's emotion shifts along: {schedule}. "
        "Place it anywhere in the dialog. Outside the span the attribute may be stable or drift slightly. "
        "Annotate *every* turn with 'user_attribute_value' as the USER's emotion at that moment."
    )
    return meta, instruction

# ----------------------- Generation logic -------------------------

def mk_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    project_id = os.environ.get("OPENAI_PROJECT")
    if project_id:
        print(f"Using OpenAI project: {project_id}")
        return OpenAI(project=project_id)
    return OpenAI()

def _recover_json(client: OpenAI, messages, seed: int, cfg: GenConfig):
    """Attempt a one-shot recovery with stricter JSON instruction."""
    recover_messages = messages + [
        {"role": "user", "content": "Your last output was not valid JSON. Re-output the SAME content as strict JSON only, no markdown fences."}
    ]
    resp2 = client.chat.completions.create(
        model=cfg.model,
        temperature=0.2,
        messages=recover_messages,
        seed=seed,
        response_format={"type": "json_object"},
    )
    out2 = resp2.choices[0].message.content
    return json.loads(out2)

def generate_single_dialog(
    client: OpenAI,
    cfg: GenConfig,
    style: str,
    min_turns: int,
    max_turns: int,
    seed: int,
) -> Dict[str, Any]:
    n_turns = random.randint(min_turns, max_turns)
    meta, attr_instruction = build_emotion_instruction(cfg)

    persona_hint = random.choice([
        "A 28-year-old barista who paints on weekends and is applying for art school.",
        "A mid-career data analyst juggling night classes in statistics.",
        "A high-school teacher preparing students for a debate tournament.",
        "A freelance travel blogger planning a series on remote islands.",
        "A new parent optimizing sleep routines with limited time.",
        "A small business owner trying to improve an online storefront.",
        "A graduate student preparing a literature review on renewable energy."
    ])
    narrative_hint = random.choice([
        "They've set aside 30 minutes to get practical, step-by-step help today.",
        "They are frustrated by slow progress and want momentum without fluff.",
        "They're exploring ideas and open to creative, lateral suggestions.",
        "They need help drafting something concrete by the end of the chat.",
        "They are practicing a tricky conversation they expect to have soon.",
        "They're comparing two approaches and need to pick one rationally."
    ])

    user_message = f"""
{attr_instruction}

Persona: {persona_hint}
Narrative premise: {narrative_hint}

Dialog requirements:
- Total turns (user+assistant combined): {n_turns} to {n_turns+2}. Alternate strictly, starting with USER.
- Keep responses concise but natural; avoid monologues.
- For emotion: fill 'user_attribute_value' on every turn with the USER's emotion at that moment (assistant turns infer the user's emotional state).
- Avoid meta-talk about being an LLM. The assistant is helpful and realistic.
- Keep content domain-general; no risky medical/legal advice.

Style control:
{build_style_instruction(style)}
""".strip()

    messages = [
        {"role": "system", "content": cfg.system_preamble},
        {"role": "user", "content": user_message},
    ]

    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        messages=messages,
        seed=seed,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content

    try:
        cleaned_raw = raw.encode('utf-8', errors='ignore').decode('utf-8')
        data = json.loads(cleaned_raw)
    except (json.JSONDecodeError, UnicodeError):
        data = _recover_json(client, messages, seed, cfg)

    # Attach runtime meta
    data["meta"] = {
        **data.get("meta", {}),
        **meta,
        "style": style,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": seed,
        "timestamp": int(time.time()),
    }

    return data

# ----------------------- Main -------------------------

def main():
    _try_set_utf8_stdio()
    parser = argparse.ArgumentParser(description="Generate emotion-focused dialog dataset using OpenAI API")
    parser.add_argument("--out", type=str, default="emotion_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Chat Completions model")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--append", action="store_true", help="Append instead of overwrite")
    parser.add_argument("--styles", type=str, default=",".join(DEFAULT_STYLES))
    parser.add_argument("--min_turns", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = GenConfig(
        model=args.model,
        temperature=args.temperature,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    client = mk_client()

    print(f"Generating {args.n_samples} emotion dialogs -> {args.out}")
    print(f"Each dialog will have {args.min_turns}-{args.max_turns} turns")
    print(f"Emotion change spans: {cfg.min_change_span}-{cfg.max_change_span} turns")
    print(f"Available emotions: {len(EMOTION_VALUES)} values")
    print(f"Styles: {styles}")
    written = 0

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mode = "a" if args.append else "w"
    if args.append and os.path.exists(args.out):
        print(f"Appending to existing file: {args.out}")
    else:
        print(f"Creating new file: {args.out}")

    with open(args.out, mode, encoding="utf-8") as f:
        for i in range(args.n_samples):
            style = random.choice(styles)
            seed = random.randint(1, 1_000_000)
            try:
                sample = generate_single_dialog(
                    client=client,
                    cfg=cfg,
                    style=style,
                    min_turns=args.min_turns,
                    max_turns=args.max_turns,
                    seed=seed,
                )
                # Light validation
                assert isinstance(sample.get("dialog"), list), "Missing dialog list"
                span_len = int(sample["meta"].get("change_span_length", 0))
                if not (cfg.min_change_span <= span_len <= cfg.max_change_span):
                    print(f"[warn] span_len out of range: {span_len}")

                # Show emotion schedule for first few samples
                if i < 3:
                    schedule = sample["meta"].get("change_schedule", [])
                    emotion_str = " → ".join(schedule)
                    print(f"Sample {i + 1}: {emotion_str}")

                # Write
                json_str = json.dumps(sample, ensure_ascii=False)
                f.write(json_str + "\n")
                written += 1
                print(f"[OK] Sample #{i+1} (style={style}, turns={len(sample.get('dialog', []))}, span={span_len})")

            except Exception as e:
                # Printing must never cause us to drop a sample
                msg = f"[WARN] Sample #{i+1} raised: {e!r}"
                try:
                    print(msg)
                except Exception:
                    # absolute fallback
                    import sys
                    sys.stdout.buffer.write((msg + "\n").encode("utf-8", "replace"))
                continue

    print(f"\n✅ Done. Wrote {written} / {args.n_samples} dialogs to {args.out}")
    print(f"📁 Output saved to: {args.out}")
    print(f"🎭 Each sample contains emotion transitions with rich metadata")

if __name__ == "__main__":
    main() 