#!/usr/bin/env python3
# llm_judge_steering_detection_async.py - Judge identifies which completion was steered

import os, csv, json, time, random, argparse, sys, re, unicodedata
import logging, asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================
#        SANITIZATION
# =========================
SMARTS = {
    "'": "'", "'": "'", "'": "'", "'": "'",
    '"': '"', '"': '"', '"': '"',
    "-": "-", "-": "-", "...": "...",
}
CTRL_RE = re.compile(r"[-\u001F\u007F]")

def to_utf8(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    for k, v in SMARTS.items():
        s = s.replace(k, v)
    s = CTRL_RE.sub("", s)
    try:
        return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"UTF-8 encoding error: {e}")
        return s

# =========================
#         PROMPTS
# =========================
JUDGE_SYSTEM = to_utf8(
    "You are an expert at detecting subtle differences in text responses. You will be shown two assistant responses "
    "to the same user prompt. One response is a standard baseline, and the other has been steered to respond as if "
    "the user is feeling a specific emotion. Your job is to identify which response was steered and which was baseline. "
    "Focus on tone, word choice, emotional sensitivity, and approach. Respond strictly in JSON format."
)

JUDGE_TEMPLATE = (
    "User Prompt:\n{prompt}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "One of these responses is steered to respond as if the user is feeling: {emotion_description}\n"
    "The other is a standard baseline response.\n\n"
    "Which response was steered toward the emotion described above?\n\n"
    "Rules:\n"
    "1) Choose either 'A' or 'B' for which response was steered\n"
    "2) Provide a brief rationale explaining your choice (1-2 sentences)\n"
    "3) Give a confidence score (0-1)\n\n"
    "Output strictly as JSON with this schema:\n"
    "{{\n"
    "  \"steered_response\": \"A\" or \"B\",\n"
    "  \"confidence\": number between 0 and 1,\n"
    "  \"rationale\": string\n"
    "}}\n"
)

# =========================
#        DATA MODEL
# =========================
@dataclass
class Row:
    prompt_id: str
    prompt: str
    completion: str

@dataclass
class DetectionTask:
    idx: int
    baseline_row: Row
    steered_row: Row
    emotion_description: str
    order: str  # "AB" means baseline=A, steered=B; "BA" means baseline=B, steered=A

def read_csv_rows(path: str, alpha_filter: Optional[float] = None) -> List[Row]:
    rows: List[Row] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            cols = set(r.fieldnames or [])
            pid_key = "prompt_id" if "prompt_id" in cols else ("id" if "id" in cols else None)
            prompt_key = "prompt" if "prompt" in cols else ("question" if "question" in cols else None)
            comp_key = "completion" if "completion" in cols else ("answer" if "answer" in cols else None)
            alpha_key = "alpha" if "alpha" in cols else None
            if not prompt_key or not comp_key:
                raise ValueError(f"{path}: need columns 'prompt' and 'completion' (or synonyms).")
            for i, row in enumerate(r):
                # Filter by alpha if specified
                if alpha_filter is not None and alpha_key and row.get(alpha_key):
                    try:
                        row_alpha = float(row[alpha_key])
                        if abs(row_alpha - alpha_filter) > 1e-6:  # Use small epsilon for float comparison
                            continue
                    except (ValueError, TypeError):
                        continue  # Skip rows with invalid alpha values
                
                pid = row[pid_key] if (pid_key and row.get(pid_key)) else str(i)
                rows.append(Row(
                    prompt_id=to_utf8(pid),
                    prompt=to_utf8(row[prompt_key]),
                    completion=to_utf8(row[comp_key])
                ))
    except Exception as e:
        logger.error(f"Error reading CSV {path}: {e}")
        raise
    return rows

def align_by_prompt_id(baseline: List[Row], steered: List[Row], emotion_description: str) -> List[DetectionTask]:
    """Align baseline and steered completions by prompt_id"""
    baseline_map = {row.prompt_id: row for row in baseline}
    aligned_tasks = []
    
    for i, steered_row in enumerate(steered):
        if steered_row.prompt_id in baseline_map:
            baseline_row = baseline_map[steered_row.prompt_id]
            # Randomize order to avoid bias
            order = "AB" if random.random() < 0.5 else "BA"
            aligned_tasks.append(DetectionTask(i, baseline_row, steered_row, emotion_description, order))
        else:
            logger.warning(f"No baseline found for prompt_id: {steered_row.prompt_id}")
    
    return aligned_tasks

# =========================
#       JSON HELPERS
# =========================
_KEY_SPACE_RE = re.compile(r"\s+")
def _normalize_key_string(k: str) -> str:
    k = str(k).replace("\r", "").replace("\n", "")
    k = k.strip()
    if (len(k) >= 2) and ((k[0] == k[-1] == '"') or (k[0] == k[-1] == "'")):
        k = k[1:-1].strip()
    k = _KEY_SPACE_RE.sub(" ", k).strip()
    return k

def _normalize_keys(obj):
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            kk = _normalize_key_string(k).lower()
            clean[kk] = _normalize_keys(v)
        return clean
    if isinstance(obj, list):
        return [_normalize_keys(x) for x in obj]
    return obj

def _first_json_obj(text: str):
    if not text:
        return None
    s = str(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
    if not m:
        m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        logger.warning("No JSON object found in response")
        return None
    frag = m.group(1) if m.lastindex else m.group(0)
    frag = re.sub(r",(\s*[}\]])", r"\1", frag)  # trailing comma clean
    try:
        obj = json.loads(frag)
    except Exception:
        frag2 = re.sub(r"[-\u001F\u007F]", frag)
        try:
            obj = json.loads(frag2)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None
    return _normalize_keys(obj)

def canonicalize_judgment(data: object) -> Optional[dict]:
    if data is None:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
        data = next(iter(data.values()))
    if not isinstance(data, dict):
        logger.warning("Invalid judgment data structure")
        return None

    kmap = {}
    for k, v in data.items():
        nk = _normalize_key_string(k).lower()
        kmap[nk] = v

    def pick(*cands, fuzzy=None):
        for c in cands:
            c_norm = _normalize_key_string(c).lower()
            if c_norm in kmap:
                return kmap[c_norm]
        if fuzzy:
            for k in list(kmap.keys()):
                if all(token in k for token in fuzzy):
                    return kmap[k]
        return None

    steered = pick("steered_response", "steered", "choice", "answer")
    conf = pick("confidence", "conf", "score")
    rat = pick("rationale", "reason", "explanation", "justification")

    def norm_choice(x):
        if isinstance(x, str):
            x_upper = x.strip().upper()
            if x_upper in ["A", "B"]:
                return x_upper
        logger.warning(f"Invalid choice value: {x}")
        return None

    def norm_float(x):
        try:
            return float(x)
        except Exception:
            try:
                xs = str(x).strip()
                if xs.endswith("%"):
                    return float(xs[:-1]) / 100.0
                return float(xs)
            except Exception:
                logger.warning("Invalid confidence value")
                return None

    return {
        "steered_response": norm_choice(steered),
        "confidence": norm_float(conf),
        "rationale": to_utf8(rat) if rat is not None else ""
    }

# =========================
#      ASYNC API HELPERS
# =========================
async def judge_detection_async(session: aiohttp.ClientSession, api_key: str, model: str, 
                               system_prompt: str, user_prompt: str, temperature: float = 1.0, 
                               max_retries: int = 3) -> tuple:
    """Return (data_dict, raw_text) on success, or (None, error_message) on parse failure."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": to_utf8(system_prompt)},
            {"role": "user", "content": to_utf8(user_prompt)}
        ]
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", 
                                  headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    raw = to_utf8(result["choices"][0]["message"]["content"])
                    data = canonicalize_judgment(_first_json_obj(raw))
                    if data is not None:
                        return data, raw
                    else:
                        return None, "parse_error"
                else:
                    error_text = await response.text()
                    logger.warning(f"Attempt {attempt + 1} failed: HTTP {response.status} - {error_text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.5 * (2 ** attempt))
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1.5 * (2 ** attempt))
    
    return None, "max_retries_exceeded"

async def process_detection_batch(session: aiohttp.ClientSession, api_key: str, model: str, 
                                 tasks: List[DetectionTask], temperature: float = 1.0,
                                 dry_run: bool = False) -> List[dict]:
    """Process a batch of detection tasks concurrently"""
    if dry_run:
        results = []
        for task in tasks:
            # Random guess for dry run
            guess = random.choice(["A", "B"])
            results.append({
                "task": task,
                "data": {"steered_response": guess, "confidence": 0.5, "rationale": "dry_run"},
                "raw": json.dumps({"steered_response": guess, "confidence": 0.5, "rationale": "dry_run"}),
                "error": None
            })
        return results
    
    async def process_single_task(task: DetectionTask):
        # Set up responses based on order
        if task.order == "AB":
            response_a = task.baseline_row.completion
            response_b = task.steered_row.completion
        else:  # "BA"
            response_a = task.steered_row.completion
            response_b = task.baseline_row.completion
        
        user_msg = JUDGE_TEMPLATE.format(
            prompt=to_utf8(task.baseline_row.prompt),
            response_a=to_utf8(response_a),
            response_b=to_utf8(response_b),
            emotion_description=to_utf8(task.emotion_description)
        )
        
        data, raw = await judge_detection_async(
            session, api_key, model, JUDGE_SYSTEM, user_msg, temperature
        )
        
        return {
            "task": task,
            "data": data,
            "raw": raw,
            "error": raw if data is None else None
        }
    
    # Process all tasks concurrently
    results = await asyncio.gather(*[process_single_task(task) for task in tasks])
    return results

# =========================
#          MAIN
# =========================
async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_file", required=True, help="CSV file with baseline completions")
    ap.add_argument("--steered_file", required=True, help="CSV file with steered completions")
    ap.add_argument("--emotion_description", required=True, help="Description of the emotion the steering targets")
    ap.add_argument("--out_csv", required=True, help="Output CSV file")
    ap.add_argument("--summary_json", default=None, help="Summary JSON file")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model to use for judging")
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature for model")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--max_comparisons", type=int, default=0, help="Maximum number of comparisons to evaluate")
    ap.add_argument("--batch_size", type=int, default=10, help="Number of concurrent API calls")
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable for API key")
    ap.add_argument("--dry_run", action="store_true", help="Dry run without API calls")
    ap.add_argument(
        "--on_parse_fail",
        choices=["warn", "skip"],
        default="warn",
        help="Behavior when JSON parsing fails: 'warn' = write a flagged row; 'skip' = drop comparison"
    )
    ap.add_argument("--baseline_alpha", type=float, default=0.0, help="Alpha filter for baseline (default 0.0)")
    ap.add_argument("--steered_alpha", type=float, default=1.0, help="Alpha filter for steered (default 1.0)")
    args = ap.parse_args()

    random.seed(args.seed)
    
    try:
        baseline_rows = read_csv_rows(args.baseline_file, args.baseline_alpha)
        steered_rows = read_csv_rows(args.steered_file, args.steered_alpha)
        aligned_tasks = align_by_prompt_id(baseline_rows, steered_rows, args.emotion_description)
    except Exception as e:
        logger.error(f"Failed to read or align CSV files: {e}")
        sys.exit(1)
    
    if args.max_comparisons > 0:
        aligned_tasks = aligned_tasks[:args.max_comparisons]

    api_key = None
    if not args.dry_run:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            logger.error(f"Missing API key in env var {args.api_key_env}")
            sys.exit(2)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    evaluated = 0
    parse_failed_ct = 0
    skipped_ct = 0
    correct_detections = 0

    fieldnames = [
        "comparison_idx", "prompt_id", "order", "judge_choice", "correct", "confidence", "rationale",
        "emotion_description", "prompt", "baseline_completion", "steered_completion", 
        "raw_json", "parse_failed", "error_message"
    ]

    # Process in batches
    connector = aiohttp.TCPConnector(limit=args.batch_size)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Process tasks in batches
            for i in range(0, len(aligned_tasks), args.batch_size):
                batch = aligned_tasks[i:i + args.batch_size]
                logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(aligned_tasks) + args.batch_size - 1)//args.batch_size} ({len(batch)} comparisons)")
                
                results = await process_detection_batch(
                    session, api_key, args.model, batch, args.temperature, args.dry_run
                )
                
                for result in results:
                    task = result["task"]
                    data = result["data"]
                    raw = result["raw"]
                    error = result["error"]
                    
                    parse_failed_flag = 0
                    if data is None:
                        parse_failed_ct += 1
                        err_msg = to_utf8(error or "parse_error")

                        if args.on_parse_fail == "skip":
                            skipped_ct += 1
                            logger.warning(f"Parse failed on comparison {task.idx}; skipping.")
                            continue

                        if args.on_parse_fail == "warn":
                            writer.writerow({
                                "comparison_idx": task.idx,
                                "prompt_id": task.baseline_row.prompt_id,
                                "order": task.order,
                                "judge_choice": "",
                                "correct": "",
                                "confidence": "",
                                "rationale": "",
                                "emotion_description": task.emotion_description,
                                "prompt": to_utf8(task.baseline_row.prompt),
                                "baseline_completion": to_utf8(task.baseline_row.completion),
                                "steered_completion": to_utf8(task.steered_row.completion),
                                "raw_json": to_utf8(raw or ""),
                                "parse_failed": 1,
                                "error_message": err_msg,
                            })
                            logger.warning(f"Parse failed on comparison {task.idx}; recorded as warning.")
                            continue

                    evaluated += 1
                    judge_choice = data.get("steered_response", None)
                    conf = data.get("confidence", None)
                    rat = to_utf8(data.get("rationale", ""))

                    # Determine if judge was correct
                    # If order is "AB": baseline=A, steered=B, so correct answer is "B"
                    # If order is "BA": baseline=B, steered=A, so correct answer is "A"
                    correct_answer = "B" if task.order == "AB" else "A"
                    is_correct = (judge_choice == correct_answer)
                    if is_correct:
                        correct_detections += 1

                    writer.writerow({
                        "comparison_idx": task.idx,
                        "prompt_id": task.baseline_row.prompt_id,
                        "order": task.order,
                        "judge_choice": judge_choice if judge_choice is not None else "",
                        "correct": 1 if is_correct else 0,
                        "confidence": conf if conf is not None else "",
                        "rationale": rat,
                        "emotion_description": task.emotion_description,
                        "prompt": to_utf8(task.baseline_row.prompt),
                        "baseline_completion": to_utf8(task.baseline_row.completion),
                        "steered_completion": to_utf8(task.steered_row.completion),
                        "raw_json": to_utf8("" if args.dry_run else (raw or "")),
                        "parse_failed": parse_failed_flag,
                        "error_message": "" if not parse_failed_flag else "parse_failed",
                    })

    denom = max(1, evaluated)
    accuracy = (correct_detections / denom) if denom else 0.0
    
    summary = {
        "baseline_file": args.baseline_file,
        "steered_file": args.steered_file,
        "emotion_description": args.emotion_description,
        "baseline_alpha": args.baseline_alpha,
        "steered_alpha": args.steered_alpha,
        "n_comparisons_input": len(aligned_tasks),
        "n_comparisons_evaluated": int(evaluated),
        "n_failed_parse": int(parse_failed_ct),
        "n_skipped": int(skipped_ct),
        "n_correct_detections": int(correct_detections),
        "detection_accuracy": accuracy,
        "detection_accuracy_percent": accuracy * 100,
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "on_parse_fail": args.on_parse_fail,
    }

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {args.out_csv}")
    if args.summary_json:
        logger.info(f"Wrote {args.summary_json}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING","UTF-8")
    os.environ.setdefault("LC_ALL","C.UTF-8")
    os.environ.setdefault("LANG","C.UTF-8")
    main() 