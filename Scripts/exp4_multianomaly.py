import json
import re
import requests
import pathlib

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "qwen/qwen3-8b"   
API_KEY     = "YOUR_API_KEY"
DATA_FILE   = "Data/synthetic_multianomaly.jsonl"
OUTPUT_FILE = "Results/Qwen/Experiment_4.jsonl"
INDEX_TOLERANCE = 2   # ±2 index positions counts as a match
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are an expert analyst of time-series data.
You will receive a time series and a question about anomalies in it.

PHASE 1 — SCAN THE SERIES
Read through the entire series carefully. Look for points where the value
deviates sharply from the surrounding trend and then returns to normal.
Each such point is a separate anomaly. Do not assume a fixed number.

PHASE 2 — IDENTIFY EACH ANOMALY
For each anomaly you find:
- Record the exact time index t
- Compute the percentage change from the previous point: ((value[t] - value[t-1]) / |value[t-1]|) × 100
Do NOT skip anomalies. Do NOT merge nearby points into one.

PHASE 3 — COUNT AND REPORT
Count all anomalies you found. Report each one individually.

Return ONLY valid JSON. No explanation. No extra text. No markdown fences.
{
  "anomaly_count": int,
  "anomalies": [
    {"index": int, "percent_change": float},
    ...
  ]
}
"""

def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)

def build_prompt(entry):
    return (
        PROMPT_TEMPLATE
        + "\n\n"
        + f"Time Series:\n{format_series(entry['series'])}\n\n"
        + f"Question:\n{entry['question']}\n"
    )

def call_llm(prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            #"response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"API error: {data['error']}")
    content = data["choices"][0]["message"]["content"]
    if content is None:
        raise ValueError("API returned None content")
    return content

def extract_json(text):
    if text is None:
        raise ValueError("Input text is None")
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Could not parse JSON")

def load_done_ids(filepath):
    done = set()
    if pathlib.Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done

def compute_f1(predicted_indices, gt_indices, tolerance=INDEX_TOLERANCE):
    """
    Match predicted indices to ground truth indices within ±tolerance.
    Each GT index can only be matched once. Returns precision, recall, F1.
    """
    matched_gt = set()
    true_positives = 0

    for pred in predicted_indices:
        for j, gt in enumerate(gt_indices):
            if j not in matched_gt and abs(pred - gt) <= tolerance:
                true_positives += 1
                matched_gt.add(j)
                break

    precision = true_positives / len(predicted_indices) if predicted_indices else 0.0
    recall    = true_positives / len(gt_indices) if gt_indices else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 3), round(recall, 3), round(f1, 3)

def main():
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(OUTPUT_FILE)
    print(f"Resuming — {len(done_ids)} entries already processed.")

    # Running metrics
    count_correct = 0
    f1_sum = 0.0
    total = len(done_ids)

    # Reload existing results for running totals
    if pathlib.Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("count_correct"):
                    count_correct += 1
                f1_sum += r.get("f1", 0.0)

    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    for entry in entries:
        entry_id = entry["id"]

        if entry_id in done_ids:
            continue

        prompt = build_prompt(entry)

        try:
            raw = call_llm(prompt)
        except Exception as e:
            print(f"HTTP error on {entry_id}: {e}")
            try:
                print(f"Response body: {e.response.text}")
            except Exception:
                pass
            continue

        try:
            result = extract_json(raw)
        except Exception:
            print(f"Failed to parse JSON for {entry_id}. Raw: {(raw or '')}")
            continue

        pred_count   = result.get("anomaly_count", 0)
        pred_anomalies = result.get("anomalies", [])
        pred_indices = [a["index"] for a in pred_anomalies if "index" in a]

        gt            = entry["ground_truth"]
        gt_count      = gt["anomaly_count"]
        gt_indices    = [a["index"] for a in gt["anomalies"]]
        series_length = entry["meta"]["series_length"]

        count_match = pred_count == gt_count
        precision, recall, f1 = compute_f1(pred_indices, gt_indices)

        record = {
            "id":             entry_id,
            "model":          MODEL_NAME,
            "series_length":  series_length,
            "gt_count":       gt_count,
            "pred_count":     pred_count,
            "count_correct":  count_match,
            "gt_indices":     gt_indices,
            "pred_indices":   pred_indices,
            "precision":      precision,
            "recall":         recall,
            "f1":             f1,
        }

        with open(OUTPUT_FILE, "a") as out:
            out.write(json.dumps(record) + "\n")

        if count_match:
            count_correct += 1
        f1_sum += f1
        total += 1

        print(
            f"[{total}] {entry_id} | len={series_length} | "
            f"GT={gt_count} Pred={pred_count} count={'✓' if count_match else '✗'} | "
            f"F1={f1:.3f}"
        )

    if total == 0:
        print("No samples processed.")
        return

    print(f"\n{'='*50}")
    print(f"Total samples:      {total}")
    print(f"Count accuracy:     {count_correct}/{total} = {count_correct/total:.2f}")
    print(f"Mean F1 (indices):  {f1_sum/total:.3f}")

    # Breakdown by series length
    if pathlib.Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            all_results = [json.loads(l) for l in f if l.strip()]

        from collections import defaultdict
        by_length = defaultdict(list)
        for r in all_results:
            by_length[r["series_length"]].append(r)

        print(f"\nBreakdown by series length:")
        for sl in sorted(by_length.keys()):
            group = by_length[sl]
            cc = sum(1 for r in group if r["count_correct"])
            mf1 = sum(r["f1"] for r in group) / len(group)
            print(f"  length={sl:4d}: {len(group):2d} samples | "
                  f"count_acc={cc/len(group):.2f} | mean_F1={mf1:.3f}")

if __name__ == "__main__":
    main()