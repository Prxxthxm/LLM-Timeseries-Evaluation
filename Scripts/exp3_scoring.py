import json
import re
import requests
import pathlib

MODEL_NAME      = "google/gemma-2-9b-it"  
API_KEY         = "YOUR_API_KEY"
DATA_FILE       = "Data/randomized_dataset.jsonl"
ANSWER_KEY_FILE = "Data/answer_key.jsonl"
OUTPUT_FILE     = "Results/Gemma/Experiment_3.jsonl"

JUDGE_TEMPLATE = """You are an expert evaluator of time-series explanations.
You will receive:
- A time series (index: value pairs)
- A question about the time series
- A single candidate explanation

Your goal: classify the explanation using the criteria below.

PHASE 1 — UNDERSTAND THE QUESTION
Identify what the question is asking for:
- Anomaly / spike detection (sudden temporary deviation)
- Regime shift (sustained step change in level)
- Trend (gradual increase or decrease over many steps)
- Volatility change (increase or decrease in variance)
- Extreme value (maximum or minimum)
- Percentage or magnitude of change between specific points
- General behavior description

PHASE 2 — ANALYZE THE TIME SERIES
Identify the relevant evidence needed to answer the question.
For percentage changes: compute manually as ((new - old) / |old|) × 100.
CRITICAL: Do NOT trust the numbers in the explanation.
Verify all numeric claims yourself against the raw data.

PHASE 3 — EVALUATE THE EXPLANATION
Check the explanation against the data on these dimensions:
[A] Does it correctly identify the right pattern, location, and direction?
[B] Are its numeric claims correct within ±2% tolerance?
[C] Does it directly answer the question?
[D] Is its reasoning internally consistent?
[E] Does it introduce unsupported claims not derivable from the data?

PHASE 4 — ASSIGN LABEL
Based on your evaluation assign exactly one label:
0 = Completely incorrect (wrong pattern, wrong location, or contradicts the data)
1 = Correct reasoning and direction but contains numeric errors
2 = Fully correct (right pattern, right location, accurate numbers, answers the question)

Return ONLY valid JSON. No explanation. No extra text. No markdown fences.
{"label": int}
"""

def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)

def build_prompt(entry, explanation):
    return (
        JUDGE_TEMPLATE
        + "\n\n"
        + f"Time Series:\n{format_series(entry['series'])}\n\n"
        + f"Question:\n{entry['question']}\n\n"
        + f"Explanation:\n{explanation}\n"
    )

def call_llm(prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"API error: {data['error']}")
    return data["choices"][0]["message"]["content"]

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

def load_done_keys(filepath):
    done = set()
    if pathlib.Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add((obj["id"], obj["candidate_idx"]))
    return done

def main():
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    answer_key = {}
    with open(ANSWER_KEY_FILE) as f:
        for line in f:
            obj = json.loads(line)
            for idx, order_val in enumerate(obj["order"]):
                answer_key[(obj["id"], idx)] = 2 - order_val

    done_keys = load_done_keys(OUTPUT_FILE)
    print(f"Resuming — {len(done_keys)} candidate evaluations already processed.")

    correct = sum(1 for line in open(OUTPUT_FILE) if json.loads(line).get("correct")) \
              if pathlib.Path(OUTPUT_FILE).exists() else 0
    total = len(done_keys)

    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    entries = [e for e in entries if not e["id"].startswith("real_")]

    for entry in entries:
        entry_id = entry["id"]

        for candidate_idx, explanation in enumerate(entry["candidates"]):
            key = (entry_id, candidate_idx)

            if key in done_keys:
                continue

            try:
                raw_output = call_llm(build_prompt(entry, explanation))
            except Exception as e:
                print(f"HTTP error on {entry_id} candidate {candidate_idx}: {e}")
                try:
                    print(f"Response body: {e.response.text}")
                except Exception:
                    pass
                continue

            try:
                result = extract_json(raw_output)
            except Exception:
                print(f"Failed to parse JSON for {entry_id} candidate {candidate_idx}. Raw output: {(raw_output or '')[:200]}")
                continue

            predicted_label = result.get("label")
            if predicted_label is None:
                print(f"Missing label for {entry_id} candidate {candidate_idx}")
                continue

            ground_truth = answer_key.get(key)
            if ground_truth is None:
                continue

            is_correct = predicted_label == ground_truth

            with open(OUTPUT_FILE, "a") as out:
                out.write(json.dumps({
                    "id":            entry_id,
                    "model":         MODEL_NAME,
                    "candidate_idx": candidate_idx,
                    "predicted":     predicted_label,
                    "ground_truth":  ground_truth,
                    "correct":       is_correct,
                }) + "\n")

            if is_correct:
                correct += 1
            total += 1
            print(f"[{total}] Processed {entry_id} candidate {candidate_idx} | Predicted: {predicted_label} | GT: {ground_truth} | Correct: {is_correct}")

    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2f}" if total else "No samples processed")

if __name__ == "__main__":
    main()
