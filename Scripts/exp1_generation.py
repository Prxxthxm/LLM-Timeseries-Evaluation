import json
import re
import requests
import pathlib

MODEL_NAME  = "qwen/qwen3-8b"
API_KEY     = "YOUR_API_KEY"
DATA_FILE   = "Data/synthetic.jsonl"
OUTPUT_FILE = "Results/Qwen/Experiment_1.jsonl" 

GENERATE_TEMPLATE = """You are an expert analyst of time-series data.
You will receive:
- A time series (index: value pairs)
- A question about the time series

Your goal: write a concise explanation that directly answers the question.

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
Inspect the data carefully to find evidence relevant to the question.
For percentage changes: compute manually as ((new - old) / |old|) × 100.
For anomalies: identify the exact index t and verify the value deviates sharply and returns afterward.
For regime shifts: identify where the mean level changes and stays changed.
For trends: assess direction and approximate rate across the full series.

PHASE 3 — WRITE THE EXPLANATION
Write a single concise explanation that:
- Directly answers the question
- References the specific time index and values from the data
- Includes any relevant numeric quantities (percentage change, magnitude, etc.)
- Does not introduce claims not supported by the data

Return ONLY valid JSON. No extra text. No markdown fences.
{"explanation": "your explanation here"}
"""

def format_series(series, precision=2):
    return ", ".join(f"{p['t']}:{round(p['value'], precision)}" for p in series)

def build_prompt(entry):
    return (
        GENERATE_TEMPLATE
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
    return data["choices"][0]["message"]["content"]

def extract_json(text):
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

def main():
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(OUTPUT_FILE)
    print(f"Resuming — {len(done_ids)} entries already processed.")
    total = len(done_ids)

    with open(DATA_FILE) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    for entry in entries:
        entry_id = entry["id"]

        if entry_id in done_ids:
            continue

        try:
            raw_output = call_llm(build_prompt(entry))
        except Exception as e:
            print(f"HTTP error on {entry_id}: {e}")
            try:
                print(f"Response body: {e.response.text}")
            except Exception:
                pass
            continue

        try:
            result = extract_json(raw_output)
        except Exception:
            print(f"Failed to parse JSON for {entry_id}. Raw output: {raw_output[:200]}")
            continue

        explanation = result.get("explanation")
        if not explanation:
            print(f"Missing explanation for {entry_id}")
            continue

        with open(OUTPUT_FILE, "a") as out:
            out.write(json.dumps({
                "id":          entry_id,
                "model":       MODEL_NAME,
                "question":    entry["question"],
                "explanation": explanation,
            }) + "\n")

        total += 1
        print(f"[{total}] Processed {entry_id}")

    print(f"\nDone. {total} explanations generated.")

if __name__ == "__main__":
    main()
