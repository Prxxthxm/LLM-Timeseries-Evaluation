import numpy as np
import json
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from Utils.stats import percent_change, compute_zscore, slope

np.random.seed(67)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_FILE  = str(PROJECT_ROOT / "Data" / "synthetic.jsonl")
NEW_LENGTHS = [200, 300, 500]
NEW_PER_LENGTH = 10

TYPE_TO_PREFIX = {
    "linear_spike":            "ls",
    "seasonal_drop":           "sd",
    "structural_break":        "sb",
    "multi_metric_consistency":"mm",
    "relative_extremum":       "re",
    "mean_shift_query":        "ms",
    "volatility_shift":        "vs",
}

def generate_linear_spike(case_id, series_length=100):
    t = np.arange(series_length)
    y = 0.5 * t + np.random.normal(0, 1, series_length)
    spike_idx = np.random.randint(int(series_length * 0.2), int(series_length * 0.8))
    y[spike_idx] += 5 * np.std(y)
    pct = percent_change(y[spike_idx - 1], y[spike_idx])
    z   = compute_zscore(y, spike_idx)
    return {
        "id":   f"ls_{case_id}",
        "meta": {"type": "linear_spike", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Is there an anomaly? If yes, identify when and quantify the change.",
        "ground_truth": {
            "anomaly_index":  int(spike_idx),
            "percent_change": round(float(pct), 2),
            "z_score":        round(float(z), 2),
        },
    }

def generate_seasonal_drop(case_id, series_length=100):
    t = np.arange(series_length)
    y = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.2, series_length)
    drop_idx = np.random.randint(int(series_length * 0.2), int(series_length * 0.8))
    y[drop_idx] -= 3
    pct = percent_change(y[drop_idx - 1], y[drop_idx])
    z   = compute_zscore(y, drop_idx)
    return {
        "id":   f"sd_{case_id}",
        "meta": {"type": "seasonal_drop", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Is there an anomaly? If yes, identify when and quantify the change.",
        "ground_truth": {
            "anomaly_index":  int(drop_idx),
            "percent_change": round(float(pct), 2),
            "z_score":        round(float(z), 2),
        },
    }

def generate_structural_break(case_id, series_length=100):
    break_idx = series_length // 2
    y = np.ones(series_length) * 10
    y[break_idx:] += 10
    y += np.random.normal(0, 0.5, series_length)
    mean_before = float(np.mean(y[:break_idx]))
    mean_after  = float(np.mean(y[break_idx:]))
    pct_shift   = percent_change(mean_before, mean_after)
    return {
        "id":   f"sb_{case_id}",
        "meta": {"type": "structural_break", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Is there a structural change? If yes, quantify the shift.",
        "ground_truth": {
            "break_index":   break_idx,
            "mean_before":   round(mean_before, 2),
            "mean_after":    round(mean_after, 2),
            "percent_shift": round(pct_shift, 2),
        },
    }

def generate_relative_extremum(case_id, series_length=100):
    t = np.arange(series_length)
    y = 0.2 * t + np.random.normal(0, 1, series_length)
    lo = int(series_length * 0.2)
    hi = int(series_length * 0.8)
    spike_indices  = sorted(np.random.choice(range(lo, hi), 3, replace=False))
    spike_magnitudes = []
    for idx in spike_indices:
        mag = np.random.uniform(3, 6)
        y[idx] += mag
        spike_magnitudes.append(mag)
    best_pos   = int(np.argmax(spike_magnitudes))
    best_index = spike_indices[best_pos]
    best_mag   = spike_magnitudes[best_pos]
    return {
        "id":   f"re_{case_id}",
        "meta": {"type": "relative_extremum", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": "Identify the largest spike in the series and justify your answer with numeric evidence.",
        "ground_truth": {
            "spike_indices":         [int(i) for i in spike_indices],
            "largest_spike_index":   int(best_index),
            "largest_spike_magnitude": round(float(best_mag), 2),
        },
    }

def generate_multi_metric(case_id, series_length=100):
    t = np.arange(series_length)
    y = 0.3 * t + np.random.normal(0, 1, series_length)
    spike_idx = np.random.randint(int(series_length * 0.3), int(series_length * 0.7))
    y[spike_idx] += 5 * np.std(y)
    y[spike_idx+1:] += np.random.normal(0, 2, series_length - spike_idx - 1)
    pct        = percent_change(y[spike_idx - 1], y[spike_idx])
    z          = compute_zscore(y, spike_idx)
    var_before = np.var(y[:spike_idx])
    var_after  = np.var(y[spike_idx+1:])
    return {
        "id":   f"mm_{case_id}",
        "meta": {"type": "multi_metric_consistency", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": (
            "Analyze the series. Is there an anomaly? "
            "If yes, quantify the spike and describe any change in variance before and after it."
        ),
        "ground_truth": {
            "anomaly_index":  int(spike_idx),
            "percent_change": round(float(pct), 2),
            "z_score":        round(float(z), 2),
            "variance_before": round(float(var_before), 2),
            "variance_after":  round(float(var_after), 2),
        },
    }

def generate_mean_shift_query(case_id, series_length=100):
    t    = np.arange(series_length)
    base = 0.1 * t + np.random.normal(0, 0.5, series_length)
    shift = np.random.uniform(10, 20)
    mid   = series_length // 2
    base[mid:] += shift
    mean_first  = float(np.mean(base[:mid]))
    mean_second = float(np.mean(base[mid:]))
    med_first   = float(np.median(base[:mid]))
    med_second  = float(np.median(base[mid:]))
    pct_diff    = percent_change(mean_first, mean_second)
    return {
        "id":   f"ms_{case_id}",
        "meta": {"type": "mean_shift_query", "series_length": series_length},
        "series": [{"t": int(i), "value": float(base[i])} for i in range(series_length)],
        "question": (
            f"Compare the mean of the first half of the series (t=0 to t={mid - 1}) "
            f"with the mean of the second half (t={mid} to t={series_length - 1}). "
            f"Quantify the difference and state which half has a higher mean."
        ),
        "ground_truth": {
            "mean_first_half":   round(mean_first, 2),
            "mean_second_half":  round(mean_second, 2),
            "median_first_half": round(med_first, 2),
            "median_second_half":round(med_second, 2),
            "percent_difference":round(float(pct_diff), 2),
            "higher_half":       "second",
        },
    }

def generate_volatility_shift(case_id, series_length=100):
    break_idx  = np.random.randint(int(series_length * 0.3), int(series_length * 0.7))
    low_std    = np.random.uniform(0.5, 1.5)
    high_std   = np.random.uniform(4.0, 7.0)
    if np.random.rand() > 0.5:
        std_before, std_after = low_std, high_std
        higher_segment = "second"
    else:
        std_before, std_after = high_std, low_std
        higher_segment = "first"
    t     = np.arange(series_length)
    base  = 0.2 * t
    noise = np.concatenate([
        np.random.normal(0, std_before, break_idx),
        np.random.normal(0, std_after,  series_length - break_idx),
    ])
    y = base + noise
    return {
        "id":   f"vs_{case_id}",
        "meta": {"type": "volatility_shift", "series_length": series_length},
        "series": [{"t": int(i), "value": float(y[i])} for i in range(series_length)],
        "question": (
            "Does the volatility of the series change at any point? "
            "If yes, identify approximately when and compare the variability before and after."
        ),
        "ground_truth": {
            "break_index":              int(break_idx),
            "std_before":               round(float(np.std(y[:break_idx])), 2),
            "std_after":                round(float(np.std(y[break_idx:])), 2),
            "higher_volatility_segment": higher_segment,
        },
    }

GENERATORS = {
    "linear_spike":            generate_linear_spike,
    "seasonal_drop":           generate_seasonal_drop,
    "structural_break":        generate_structural_break,
    "multi_metric_consistency":generate_multi_metric,
    "relative_extremum":       generate_relative_extremum,
    "mean_shift_query":        generate_mean_shift_query,
    "volatility_shift":        generate_volatility_shift,
}

def get_existing_ids():
    """Read existing entries and return set of all existing IDs."""
    existing = set()
    if pathlib.Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    existing.add(case.get("id", ""))
    return existing

def get_next_case_num(type_str, existing_ids):
    """Get the next case number for a type that doesn't already exist."""
    prefix = TYPE_TO_PREFIX[type_str]
    num = 0
    while f"{prefix}_{num}" in existing_ids:
        num += 1
    return num

def main():
    pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    existing_ids = get_existing_ids()
    if existing_ids:
        print(f"Found {len(existing_ids)} existing entries")
    added = 0
    with open(OUTPUT_FILE, "a") as f:
        for type_str, gen_fn in GENERATORS.items():
            case_num = get_next_case_num(type_str, existing_ids)
            for length in NEW_LENGTHS:
                for _ in range(NEW_PER_LENGTH):
                    case_id = f"{TYPE_TO_PREFIX[type_str]}_{case_num}"
                    if case_id not in existing_ids:
                        case = gen_fn(case_num, series_length=length)
                        f.write(json.dumps(case) + "\n")
                        existing_ids.add(case_id)
                        print(f"  {case_id} | length={length}")
                        added += 1
                    case_num += 1
    print(f"\nAdded {added} new entries.")
    with open(OUTPUT_FILE) as f:
        total = sum(1 for l in f if l.strip())
    print(f"Total entries: {total}")


if __name__ == "__main__":
    main()
