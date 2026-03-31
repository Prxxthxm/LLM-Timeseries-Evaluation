# LLM-Timeseries-Evaluation

*Anonymous submission — under double-blind review at COLM 2026.*

Reference-free evaluation of time-series explanations using an LLM-as-a-Judge framework with structured rubric-based scoring.

---

## Overview

This repository contains the code, data, and results for the paper **"LLM-as-a-Judge for Time Series Explanations"**. We study large language models as both generators and evaluators of time-series explanations in a reference-free setting, using a rubric-guided prompting framework that conditions directly on raw numerical data.

---

## Repository Structure

```
LLM-Timeseries-Evaluation/
├── Data/       # Synthetic benchmark datasets (350 main instances + 100 anomaly detection instances)
├── Results/    # Experimental outputs and evaluation logs
├── Scripts/    # Experiment scripts (generation, ranking, scoring, anomaly detection)
├── Utils/      # Shared utilities for prompting, parsing, and metrics
└── LICENSE
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

```bash
# Explanation Generation
python Scripts/generation.py --model qwen3-8b --output Results/generation/

# Relative Ranking
python Scripts/ranking.py --model qwen3-8b --output Results/ranking/

# Independent Scoring
python Scripts/scoring.py --model qwen3-8b --output Results/scoring/

# Multi-Anomaly Detection
python Scripts/anomaly_detection.py --model qwen3-8b --output Results/anomaly/
```

All experiments are run zero-shot with structured JSON output. Refer to individual scripts for full argument options.

---

## License

MIT — see [LICENSE](LICENSE) for details.
