# Cognitive Inertia in LLMs

A compact research pipeline for testing **contextual inertia** in language models on MATH-500: can a model's quality be shifted up or down by injecting prior solved examples from another model profile?

## Repository Layout

```text
.
├── data/
│   ├── raw/                  # Source datasets
│   └── results/              # Generated responses, scores, and metrics
├── docs/                     # Study reports
├── outputs/plots/            # Visualization artifacts
├── prompts/                  # System/developer prompts
├── scripts/                  # Reproducible experiment scripts
└── src/cognitive_inertia/    # Shared library utilities
```

## Quick Start

1. Create environment and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn tqdm datasets python-dotenv openai
   ```
2. Set `OPENAI_API_KEY` in your environment (`.env` supported).
3. Run scripts from the repository root.

## End-to-End Workflow

### 1) Download dataset
```bash
python scripts/download_dataset.py
```

### 2) Generate baseline responses
```bash
python scripts/run_baseline_inference.py
```

### 3) Score baseline responses
```bash
python scripts/score_baseline_responses.py
```

### 4) Run history-hacking experiments
```bash
python scripts/generate_stupid_with_smart_history.py
python scripts/score_stupid_with_smart_history.py
python scripts/generate_smart_with_stupid_history.py
python scripts/score_smart_with_stupid_history.py
```

### 5) Visualize and analyze
```bash
python scripts/plot_baseline_scores.py
python scripts/plot_hacking_impact.py
python scripts/analyze_study_results.py
```

## Why this structure is cleaner

- **Separation of concerns:** data, code, outputs, and docs are now isolated.
- **Path centralization:** file locations are managed in `src/cognitive_inertia/paths.py`.
- **Naming clarity:** script names now describe actions and directionality explicitly.

## Future Steps (high-value, low-noise)

1. **Calibrated scoring:** add a second independent grader model and agreement tracking.
2. **Intervention policy learning:** train a small classifier to decide when history injection helps.
3. **Counterfactual history controls:** use semantically similar but incorrect traces to separate style imitation from reasoning transfer.
4. **Temporal robustness:** repeat runs over multiple days/model snapshots to quantify drift.
5. **Mechanistic probes:** inspect token-level uncertainty shifts before/after history injection.

## Insight

The most interesting pattern in this setup is asymmetry: weak-model trajectories are more pliable than strong-model trajectories. That suggests context-history conditioning behaves more like a *performance prior* on uncertain problem regions than a universal behavior override.
