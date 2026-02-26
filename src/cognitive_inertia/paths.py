from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

MATH_DATASET_PATH = DATA_RAW_DIR / "MATH-500_test.csv"
BASELINE_RESPONSES_PATH = DATA_RESULTS_DIR / "results_single_response.csv"
BASELINE_SCORES_PATH = DATA_RESULTS_DIR / "results_scores.csv"

STUPID_TO_SMART_RESPONSES_PATH = DATA_RESULTS_DIR / "responses_from_stupid_to_smart.csv"
STUPID_TO_SMART_SCORES_PATH = DATA_RESULTS_DIR / "responses_from_stupid_to_smart_scores.csv"
SMART_TO_STUPID_RESPONSES_PATH = DATA_RESULTS_DIR / "responses_from_smart_to_stupid.csv"
SMART_TO_STUPID_SCORES_PATH = DATA_RESULTS_DIR / "responses_from_smart_to_stupid_scores.csv"

BASELINE_PLOT_PATH = PLOTS_DIR / "eval_plot.png"
ALL_IN_ONE_PLOT_PATH = PLOTS_DIR / "eval_all_in_one_plot.png"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
