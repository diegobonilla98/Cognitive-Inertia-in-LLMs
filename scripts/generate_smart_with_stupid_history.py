from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tqdm

from cognitive_inertia.llm import call_gpt52_smart_messages
from cognitive_inertia.paths import BASELINE_SCORES_PATH, SMART_TO_STUPID_RESPONSES_PATH

max_workers = 8
checkpoint_interval = 2
scores_path = BASELINE_SCORES_PATH
output_path = SMART_TO_STUPID_RESPONSES_PATH

history_examples = 10
smart_target_score = 100
dumb_history_max_score = 99

dataset = pd.read_csv(scores_path)

targets = dataset[dataset["smart_score"] == smart_target_score].copy()
stupid_low = dataset[
    (dataset["stupid_score"] <= dumb_history_max_score)
    & dataset["stupid_response"].notna()
    & dataset["problem"].notna()
].copy()
stupid_pool_by_subject = {subject: df for subject, df in stupid_low.groupby("subject")}


def select_low_score_history(target_row: pd.Series) -> pd.DataFrame:
    subject = target_row["subject"]
    subject_pool = stupid_pool_by_subject.get(subject)
    if subject_pool is None or subject_pool.empty:
        return pd.DataFrame()
    filtered_pool = subject_pool[subject_pool["unique_id"] != target_row["unique_id"]]
    if filtered_pool.empty:
        filtered_pool = subject_pool
    sample_size = min(history_examples, len(filtered_pool))
    return filtered_pool.nsmallest(sample_size, ["stupid_score", "unique_id"])


def process_row(row: pd.Series):
    unique_id = row["unique_id"]
    problem = row["problem"]
    subject = row["subject"]
    level = row["level"]
    answer = row["answer"]
    solution = row["solution"]

    history_df = select_low_score_history(row)
    if history_df.empty:
        print(f"Skipping {unique_id}: no low-score stupid history found for subject={subject}")
        return None

    messages = []
    history_ids = []
    for _, history_row in history_df.iterrows():
        messages.append({"role": "user", "content": str(history_row["problem"])})
        messages.append({"role": "assistant", "content": str(history_row["stupid_response"])})
        history_ids.append(str(history_row["unique_id"]))
    messages.append({"role": "user", "content": str(problem)})

    try:
        hacked_response = call_gpt52_smart_messages(messages)
    except Exception as e:
        print(f"Error on unique_id={unique_id}: {e}")
        return None

    return {
        "unique_id": unique_id,
        "problem": problem,
        "hacked_smart_response": hacked_response,
        "original_smart_response": row["smart_response"],
        "original_smart_score": row["smart_score"],
        "original_stupid_score": row["stupid_score"],
        "subject": subject,
        "level": level,
        "answer": answer,
        "solution": solution,
        "history_source": "same_subject_stupid_low_score",
        "history_size": len(history_ids),
        "history_unique_ids": "|".join(history_ids),
    }


results = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_row, row): row for _, row in targets.iterrows()}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        out = future.result()
        if out is not None:
            results.append(out)
            if len(results) % checkpoint_interval == 0:
                df = pd.DataFrame(results)
                df = df.sort_values("unique_id").reset_index(drop=True)
                df.to_csv(output_path, index=False)

df = pd.DataFrame(results)
df = df.sort_values("unique_id").reset_index(drop=True)
df.to_csv(output_path, index=False)
