from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import pandas as pd
import tqdm

from cognitive_inertia.llm import call_4o_mini_stupid_messages
from cognitive_inertia.paths import BASELINE_SCORES_PATH, STUPID_TO_SMART_RESPONSES_PATH

max_workers = 8
checkpoint_interval = 2
scores_path = BASELINE_SCORES_PATH
output_path = STUPID_TO_SMART_RESPONSES_PATH

history_examples = 10
failure_threshold = 100
random_seed = 42

dataset = pd.read_csv(scores_path)

targets = dataset[dataset["stupid_score"] < failure_threshold].copy()
smart_perfect = dataset[
    (dataset["smart_score"] == 100)
    & dataset["smart_response"].notna()
    & dataset["problem"].notna()
].copy()
smart_pool_by_subject = {subject: df for subject, df in smart_perfect.groupby("subject")}


def sample_history_examples(target_row: pd.Series) -> pd.DataFrame:
    subject = target_row["subject"]
    subject_pool = smart_pool_by_subject.get(subject)
    if subject_pool is None or subject_pool.empty:
        return pd.DataFrame()
    filtered_pool = subject_pool[subject_pool["unique_id"] != target_row["unique_id"]]
    if filtered_pool.empty:
        filtered_pool = subject_pool
    sample_size = min(history_examples, len(filtered_pool))
    local_rng = random.Random(random_seed + int(target_row["unique_id"]))
    sampled_idx = local_rng.sample(list(filtered_pool.index), k=sample_size)
    return filtered_pool.loc[sampled_idx]


def process_row(row: pd.Series):
    unique_id = row["unique_id"]
    problem = row["problem"]
    subject = row["subject"]
    level = row["level"]
    answer = row["answer"]
    solution = row["solution"]

    history_df = sample_history_examples(row)
    if history_df.empty:
        print(f"Skipping {unique_id}: no smart perfect history found for subject={subject}")
        return None

    messages = []
    history_ids = []
    for _, history_row in history_df.iterrows():
        messages.append({"role": "user", "content": str(history_row["problem"])})
        messages.append({"role": "assistant", "content": str(history_row["smart_response"])})
        history_ids.append(str(history_row["unique_id"]))
    messages.append({"role": "user", "content": str(problem)})

    try:
        hacked_response = call_4o_mini_stupid_messages(messages)
    except Exception as e:
        print(f"Error on unique_id={unique_id}: {e}")
        return None

    return {
        "unique_id": unique_id,
        "problem": problem,
        "hacked_stupid_response": hacked_response,
        "original_stupid_response": row["stupid_response"],
        "original_stupid_score": row["stupid_score"],
        "original_smart_score": row["smart_score"],
        "subject": subject,
        "level": level,
        "answer": answer,
        "solution": solution,
        "history_source": "same_subject_smart_score_100",
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
