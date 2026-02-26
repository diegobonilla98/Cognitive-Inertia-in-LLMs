from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from concurrent.futures import ThreadPoolExecutor, as_completed
from cognitive_inertia.llm import call_gpt52_smart
from cognitive_inertia.paths import BASELINE_RESPONSES_PATH, BASELINE_SCORES_PATH, MATH_DATASET_PATH
import pandas as pd
import tqdm

max_workers = 8
checkpoint_interval = 10
checkpoint_path = BASELINE_RESPONSES_PATH
output_path = BASELINE_SCORES_PATH
math_dataset_path = MATH_DATASET_PATH

SCORER_INSTRUCTIONS = "You are a math teacher. You are given a problem and a solution. You need to evaluate the solution and give a score between 0 and 100. Only answer with the score. Nothing more. It will be automatically parsed as int(your_response) so don't include any other text."

dataset = pd.read_csv(checkpoint_path)
if "problem" not in dataset.columns:
    math_df = pd.read_csv(math_dataset_path)[["unique_id", "problem"]]
    dataset = dataset.merge(math_df, on="unique_id", how="left")


def process_row(row):
    unique_id = row["unique_id"]
    problem = row["problem"]
    smart_response = row["smart_response"]
    stupid_response = row["stupid_response"]
    subject = row["subject"]
    level = row["level"]
    answer = row["answer"]
    solution = row["solution"]
    scores = {"smart": None, "stupid": None}
    for idx, response in enumerate([smart_response, stupid_response]):
        prompt = f"""
Problem: {problem}
Ground Truth Solution: {solution}
Ground Truth Answer: {answer}
Ground Truth Subject: {subject}
Ground Truth Level: {level}/5

-------------------------------

Student Response: {response}
"""
        try:
            score = call_gpt52_smart(prompt, SCORER_INSTRUCTIONS)
            score = int(score.strip())
        except Exception as e:
            print(f"Error: {e}")
            return None
        scores["smart" if idx == 0 else "stupid"] = score
    return {
        "unique_id": unique_id,
        "problem": problem,
        "smart_response": smart_response,
        "smart_score": scores["smart"],
        "stupid_response": stupid_response,
        "stupid_score": scores["stupid"],
        "subject": subject,
        "level": level,
        "answer": answer,
        "solution": solution,
    }


results = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_row, row): row for _, row in dataset.iterrows()}
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
