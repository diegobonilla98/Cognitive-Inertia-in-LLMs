from concurrent.futures import ThreadPoolExecutor, as_completed
from send_llm import call_gpt52_smart, call_4o_mini_stupid
import pandas as pd
import tqdm

max_workers = 8
checkpoint_interval = 2
checkpoint_path = "results_single_response.csv"

dataset = pd.read_csv("MATH-500_test.csv")


def process_row(row):
    problem = row["problem"]
    solution = row["solution"]
    answer = row["answer"]
    subject = row["subject"]
    level = row["level"]
    unique_id = row["unique_id"]
    try:
        smart_response = call_gpt52_smart(problem)
        stupid_response = call_4o_mini_stupid(problem)
    except Exception as e:
        print(f"Error: {e}")
        return None
    return {
        "smart_response": smart_response,
        "stupid_response": stupid_response,
        "subject": subject,
        "level": level,
        "answer": answer,
        "solution": solution,
        "unique_id": unique_id,
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
                df.to_csv(checkpoint_path, index=False)

df = pd.DataFrame(results)
df = df.sort_values("unique_id").reset_index(drop=True)
df.to_csv(checkpoint_path, index=False)
