from datasets import load_dataset

dataset_name = "HuggingFaceH4/MATH-500"
output_path = "MATH-500_test.csv"

dataset = load_dataset(dataset_name, split="test")
width = max(3, len(str(len(dataset))))
new_ids = [f"{i:0{width}d}" for i in range(len(dataset))]
dataset = dataset.remove_columns("unique_id").add_column("unique_id", new_ids)
dataset.to_csv(output_path, index=False)
print(f"Saved {len(dataset)} rows to {output_path}")
