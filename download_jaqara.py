from datasets import load_dataset
import os

ds = load_dataset("hotchpotch/JQaRA", split="test")

# 保存先ディレクトリ
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

# 出力ファイルパス
output_path = os.path.join(output_dir, "jqara_test.jsonl")

# JSONL 形式で保存
ds.to_json(output_path, orient="records", lines=True, force_ascii=False)

print(f"✅ Saved to {output_path}")
