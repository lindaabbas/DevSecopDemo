"""
make_h3_benchmark.py — Run this ONCE locally (not in CI) to create a small,
fixed benchmark sample file from your real test set. Commit the resulting
h3_benchmark_samples.csv to the repository so every GitHub Actions run
measures H3 timing on the exact same samples (reproducible comparisons
across commits).

Usage:
    python make_h3_benchmark.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

N_BENCHMARK_SAMPLES = 30  # small enough to keep CI runs fast

df = pd.read_csv('dataset_with_content.csv')
df.columns = df.columns.str.strip().str.lower()
df = df.dropna(subset=['content', 'label']).reset_index(drop=True)
df['label'] = df['label'].astype(int)
if 'id' not in df.columns:
    df['id'] = df.index

# Same split logic as every other notebook, so this draws from the real
# held-out test partition, not train/val.
y_all = df['label'].values
idx_all = df.index.values
idx_tv, idx_test = train_test_split(idx_all, test_size=0.20, random_state=42, stratify=y_all)

test_df = df.loc[idx_test]
bench = test_df.sample(n=min(N_BENCHMARK_SAMPLES, len(test_df)), random_state=42)
bench = bench[['id', 'content', 'label']]
bench.to_csv('h3_benchmark_samples.csv', index=False)

print(f"Saved h3_benchmark_samples.csv with {len(bench)} samples "
      f"({(bench['label']==1).sum()} vulnerable, {(bench['label']==0).sum()} safe).")
print("Commit this file to the repository -- it stays fixed across CI runs.")
