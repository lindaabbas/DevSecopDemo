"""
measure_h3_pipeline.py — Measures real Time-to-Detection (TTD) for the
baseline (always-run-all-three) pipeline vs. the adaptive RL+Bandit-gated
pipeline, using real wall-clock timing on the machine it runs on.

Designed to run inside GitHub Actions so the reported numbers come from
the actual CI runner hardware, not a local machine -- directly answering
H3 as literally stated ("... CI/CD pipeline execution time ...").

Usage:
    python measure_h3_pipeline.py

Requires:
    - inference_utils.py in the same folder
    - models/ populated (bandit_threshold.json, rl artifacts, and the
      three transformer *_finetuned/ folders)
    - h3_benchmark_samples.csv in the same folder (a small FIXED set of
      labeled code samples shipped with the repo -- NOT the full dataset,
      so this stays fast and reproducible on every CI run)
    - models/hybrid_config.json (produced by Hybrid.ipynb -- contains
      best_llm, best_rl, weights, and threshold)

Output:
    results/h3_ttd_pipeline_overhead.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import inference_utils as iu

CONFIDENCE_MARGIN = 0.15
BENCHMARK_FILE = 'h3_benchmark_samples.csv'  # columns: id, content, label


def main():
    if not os.path.exists(BENCHMARK_FILE):
        raise FileNotFoundError(
            f"{BENCHMARK_FILE} not found. This should be a small, fixed CSV "
            f"(columns: id, content, label) shipped in the repo -- see the "
            f"accompanying instructions for how to create it once from your "
            f"real test set."
        )
    if not os.path.exists('models/hybrid_config.json'):
        raise FileNotFoundError(
            "models/hybrid_config.json not found -- run Hybrid.ipynb's weight "
            "search first and commit/publish its output."
        )

    with open('models/hybrid_config.json') as f:
        config = json.load(f)
    best_llm, best_rl = config['llm_component'], config['rl_component']
    w_bandit, w_llm, w_rl = config['weights']['bandit'], config['weights']['llm'], config['weights']['rl']
    total_w = w_bandit + w_llm + w_rl
    best_thr = config['threshold']

    llm_display_map = {'codebert_prob': 'CodeBERT', 'graphcodebert_prob': 'GraphCodeBERT',
                        'codegpt_prob': 'CodeGPT'}
    rl_display_map = {'dqn_prob': 'DQN', 'qlearn_prob': 'Q-Learning'}
    chosen_llm_display = llm_display_map[best_llm]
    chosen_rl_display = rl_display_map[best_rl]

    bench = pd.read_csv(BENCHMARK_FILE)
    n = len(bench)
    print(f'Timing {n} fixed benchmark samples on this runner '
          f'(baseline = always full scan, adaptive = RL+Bandit-gated)...')

    per_component_times = {'bandit': [], 'rl': [], 'llm': []}
    baseline_times, adaptive_times, used_llm_flags = [], [], []
    baseline_preds, adaptive_preds, y_timing = [], [], []

    for _, row in bench.iterrows():
        code_str = row['content']
        y_timing.append(int(row['label']))

        t0 = time.perf_counter()
        b = iu.bandit_score_live(code_str)
        t_bandit = time.perf_counter() - t0

        t1 = time.perf_counter()
        rl_scores = iu.rl_predict_live(code_str)
        rl = rl_scores.get(chosen_rl_display, 0.5)
        rl = rl if rl is not None else 0.5
        t_rl = time.perf_counter() - t1

        t_fast = t_bandit + t_rl
        fast_score = (b + rl) / 2

        t2 = time.perf_counter()
        llm_scores = iu.transformer_predict_live(code_str)
        llm = llm_scores.get(chosen_llm_display, 0.5)
        llm = llm if llm is not None else 0.5
        t_llm = time.perf_counter() - t2

        per_component_times['bandit'].append(t_bandit)
        per_component_times['rl'].append(t_rl)
        per_component_times['llm'].append(t_llm)

        full_score = (w_bandit * b + w_llm * llm + w_rl * rl) / total_w
        baseline_times.append(t_fast + t_llm)
        baseline_preds.append(int(full_score >= best_thr))

        if abs(fast_score - 0.5) < CONFIDENCE_MARGIN:
            adaptive_times.append(t_fast + t_llm)
            used_llm_flags.append(True)
            adaptive_score = full_score
        else:
            adaptive_times.append(t_fast)
            used_llm_flags.append(False)
            adaptive_score = fast_score
        adaptive_preds.append(int(adaptive_score >= best_thr))

    y_timing = np.array(y_timing)
    baseline_total, adaptive_total = sum(baseline_times), sum(adaptive_times)
    pct_time_saved = (1 - adaptive_total / baseline_total) * 100 if baseline_total > 0 else 0.0
    pct_skipped_llm = (1 - sum(used_llm_flags) / len(used_llm_flags)) * 100

    baseline_acc = accuracy_score(y_timing, baseline_preds)
    adaptive_acc = accuracy_score(y_timing, adaptive_preds)
    baseline_f1 = f1_score(y_timing, baseline_preds, zero_division=0)
    adaptive_f1 = f1_score(y_timing, adaptive_preds, zero_division=0)

    print(f'\nBaseline total: {baseline_total*1000:.1f} ms | '
          f'Adaptive total: {adaptive_total*1000:.1f} ms | '
          f'Time saved: {pct_time_saved:.1f}%')
    print(f'Baseline F1={baseline_f1:.4f} | Adaptive F1={adaptive_f1:.4f}')

    result = {
        'environment': 'GitHub Actions CI runner',
        'n_samples': int(n),
        'confidence_margin': CONFIDENCE_MARGIN,
        'ttd_ms_per_sample': {
            comp: {'mean': round(float(np.mean(t) * 1000), 3),
                   'median': round(float(np.median(t) * 1000), 3),
                   'std': round(float(np.std(t) * 1000), 3)}
            for comp, t in per_component_times.items()
        },
        'pipeline_overhead': {
            'baseline_total_ms': round(float(baseline_total * 1000), 2),
            'adaptive_total_ms': round(float(adaptive_total * 1000), 2),
            'pct_time_saved': round(float(pct_time_saved), 2),
            'pct_transformer_calls_skipped': round(float(pct_skipped_llm), 2),
        },
        'detection_performance': {
            'baseline': {'accuracy': round(float(baseline_acc), 4), 'f1': round(float(baseline_f1), 4)},
            'adaptive': {'accuracy': round(float(adaptive_acc), 4), 'f1': round(float(adaptive_f1), 4)},
        },
    }

    os.makedirs('results', exist_ok=True)
    with open('results/h3_ttd_pipeline_overhead.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('\nSaved: results/h3_ttd_pipeline_overhead.json')


if __name__ == '__main__':
    main()
