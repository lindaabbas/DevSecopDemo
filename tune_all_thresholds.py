import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

METHODS = [
    {'key': 'bandit',        'scores_path': 'models/bandit_scores.csv',        'col': 'bandit_score',
     'metrics_path': 'results/bandit_metrics.json', 'metrics_key': None},
    {'key': 'dqn',           'scores_path': 'models/rl_scores.csv',            'col': 'dqn_prob',
     'metrics_path': 'results/rl_metrics.json', 'metrics_key': 'dqn'},
    {'key': 'qlearn',        'scores_path': 'models/rl_scores.csv',            'col': 'qlearn_prob',
     'metrics_path': 'results/rl_metrics.json', 'metrics_key': 'tabular_q_learning'},
    {'key': 'codebert',      'scores_path': 'models/codebert_scores.csv',      'col': 'codebert_prob',
     'metrics_path': 'results/llm_metrics.json', 'metrics_key': 'CodeBERT'},
    {'key': 'graphcodebert', 'scores_path': 'models/graphcodebert_scores.csv', 'col': 'graphcodebert_prob',
     'metrics_path': 'results/llm_metrics.json', 'metrics_key': 'GraphCodeBERT'},
    {'key': 'codegpt',       'scores_path': 'models/codegpt_scores.csv',       'col': 'codegpt_prob',
     'metrics_path': 'results/llm_metrics.json', 'metrics_key': 'CodeGPT'},
    {'key': 'hybrid',        'scores_path': 'models/hybrid_scores.csv',        'col': 'ensemble_score',
     'metrics_path': 'results/hybrid_metrics.json', 'metrics_key': None},
]


def tune_one_method(m, df):
    if not os.path.exists(m['scores_path']):
        print(f"  [skip] {m['key']}: {m['scores_path']} not found")
        return None
    scores_df = pd.read_csv(m['scores_path'])
    if m['col'] not in scores_df.columns:
        print(f"  [skip] {m['key']}: column '{m['col']}' not in {m['scores_path']}")
        return None
    merged = scores_df.merge(df[['id', 'label']], on='id', how='inner')
    val_df = merged[merged['split'] == 'val']
    test_df = merged[merged['split'] == 'test']
    if len(val_df) == 0 or len(test_df) == 0:
        print(f"  [skip] {m['key']}: missing val or test rows")
        return None

    val_probs, val_trues = val_df[m['col']].values, val_df['label'].values
    test_probs, test_trues = test_df[m['col']].values, test_df['label'].values

    # ── tune threshold on VAL only ────────────────────────────────────────
    best_thr, best_val_f1 = 0.5, -1
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (val_probs >= thr).astype(int)
        f1v = f1_score(val_trues, preds, zero_division=0)
        if f1v > best_val_f1:
            best_val_f1, best_thr = f1v, thr

    old_preds = (test_probs >= 0.5).astype(int)
    new_preds = (test_probs >= best_thr).astype(int)

    old_f1 = f1_score(test_trues, old_preds, zero_division=0)
    result = {
        'key': m['key'],
        'tuned_threshold': round(float(best_thr), 4),
        'old_accuracy': round(float(accuracy_score(test_trues, old_preds)), 4),
        'old_f1': round(float(old_f1), 4),
        'new_accuracy': round(float(accuracy_score(test_trues, new_preds)), 4),
        'new_precision': round(float(precision_score(test_trues, new_preds, zero_division=0)), 4),
        'new_recall': round(float(recall_score(test_trues, new_preds, zero_division=0)), 4),
        'new_f1': round(float(f1_score(test_trues, new_preds, zero_division=0)), 4),
    }
    return result


def update_metrics_json(m, result):
    if not os.path.exists(m['metrics_path']):
        return
    with open(m['metrics_path']) as f:
        metrics = json.load(f)
    target = metrics if m['metrics_key'] is None else metrics.setdefault(m['metrics_key'], {})
    target['accuracy'] = result['new_accuracy']
    target['precision'] = result['new_precision']
    target['recall'] = result['new_recall']
    target['f1'] = result['new_f1']
    target['threshold'] = result['tuned_threshold']
    with open(m['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)


def run_threshold_tuning(df):
    all_results = {}
    rows = []
    for m in METHODS:
        r = tune_one_method(m, df)
        if r is None:
            continue
        all_results[m['key']] = {'threshold': r['tuned_threshold']}
        update_metrics_json(m, r)
        rows.append(r)

    with open('results/tuned_thresholds.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    comp_df = pd.DataFrame(rows).set_index('key')
    print(comp_df.to_string())
    print(f"\nSaved: results/tuned_thresholds.json")
    print("Updated: all results/*_metrics.json files with tuned-threshold values")
    return comp_df


# ── Run it ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    CSV = 'dataset_with_content.csv'
    if not os.path.exists(CSV):
        raise FileNotFoundError('Run the dataset-collection notebook first.')
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip().str.lower()
    if 'id' not in df.columns:
        df['id'] = df.index

    print('Tuning decision thresholds for every available method (validation split only)...\n')
    comparison_df = run_threshold_tuning(df)
    print('\nDone. Each results/*_metrics.json now reflects tuned-threshold performance,')
    print('and results/tuned_thresholds.json is ready for inference_utils.py to use.')
