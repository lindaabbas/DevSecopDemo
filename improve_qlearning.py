"""
improve_qlearning.py
=====================
Two fixes for Tabular Q-Learning, run once after RL.ipynb has already
produced models/rl_scaler.pkl and models/qtable.npy:

1. Adds the missing AUC metric (was never computed in RL.ipynb originally).
2. Genuinely improves the model itself (not just its decision threshold):
   sweeps state-space granularity (27 -> 125 states) and the learning rate
   (ALPHA), picking the combination with the best F1 on the VALIDATION
   split only, then reports honest test-set numbers with that config.

This does NOT touch DQN, CodeBERT, GraphCodeBERT, CodeGPT, or Bandit --
only the Tabular Q-Learning component.

Run from the project root (same folder as dataset_with_content.csv,
models/, results/):
    python improve_qlearning.py
"""
import os
import re
import ast
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ── Same 11-feature extraction as RL.ipynb -- must stay in sync ──────────
RISKY_CALLS = ['eval', 'exec', 'pickle.loads', 'os.system', 'subprocess.']
CRYPTO_PATTERN = re.compile(r'\b(md5|sha1)\s*\(', re.IGNORECASE)
SQLI_PATTERN = re.compile(r"""(SELECT|INSERT|UPDATE|DELETE)[^;]*['"]\s*\+""", re.IGNORECASE)
CRED_PATTERN = re.compile(r'\b(password|secret|api_key|token)\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE)


def extract_features(code: str) -> np.ndarray:
    dangerous_call_count = sum(len(re.findall(re.escape(c), code)) for c in RISKY_CALLS)
    weak_crypto_flag = 1.0 if CRYPTO_PATTERN.search(code) else 0.0
    loc = len(code.splitlines())
    try:
        tree = ast.parse(code)
        node_count = sum(1 for _ in ast.walk(tree))
        n_imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
        func_params = sum(len(n.args.args) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)
        cyclomatic = 1 + sum(1 for n in ast.walk(tree) if isinstance(n, branch_nodes))
        def depth(node, d=0):
            return max([depth(c, d + 1) for c in ast.iter_child_nodes(node)], default=d)
        max_nesting = depth(tree)
    except SyntaxError:
        node_count = n_imports = func_params = cyclomatic = max_nesting = 0
    sql_concat_count = len(SQLI_PATTERN.findall(code))
    cred_flag = 1.0 if CRED_PATTERN.search(code) else 0.0
    n_comment_lines = sum(1 for ln in code.splitlines() if ln.strip().startswith('#'))
    comment_to_code_ratio = n_comment_lines / loc if loc > 0 else 0.0
    return np.array([
        dangerous_call_count, weak_crypto_flag,
        cyclomatic, loc, node_count, max_nesting, func_params,
        sql_concat_count, cred_flag, n_imports, comment_to_code_ratio,
    ], dtype=np.float32)


REWARD = {(1, 1): 2.0, (0, 0): 1.5, (0, 1): -1.0, (1, 0): -1.3}


def discretize(feat_row, n_levels=3):
    """Generalized version of RL.ipynb's discretize -- n_levels=3 reproduces
    the original 27-state space exactly; n_levels=5 gives 125 states."""
    groups = [feat_row[:2].sum(), feat_row[2:7].sum(), feat_row[7:11].sum()]
    if n_levels == 3:
        edges = [0.5, 3.0]
    else:
        edges = np.linspace(0.3, 4.0, n_levels - 1)
    val = 0
    for g in groups:
        b = int(np.searchsorted(edges, g))
        val = val * n_levels + min(b, n_levels - 1)
    return val


def train_qlearning(states_train, y_train, n_states, alpha, n_episodes=30, seed=42):
    rng = np.random.RandomState(seed)
    Q = np.zeros((n_states, 2))
    n_tr = len(states_train)
    for ep in range(n_episodes):
        eps = 1.0 - (1.0 - 0.05) * (ep / (n_episodes - 1))
        order = rng.permutation(n_tr)
        for i in order:
            s, yt = states_train[i], int(y_train[i])
            a = rng.randint(2) if rng.rand() < eps else int(np.argmax(Q[s]))
            r = REWARD[(yt, a)]
            Q[s, a] += alpha * (r - Q[s, a])
    return Q


def softmax_prob1(Q_row):
    e = np.exp(Q_row - Q_row.max())
    return (e / e.sum())[1]


def evaluate(Q, states, y_true):
    probs = np.array([softmax_prob1(Q[s]) for s in states])
    preds = (probs >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'auc': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan'),
    }, probs


def main():
    CSV = 'dataset_with_content.csv'
    if not os.path.exists(CSV):
        raise FileNotFoundError('Run the dataset-collection notebook first.')
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['content', 'label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    if 'id' not in df.columns:
        df['id'] = df.index
    y_all = df['label'].values

    print('Extracting features for all samples (same 11-feature set as RL.ipynb)...')
    X_all = np.vstack([extract_features(c) for c in df['content']])

    idx_all = np.arange(len(df))
    idx_tv, idx_test = train_test_split(idx_all, test_size=0.20, random_state=42, stratify=y_all)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=42, stratify=y_all[idx_tv])

    with open('models/rl_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_train_s = scaler.transform(X_all[idx_train])
    X_val_s = scaler.transform(X_all[idx_val])
    X_test_s = scaler.transform(X_all[idx_test])
    y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]

    # ── Baseline: reproduce the ORIGINAL 27-state, alpha=0.1 config exactly,
    #    just adding the AUC that was never computed ──────────────────────
    states_train_27 = np.array([discretize(r, 3) for r in X_train_s])
    states_test_27 = np.array([discretize(r, 3) for r in X_test_s])
    Q_orig = train_qlearning(states_train_27, y_train, 27, alpha=0.1)
    orig_metrics, _ = evaluate(Q_orig, states_test_27, y_test)
    print(f"\nOriginal (27 states, alpha=0.1): "
          f"Acc={orig_metrics['accuracy']:.4f} F1={orig_metrics['f1']:.4f} "
          f"AUC={orig_metrics['auc']:.4f}  <- AUC now added, was missing before")

    # ── Sweep: finer state space (125) x several learning rates, picked on
    #    VALIDATION only (never touches test until the final report) ──────
    print('\nSweeping state-space granularity and learning rate on validation...')
    best_cfg, best_val_f1, best_Q, best_n_levels = None, -1, None, 3
    for n_levels in [3, 5]:
        states_train = np.array([discretize(r, n_levels) for r in X_train_s])
        states_val = np.array([discretize(r, n_levels) for r in X_val_s])
        for alpha in [0.05, 0.1, 0.2, 0.3]:
            n_states = n_levels ** 3
            Q_cand = train_qlearning(states_train, y_train, n_states, alpha=alpha)
            val_metrics, _ = evaluate(Q_cand, states_val, y_val)
            print(f"  n_levels={n_levels} ({n_states} states), alpha={alpha}: "
                  f"val F1={val_metrics['f1']:.4f}")
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_cfg = (n_levels, alpha)
                best_n_levels = n_levels

    n_levels, alpha = best_cfg
    print(f"\nBest config on validation: n_levels={n_levels} ({n_levels**3} states), alpha={alpha}")

    states_train_final = np.array([discretize(r, n_levels) for r in X_train_s])
    states_val_final = np.array([discretize(r, n_levels) for r in X_val_s])
    states_test_final = np.array([discretize(r, n_levels) for r in X_test_s])
    Q_final = train_qlearning(states_train_final, y_train, n_levels ** 3, alpha=alpha)

    test_metrics, test_probs = evaluate(Q_final, states_test_final, y_test)
    _, train_probs = evaluate(Q_final, states_train_final, y_train)
    _, val_probs = evaluate(Q_final, states_val_final, y_val)

    print(f"\n=== Improved Tabular Q-Learning -- Test Set ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nComparison: original F1={orig_metrics['f1']:.4f} -> improved F1={test_metrics['f1']:.4f}")

    # ── Save updated qlearn_prob scores for every split (merged on 'id'
    #    only, not 'id'+'split' -- avoids any mismatch risk if the existing
    #    file's split labels were assigned differently for any reason) ────
    rl_scores = pd.read_csv('models/rl_scores.csv')
    new_scores = pd.concat([
        pd.DataFrame({'id': df['id'].iloc[idx_train].values, 'split': 'train', 'qlearn_prob': train_probs}),
        pd.DataFrame({'id': df['id'].iloc[idx_val].values, 'split': 'val', 'qlearn_prob': val_probs}),
        pd.DataFrame({'id': df['id'].iloc[idx_test].values, 'split': 'test', 'qlearn_prob': test_probs}),
    ], ignore_index=True)
    other_cols = [c for c in rl_scores.columns if c not in ('id', 'split', 'qlearn_prob')]
    rl_scores_other = rl_scores[['id'] + other_cols].drop_duplicates(subset='id')
    rl_scores = new_scores.merge(rl_scores_other, on='id', how='left')
    n_missing_other = rl_scores[other_cols].isna().any(axis=1).sum() if other_cols else 0
    if n_missing_other > 0:
        print(f"  WARNING: {n_missing_other} rows could not be matched back to the "
              f"existing rl_scores.csv by id -- check that 'id' is a stable, unique key.")
    rl_scores.to_csv('models/rl_scores.csv', index=False)
    print("\nUpdated: models/rl_scores.csv (qlearn_prob column replaced, other columns preserved by id)")

    np.save('models/qtable_improved.npy', Q_final)
    print("Saved: models/qtable_improved.npy (kept alongside the original qtable.npy)")

    # ── Update rl_metrics.json ────────────────────────────────────────────
    with open('results/rl_metrics.json') as f:
        rm = json.load(f)
    rm['tabular_q_learning'] = {
        'accuracy': round(float(test_metrics['accuracy']), 4),
        'precision': round(float(test_metrics['precision']), 4),
        'recall': round(float(test_metrics['recall']), 4),
        'f1': round(float(test_metrics['f1']), 4),
        'auc': round(float(test_metrics['auc']), 4),
        'n_states': n_levels ** 3,
        'alpha': alpha,
    }
    with open('results/rl_metrics.json', 'w') as f:
        json.dump(rm, f, indent=2)
    print("Updated: results/rl_metrics.json (tabular_q_learning entry)")

    print("\nDone. Re-run tune_all_thresholds.py afterward if you want a tuned "
          "decision threshold on top of this improved model.")


if __name__ == '__main__':
    main()
