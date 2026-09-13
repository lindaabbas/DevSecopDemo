"""
inference_utils.py — Live prediction for every method, on arbitrary
user-pasted Python code.
=============================================================================
Used by app.py's "Check Code" tab. Loads every artifact that's actually on
disk (gracefully skipping methods whose notebook hasn't been run yet) and
exposes one function:

    results = predict_all(code_text)
    # -> {'Bandit': {'label': 'Vulnerable', 'score': 0.73}, 'CodeBERT': {...}, ...}

Everything is cached at module level so repeated calls (e.g. from a
Streamlit button) don't reload models from disk each time.
"""
import os
import re
import json
import subprocess
import sys
import tempfile
import ast

import numpy as np

# ── Lazy/optional heavy imports (only loaded if their artifacts exist) ────
_torch = None
_transformers = None


def _lazy_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _lazy_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _transformers = (AutoTokenizer, AutoModelForSequenceClassification)
    return _transformers


# =============================================================================
# Bandit (real CLI, no model to load)
# =============================================================================
_SEV_WEIGHT = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.2}


def bandit_score_live(code: str) -> float:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                          encoding='utf-8') as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        result = subprocess.run(
            [sys.executable, '-m', 'bandit', '-f', 'json', '-ll', tmp_path],
            capture_output=True, text=True, timeout=15)
        os.unlink(tmp_path)
        data = json.loads(result.stdout)
        issues = data.get('results', [])
        total = sum(_SEV_WEIGHT.get(i.get('issue_severity', 'LOW').upper(), 0.2) for i in issues)
        return min(1.0, total / (len(issues) * 0.5 + 2)) if issues else 0.0
    except Exception:
        return 0.0


def _load_bandit_threshold():
    path = 'models/bandit_threshold.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('threshold', 0.5)
    return 0.5


_tuned_thresholds_cache = None

def _load_tuned_threshold(key: str, default: float = 0.5) -> float:
    """Loads a per-method decision threshold tuned on the validation split
    (see tune_all_thresholds.py). Falls back to 0.5 if the file or this
    specific key isn't present yet (e.g. tuning hasn't been run)."""
    global _tuned_thresholds_cache
    if _tuned_thresholds_cache is None:
        path = 'results/tuned_thresholds.json'
        if os.path.exists(path):
            with open(path) as f:
                _tuned_thresholds_cache = json.load(f)
        else:
            _tuned_thresholds_cache = {}
    return _tuned_thresholds_cache.get(key, {}).get('threshold', default)


# =============================================================================
# RL (11 hand-engineered features -> DQN and/or Tabular Q-table)
# =============================================================================
_RISKY_CALLS = ['eval', 'exec', 'pickle.loads', 'os.system', 'subprocess.']
_CRYPTO_PATTERN = re.compile(r'\b(md5|sha1)\s*\(', re.IGNORECASE)
_SQLI_PATTERN = re.compile(r"""(SELECT|INSERT|UPDATE|DELETE)[^;]*['"]\s*\+""", re.IGNORECASE)
_CRED_PATTERN = re.compile(r'\b(password|secret|api_key|token)\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE)


def extract_features(code: str) -> np.ndarray:
    """Same 11-feature extraction used in RL.ipynb -- must stay in sync."""
    dangerous_call_count = sum(len(re.findall(re.escape(c), code)) for c in _RISKY_CALLS)
    weak_crypto_flag = 1.0 if _CRYPTO_PATTERN.search(code) else 0.0

    loc = len(code.splitlines())
    try:
        tree = ast.parse(code)
        node_count = sum(1 for _ in ast.walk(tree))
        n_imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
        func_params = sum(len(n.args.args) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)
        cyclomatic = 1 + sum(1 for n in ast.walk(tree) if isinstance(n, branch_nodes))

        def depth(node, d=0):
            child_depths = [depth(c, d + 1) for c in ast.iter_child_nodes(node)]
            return max(child_depths, default=d)
        max_nesting = depth(tree)
    except SyntaxError:
        node_count = n_imports = func_params = cyclomatic = max_nesting = 0

    sql_concat_count = len(_SQLI_PATTERN.findall(code))
    cred_flag = 1.0 if _CRED_PATTERN.search(code) else 0.0
    n_comment_lines = sum(1 for ln in code.splitlines() if ln.strip().startswith('#'))
    comment_to_code_ratio = n_comment_lines / loc if loc > 0 else 0.0

    return np.array([
        dangerous_call_count, weak_crypto_flag,
        cyclomatic, loc, node_count, max_nesting, func_params,
        sql_concat_count, cred_flag, n_imports, comment_to_code_ratio,
    ], dtype=np.float32)


class DQNetwork:
    """Lazily-defined so this module imports fine even without torch installed."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is None:
            torch = _lazy_torch()
            import torch.nn as nn

            class _DQNetwork(nn.Module):
                def __init__(self, state_dim=11, n_actions=2):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(state_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, n_actions),
                    )
                def forward(self, x):
                    return self.net(x)
            cls._cls = _DQNetwork
        return cls._cls


_rl_artifacts = None


def _load_rl_artifacts():
    global _rl_artifacts
    if _rl_artifacts is not None:
        return _rl_artifacts
    artifacts = {}
    scaler_path, dqn_path, qtable_path = 'models/rl_scaler.pkl', 'models/dqn_policy.pt', 'models/qtable.npy'
    if os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
    if os.path.exists(dqn_path) and 'scaler' in artifacts:
        torch = _lazy_torch()
        net = DQNetwork.get_class()(state_dim=11)
        net.load_state_dict(torch.load(dqn_path, map_location='cpu', weights_only=True))
        net.eval()
        artifacts['dqn'] = net
    if os.path.exists(qtable_path) and 'scaler' in artifacts:
        artifacts['qtable'] = np.load(qtable_path)
    _rl_artifacts = artifacts
    return artifacts


def _discretize(feat_row):
    groups = [feat_row[:2].sum(), feat_row[2:7].sum(), feat_row[7:11].sum()]
    bins = []
    for g in groups:
        if g <= 0.5:
            bins.append(0)
        elif g <= 3.0:
            bins.append(1)
        else:
            bins.append(2)
    return bins[0] * 9 + bins[1] * 3 + bins[2]


def rl_predict_live(code: str) -> dict:
    """Returns {'DQN': prob_or_None, 'Q-Learning': prob_or_None}."""
    artifacts = _load_rl_artifacts()
    out = {'DQN': None, 'Q-Learning': None}
    if 'scaler' not in artifacts:
        return out
    feat = extract_features(code).reshape(1, -1)
    feat_scaled = artifacts['scaler'].transform(feat)

    if 'dqn' in artifacts:
        torch = _lazy_torch()
        with torch.no_grad():
            logits = artifacts['dqn'](torch.FloatTensor(feat_scaled))
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        out['DQN'] = prob

    if 'qtable' in artifacts:
        state = _discretize(feat_scaled[0])
        q_row = artifacts['qtable'][state]
        exp_q = np.exp(q_row - q_row.max())
        softmax_q = exp_q / exp_q.sum()
        out['Q-Learning'] = float(softmax_q[1])

    return out


# =============================================================================
# Transformer models (CodeBERT / GraphCodeBERT / CodeGPT) -- auto-discovered
# =============================================================================
_transformer_artifacts = {}

_KNOWN_TRANSFORMER_KEYS = {
    'codebert': 'CodeBERT', 'graphcodebert': 'GraphCodeBERT', 'codegpt': 'CodeGPT',
}


def _discover_transformer_models():
    found = {}
    for key, display_name in _KNOWN_TRANSFORMER_KEYS.items():
        model_dir = f'models/{key}_finetuned'
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'config.json')):
            found[key] = (display_name, model_dir)
    return found


def _load_transformer(key, model_dir):
    if key in _transformer_artifacts:
        return _transformer_artifacts[key]
    AutoTokenizer, AutoModelForSequenceClassification = _lazy_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    _transformer_artifacts[key] = (tokenizer, model)
    return tokenizer, model


def transformer_predict_live(code: str, max_len: int = 256) -> dict:
    """Returns {'CodeBERT': prob_or_None, 'GraphCodeBERT': ..., 'CodeGPT': ...}
    for whichever fine-tuned models are actually present on disk."""
    torch = _lazy_torch()
    out = {name: None for name in _KNOWN_TRANSFORMER_KEYS.values()}
    for key, (display_name, model_dir) in _discover_transformer_models().items():
        try:
            tokenizer, model = _load_transformer(key, model_dir)
            enc = tokenizer(code, truncation=True, max_length=max_len,
                             padding='max_length', return_tensors='pt')
            with torch.no_grad():
                logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).logits
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            out[display_name] = prob
        except Exception as e:
            out[display_name] = None
    return out


# =============================================================================
# Hybrid ensemble (combines whichever components were selected during training)
# =============================================================================
def _load_hybrid_config():
    path = 'models/hybrid_config.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# =============================================================================
# Orchestration: run everything available, return a uniform result dict
# =============================================================================
def predict_all(code: str) -> dict:
    """Runs every method whose artifacts are present on disk and returns:
    {method_display_name: {'score': float or None, 'label': 'Vulnerable'/'Safe'/None,
                            'available': bool}}
    Methods whose model files aren't found yet get 'available': False instead
    of crashing the whole check.
    """
    results = {}

    # Bandit
    try:
        score = bandit_score_live(code)
        thr = _load_bandit_threshold()
        results['Bandit'] = {'score': round(score, 4), 'label': 'Vulnerable' if score >= thr else 'Safe',
                              'available': True}
    except Exception as e:
        results['Bandit'] = {'score': None, 'label': None, 'available': False, 'error': str(e)}

    # RL
    rl_scores = rl_predict_live(code)
    _rl_threshold_keys = {'DQN': 'dqn', 'Q-Learning': 'qlearn'}
    for name, score in rl_scores.items():
        key = f'RL ({name})'
        if score is None:
            results[key] = {'score': None, 'label': None, 'available': False}
        else:
            thr = _load_tuned_threshold(_rl_threshold_keys.get(name, ''))
            results[key] = {'score': round(score, 4), 'label': 'Vulnerable' if score >= thr else 'Safe',
                             'available': True}

    # Transformers
    llm_scores = transformer_predict_live(code)
    _tf_threshold_keys = {'CodeBERT': 'codebert', 'GraphCodeBERT': 'graphcodebert', 'CodeGPT': 'codegpt'}
    for name, score in llm_scores.items():
        if score is None:
            results[name] = {'score': None, 'label': None, 'available': False}
        else:
            thr = _load_tuned_threshold(_tf_threshold_keys.get(name, ''))
            results[name] = {'score': round(score, 4), 'label': 'Vulnerable' if score >= thr else 'Safe',
                              'available': True}

    # Hybrid ensemble -- needs the SAME components chosen during Hybrid.ipynb training
    config = _load_hybrid_config()
    if config is not None:
        llm_key_map = {'codebert_prob': 'CodeBERT', 'graphcodebert_prob': 'GraphCodeBERT',
                       'codegpt_prob': 'CodeGPT'}
        rl_key_map = {'dqn_prob': 'DQN', 'qlearn_prob': 'Q-Learning'}
        llm_name = llm_key_map.get(config['llm_component'])
        rl_name = rl_key_map.get(config['rl_component'])
        bandit_res = results.get('Bandit', {})
        llm_res = results.get(llm_name, {}) if llm_name else {}
        rl_res = results.get(f'RL ({rl_name})', {}) if rl_name else {}

        if bandit_res.get('available') and llm_res.get('available') and rl_res.get('available'):
            w = config['weights']
            total = w['bandit'] + w['llm'] + w['rl']
            ens_score = (w['bandit'] * bandit_res['score'] + w['llm'] * llm_res['score'] +
                         w['rl'] * rl_res['score']) / total
            results['Hybrid Ensemble'] = {
                'score': round(ens_score, 4),
                'label': 'Vulnerable' if ens_score >= config['threshold'] else 'Safe',
                'available': True,
                'components_used': f"Bandit + {llm_name} + RL({rl_name})",
            }
        else:
            results['Hybrid Ensemble'] = {'score': None, 'label': None, 'available': False,
                                           'note': 'One or more required components unavailable.'}
    else:
        results['Hybrid Ensemble'] = {'score': None, 'label': None, 'available': False,
                                       'note': 'Run Hybrid.ipynb first to determine ensemble weights.'}

    return results
