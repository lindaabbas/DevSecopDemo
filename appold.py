"""
app.py — DevSecOps Hybrid Framework Results Dashboard
=======================================================
Streamlit GUI that loads every artifact produced by the project's
notebooks (dataset collection, Bandit, RL, and -- once built -- CodeBERT
and the Hybrid ensemble) and displays them together: metrics comparison,
combined ROC/AUC curves (the hypothesis-proof visual), confusion
matrices, and the RL learning curves.

Run with:  streamlit run app.py
Expects to be run from the project root (same folder as
dataset_with_content.csv, models/, results/).
"""
import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import inference_utils as iu

st.set_page_config(page_title="DevSecOps Hybrid Framework — Results", layout="wide")

# ── Registry of known method score files ────────────────────────────────────
# Add an entry here whenever a new notebook starts saving per-sample scores;
# the app will pick it up automatically and everything below just works.
METHOD_REGISTRY = {
    'Bandit':            {'scores_path': 'models/bandit_scores.csv',  'columns': ['bandit_score'],
                           'metrics_path': 'results/bandit_metrics.json'},
    'RL (DQN)':          {'scores_path': 'models/rl_scores.csv',      'columns': ['dqn_prob'],
                           'metrics_path': 'results/rl_metrics.json', 'metrics_key': 'dqn'},
    'RL (Q-Learning)':   {'scores_path': 'models/rl_scores.csv',      'columns': ['qlearn_prob'],
                           'metrics_path': 'results/rl_metrics.json', 'metrics_key': 'tabular_q_learning'},
    'CodeBERT':          {'scores_path': 'models/codebert_scores.csv',      'columns': ['codebert_prob'],
                           'metrics_path': 'results/llm_metrics.json', 'metrics_key': 'CodeBERT'},
    'GraphCodeBERT':     {'scores_path': 'models/graphcodebert_scores.csv', 'columns': ['graphcodebert_prob'],
                           'metrics_path': 'results/llm_metrics.json', 'metrics_key': 'GraphCodeBERT'},
    'CodeGPT':           {'scores_path': 'models/codegpt_scores.csv',       'columns': ['codegpt_prob'],
                           'metrics_path': 'results/llm_metrics.json', 'metrics_key': 'CodeGPT'},
    'Hybrid Ensemble':   {'scores_path': 'models/hybrid_scores.csv',  'columns': ['ensemble_score'],
                           'metrics_path': 'results/hybrid_metrics.json'},
}

METHOD_COLORS = {
    'Bandit': '#ef4444', 'RL (DQN)': '#22c55e', 'RL (Q-Learning)': '#84cc16',
    'CodeBERT': '#3b82f6', 'GraphCodeBERT': '#06b6d4', 'CodeGPT': '#f59e0b',
    'Hybrid Ensemble': '#a855f7',
}


# ── Loaders (cached so the app stays snappy) ────────────────────────────────
@st.cache_data
def load_dataset(csv_path='dataset_with_content.csv'):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    return df

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_method_scores(scores_path, score_columns, split='test'):
    if not os.path.exists(scores_path):
        return None
    scores_df = pd.read_csv(scores_path)
    if 'split' in scores_df.columns:
        scores_df = scores_df[scores_df['split'] == split]
    return scores_df

def collect_roc_data(df):
    """{display_name: (fpr, tpr, auc_value, n_samples)} for every method whose
    score file actually exists on disk -- others are silently skipped, since
    not every notebook in the pipeline may have been run yet."""
    available = {}
    for name, info in METHOD_REGISTRY.items():
        scores_df = load_method_scores(info['scores_path'], info['columns'])
        if scores_df is None:
            continue
        merged = scores_df.merge(df[['id', 'label']], on='id', how='inner')
        for col in info['columns']:
            if col not in merged.columns:
                continue
            y_true, y_score = merged['label'].values, merged[col].values
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            available[name] = (fpr, tpr, auc_val, len(y_true))
    return available

def get_metric(metrics_json, key, metrics_key=None):
    if metrics_json is None:
        return None
    d = metrics_json.get(metrics_key, metrics_json) if metrics_key else metrics_json
    return d.get(key) if isinstance(d, dict) else None


# ── Load everything up front ─────────────────────────────────────────────
df = load_dataset()

st.title("🛡️ DevSecOps Hybrid Framework — Results Dashboard")

if df is None:
    st.error(
        "`dataset_with_content.csv` not found in the current folder. "
        "Run the dataset-collection notebook first, and launch this app "
        "from the same project folder (`streamlit run app.py`)."
    )
    st.stop()

st.caption(
    f"Dataset: **{len(df):,}** samples "
    f"({(df['label']==1).sum():,} vulnerable, {(df['label']==0).sum():,} safe) "
    f"— loaded from `dataset_with_content.csv`"
)

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_check, tab_overview, tab_metrics, tab_roc, tab_confusion, tab_rl = st.tabs(
    ["🔍 Check Code", "📊 Dataset Overview", "📋 Metrics Comparison",
     "📈 ROC / AUC (Hypothesis Proof)", "🔲 Confusion Matrices", "🎯 RL Learning Curves"]
)

# ── Tab 0: Check Code — live prediction on user-pasted code ──────────────
with tab_check:
    st.subheader("Paste Python code to check")
    st.caption(
        "Runs every method whose trained artifacts are currently on disk. "
        "Methods whose notebook hasn't been run yet are shown as unavailable "
        "instead of breaking the check."
    )

    code_input = st.text_area(
        "Python code", height=250, placeholder="def login(user, password):\n    ...",
        label_visibility="collapsed",
    )

    if st.button("Check", type="primary", width="stretch"):
        if not code_input.strip():
            st.warning("Paste some Python code first.")
        else:
            with st.spinner("Running every available method..."):
                try:
                    results = iu.predict_all(code_input)
                except Exception as e:
                    results = None
                    st.error(f"Something went wrong while checking the code: {e}")

            if results:
                # Highlight the headline verdict (Hybrid Ensemble if available,
                # otherwise the first available method) prominently up top.
                headline_name = 'Hybrid Ensemble' if results.get('Hybrid Ensemble', {}).get('available') else None
                if headline_name is None:
                    for name, r in results.items():
                        if r.get('available'):
                            headline_name = name
                            break

                if headline_name:
                    r = results[headline_name]
                    color = "🔴" if r['label'] == 'Vulnerable' else "🟢"
                    st.markdown(f"### {color} {headline_name} verdict: **{r['label']}** "
                                f"(score: {r['score']:.3f})")
                    if 'components_used' in r:
                        st.caption(f"Ensemble combined: {r['components_used']}")
                else:
                    st.warning("No method's artifacts were found on disk yet -- run the "
                               "notebooks first (Bandit.ipynb, RL.ipynb, LLMs.ipynb, Hybrid.ipynb).")

                st.subheader("Every method's individual result")
                cols = st.columns(3)
                for i, (name, r) in enumerate(results.items()):
                    with cols[i % 3]:
                        if not r.get('available'):
                            st.metric(label=name, value="Unavailable")
                            st.caption(r.get('note', "Run its notebook first."))
                        else:
                            emoji = "🔴" if r['label'] == 'Vulnerable' else "🟢"
                            st.metric(label=f"{emoji} {name}", value=r['label'],
                                      delta=f"score {r['score']:.3f}", delta_color="off")

                with st.expander("Raw output (JSON)"):
                    st.json(results)

# ── Tab 1: Dataset Overview ──────────────────────────────────────────────
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sources")
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                   colors=plt.cm.Set2.colors, startangle=90)
            ax.set_title(f"Dataset Sources (Total: {len(df):,})")
            st.pyplot(fig)
        else:
            st.info("No `source` column found in the dataset.")

    with col2:
        st.subheader("Vulnerable vs Safe")
        label_counts = df['label'].value_counts().rename({0: 'Safe', 1: 'Vulnerable'})
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
               colors=['#22c55e', '#ef4444'], startangle=90)
        ax.set_title("Label Distribution")
        st.pyplot(fig)

    if 'label_basis' in df.columns:
        st.subheader("Ground Truth Source")
        st.dataframe(df['label_basis'].value_counts().rename_axis('Ground Truth Source')
                     .reset_index(name='Count'), width='stretch')

# ── Tab 2: Metrics Comparison ─────────────────────────────────────────────
with tab_metrics:
    st.subheader("Per-method metrics (test set)")

    rows = []
    for name, info in METHOD_REGISTRY.items():
        metrics_json = load_json(info['metrics_path'])
        if metrics_json is None:
            continue
        metrics_key = info.get('metrics_key')
        acc = get_metric(metrics_json, 'accuracy', metrics_key)
        prec = get_metric(metrics_json, 'precision', metrics_key)
        rec = get_metric(metrics_json, 'recall', metrics_key)
        f1 = get_metric(metrics_json, 'f1', metrics_key)
        auc_val = get_metric(metrics_json, 'auc', metrics_key)
        if acc is None and f1 is None:
            continue
        rows.append({'Method': name, 'Accuracy': acc, 'Precision': prec,
                      'Recall': rec, 'F1': f1, 'AUC': auc_val})

    if not rows:
        st.warning(
            "No metrics files found yet. Run `Bandit.ipynb` and `RL.ipynb` "
            "(and later `LLMs.ipynb` / `Hybrid.ipynb`) to populate this table."
        )
    else:
        metrics_df = pd.DataFrame(rows).set_index('Method')
        st.dataframe(metrics_df.style.format("{:.4f}", na_rep="—").highlight_max(axis=0, color='#065f46'),
                     width='stretch')

        st.subheader("Visual comparison")
        plot_df = metrics_df.dropna(axis=1, how='all').dropna(how='all')
        metric_cols = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'] if c in plot_df.columns]
        if metric_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(plot_df))
            width = 0.8 / len(metric_cols)
            for i, col in enumerate(metric_cols):
                ax.bar(x + i * width, plot_df[col].fillna(0), width, label=col)
            ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
            ax.set_xticklabels(plot_df.index, rotation=15, ha='right')
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.set_title("Metric comparison across available methods")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)

# ── Tab 3: ROC / AUC — the hypothesis-proof visual ───────────────────────
with tab_roc:
    st.subheader("Combined ROC curves (test set)")
    st.caption(
        "This is the direct visual evidence for H1 (does the proposed approach "
        "improve detection over Bandit alone?): a curve further toward the "
        "top-left, with a higher AUC, means better discrimination between "
        "vulnerable and safe code at every possible decision threshold — not "
        "just at the one threshold used for the accuracy/F1 table above."
    )

    roc_data = collect_roc_data(df)

    if not roc_data:
        st.warning(
            "No per-sample score files found yet. `Bandit.ipynb` and "
            "`RL.ipynb` both save these automatically "
            "(`models/bandit_scores.csv`, `models/rl_scores.csv`) — run them "
            "first, then reload this page."
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        for name, (fpr, tpr, auc_val, n) in sorted(roc_data.items(), key=lambda kv: -kv[1][2]):
            color = METHOD_COLORS.get(name, None)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f}, n={n})",
                     color=color, linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess (AUC = 0.50)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — All Available Methods')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.subheader("AUC ranking")
        auc_table = pd.DataFrame(
            [(name, v[2], v[3]) for name, v in roc_data.items()],
            columns=['Method', 'AUC', 'Test samples']
        ).sort_values('AUC', ascending=False).reset_index(drop=True)
        st.dataframe(auc_table, width='stretch')

        missing = [n for n in METHOD_REGISTRY if n not in roc_data]
        if missing:
            st.info(f"Not yet available (notebook not run, or file missing): {', '.join(missing)}")

# ── Tab 4: Confusion matrices ─────────────────────────────────────────────
with tab_confusion:
    st.subheader("Confusion matrices")
    cm_images = {
        'Bandit': 'results/bandit_confusion_matrix.png',
        'RL': 'results/rl_confusion_matrix.png',
        'CodeBERT': 'results/llm_confusion_matrix.png',
        'Hybrid Ensemble': 'results/hybrid_confusion_matrix.png',
    }
    found_any = False
    cols = st.columns(2)
    for i, (name, path) in enumerate(cm_images.items()):
        if os.path.exists(path):
            found_any = True
            with cols[i % 2]:
                st.image(path, caption=name, width='stretch')
    if not found_any:
        st.warning("No confusion-matrix images found yet.")

# ── Tab 5: RL learning curves ──────────────────────────────────────────────
with tab_rl:
    st.subheader("RL training dynamics")
    st.caption(
        "Evidence that the RL agents genuinely need time to learn (epsilon-greedy "
        "exploration decaying over episodes) rather than converging instantly, "
        "which would indicate plain supervised learning rather than RL."
    )
    curve_path = 'results/rl_learning_curves.png'
    if os.path.exists(curve_path):
        st.image(curve_path, width='stretch')
    else:
        st.warning("`results/rl_learning_curves.png` not found — run `RL.ipynb` first.")

    rl_metrics = load_json('results/rl_metrics.json')
    if rl_metrics:
        st.subheader("Training protocol")
        st.json(rl_metrics.get('training_protocol', {}))
        st.caption(rl_metrics.get('scope_note', ''))

st.divider()
st.caption(
    "This dashboard reads directly from `dataset_with_content.csv`, `models/`, "
    "and `results/` in the current folder — re-run any notebook and refresh "
    "the page (or restart Streamlit) to see updated results."
)
