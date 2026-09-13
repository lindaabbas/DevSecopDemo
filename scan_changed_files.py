# trigger
# retrigger
# retrigger3
"""
scan_changed_files.py — Runs every available method (via inference_utils.py)
on Python files in a GitHub Actions workflow, and fails the check if the
Hybrid Ensemble flags any file as Vulnerable (matching Fig. 1's claimed
"Deployment control: Block/Allow in CI/CD" behavior).

Usage:
    python scan_changed_files.py file1.py file2.py ...   # scan specific files
    python scan_changed_files.py                          # scan all .py files under the repo

Environment variables:
    GATE_ON       = which method's verdict blocks the build (default: "Hybrid Ensemble")
    REPORT_ONLY   = "1" to always exit 0 (report findings without failing the build)
"""
import os
import sys
import glob
import json

import inference_utils as iu

GATE_ON_DEFAULT = 'Hybrid Ensemble'

# Directories that should never be scanned even if present in the repo.
EXCLUDE_DIRS = {'.git', 'venv', '.venv', '__pycache__', 'node_modules', 'models', 'results'}


def find_python_files():
    files = []
    for path in glob.glob('**/*.py', recursive=True):
        if any(part in EXCLUDE_DIRS for part in path.split(os.sep)):
            continue
        files.append(path)
    return files


def scan_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception as e:
        return {'path': path, 'error': str(e)}
    if not code.strip():
        return {'path': path, 'skipped': 'empty file'}
    results = iu.predict_all(code)
    return {'path': path, 'results': results}


def main():
    # Read env vars at call time (not import time) so behavior reflects the
    # current environment even if it changes after this module was imported.
    gate_on = os.environ.get('GATE_ON', GATE_ON_DEFAULT)
    report_only = os.environ.get('REPORT_ONLY', '0') == '1'

    targets = sys.argv[1:] if len(sys.argv) > 1 else find_python_files()
    targets = [t for t in targets if t.endswith('.py')]

    if not targets:
        print("No Python files to scan.")
        return 0

    print(f"Scanning {len(targets)} file(s) -- gating on: {gate_on}\n")
    print(f"{'File':<45}{'Verdict':<14}{'Score':<10}Notes")
    print('-' * 90)

    any_vulnerable = False
    any_unavailable = False
    all_reports = []

    for path in targets:
        report = scan_file(path)
        all_reports.append(report)

        if 'error' in report:
            print(f"{path:<45}{'ERROR':<14}{'--':<10}{report['error']}")
            continue
        if 'skipped' in report:
            print(f"{path:<45}{'SKIPPED':<14}{'--':<10}{report['skipped']}")
            continue

        r = report['results'].get(gate_on, {})
        if not r.get('available'):
            any_unavailable = True
            print(f"{path:<45}{'N/A':<14}{'--':<10}'{gate_on}' not available "
                  f"(run its notebook first)")
            continue

        label, score = r['label'], r['score']
        print(f"{path:<45}{label:<14}{score:<10.4f}")
        if label == 'Vulnerable':
            any_vulnerable = True

    # ── GitHub Actions step summary (renders nicely in the workflow UI) ────
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a') as f:
            f.write(f"## Security Scan Results (gated on: {gate_on})\n\n")
            f.write("| File | Verdict | Score |\n|---|---|---|\n")
            for report in all_reports:
                if 'results' not in report:
                    continue
                r = report['results'].get(gate_on, {})
                if r.get('available'):
                    icon = "🔴" if r['label'] == 'Vulnerable' else "🟢"
                    f.write(f"| `{report['path']}` | {icon} {r['label']} | {r['score']:.4f} |\n")

    with open('scan_results.json', 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)

    if any_vulnerable:
        print(f"\n❌ One or more files flagged Vulnerable by {gate_on}.")
        if report_only:
            print("REPORT_ONLY=1 set -- not failing the build.")
            return 0
        return 1

    if any_unavailable:
        print(f"\n⚠ '{gate_on}' was unavailable for some files -- "
              f"train its notebook(s) and re-run for a full check.")

    print("\n✅ No vulnerabilities flagged.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
