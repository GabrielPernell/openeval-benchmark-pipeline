"""Faithful reimplementation of DocHop's heuristic scoring.

Ported from vlmeval/dataset/dochop.py in ZhuoranYu/dochop-vlmevalkit so the
scores we attach are the authors' own, not our interpretation. Verified by
reproducing their published per-model accuracies exactly.
"""
import math
import re


def clean_text(s):
    s = str(s)
    s = re.sub(r'^\s*answer\s*:\s*', '', s, flags=re.I)
    return s.strip()


def extract_number(text):
    if not text:
        return None
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except Exception:
        return None


def is_correct(gt_str, pred_str):
    if clean_text(gt_str).lower() == clean_text(pred_str).lower():
        return True
    gt_val, pred_val = extract_number(gt_str), extract_number(pred_str)
    if gt_val is not None and pred_val is not None:
        return math.isclose(gt_val, pred_val, rel_tol=0.01, abs_tol=1e-3)
    return False


def extract_answer(response):
    if not response:
        return ""
    matches = re.findall(r"the answer is[:\s]*(.+?)$", str(response),
                         re.IGNORECASE | re.MULTILINE)
    if matches:
        raw = matches[-1].strip()
        if raw.endswith('.'):
            raw = raw[:-1]
        return raw.replace('*', '').replace('`', '').strip()
    lines = str(response).strip().split('\n')
    return lines[-1].strip() if lines else ""
