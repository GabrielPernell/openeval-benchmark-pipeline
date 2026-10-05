"""Faithful reimplementation of VisFactor's scoring and prompt building.

Ported from vlmeval/dataset/visfactor.py in CUHK-ARISE/VisFactor so the scores
and prompts we attach are the authors' own, not our interpretation. Verified
against the lab's released results: it reproduces every stored per-row
`correct` value and every model's published overall score exactly.

Scoring has two levels. Each row is marked right or wrong (`score_row`); rows
sharing a (category_id, eval_index) form one test question, which counts only
if all its rows are right; a subtest's score is the share of its questions
that count, and the overall score is the mean over the 20 subtests
(`overall`).
"""
import collections
import re

# Subtests whose answer is true/false.
_TF_SUBTESTS = ['CF1', 'CF2', 'MV1', 'MV2', 'MV3', 'P3', 'RL2', 'S1', 'S2', 'SS2', 'VZ1', 'VZ2']


def extract_last_json_answer(s):
    pattern = r'\{\s*"answer"\s*:\s*(.+?)\s*\}'
    matches = list(re.finditer(pattern, s))
    if not matches:
        return ''
    raw_value = matches[-1].group(1).strip()
    if (raw_value.startswith('"') and raw_value.endswith('"')) or \
       (raw_value.startswith("'") and raw_value.endswith("'")):
        raw_value = raw_value[1:-1]
    return raw_value


def extract_last_numbers(s):
    return [num for num in re.findall(r'\d+', s)]


def extract_last_uppercase_letter(s):
    for char in reversed(s):
        if char.isupper():
            return char
    return None


def score_row(category_id, prediction_raw, answer, additional):
    """-> (extracted prediction, correct). `additional` is the raw column value;
    pandas reads an empty one as nan, which is what the authors compared."""
    cid = str(category_id)
    additional = 'nan' if additional in (None, '') else str(additional)
    prediction = extract_last_json_answer(str(prediction_raw))
    answer = str(answer)

    if cid in _TF_SUBTESTS or (cid == 'VZ3' and not additional.isdigit()):
        p = prediction.lower()
        if p in ['t', 'y', '1', 'true', 'yes']:
            pred = 'T'
        elif p in ['f', 'n', '0', 'false', 'no']:
            pred = 'F'
        else:
            pred = ''
        return pred, pred == answer
    if cid in ['CS1', 'CS2', 'CS3']:
        return prediction, prediction.lower() in [i.lower() for i in answer.split(',')]
    if cid in ['CF3', 'I3', 'MA1', 'SS3', 'VZ3']:
        if prediction == '':
            prediction = str(prediction_raw)
        if cid == 'VZ3':
            prediction = extract_last_uppercase_letter(prediction)
        elif cid == 'CF3':
            nums = extract_last_numbers(prediction)
            prediction = '' if len(nums) < 2 else f'({nums[-2]}, {nums[-1]})'
        else:
            nums = extract_last_numbers(prediction)
            prediction = '' if len(nums) < 1 else nums[-1]
        return prediction, prediction == answer
    raise ValueError(f"unknown VisFactor subtest {cid!r}")


def overall(rows, correct):
    """rows: dicts with category_id and eval_index; correct: parallel bools.
    -> ({subtest: score}, overall score)"""
    groups = collections.defaultdict(list)
    for row, c in zip(rows, correct):
        groups[(str(row["category_id"]), str(row["eval_index"]))].append(c)
    per_subtest = collections.defaultdict(list)
    for (cid, _), cs in groups.items():
        per_subtest[cid].append(1 if all(cs) else 0)
    subtests = {cid: sum(v) / len(v) for cid, v in per_subtest.items()}
    return subtests, sum(subtests.values()) / len(subtests)


def build_prompt(question, additional, n_images):
    """The authors' build_prompt, standard (non-CoT) variant: `<br>` becomes a
    newline, <ADDITIONAL_n> slots are filled, and the text is split around the
    <IMAGE_n> tags. -> list of ("text", str) / ("image", n) parts in order."""
    msgs = question.replace('<br>', '\n')
    if additional not in (None, '', 'nan'):
        fills = str(additional).replace('<br>', '\n').split(';')

        def replacer(match):
            i = int(match.group(1))
            return fills[i] if 0 <= i < len(fills) else match.group(0)
        msgs = re.sub(r"<ADDITIONAL_(\d+)>", replacer, msgs)

    parts = []
    for part in re.split(r'(<IMAGE_\d+>)', msgs):
        if part == '':
            continue
        if part[0] != '<':
            parts.append(("text", part))
        else:
            # The authors read the single digit before '>'.
            k = int(part[-2])
            if k >= n_images:
                raise ValueError(f"prompt refers to image {k} of {n_images}")
            parts.append(("image", k))
    return parts
