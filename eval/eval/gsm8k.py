import re
import json


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    The model is prompted to end its answer with ``#### <number>``.
    If that marker is absent we fall back to the last standalone number
    in the response.
    """
    # Primary: look for the "#### <number>" pattern
    match = re.search(r"####\s*([\d,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: last standalone number in the text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def normalize_answer(ans: str) -> str:
    """Normalise a numeric string for comparison (strip commas and leading zeros)."""
    ans = ans.replace(",", "").strip()
    try:
        # Use float → int conversion to handle "8.0" == "8"
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return ans


def compute_scores_gsm8k(jobs: list, cache_path: str) -> float:
    """Score GSM8K predictions and return accuracy.

    Parameters
    ----------
    jobs:
        List of dicts, each with at least ``"prompt"``, ``"answer"`` (ground
        truth), and ``"gen"`` (a list with one generated string).
    cache_path:
        Path where detailed per-item results are written (JSONL).

    Returns
    -------
    float
        Exact-match accuracy (fraction of correct answers).
    """
    correct = 0
    total = 0

    for job in jobs:
        assert len(job.get("gen", [])) == 1, (
            "Each job should contain exactly one generation output"
        )

        pred_raw = job["gen"][0]
        gt = normalize_answer(str(job["answer"]))

        pred = extract_answer(pred_raw)
        if pred is not None:
            pred_norm = normalize_answer(pred)
            is_correct = pred_norm == gt
        else:
            pred_norm = None
            is_correct = False

        job.update(
            {
                "pred": pred_norm,
                "gt": gt,
                "acc": 1.0 if is_correct else 0.0,
            }
        )

        correct += int(is_correct)
        total += 1

    save_cache(jobs, cache_path)
    return correct / total if total > 0 else 0.0


def save_cache(jobs: list, cache_path: str) -> None:
    with open(cache_path, "w", encoding="utf-8") as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + "\n")
            g.flush()
