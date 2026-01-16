import numpy as np
import pandas as pd

KEIN_KANDIDAT = -1.7976931348623157e308  # keine Übereinstimmung mit anderem Datensatz
REVIEW_RATIO = 0.01  # 2% der Datensätze sollen als "POSSIBLE" markiert werden

def load_scores(path: str):
    df = pd.read_csv(path)
    y = df["is_duplicate"].astype(bool).to_numpy()
    s = df["bestmatchedweight"].to_numpy(dtype=float)

    # KEIN_KANDIDAT => -inf (kein Kandidat)
    s = np.where(s <= KEIN_KANDIDAT / 2, -np.inf, s)
    return s, y

def optimize_thresholds(scores, y_true, review_ratio=REVIEW_RATIO):
    n = len(scores)
    review_target = int(round(n * review_ratio))  # z.B. 500 bei 10k

    # UPPER: FP=0 in MATCH
    nondup = scores[~y_true]
    nondup = nondup[np.isfinite(nondup)]
    max_nondup = np.max(nondup) if len(nondup) else -np.inf
    upper = np.nextafter(max_nondup, np.inf)  # knapp drüber => FP=0

    # LOWER: kleinstes LOWER, sodass POSSIBLE <= review_target
    # LOWER <= POSSIBLE < UPPER
    finite_candidates = np.unique(scores[np.isfinite(scores) & (scores < upper)])
    finite_candidates.sort()

    chosen_lower = None
    for lower in finite_candidates:
        possible = int(np.sum((scores > lower) & (scores < upper)))
        if possible <= review_target:
            chosen_lower = float(lower)
            break

    if chosen_lower is None:
        # Falls selbst der höchste LOWER (knapp unter UPPER) nicht reicht
        chosen_lower = float(finite_candidates[-1]) if len(finite_candidates) else -np.inf

    pred = np.where(scores >= upper, "MATCH",
            np.where(scores <= chosen_lower, "NON_MATCH", "POSSIBLE"))

    tp = int(np.sum((pred == "MATCH") & (y_true == True)))
    fp = int(np.sum((pred == "MATCH") & (y_true == False)))
    fn = int(np.sum((pred == "NON_MATCH") & (y_true == True)))
    tn = int(np.sum((pred == "NON_MATCH") & (y_true == False)))
    possible = int(np.sum(pred == "POSSIBLE"))

    # FN-Aufschlüsselung: "keine Übereinstimmung" 
    fn_no_candidate = int(np.sum((pred == "NON_MATCH") & (y_true == True) & (~np.isfinite(scores))))
    fn_threshold = fn - fn_no_candidate

    return {
        "lower": chosen_lower,
        "upper": float(upper),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "POSSIBLE": possible,
        "FN_no_candidate": fn_no_candidate,
        "FN_threshold_caused": fn_threshold,
    }

if __name__ == "__main__":
    files = {
        "DL": "./eval/eval_dl_10k.csv",
        "JW": "./eval/eval_jw_10k.csv",
        "NGRAM": "./eval/eval_ngram_10k.csv",
    }

    for name, path in files.items():
        scores, y = load_scores(path)
        r = optimize_thresholds(scores, y, review_ratio=0.05)
        print(f"\n== {name} ==")
        print(f"LOWER (NON_MATCH <=): {r['lower']:.10f}")
        print(f"UPPER (MATCH >=):     {r['upper']:.10f}")
        print(f"TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']} POSSIBLE={r['POSSIBLE']}")
        print(f"FN(no-candidate)={r['FN_no_candidate']}  FN(threshold)={r['FN_threshold_caused']}")
