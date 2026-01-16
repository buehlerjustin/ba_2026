#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd

EVAL_GLOB = os.path.join("eval", "eval_*_10k.csv")

def metrics_from_eval_csv(path: str):
    df = pd.read_csv(path)

    # Ground truth
    is_dup = df["is_duplicate"].astype(bool).to_numpy()

    assigned = df["assignedpatient_patientjpaid"].to_numpy()
    seen = set()
    merged = np.zeros(len(assigned), dtype=bool)
    for i, aid in enumerate(assigned):
        merged[i] = aid in seen
        seen.add(aid)

    TP = int((is_dup & merged).sum())
    FP = int((~is_dup & merged).sum())
    FN = int((is_dup & ~merged).sum())
    TN = int((~is_dup & ~merged).sum())
    N = len(df)

    assert TP + FP + FN + TN == N, "Summe von TP/FP/FN/TN entspricht nicht N"

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return TP, FP, TN, FN, precision, recall, f1, N

def main():
    files = sorted(glob.glob(EVAL_GLOB))
    if not files:
        raise SystemExit(f"Keine Dateien gefunden: {EVAL_GLOB}")

    for f in files:
        TP, FP, TN, FN, p, r, f1, N = metrics_from_eval_csv(f)
        print(f"{f}")
        print(f"    TP={TP} FP={FP} TN={TN} FN={FN} (sum={TP+FP+TN+FN}={N})")
        print(f"    Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

if __name__ == "__main__":
    main()
