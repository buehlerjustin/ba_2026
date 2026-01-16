#!/usr/bin/env python3
import glob
import os
import pandas as pd

GOLD_PATH = os.path.join("data", "mut_test_records_10000_with_metadata.csv")
GOLD_SEP = ";"

OUT_DIR = "eval"

NEEDED_CHECK_COLS = [
    "assignedpatient_patientjpaid",
    "bestmatchedweight",
    "bestmatchedpatient_patientjpaid",
]

TARGETS = {
    "ngram": {"pattern": os.path.join("results_*", "db_check_ngram_10k.csv"), "out": "eval_ngram_10k.csv"},
    "dl":    {"pattern": os.path.join("results_*", "db_check_dl_10k.csv"),    "out": "eval_dl_10k.csv"},
    "jw":    {"pattern": os.path.join("results_*", "db_check_jw_10k.csv"),    "out": "eval_jw_10k.csv"},
}

def find_single_file(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Keine Datei gefunden für Pattern: {pattern}")
    if len(matches) > 1:
        raise FileExistsError(f"Mehr als eine Datei gefunden für Pattern: {pattern}\n{matches}")
    return matches[0]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gold = pd.read_csv(GOLD_PATH, sep=GOLD_SEP)

    for key, cfg in TARGETS.items():
        check_path = find_single_file(cfg["pattern"])
        check = pd.read_csv(check_path)

        missing = [c for c in NEEDED_CHECK_COLS if c not in check.columns]
        if missing:
            raise ValueError(f"{check_path} fehlt Spalten: {missing}")

        check_sel = check[NEEDED_CHECK_COLS].copy()

        for c in ["assignedpatient_patientjpaid", "bestmatchedpatient_patientjpaid"]:
            check_sel[c] = pd.to_numeric(check_sel[c], errors="coerce").astype("Int64")
        
        if len(check_sel) != len(gold):
            raise ValueError(
                f"Längen passen nicht: gold={len(gold)} vs {check_path}={len(check_sel)}"
            )

        #Konkatenieren
        eval_df = pd.concat([gold.reset_index(drop=True), check_sel.reset_index(drop=True)], axis=1)

        out_path = os.path.join(OUT_DIR, cfg["out"])
        eval_df.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}  (from {check_path})")

if __name__ == "__main__":
    main()
