#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import pandas as pd


BASE_FILE = "./data/mut_test_records_10000_with_metadata.csv"
FILES_TO_CHECK = [
    "./results_ngram/db_check_ngram_10k.csv",
    "./results_jw/db_check_jw_10k.csv",
    "./results_dl/db_check_dl_10k.csv",
    "./eval/eval_ngram_10k.csv",
    "./eval/eval_jw_10k.csv",
    "./eval/eval_dl_10k.csv",
]
FLOAT_TOL = 1e-12  # Toleranz für bestmatchedweight (wegen Rundung/Format)


FIELDS = ["vorname", "nachname", "geburtsname", "geburtstag", "geburtsmonat", "geburtsjahr", "plz", "ort"]
TRUE_SET = {"true", "1", "t", "yes", "y"}


def sniff_delimiter_and_encoding(path: Path, sample_bytes: int = 20000):
    raw = path.read_bytes()[:sample_bytes]
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin1", errors="ignore")
        encoding = "latin1"

    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter, encoding
    except Exception:
        return ",", encoding


def read_csv_auto(path: Path) -> pd.DataFrame:
    delim, enc = sniff_delimiter_and_encoding(path)
    # dtype=str + keep_default_na=False -> keine führenden Nullen verlieren, leere Felder bleiben ""
    return pd.read_csv(path, sep=delim, encoding=enc, dtype=str, keep_default_na=False)


def norm_str(v) -> str:
    return "" if v is None else str(v).strip()


def norm_day_month(v) -> str:
    s = norm_str(v)
    if s == "":
        return ""
    try:
        return f"{int(s):02d}"
    except ValueError:
        # falls schon "09" etc.
        return s.zfill(2) if s.isdigit() else s


def norm_year(v) -> str:
    s = norm_str(v)
    if s == "":
        return ""
    try:
        return f"{int(s):04d}"
    except ValueError:
        return s


def norm_plz(v) -> str:
    s = norm_str(v)
    return s.zfill(5) if s.isdigit() and len(s) < 5 else s


def signature_from_dict(d: dict) -> str:
    parts = []
    for k in FIELDS:
        v = d.get(k, "")
        if k in ("geburtstag", "geburtsmonat"):
            parts.append(norm_day_month(v))
        elif k == "geburtsjahr":
            parts.append(norm_year(v))
        elif k == "plz":
            parts.append(norm_plz(v))
        else:
            parts.append(norm_str(v))
    return "|".join(parts)


def signatures_from_base_like(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten für Entitätsvergleich: {missing}")
    return df.apply(lambda r: signature_from_dict({k: r[k] for k in FIELDS}), axis=1)


def signatures_from_db_check(df: pd.DataFrame) -> pd.Series:
    if "inputfieldsstring" not in df.columns:
        raise ValueError("db_check-Datei hat keine Spalte 'inputfieldsstring'.")
    def one(js: str) -> str:
        try:
            d = json.loads(js)
        except Exception as e:
            raise ValueError(f"Ungültiges JSON in inputfieldsstring: {js[:120]}... ({e})")
        return signature_from_dict(d)
    return df["inputfieldsstring"].apply(one)


def fail(msg: str):
    raise SystemExit(f"\n FEHLER: {msg}\n")


def report_mismatches(base_sig: pd.Series, other_sig: pd.Series, label: str, max_examples: int = 10):
    if len(base_sig) != len(other_sig):
        fail(f"{label}: unterschiedliche Zeilenanzahl: base={len(base_sig)} vs other={len(other_sig)}")

    mism = (base_sig != other_sig)
    n = int(mism.sum())
    if n == 0:
        print(f"{label}: Entitäten stimmen zeilenweise (Reihenfolge korrekt).")
        return

    print(f"{label}: {n} Zeilen unterscheiden sich (Reihenfolge/Content stimmt nicht). Beispiele:")
    idxs = list(mism[mism].index[:max_examples])
    for i in idxs:
        print(f"  - Zeile {i+1}:")
        print(f"    base : {base_sig.iloc[i]}")
        print(f"    other: {other_sig.iloc[i]}")
    fail(f"{label}: Reihenfolge/Entität nicht identisch. (Siehe Beispiele oben.)")


def check_duplicate_integrity(base_df: pd.DataFrame):
    # Prüft: Duplikate zeigen auf existierende Originale; Original-IDs der Originale sind eindeutig.
    if "is_duplicate" not in base_df.columns or "original_id" not in base_df.columns:
        fail("Base-Datei: Spalten 'is_duplicate' und/oder 'original_id' fehlen.")

    is_dup = base_df["is_duplicate"].astype(str).str.lower().isin(TRUE_SET)
    originals = base_df.loc[~is_dup].copy()
    dups = base_df.loc[is_dup].copy()

    try:
        orig_ids = originals["original_id"].astype(int)
        dup_orig_ids = dups["original_id"].astype(int)
    except Exception:
        fail("original_id ist nicht durchgehend in int konvertierbar. Bitte prüfen.")

    if orig_ids.duplicated().any():
        bad = originals.loc[orig_ids.duplicated(), "original_id"].head(10).tolist()
        fail(f"Base-Datei: original_id bei Originalen ist nicht eindeutig. Beispiele: {bad}")

    orig_set = set(orig_ids.tolist())
    missing = dups.loc[~dup_orig_ids.isin(orig_set)]
    if not missing.empty:
        ex = missing[["original_id"]].head(10).to_dict("records")
        fail(f"Base-Datei: Duplikate verweisen auf nicht existierende Originale. Beispiele: {ex}")

    print("Base-Datei: Duplikat-/Original-Referenzen sind konsistent.")


def numeric_equal(a: str, b: str, tol: float) -> bool:
    a, b = norm_str(a), norm_str(b)
    if a == b:
        return True
    if a == "" and b == "":
        return True
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) <= tol
    except Exception:
        return False


def check_eval_vs_db(eval_df: pd.DataFrame, db_df: pd.DataFrame, label: str):

    for col in ["assignedpatient_patientjpaid", "bestmatchedpatient_patientjpaid", "bestmatchedweight"]:
        if col not in eval_df.columns:
            fail(f"{label}: eval-Datei fehlt Spalte '{col}'")

    if len(eval_df) != len(db_df):
        fail(f"{label}: unterschiedliche Zeilenanzahl eval={len(eval_df)} vs db={len(db_df)}")

    for col in ["assignedpatient_patientjpaid", "bestmatchedpatient_patientjpaid"]:
        mism = (eval_df[col].astype(str).fillna("") != db_df[col].astype(str).fillna(""))
        if mism.any():
            i = int(mism.idxmax())
            fail(
                f"{label}: Unterschied in '{col}' bei Zeile {i+1}\n"
                f"  eval: {eval_df.at[i, col]!r}\n"
                f"  db  : {db_df.at[i, col]!r}"
            )
    # bestmatchedweight mit Toleranz prüfen
    w_eval = eval_df["bestmatchedweight"].astype(str).fillna("")
    w_db = db_df["bestmatchedweight"].astype(str).fillna("")
    bad_idxs = []
    for i, (a, b) in enumerate(zip(w_eval.tolist(), w_db.tolist())):
        if not numeric_equal(a, b, FLOAT_TOL):
            bad_idxs.append(i)
            if len(bad_idxs) >= 10:
                break

    if bad_idxs:
        i = bad_idxs[0]
        fail(
            f"{label}: Unterschied in 'bestmatchedweight' (außerhalb Toleranz {FLOAT_TOL}) bei Zeile {i+1}\n"
            f"  eval: {w_eval.iloc[i]!r}\n"
            f"  db  : {w_db.iloc[i]!r}"
        )

    print(f"{label}: eval_* und db_check_* sind bzgl. DB-Feldern konsistent (bestmatchedweight mit Toleranz).")


def main():
    base_path = Path(BASE_FILE)
    if not base_path.exists():
        fail(f"Base-Datei nicht gefunden: {base_path.resolve()}")

    base_df = read_csv_auto(base_path)
    print(f"== Base geladen: {base_path.name} ({len(base_df)} Zeilen) ==")

    check_duplicate_integrity(base_df)

    base_sig = signatures_from_base_like(base_df)

    # Prüfe alle Dateien gegen Base-Reihenfolge/Entität
    for fname in FILES_TO_CHECK:
        p = Path(fname)
        if not p.exists():
            fail(f"Datei nicht gefunden: {p.resolve()}")

        df = read_csv_auto(p)

        if "inputfieldsstring" in df.columns:
            other_sig = signatures_from_db_check(df)
            report_mismatches(base_sig, other_sig, label=p.name)

        else:
            other_sig = signatures_from_base_like(df)
            report_mismatches(base_sig, other_sig, label=p.name)

            if p.name.startswith("eval_"):
                suffix = p.name.replace("eval_", "")
                db_name = f"db_check_{suffix}"
                db_path = Path(db_name)
                if db_path.exists():
                    db_df = read_csv_auto(db_path)
                    check_eval_vs_db(df, db_df, label=f"{p.name} <-> {db_name}")

    print("\n Reihenfolge & Entitäten sind über alle geprüften Dateien konsistent.")


if __name__ == "__main__":
    main()
