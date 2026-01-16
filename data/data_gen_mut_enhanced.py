#!/usr/bin/env python3
"""
Synthetische Testdaten für Record Linkage / Dedup (Fokus: Vorname, Nachname, Geburtsname)

Erzeugt:
- Clean Records (Originale)
- Duplicates mit realistischeren Fehlerprofilen (mehrere Fehler pro Duplikat, gewichtete Fehlertypen)
- Hard Negatives (Nicht-Matches, die absichtlich ähnlich sind)

Outputs (CSV, ';'-separiert, UTF-8) in ./data/
- mut_test_records_<N>_with_metadata.csv
- mut_test_records_<N>.csv

Hinweis: Die Felder geburtstag/monat/jahr, plz, ort sind nur Kontext.
Für Ähnlichkeitsmaße sind vorname/nachname/geburtsname relevant.
"""

import os
import random
import re
import string
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker


# ---------------------------
# Konfiguration
# ---------------------------

SEED = 2048

# Gesamtgröße
AMOUNT_OF_RECORDS = 10

# Anteile (sollten sich zu 1.0 addieren)
CLEAN_SHARE = 0.80        # Originale (entity-unique)
DUPLICATE_SHARE = 0.20    # echte Duplikate (positive Paare)
#HARD_NEG_SHARE = 0.00     # ähnliche Nicht-Matches

# Multi-Error: wie viele Fehler pro Duplikat?
# 50% -> 1 Fehler, 35% -> 2 Fehler, 15% -> 3 Fehler
MULTI_ERROR_DIST = [(1, 0.50), (2, 0.35), (3, 0.15)]

# Wenn True, wird eine Eingabe-Normalisierung simuliert (Umlaute/Diakritika/Whitespace).
SIMULATE_INPUT_NORMALIZATION = True

# Wenn True, bekommen Duplikate (und Hard Negatives) eine neue clinicExtId (typisch bei Dubletten im System).
NEW_EXT_ID_FOR_NON_ORIGINALS = True

# Locale: für konsistente Namen (keine DE/GB-Mischung)
FAKER_LOCALE = "de_DE"

# Gewichtete Feldwahl für Mutationen
FIELD_WEIGHTS = {
    "vorname": 0.35,
    "nachname": 0.45,
    "geburtsname": 0.20,
}

# ---------------------------
# Seed / Faker
# ---------------------------

random.seed(SEED)
np.random.seed(SEED)
fake = Faker(FAKER_LOCALE)
fake.seed_instance(SEED)


# ---------------------------
# Normalisierung (optional)
# ---------------------------

UMLAUT_MAP = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
    "ß": "ss",
})

def strip_diacritics(s: str) -> str:
    # Entfernt Kombinationszeichen (Diakritika), z.B. "Š" -> "S"
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_name(s: str) -> str:
    s = s.translate(UMLAUT_MAP)
    s = strip_diacritics(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------
# Keyboard-Typo (QWERTZ)
# ---------------------------

KEYBOARD_ADJACENCY = {
    "a": "qwsy", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "rdsw",
    "f": "rtgvcd", "g": "tzhbvf", "h": "zubngj", "i": "ujko", "j": "uikmnh",
    "k": "iolmj", "l": "opk", "m": "njk", "n": "bhjm", "o": "pki",
    "p": "o", "q": "wa", "r": "tfde", "s": "wedaxz", "t": "rzgf",
    "u": "zhioj", "v": "cfgb", "w": "qesa", "x": "sdcy", "y": "asx", "z": "ughit",
}

# Pollock & Zamora (1984): Pos1 7.8% | Pos2 11.7% | Pos3 19.2% | Rest 61.3% gleichverteilt
def get_weighted_index(word_len: int) -> int:
    if word_len <= 0:
        return 0
    base = [0.078, 0.117, 0.192]
    if word_len >= 4:
        remaining_prob = 0.613
        dist_per_pos = remaining_prob / (word_len - 3)
        weights = base + [dist_per_pos] * (word_len - 3)
    else:
        temp = base[:word_len]
        total = sum(temp)
        weights = [w / total for w in temp]
    indices = list(range(word_len))
    return int(np.random.choice(indices, p=weights))

def mutate_keyboard_typo(value: str) -> Optional[str]:
    if len(value) < 3:
        return None
    idx = get_weighted_index(len(value))
    if value[idx].isspace() or value[idx] == "-":
        return None
    char = value[idx].lower()
    if char in KEYBOARD_ADJACENCY:
        replacement = random.choice(KEYBOARD_ADJACENCY[char])
        if value[idx].isupper():
            replacement = replacement.upper()
        return value[:idx] + replacement + value[idx+1:]
    return None


# ---------------------------
# Phonetik / Substitution (case-preserving)
# (nur mild gewichtet; bei normalisierten Inputs sind diese Fehler seltener)
# ---------------------------

PHONETIC_MAP = {
    "sch": "sh",
    "ph": "f",
    "th": "t",
    "ck": "k",
    "z": "s",
    "c": "k",
    "v": "f",
}

SUB_MAP = {
    "rn": "m",
    "vv": "w",
    "cl": "d",
}

def replace_once_ci(text: str, old: str, new: str) -> Optional[str]:
    m = re.search(re.escape(old), text, flags=re.IGNORECASE)
    if not m:
        return None
    return text[:m.start()] + new + text[m.end():]

def mutate_phonetic(value: str) -> Optional[str]:
    v = value
    for old, new in PHONETIC_MAP.items():
        out = replace_once_ci(v, old, new)
        if out is not None and out != v:
            return out
    return None

def mutate_substitution(value: str) -> Optional[str]:
    v = value
    for old, new in SUB_MAP.items():
        out = replace_once_ci(v, old, new)
        if out is not None and out != v:
            return out
    return None


# ---------------------------
# Name-only Mutationen (realistische Klassiker)
# ---------------------------

ALPH = string.ascii_lowercase

def mutate_delete_char(value: str) -> Optional[str]:
    if len(value) < 4:
        return None
    idx = get_weighted_index(len(value))
    if value[idx].isspace() or value[idx] == "-":
        return None
    return value[:idx] + value[idx+1:]

def mutate_insert_char(value: str) -> Optional[str]:
    if len(value) < 3:
        return None
    idx = get_weighted_index(len(value))
    ch = random.choice(ALPH)
    return value[:idx] + ch + value[idx:]

def mutate_transpose(value: str) -> Optional[str]:
    if len(value) < 4:
        return None
    idx = get_weighted_index(len(value) - 1)
    if value[idx].isspace() or value[idx+1].isspace() or value[idx] == "-" or value[idx+1] == "-":
        return None
    if value[idx] == value[idx+1]:
        return None
    return value[:idx] + value[idx+1] + value[idx] + value[idx+2:]

def mutate_double_char(value: str) -> Optional[str]:
    if len(value) < 3:
        return None
    idx = get_weighted_index(len(value))
    if value[idx].isspace() or value[idx] == "-":
        return None
    return value[:idx] + value[idx] + value[idx:]


def split_tokens(name: str) -> List[str]:
    # trennt an Space oder Hyphen
    return [t for t in re.split(r"[ -]+", name) if t]

def mutate_drop_token(value: str) -> Optional[str]:
    toks = split_tokens(value)
    if len(toks) < 2:
        return None
    # häufig wird der zweite Teil eines Doppelnamens "weggeworfen"
    if random.random() < 0.65:
        return toks[0]
    return random.choice(toks)

def mutate_punctuation_swap(value: str) -> Optional[str]:
    # nur sinnvoll bei Doppelnamen
    if "-" in value:
        return value.replace("-", " ")
    if " " in value:
        return value.replace(" ", "-")
    return None

def mutate_initial(value: str) -> Optional[str]:
    toks = split_tokens(value)
    if not toks:
        return None
    t = toks[0]
    if len(t) < 2:
        return None
    return t[0] + "."

NICK = {
    "Alexander": ["Alex"],
    "Maximilian": ["Max"],
    "Katharina": ["Kathi", "Kata"],
    "Stefanie": ["Steffi"],
    "Johannes": ["Hannes"],
    "Benjamin": ["Ben"],
    "Christine": ["Chris"],
    "Michael": ["Mike"],
    "Andreas": ["Andi"],
}

def mutate_nickname(value: str) -> Optional[str]:
    toks = split_tokens(value)
    if not toks:
        return None
    base = toks[0]
    if base in NICK:
        new0 = random.choice(NICK[base])
        rest = toks[1:]
        return " ".join([new0] + rest) if rest else new0
    return None


# ---------------------------
# Record-Erzeugung
# ---------------------------

def maybe_compound_first(first: str) -> str:
    if random.random() < 0.10:  # 10% Doppelvorname
        sep = random.choice(["-", " "])
        first = f"{first}{sep}{fake.first_name()}"
    return first

def create_base_record(idx: int) -> Dict:
    last = fake.last_name()
    # 15% Doppelname (wie im Originalskript)
    if random.random() < 0.15:
        sep = random.choice([" ", "-"])
        last = f"{last}{sep}{fake.last_name()}"

    first = maybe_compound_first(fake.first_name())

    if SIMULATE_INPUT_NORMALIZATION:
        first = normalize_name(first)
        last = normalize_name(last)

    dob = fake.date_of_birth(maximum_age=110)

    rec = {
        "clinicExtId": str(random.randint(1000000, 99999999)),
        "vorname": first,
        "nachname": last,
        "geburtsname": last,   # Standard -->identisch
        "geburtstag": f"{dob.day:02d}",
        "geburtsmonat": f"{dob.month:02d}",
        "geburtsjahr": str(dob.year),
        "plz": fake.postcode(),
        "ort": fake.city(),
        "is_duplicate": False,
        "original_id": idx,
        "mutation_type": "Original",
    }
    return rec


# ---------------------------
# Multi-Error Duplicate Engine
# ---------------------------

def sample_num_errors() -> int:
    vals = [k for k, _ in MULTI_ERROR_DIST]
    weights = [p for _, p in MULTI_ERROR_DIST]
    return int(random.choices(vals, weights=weights, k=1)[0])

def weighted_field_choice() -> str:
    fields = list(FIELD_WEIGHTS.keys())
    weights = list(FIELD_WEIGHTS.values())
    return random.choices(fields, weights=weights, k=1)[0]

# Operationen: (name, weight, fn(value) -> Optional[str], applicable_fields)
Op = Tuple[str, float, Callable[[str], Optional[str]], List[str]]

OPS: List[Op] = [
    ("keyboard",   0.25, mutate_keyboard_typo,    ["vorname", "nachname", "geburtsname"]),
    ("delete",     0.15, mutate_delete_char,      ["vorname", "nachname", "geburtsname"]),
    ("transpose",  0.10, mutate_transpose,        ["vorname", "nachname", "geburtsname"]),
    ("insert",     0.10, mutate_insert_char,      ["vorname", "nachname", "geburtsname"]),
    ("doublechar", 0.08, mutate_double_char,      ["vorname", "nachname", "geburtsname"]),
    ("phonetic",   0.05, mutate_phonetic,         ["vorname", "nachname", "geburtsname"]),
    ("substitute", 0.05, mutate_substitution,     ["vorname", "nachname", "geburtsname"]),
    ("droptoken",  0.10, mutate_drop_token,       ["nachname", "geburtsname", "vorname"]),
    ("initial",    0.04, mutate_initial,          ["vorname"]),
    ("nickname",   0.03, mutate_nickname,         ["vorname"]),
    ("punctswap",  0.03, mutate_punctuation_swap, ["nachname", "geburtsname"]),
    # "marriage" wird als Spezialfall unten (selten) behandelt
]

def choose_op() -> Op:
    weights = [w for _, w, _, _ in OPS]
    idx = random.choices(range(len(OPS)), weights=weights, k=1)[0]
    return OPS[idx]

def apply_mutations_to_record(original: Dict) -> Dict:
    rec = original.copy()
    tags: List[str] = []
    k = sample_num_errors()

    # Versuche, k Mutationen zu setzen (mit Sicherheitsbremse)
    attempts = 0
    max_attempts = 30

    while len(tags) < k and attempts < max_attempts:
        attempts += 1
        op_name, _, op_fn, fields = choose_op()
        field = random.choice(fields)  # fields already filtered per op

        before = rec[field]
        after = op_fn(str(before))
        if after is None or after == before:
            continue

        if SIMULATE_INPUT_NORMALIZATION:
            after = normalize_name(after)

        rec[field] = after
        tags.append(f"{op_name}_{field}")

    # Seltene, aber realistische Spezialfälle:
    # 2%: "Heirat": nachname ändert sich, geburtsname bleibt
    if random.random() < 0.02:
        new_last = fake.last_name()
        if SIMULATE_INPUT_NORMALIZATION:
            new_last = normalize_name(new_last)
        # 30% Doppelname, 70% kompletter Wechsel
        if random.random() < 0.30:
            sep = random.choice([" ", "-"])
            rec["nachname"] = normalize_name(f"{rec['nachname']}{sep}{new_last}") if SIMULATE_INPUT_NORMALIZATION else f"{rec['nachname']}{sep}{new_last}"
            tags.append("marriage_doublename_nachname")
        else:
            rec["nachname"] = new_last
            tags.append("marriage_fullchange_nachname")

    # 1%: Vor-/Nachname vertauscht (Import-/Mapping-Fehler)
    if random.random() < 0.01:
        rec["vorname"], rec["nachname"] = rec["nachname"], rec["vorname"]
        tags.append("swap_vorname_nachname")

    rec["is_duplicate"] = True
    rec["mutation_type"] = "+".join(tags) if tags else "NoOp"

    if NEW_EXT_ID_FOR_NON_ORIGINALS:
        rec["clinicExtId"] = str(random.randint(1000000, 99999999))

    return rec


# ---------------------------
# Hard Negatives
# ---------------------------

def make_hard_negative(a: Dict, b: Dict, new_id: int) -> Dict:
    """
    Erzeugt einen neuen Record (keine Dublette), der absichtlich ähnlich zu a ist,
    aber eine eigene Entity (original_id = new_id) bleibt.
    """
    rec = b.copy()

    mode = random.choice(["same_last", "same_first", "share_last_token"])

    if mode == "same_last":
        rec["nachname"] = a["nachname"]
        # geburtsname wird in echten Daten oft aus nachname abgeleitet -> hier gleichziehen
        rec["geburtsname"] = a["geburtsname"]
    elif mode == "same_first":
        rec["vorname"] = a["vorname"]
    else:  # share_last_token
        toks = split_tokens(a["nachname"])
        if toks:
            rec["nachname"] = toks[0] if random.random() < 0.7 else random.choice(toks)
            rec["geburtsname"] = rec["nachname"]

    # Optional: eine leichte Schreibvariante (damit es noch ähnlicher wird)
    if random.random() < 0.30:
        field = weighted_field_choice()
        before = rec[field]
        after = random.choice([mutate_delete_char, mutate_transpose, mutate_keyboard_typo])(str(before))
        if after is not None and after != before:
            rec[field] = normalize_name(after) if SIMULATE_INPUT_NORMALIZATION else after

    rec["is_duplicate"] = False
    rec["original_id"] = new_id
    rec["mutation_type"] = f"HardNegative_{mode}"

    if NEW_EXT_ID_FOR_NON_ORIGINALS:
        rec["clinicExtId"] = str(random.randint(1000000, 99999999))

    return rec


# ---------------------------
# Main: Dataset bauen & exportieren
# ---------------------------

def main() -> None:
    clean_n = int(AMOUNT_OF_RECORDS * CLEAN_SHARE)
    dup_n = int(AMOUNT_OF_RECORDS * DUPLICATE_SHARE)
    hard_n = AMOUNT_OF_RECORDS - clean_n - dup_n

    if hard_n < 0:
        raise ValueError("Anteile CLEAN_SHARE + DUPLICATE_SHARE + HARD_NEG_SHARE ergeben > 1.0")

    data: List[Dict] = [create_base_record(i) for i in range(clean_n)]

    # Duplicates erzeugen
    for _ in range(dup_n):
        base = random.choice(data[:clean_n])
        dup = apply_mutations_to_record(base)
        dup["original_id"] = base["original_id"]  # bleibt Match-Entity
        data.append(dup)

    # Hard Negatives erzeugen (aus 2 verschiedenen Originalen)
    next_id = clean_n  # neue Entity-IDs für Hard Negs (und ggf. weitere Nicht-Duplikate)
    for _ in range(hard_n):
        a, b = random.sample(data[:clean_n], 2)
        hn = make_hard_negative(a, b, new_id=next_id)
        next_id += 1
        data.append(hn)

    df = pd.DataFrame(data)

    idat_cols = [
        "clinicExtId", "vorname", "nachname", "geburtsname",
        "geburtstag", "geburtsmonat", "geburtsjahr", "plz", "ort"
    ]
    metadata_cols = ["is_duplicate", "original_id", "mutation_type"]

    os.makedirs(".", exist_ok=True)

    df[idat_cols + metadata_cols].to_csv(
        f"./data/mut_test_records_{AMOUNT_OF_RECORDS}_with_metadata.csv",
        index=False, sep=";", encoding="utf-8"
    )
    df[idat_cols].to_csv(
        f"./data/mut_test_records_{AMOUNT_OF_RECORDS}.csv",
        index=False, sep=";", encoding="utf-8"
    )

    print("Fertig - Dateien wurden erstellt.")
    print(f"Originale: {clean_n}, Duplikate: {dup_n}, HardNegatives: {hard_n}")
    print(f"Normalisierung: {SIMULATE_INPUT_NORMALIZATION}, neue ExtId: {NEW_EXT_ID_FOR_NON_ORIGINALS}")

if __name__ == "__main__":
    main()
