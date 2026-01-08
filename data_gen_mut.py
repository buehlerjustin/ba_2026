import random
import pandas as pd
import numpy as np
from faker import Faker

# Herkünfte der Daten + Seeding für Reproduzierbarkeit
SEED = 2048
random.seed(SEED)
np.random.seed(SEED)
fake = Faker(['de_DE', 'en_GB'])
fake.seed_instance(SEED)

PHONETIC_MAP = {"sch": "sh", "dt": "tt", "ai": "ei", "ey": "ay", "ph": "f", "c": "k", 
                "z": "s", "th": "t","v": "f", "pf": "f", "y": "i", "ck": "k", "chs": "ks"}
SUB_MAP = {"rn": "m", "cl": "d", "vv": "w", "fi": "fj", "ni": "m", "ce": "de"}

# Tastatur-Map für QWERTZ
KEYBOARD_ADJACENCY = {
    'a': 'qwsy', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'rdsw',
    'f': 'rtgvcd', 'g': 'tzhbvf', 'h': 'zubngj', 'i': 'ujko', 'j': 'uikmnh',
    'k': 'iolmj', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'pki',
    'p': 'o', 'q': 'wa', 'r': 'tfde', 's': 'wedaxz', 't': 'rzgf',
    'u': 'zhioj', 'v': 'cfgb', 'w': 'qesa', 'x': 'sdcy', 'y': 'asx', 'z': 'ughit'
}

# Berechnet einen Fehlerindex basierend auf der Verteilung von Pollock & Zamora (1984).
# Pos 1: 7.8% | Pos 2: 11.7% | Pos 3: 19.2% | Rest: 61.3% gleichverteilt
def get_weighted_index(word_len):
    if word_len == 0:
        return 0
    
    weights = [0.078, 0.117, 0.192]
    
    if word_len >= 4:
        remaining_prob = 0.613
        dist_per_pos = remaining_prob / (word_len - 3)
        full_weights = weights + [dist_per_pos] * (word_len - 3)
    else:
        temp_weights = weights[:word_len]
        total = sum(temp_weights)
        full_weights = [w / total for w in temp_weights]

    indices = list(range(word_len))
    return np.random.choice(indices, p=full_weights)

def mutate_keyboard_typo(value):
    if len(value) < 3: return None
    idx = get_weighted_index(len(value))
    char = value[idx].lower()
    if char in KEYBOARD_ADJACENCY:
        replacement = random.choice(KEYBOARD_ADJACENCY[char])
        # Großschreibung beibehalten
        if value[idx].isupper(): replacement = replacement.upper()
        return value[:idx] + replacement + value[idx+1:]
    return None

def create_base_record(idx):
    last = fake.last_name()
    dob = fake.date_of_birth(maximum_age=110)
    
    return {
        "clinicExtId": str(random.randint(1000000, 99999999)),
        "vorname": fake.first_name(),
        "nachname": last,
        "geburtsname": last, # Standard --> Identisch
        "geburtstag": f"{dob.day:02d}",
        "geburtsmonat": f"{dob.month:02d}",
        "geburtsjahr": str(dob.year),
        "plz": fake.postcode(),
        "ort": fake.city(),
        "is_duplicate": False,
        "original_id": idx,
        "mutation_type": "Original"
    }

# INSGEMSAMTE Menge an Datensätzen bestimmen
amount_of_records = 200
amount_of_clean_records = int(amount_of_records * 0.8)
data = [create_base_record(i) for i in range(amount_of_clean_records)]

# Mutationen gleichmäßig verteilen
amount_of_mutations_per_type = int(amount_of_records * 0.04)
mutation_tasks = [1]*amount_of_mutations_per_type + \
                    [2]*amount_of_mutations_per_type + \
                    [3]*amount_of_mutations_per_type + \
                    [4]*amount_of_mutations_per_type + \
                    [5]*amount_of_mutations_per_type
                    
random.shuffle(mutation_tasks)

for index, mutation_type in enumerate(mutation_tasks):
    mutated = False
    while not mutated:
        original = random.choice(data[:160]).copy()
        target_field = random.choice(["vorname", "nachname", "geburtsname"])
        val = str(original[target_field])
        
        #Phonetischer Fehler
        if mutation_type == 1:
            for old, new in PHONETIC_MAP.items():
                if old in val.lower():
                    original[target_field] = val.lower().replace(old, new, 1).capitalize()
                    original["mutation_type"] = f"Phonetic_{target_field}"
                    mutated = True
                    break
        
        # Substitutionsfehler      
        elif mutation_type == 2:
            for old, new in SUB_MAP.items():
                if old in val.lower():
                    original[target_field] = val.lower().replace(old, new, 1).capitalize()
                    original["mutation_type"] = f"Substitution_{target_field}"
                    mutated = True
                    break
                
        # Tastaturfehler        
        elif mutation_type == 3:
            new_val = mutate_keyboard_typo(val)
            if new_val:
                original[target_field] = new_val
                original["mutation_type"] = f"KeyboardTypo_{target_field}"
                mutated = True
        # Fehlendes Feld (wird von Mainzelliste abgefangen)            
        # elif m_type == 3:
        #     field_to_drop = random.choice(["vorname", "geburtsname", "geburtsjahr"])
        #     original[field_to_drop] = ""
        #     original["mutation_type"] = f"Missing_{field_to_drop}"
        #     mutated = True
         
        # Verändeurng Nachname (Heirat)   
        elif mutation_type == 4:
            original["nachname"] = fake.last_name()
            original["mutation_type"] = "Marriage"
            mutated = True
            
        # Interpunktion / Spacing    
        elif mutation_type == 5:
            if len(val) > 3:
                pos = get_weighted_index(len(val))
                char = random.choice(["-", " "])
                original[target_field] = val[:pos] + char + val[pos:]
                original["mutation_type"] = f"Punctuation_{target_field}"
                mutated = True

    # Metadaten anpassen
    original["is_duplicate"] = True
    # neue ExtId Ja/Nein ?
    #original["clinicExtId"] = str(random.randint(1000000, 99999999))
    data.append(original)

df = pd.DataFrame(data)

# Reihenfolge Spalten festlegen
idat_cols = ["clinicExtId", "vorname", "nachname", "geburtsname", "geburtstag", "geburtsmonat", "geburtsjahr", "plz", "ort"]
metadata_cols = ["is_duplicate", "original_id", "mutation_type"]

# Export
# Datei 1: mit Metadatan
df[idat_cols + metadata_cols].to_csv(f"./data/mut_test_records_{amount_of_records}_with_metadata.csv", index=False, sep=';', encoding='utf-8')

# Datei 2: ohne Metadaten
df[idat_cols].to_csv(f"./data/mut_test_records_{amount_of_records}.csv", index=False, sep=';', encoding='utf-8')

print("Fertig - Dateien wurden erstellt.")
