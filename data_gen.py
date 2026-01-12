import random
import pandas as pd
from faker import Faker

# Herkünfte der Daten
fake = Faker(['de_DE'])

# Anzahl Datensätze
count = 1_000_000

def create_clean_record(idx):
    last = fake.last_name()
    dob = fake.date_of_birth(maximum_age=110)
    
    return {
        "clinicextid": str(random.randint(1000000, 99999999)),
        "vorname": fake.first_name(),
        "nachname": last,
        "geburtsname": last,
        "geburtstag": f"{dob.day:02d}",
        "geburtsmonat": f"{dob.month:02d}",
        "geburtsjahr": str(dob.year),
        "plz": fake.postcode(),
        "ort": fake.city()
    }

data = [create_clean_record(i) for i in range(count)]

df = pd.DataFrame(data)

idat_cols = ["clinicextid", "vorname", "nachname", "geburtsname", "geburtstag", "geburtsmonat", "geburtsjahr", "plz", "ort"]

df[idat_cols].to_csv(f"./data/test_records_{count}.csv", index=False, sep=';', encoding='utf-8')

print(f"Fertig, {count} Einträge")