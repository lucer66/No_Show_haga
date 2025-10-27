import csv
import random
from pathlib import Path
from datetime import datetime, timedelta

soort_consult_values = [
    "Controle patient",
    "Behandeling/ingrepen op poli 20min",
    "Nieuwe patient",
    "Nieuwe patient spoed",
    "Nieuwe Patient Spraak",
    "Opname",
    "Administratie",
    "NP ZONDER SMS",
    "Teleconsultatie voor de huisarts op afstand",
]

afspraak_code_values = [
    "CP",
    "BEH",
    "NP",
    "NPSP",
    "NPSPR",
    "OPNAME",
    "ADMIN",
    "NPNOSMS",
    "TELECON", 
]

afspraak_weights = [10, 10, 10, 10, 10, 10, 10, 10, 2]

status_values = [
    "Is voldaan",
    "Is niet voldaan",
]

status_weights = [98, 10]

cancellation_values = [
    'N',
    'NULL'
]

cancellation_display_values = [
    'NULL',
    'Patient niet verschenen (of te laat gemeld)'
]

cancellation_weights = [50, 20]

def get_random_dict(dict, weights = None):
    return random.choices(dict, weights=weights, k=1)[0]

def random_datetime():
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    """Generate a random datetime between two dates, but only between 09:00 and 18:00."""
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    random_time = random.randint(9 * 60, 18 * 60)
    final_datetime = datetime.combine(random_date, datetime.min.time()) + timedelta(minutes=random_time)
    return final_datetime

def random_datetime_in_year(year):
    """Generate a random datetime within a given year."""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)
    random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
    return start_date + timedelta(seconds=random_seconds)


script_dir = Path(__file__).parent
csv_filename = script_dir / "poliafspraken_no_show.csv"

headers = ["APP_ID", "pseudo_id", "hoofdagenda", "hoofdagenda_id", "subagenda_id", "specialty_code", "soort_consult", "afspraak_code", "start", "end", "arrival",
           "created", "minutesDuration", "status", "status_code_original", "cancelationReason_code", "cancelationReason_display", "BIRTH_YEAR",
           "address_postalCodeNumbersNL", "name", "description", 
           #, #"name_given1_callMe", "telecom1_value", "telecom2_value", "telecom3_value", "birthDate"
           ]

csv_row_count = 500
unique_guests = 50
guest_book = []

for x in range(unique_guests):
    pseudo_id = 'A' + '{:04}'.format(random.randrange(1, 10**3))
    guest_book.append(pseudo_id)

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers
    for x in range(csv_row_count):
        APP_ID = '{:04}'.format(random.randrange(1, 10**3))
        pseudo_id = guest_book[random.randint(0, unique_guests-1)]
        hoofdagenda = "KNO"
        hoofdagenda_id = 'RS' + '{:02}'.format(random.randrange(1, 10**1))
        subagenda_id = 'SUB' + '{:02}'.format(random.randrange(1, 10**2))
        specialty_code = "KNKO"
        soort_consult = get_random_dict(soort_consult_values, afspraak_weights)
        afspraak_code = get_random_dict(afspraak_code_values, afspraak_weights)

        startTime = random_datetime()
        minutesDuration = random.randrange(10, 30)
        endTime = startTime + timedelta(minutes=minutesDuration)
        arrived = startTime - timedelta(minutes=minutesDuration)
        created = startTime - timedelta(days=10)

        status = str(get_random_dict(status_values, status_weights))
        status_code_original = status
        cancelationReason_code = get_random_dict(cancellation_values, cancellation_weights)
        cancelationReason_display = get_random_dict(cancellation_display_values, cancellation_weights)

        BIRTH_YEAR = random.randint(1930, 2024)
        address_postalCodeNumbersNL = '{:04}'.format(random.randrange(1, 10**4))

        name = 'NULL'
        description = 'NULL'
        #name_text = None
        #patient_id = None
        #name_given1_callMe = None
        #telecom1_value = None
        #telecom2_value = None
        #telecom3_value = None
        #birthDate = random_datetime_in_year(BIRTH_YEAR)

        writer.writerow([APP_ID, pseudo_id, hoofdagenda, hoofdagenda_id, subagenda_id, specialty_code, soort_consult, afspraak_code, startTime, endTime, arrived, created, minutesDuration, status, status_code_original,
                         cancelationReason_code, cancelationReason_display, BIRTH_YEAR, address_postalCodeNumbersNL, 
                         name, description, #name_text, patient_id, name_given1_callMe, telecom1_value, telecom2_value, telecom3_value, birthDate
                        ])
        

#notes about random data
#currently soort_consult and afspraak_code don't match (I dont think this matters)

print(f"CSV file generated: {csv_filename}")
