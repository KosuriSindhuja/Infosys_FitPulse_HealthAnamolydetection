# src/extract_from_bson.py
"""
Stream-extract heart rate, steps and sleep records from a large fitbit.bson file.
This script is conservative: it writes only rows that match known field-name patterns.
It avoids storing the raw JSON payload to keep CSVs small.

Outputs (in data/raw/):
 - fitbit_heartrate_raw.csv  (columns: user_id,timestamp,heart_rate,source_type)
 - fitbit_steps_raw.csv      (columns: user_id,timestamp,steps,source_type)
 - fitbit_sleep_raw.csv      (columns: user_id,startTime,endTime,minutesAsleep,source_type)

Run:
    python src/extract_from_bson.py
"""
from pathlib import Path
from bson import decode_file_iter
import csv
import ujson
import sys

BASE = Path.cwd()
BSON_FILE = BASE / "fitbit.bson"
OUT_DIR = BASE / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HR_CSV = OUT_DIR / "fitbit_heartrate_raw.csv"
STEPS_CSV = OUT_DIR / "fitbit_steps_raw.csv"
SLEEP_CSV = OUT_DIR / "fitbit_sleep_raw.csv"

# Patterns of key names we will look for (adjust later if needed)
HR_KEYS = {"heart_rate","heartrate","hr","bpm","value"}          # hr value names
STEPS_KEYS = {"steps","step_count","total_steps","stepCount"}   # step names
SLEEP_KEYS = {"minutesAsleep","sleep_minutes","minutes_asleep","sleepMinutes","sleep"}  # sleep names

TS_KEYS = ("timestamp","time","startTime","dateTime","created_at","ts","date")

# Helpers
def get_user_id(rec):
    # try common top-level or nested places for a user id
    for k in ("user_id","id","userid","owner","owner_id","participant_id","subject"):
        if k in rec:
            return rec[k]
    # sometimes inside meta or data
    if "data" in rec and isinstance(rec["data"], dict):
        for k in ("user_id","userid","owner_id"):
            if k in rec["data"]:
                return rec["data"][k]
    return None

def get_timestamp_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    for k in TS_KEYS:
        if k in payload and payload[k] not in (None, ""):
            return payload[k]
    # sometimes timestamp is nested under 'date' or similar
    return None

def safe_val(payload, candidates):
    if not isinstance(payload, dict):
        return None
    for c in candidates:
        if c in payload:
            return payload[c]
    return None

# Prepare CSV writers (write headers)
def writer_setup(path, header):
    write_header = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    return f, w

hr_file, hr_writer = writer_setup(HR_CSV, ["user_id","timestamp","heart_rate","source_type"])
steps_file, steps_writer = writer_setup(STEPS_CSV, ["user_id","timestamp","steps","source_type"])
sleep_file, sleep_writer = writer_setup(SLEEP_CSV, ["user_id","startTime","endTime","minutesAsleep","source_type"])

if not BSON_FILE.exists():
    print("ERROR: fitbit.bson not found in project root. Move fitbit.bson here or update BSON_FILE path.")
    sys.exit(1)

print("Starting streaming extraction from:", BSON_FILE)
count = 0
matched = {"hr":0,"steps":0,"sleep":0}

with open(BSON_FILE, "rb") as fh:
    for rec in decode_file_iter(fh):
        count += 1
        # conservative: get payload where sensor data likely lives
        payload = None
        if isinstance(rec, dict):
            # preference to nested 'data' or 'payload'
            if "data" in rec and isinstance(rec["data"], dict):
                payload = rec["data"]
            elif "payload" in rec and isinstance(rec["payload"], dict):
                payload = rec["payload"]
            elif isinstance(rec.get("value"), dict):
                payload = rec["value"]
            else:
                # fallback to entire record (may contain keys directly)
                payload = rec

        # normalize user id & source type
        user_id = get_user_id(rec)
        source_type = rec.get("type") if isinstance(rec.get("type"), str) else None

        # quick guard: payload must be dict to inspect keys
        if isinstance(payload, dict):
            keys = set(payload.keys())

            # HEART RATE detection
            if keys & HR_KEYS:
                # timestamp (if present)
                ts = get_timestamp_from_payload(payload)
                hr_val = safe_val(payload, tuple(HR_KEYS))
                try:
                    # coerce numeric if possible
                    hr_val = float(hr_val) if hr_val is not None else None
                except:
                    hr_val = None
                hr_writer.writerow([user_id, ts, hr_val, source_type])
                matched["hr"] += 1

            # STEPS detection
            elif keys & STEPS_KEYS:
                ts = get_timestamp_from_payload(payload)
                s_val = safe_val(payload, tuple(STEPS_KEYS))
                try:
                    s_val = int(s_val) if s_val is not None else 0
                except:
                    s_val = 0
                steps_writer.writerow([user_id, ts, s_val, source_type])
                matched["steps"] += 1

            # SLEEP detection (start/end, minutes)
            elif keys & SLEEP_KEYS or ("startTime" in payload and "endTime" in payload):
                start = payload.get("startTime") or payload.get("start_time") or payload.get("start")
                end = payload.get("endTime") or payload.get("end_time") or payload.get("end")
                mins = safe_val(payload, tuple(SLEEP_KEYS))
                try:
                    mins = float(mins) if mins is not None else None
                except:
                    mins = None
                sleep_writer.writerow([user_id, start, end, mins, source_type])
                matched["sleep"] += 1

        # progress prints
        if count % 100000 == 0:
            print(f"Processed {count:,} records — matched: {matched}")

# close files
hr_file.close()
steps_file.close()
sleep_file.close()

print("Done. Records processed:", count)
print("Matched counts:", matched)
print("Wrote files to", OUT_DIR)
