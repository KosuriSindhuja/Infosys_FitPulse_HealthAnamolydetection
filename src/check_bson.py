from bson import decode_file_iter
from pathlib import Path
import sys

BSON = Path("fitbit.bson")

if not BSON.exists():
    print("fitbit.bson not found")
    sys.exit(1)

with open(BSON, "rb") as f:
    for i, rec in enumerate(decode_file_iter(f)):
        print(f"\n------ RECORD {i} ------")

        # print top-level keys
        print("Top-level keys:", list(rec.keys()))

        # print each key and type of its value
        for k, v in rec.items():
            print(f"  {k}: {type(v)}")

        print("------------------------\n")

        if i == 2:  # print first 3 records
            break
