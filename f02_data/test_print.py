# Run: python f02_data/test_print.py

import sys
sys.path.insert(0, r"E:\Bot-RL-2")

from mt5_data_loader_E import _parse_dt
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

INPUT_TZ = "Europe/Athens"
OUTPUT_TZ = "Asia/Tehran"

# =============================================================================
def print_result_1(label, value):
    try:
        result = _parse_dt(value, INPUT_TZ, OUTPUT_TZ)
        val_str = str(value) if value is not None else "None"
        print(f"{label:28}: {val_str:28} → {result}  (tz={result.tzinfo if result else None})")
    except Exception as e:
        print(f"{label:28}: {str(value):28} → ERROR: {e}")


# ===== ورودی‌های مختلف =====
print_result_1(" 1) None", None)
print_result_1(" 2) empty string", "")
print_result_1(" 3) date only", "2024-01-01")
print_result_1(" 4) ISO with Z", "2024-01-01T12:00:00Z")
print_result_1(" 5) ISO with space + Z", "2024-01-01 12:00:00Z")
print_result_1(" 6) ISO no timezone", "2024-01-01T12:00:00")
print_result_1(" 7) ISO with +03:30", "2024-01-01T12:00:00+04:00")
print_result_1(" 8) 'now'", "now")
print_result_1(" 9) datetime aware (UTC)", datetime(2024,1,1,12,0,tzinfo=timezone.utc))
print_result_1("10) datetime naive", datetime(2024,1,1,12,0))
print_result_1("11) date object", date(2024,1,1))

print("\n", _parse_dt("2024-07-16", INPUT_TZ,), "\n")


