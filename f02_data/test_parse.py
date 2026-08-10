# Run: pytest f02_data/test_parse.py -v
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from mt5_data_loader_E import _parse_dt as _parse_dt

INPUT_TZ = "Asia/Tehran"   # منطقه ورودی فرضی
OUTPUT_TZ = "UTC"          # منطقه خروجی نهایی

def test_none():
    assert _parse_dt(None, INPUT_TZ, OUTPUT_TZ) is None

def test_empty_string():
    assert _parse_dt("", INPUT_TZ, OUTPUT_TZ) is None

def test_datetime_aware():
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = _parse_dt(dt, INPUT_TZ, OUTPUT_TZ)
    # ابتدا dt باید به INPUT_TZ تبدیل شود، سپس به OUTPUT_TZ
    expected = dt.astimezone(ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_datetime_naive():
    dt = datetime(2024, 1, 1, 12, 0)
    result = _parse_dt(dt, INPUT_TZ, OUTPUT_TZ)
    expected = dt.replace(tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_date_object():
    d = date(2024, 1, 1)
    result = _parse_dt(d, INPUT_TZ, OUTPUT_TZ)
    expected = datetime.combine(d, datetime.min.time(), tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_now_string():
    result = _parse_dt("now", INPUT_TZ, OUTPUT_TZ)
    assert isinstance(result, datetime)
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_date_only_string():
    result = _parse_dt("2024-01-01", INPUT_TZ, OUTPUT_TZ)
    expected = datetime(2024, 1, 1, 0, 0, tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_iso_with_z():
    result = _parse_dt("2024-01-01T12:00:00Z", INPUT_TZ, OUTPUT_TZ)
    # منطقه Z نادیده گرفته شده و INPUT_TZ جایگزین می‌شود
    expected = datetime(2024, 1, 1, 12, 0, tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_iso_naive():
    result = _parse_dt("2024-01-01T12:00:00", INPUT_TZ, OUTPUT_TZ)
    expected = datetime(2024, 1, 1, 12, 0, tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

def test_invalid_format():
    with pytest.raises(ValueError, match="Invalid date format"):
        _parse_dt("2024/01/01", INPUT_TZ, OUTPUT_TZ)

def test_invalid_timezone():
    with pytest.raises(ValueError, match="Invalid timezone"):
        _parse_dt("2024-01-01", "Invalid/Timezone", OUTPUT_TZ)