# f10_utils/config_ops.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Union, Iterable
import yaml
import logging


""" --------------------------------------------------------------------------- OK Func1
"""
def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """دسترسی امن به مسیرهای تو در توی دیکشنری با dot-notation؛ در نبود کلید، مقدار پیش‌فرض برمی‌گردد."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


""" --------------------------------------------------------------------------- OK Func2,3
# deep_set / yaml_set (FINAL)
# -------------------------
# توضیح: این توابع برای به‌روزرسانی امنِ کلیدهای config به‌صورت برنامه‌وار استفاده می‌شوند.
# - deep_set: معکوسِ _deep_get؛ مقدار را در مسیر دلخواه قرار می‌دهد (in-place).
# - yaml_set: فایل YAML را لود می‌کند، deep_set را اعمال می‌کند و ذخیره می‌کند.
# ورودی path می‌تواند رشته (مانند "indicators.patterns[3].tol_k") یا لیست توکن‌ها باشد.
# پیام‌های ترمینال/لاگ: انگلیسی. کامنت‌ها: فارسی.
# =========================
"""

PathLike = Union[str, List[Union[str, int]]]

def _parse_path(path: PathLike) -> List[Union[str, int]]:
    """
    مسیر را به توکن‌های قابل پیمایش تبدیل می‌کند.
    مثال‌ها:
      "a.b[3].c"  -> ["a","b",3,"c"]
      ["a","b",3,"c"] -> همان
    """
    if isinstance(path, list):
        return path[:]
    s = str(path).strip()
    out: List[Union[str, int]] = []
    buf = ""
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '.':
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if ch == '[':
            if buf:
                out.append(buf); buf = ""
            j = s.find(']', i+1)
            if j == -1:
                raise ValueError(f"Malformed path (missing ']'): {s}")
            token = s[i+1:j].strip().strip('"').strip("'")
            # اندیس عددی یا کلید رشته‌ای
            if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
                out.append(int(token))
            else:
                out.append(token)
            i = j + 1
            continue
        buf += ch
        i += 1
    if buf:
        out.append(buf)
    return out


def deep_set(obj: Any, path: PathLike, value: Any, create_missing: bool = True) -> Any:
    """
    مقدار را در مسیر دلخواه قرار می‌دهد (in-place).
    - اگر create_missing=True باشد، شاخه‌های مفقود ساخته می‌شوند (dict یا list).
    - نوع شاخهٔ ساخته‌شده براساس توکن بعدی تعیین می‌شود (int -> list، else -> dict).

    مثال:
      deep_set(cfg, "indicators.patterns[3].tol_k", 0.35)
      deep_set(cfg, ["indicators","patterns",3,"tol_k"], 0.35)
    """
    tokens = _parse_path(path)
    if not tokens:
        raise ValueError("Empty path")

    cur = obj
    for idx, tk in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        # دیکشنری
        if isinstance(tk, str):
            if not isinstance(cur, dict):
                if not create_missing:
                    raise TypeError(f"Cannot descend into non-dict at {tokens[:idx+1]}")
                # تبدیل اجباری
                raise TypeError(f"Expected dict at {tokens[:idx]}, got {type(cur).__name__}")
            if tk not in cur or cur[tk] is None:
                if not create_missing:
                    raise KeyError(f"Missing key: {tk}")
                # نوع شاخهٔ جدید
                cur[tk] = [] if isinstance(nxt, int) else {}
            cur = cur[tk]
        # لیست با اندیس عددی
        elif isinstance(tk, int):
            if not isinstance(cur, list):
                if not create_missing:
                    raise TypeError(f"Cannot index non-list at {tokens[:idx+1]}")
                raise TypeError(f"Expected list at {tokens[:idx]}, got {type(cur).__name__}")
            # توسعهٔ لیست در صورت نیاز
            need_len = tk + 1
            if len(cur) < need_len:
                cur.extend([None] * (need_len - len(cur)))
            if cur[tk] is None:
                cur[tk] = [] if isinstance(nxt, int) else {}
            cur = cur[tk]
        else:
            raise TypeError(f"Unsupported token type at {tokens[:idx+1]}: {type(tk).__name__}")

    last = tokens[-1]
    if isinstance(last, str):
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot set key on non-dict at {tokens[:-1]}")
        cur[last] = value
    elif isinstance(last, int):
        if not isinstance(cur, list):
            raise TypeError(f"Cannot set index on non-list at {tokens[:-1]}")
        need_len = last + 1
        if len(cur) < need_len:
            cur.extend([None] * (need_len - len(cur)))
        cur[last] = value
    else:
        raise TypeError(f"Unsupported final token: {type(last).__name__}")

    return obj


def yaml_set(path_to_yaml: str, key_path: PathLike, value: Any, create_missing: bool = True, save_backup: bool = True) -> None:
    """
    فایل YAML را باز می‌کند، deep_set را اعمال می‌کند و ذخیره می‌کند.
    - اگر save_backup=True باشد، از فایل یک کپی .bak می‌سازد.
    """
    import os
    if not os.path.exists(path_to_yaml):
        raise FileNotFoundError(f"YAML not found: {path_to_yaml}")

    with open(path_to_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    deep_set(cfg, key_path, value, create_missing=create_missing)

    if save_backup:
        bak = path_to_yaml + ".bak"
        try:
            if os.path.exists(bak):
                os.remove(bak)
        except Exception:
            pass
        try:
            import shutil
            shutil.copy2(path_to_yaml, bak)
        except Exception as e:
            logging.warning(f"[WARN] Could not create backup: {e}")

    with open(path_to_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Config updated: {key_path} -> {value}")

