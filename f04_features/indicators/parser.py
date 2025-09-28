# f04_features/indicators/parser.py
# -*- coding: utf-8 -*-

r"""Parser برای Spec:  <name>(args)@TF  →  (name, args[], tf)

افزودنی‌های پارس Spec (Bot-RL-2)
- parse_spec_v2: پارسِ نام/آرگومان‌ها/کلیدواژه‌ها + @TF
- سازگار با نگارش‌های قدیمی (هیچ تابع قبلی حذف نمی‌شود)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import ast
import re
import logging
import inspect
from .registry import get_indicator_v2

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------
# پارسر توسعه‌یافتهٔ Spec
# -----------------------------
_SPEC_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<args>.*)\))?\s*(?:@(?P<tf>[A-Za-z0-9]+))?\s*$", re.DOTALL)
'''
@dataclass
class Spec:
    name: str
    args: List[Any]
    tf: Optional[str]
'''
# --- New Added ----------------------------------------------------- 040607
# -----------------------------
# مدل دادهٔ خروجی پارس
# -----------------------------
@dataclass
class ParsedSpec:
    name: str                 # نام اندیکاتور (registry key)
    args: List[Any]           # آرگومان‌های موقعیتی
    kwargs: Dict[str, Any]    # آرگومان‌های کلیدواژه‌ای
    timeframe: Optional[str]  # @TF اگر وجود داشت
    raw: str                  # متن خام Spec (برای لاگ/دیباگ)


# -----------------------------
# کمکی‌ها: شکستن آرگومان‌ها در سطح-بالا
# -----------------------------
def _split_top_level_commas(s: str) -> List[str]:
    """تقسیم بر اساس ویرگول‌ها در سطح-بالا (داخل []/() دست‌نخورده بماند)."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{" :
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [t for t in out if t != ""]


def _parse_value(token: str) -> Any:
    """تبدیل توکن به مقدار پایتونی؛ ایمن و سازگار با TFهای بدون کوتیشن (M1,H1,...)"""
    # ساده‌سازی: True/False/None کوچک/بزرگ
    low = token.strip()
    if low.lower() in ("true", "false", "none", "null"):
        return {"true": True, "false": False, "none": None, "null": None}[low.lower()]
    # تلاش: literal_eval
    try:
        return ast.literal_eval(low)
    except Exception:
        pass
    # تبدیل TFهای بدون کوتیشن داخل لیست‌ها: [M1,H1] → ["M1","H1"]
    if low.startswith("[") and low.endswith("]"):
        inner = low[1:-1].strip()
        parts = _split_top_level_commas(inner)
        vals = []
        for p in parts:
            p = p.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", p):
                vals.append(p)  # TF token as string
            else:
                try:
                    vals.append(ast.literal_eval(p))
                except Exception:
                    vals.append(str(p))
        return vals
    # اعداد ساده
    if re.fullmatch(r"[-+]?\d+", low):
        return int(low)
    if re.fullmatch(r"[-+]?\d*\.\d+", low):
        return float(low)
    # پیش‌فرض: رشته
    return str(low)


def _parse_args_kwargs(argstr: Optional[str]) -> Tuple[List[Any], Dict[str, Any]]:
    """پارس آرگومان‌های موقعیتی و کلیدواژه‌ای."""
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    if not argstr or argstr.strip() == "":
        return args, kwargs

    tokens = _split_top_level_commas(argstr)
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kwargs[k.strip()] = _parse_value(v.strip())
        else:
            args.append(_parse_value(tok.strip()))
    return args, kwargs

# مثال‌های قابل پشتیبانی:
#   golden_zone(0.382,0.618)@H1
#   fib_cluster(tf=[H1,H4,D1], tol_pct=0.1, prefer_ratio=0.618)
#   fib_ext_targets(atr_mult=1.5)@H1
#   ma_slope(window=20, method='ema')@M5
#   rsi_zone(period=14)@H1

'''
# ========================= Bot-RL-2 :: Phase C :: Arg-Mapping Helper (ANCHOR: ARG_MAPPING_HELPER) =========================
def _align_args_with_signature(ind_name: str, args_in: List[Any], kwargs_in: Dict[str, Any]) -> tuple[list, dict]:
    """
    نگاشت آرگومان‌های موقعیتی Spec به نام پارامترها بر اساس امضای اندیکاتور ثبت‌شده در رجیستری v2.
    - مثال: 'sma(20)' → make_sma(df, col='close', period=20)  ← period از روی ترتیب پارامتر دوم به بعد نگاشت می‌شود.
    - اگر پارامتر از قبل در kwargs باشد، مقدار موقعیتی همان پارامتر نادیده گرفته می‌شود (اولویت با kwargs است).
    - آرگومان اول توابع اندیکاتور معمولاً DataFrame (df/ohlc) است؛ در نگاشت نادیده گرفته می‌شود.
    - در صورت نبودِ اندیکاتور یا امضای غیرقابل‌تحلیل، همان args/kwargs اولیه برگردانده می‌شود.
    """
    try:
        fn = get_indicator_v2(ind_name)
        if fn is None:
            return list(args_in), dict(kwargs_in)

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # پارامترهای کاندید برای نگاشت: از «بعدِ» اولین پارامترِ df/ohlc به بعد
        # فقط POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD / KEYWORD_ONLY
        cand_names: list[str] = []
        for i, p in enumerate(params):
            if i == 0:
                # فرض: پارامتر اول، df/ohlc است و نگاشت نمی‌گیرد
                continue
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                # نام‌های رایج‌ناپذیرفتنی را نیز حذف نکنیم؛ فقط **kwargs و *args را رد می‌کنیم
                cand_names.append(p.name)

        new_kwargs = dict(kwargs_in)
        new_args: list[Any] = []
        ai = 0
        for i, pname in enumerate(cand_names):
            if ai >= len(args_in):
                break
            # اگر کاربر قبلاً همین پارامتر را به‌صورت kwargs داده، همان را نگه می‌داریم
            if pname in new_kwargs:
                continue
            # مقدار موقعیتی را به نام همین پارامتر نگاشت می‌کنیم
            new_kwargs[pname] = args_in[ai]
            ai += 1

        # باقی‌ماندهٔ args (اگر بیشتر از پارامترهای اسمی باشد) را به‌صورت positional عبور بدهیم
        while ai < len(args_in):
            new_args.append(args_in[ai])
            ai += 1

        return new_args, new_kwargs
    except Exception:
        # هرگونه مشکل در تشخیص امضا → بدون تغییر برگردان
        return list(args_in), dict(kwargs_in)
# ========================= End of ARG_MAPPING_HELPER ================================================================================
'''
# ========================= Bot-RL-2 :: Phase C :: Arg-Mapping Helper (ANCHOR: ARG_MAPPING_HELPER) =========================
def _align_args_with_signature(ind_name: str, args_in: List[Any], kwargs_in: Dict[str, Any]) -> tuple[list, dict]:
    """
    نگاشت آرگومان‌های موقعیتی Spec به نام پارامترها بر اساس امضای اندیکاتور.
    قواعد:
      - عددی‌ها (int/float) → به پارامترهای عددیِ متداول: period/n/window/length/fast/slow/signal/k/d
      - رشته‌ها (str) → به پارامترهای ستونی: col/column/field
      - باقی موارد → به ترتیب امضای تابع (fallback)
    """
    try:
        fn = get_indicator_v2(ind_name)
        if fn is None:
            return list(args_in), dict(kwargs_in)

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # پارامترهای کاندید برای نگاشت: همهٔ پارامترها به‌جز اولی (df/ohlc)
        cands = [p for i, p in enumerate(params) if i > 0 and p.kind in (
            p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY
        )]
        cand_names = [p.name for p in cands]

        # اولویت نگاشت
        numeric_pref = [n for n in ("period","n","window","length","fast","slow","signal","k","d")
                        if n in cand_names and n not in kwargs_in]
        str_pref     = [n for n in ("col","column","field")
                        if n in cand_names and n not in kwargs_in]

        new_kwargs = dict(kwargs_in)
        new_args: list[Any] = []

        for v in args_in:
            assigned = False
            # 1) عددی‌ها → پارامترهای عددی
            if isinstance(v, (int, float)) and numeric_pref:
                tgt = numeric_pref.pop(0)
                new_kwargs[tgt] = v
                assigned = True
            # 2) رشته‌ها → پارامترهای ستونی
            elif isinstance(v, str) and str_pref:
                tgt = str_pref.pop(0)
                new_kwargs[tgt] = v
                assigned = True
            # 3) fallback: به ترتیب اولین پارامتر آزاد
            if not assigned:
                for pn in cand_names:
                    if pn in new_kwargs:
                        continue
                    new_kwargs[pn] = v
                    assigned = True
                    break
            # 4) اگر هیچ‌کدام نشد، عبور به‌صورت positional
            if not assigned:
                new_args.append(v)

        return new_args, new_kwargs
    except Exception:
        # هر اشکالی در تحلیل امضا → بدون تغییر
        return list(args_in), dict(kwargs_in)
# ========================= End of ARG_MAPPING_HELPER ================================================================================


def parse_spec_v2(spec: str) -> ParsedSpec:
    """
    پارس Spec به ساختار استاندارد ParsedSpec.
    - name: نام اندیکاتور
    - args/kwargs: پارس امن
    - timeframe: اگر @TF داده شده باشد
    """
    m = _SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"Invalid spec: {spec}")
    name, argstr, tf = m.group(1), m.group(2), m.group(3)
    args, kwargs = _parse_args_kwargs(argstr)

    # --- Phase C: map positional args to named params via registry signature ---
    args, kwargs = _align_args_with_signature(name, args, kwargs)

    # نرمال‌سازیِ TF (اگر بود)
    tf_norm = tf.upper() if tf else None
    ps = ParsedSpec(name=name, args=args, kwargs=kwargs, timeframe=tf_norm, raw=spec)
    logger.debug("parse_spec_v2: %s -> %s", spec, ps)
    return ps
