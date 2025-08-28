# f04_features/indicators/parser.py
# -*- coding: utf-8 -*-
"""Parser برای Spec:  <name>(args)@TF  →  (name, args[], tf)"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, List, Optional

_SPEC_RE = re.compile(r"^\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?:\((?P<args>[^)]*)\))?(?:@(?P<tf>[A-Za-z0-9]+))?\s*$")

@dataclass
class Spec:
    name: str
    args: List[Any]
    tf: Optional[str]

def _parse_args_list(arg_str: str) -> List[Any]:
    if arg_str.strip() == "":
        return []
    out = []
    for tok in [t.strip() for t in arg_str.split(",")]:
        if tok.lower() in {"open","high","low","close","volume"}:
            out.append(tok.lower()); continue
        try:
            if "." in tok: out.append(float(tok))
            else: out.append(int(tok))
        except ValueError:
            out.append(tok)
    return out

def parse_spec(s: str) -> Spec:
    m = _SPEC_RE.match(s)
    if not m:
        raise ValueError(f"Invalid spec: {s}")
    name = (m.group("name") or "").lower()
    arg_str = m.group("args") or ""
    args = _parse_args_list(arg_str)
    tf = m.group("tf")
    return Spec(name=name, args=args, tf=(tf.upper() if tf else None))