# -*- coding: utf-8 -*-
# f02_data/symbol_specs_snapshot.py
# Status in (Bot-RL-2): New (adds snapshot+overlay writer without altering core logic)

"""
ابزار ثبت «Snapshot مشخصات نمادها» از MT5 و تولید فایل overlay برای کانفیگ.
- پیام‌های runtime انگلیسی؛ توضیحات فارسی.
- بدون وابستگی به اسکریپت‌های خارج از حوزهٔ f01..f14.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import yaml, os

#from f10_utils.config_loader import load_config   # (delayed import after bootstrap)
#from f02_data.mt5_connector import MT5Connector, MT5Credentials

LOGGER = logging.getLogger("symbol_specs_snapshot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _resolve_symbols(cfg: dict) -> list[str]:
    # ترجیح: download_defaults.symbols ، در غیر این صورت data.symbols یا project.symbols یا symbols
    cand_keys = [
        ("download_defaults", "symbols"),
        ("data", "symbols"),
        ("project", "symbols"),
        ("symbols",),
    ]
    for keys in cand_keys:
        cur = cfg
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and cur:
            return [str(s) for s in cur]
    raise SystemExit("No symbols found in config (looked at download_defaults.symbols, data.symbols, project.symbols, symbols).")


def main():
    ap = argparse.ArgumentParser(description="Snapshot MT5 symbol specifications and write overlay YAML.")
    ap.add_argument("-c", "--config", default="f01_config/config.yaml")
    ap.add_argument("--overlay-out", default=None, help="Path to write overlay YAML (default derived from config dir)")
    ap.add_argument("--snapshot-out", default=None, help="Path to write raw snapshot YAML (default derived from config paths)")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional explicit list of symbols")
    args = ap.parse_args()


    # 1) از خود YAML خام، مسیر اوورراید را پیدا کن و اگر نبود، فایل خالی بساز (bootstrap)
    raw = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    base = Path(args.config).parent
    proj = base.parent

    ovr = (raw.get("overlays") or [])
    if isinstance(ovr, str):
        ovr = [ovr]
    for p in ovr:
        q = (proj / p) if not Path(p).is_absolute() else Path(p)
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text("symbol_specs: {}\n", encoding="utf-8")
    # 2) حالا لودر را (که ENV را هم اعمال می‌کند) صدا بزن
    from f10_utils.config_loader import load_config
    cfg = load_config(args.config, enable_env_override=True)

    symbols = args.symbols or _resolve_symbols(cfg)

    # Resolve defaults from config: data_dir from paths.data_dir or parent(processed_dir); overlay in same dir as config.yaml
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    config_dir = Path(args.config).resolve().parent
    proj_root = config_dir.parent
    data_dir = (Path(paths.get("data_dir") or Path(paths.get("processed_dir", "f02_data/processed")).parent))
    data_dir = data_dir if data_dir.is_absolute() else (proj_root / data_dir)
    overlay_out = (Path(args.overlay_out) if args.overlay_out else (config_dir / "symbol_specs.yaml")).resolve()
    snapshot_out = (Path(args.snapshot_out) if args.snapshot_out else (data_dir / "specs_snapshot.yaml")).resolve()

    # Ensure output dirs
    overlay_out.parent.mkdir(parents=True, exist_ok=True)

    # 3) حالا که overlay ساخته و config با ENV لود شده، MT5Connector را import کن
    from f02_data.mt5_connector import MT5Connector
    conn = MT5Connector(config=cfg)

    if not conn.initialize():
        raise SystemExit("Could not initialize MT5Connector.")

    try:
        snapshot = conn.get_symbol_specs(symbols)
    finally:
        conn.shutdown()


    
    sp = snapshot.get("symbol_specs", {}) or {}
    for s,d in sp.items():
        try:
            p, tv, ts = float(d.get("point") or 0), float(d.get("trade_tick_value") or 0), float(d.get("trade_tick_size") or 1)
            if p and tv and ts: d["pip_value_per_lot"] = round((tv/ts)*(10.0*p), 6)  # pip := 10*point
        except Exception: pass
    overlay = {"account_currency": (snapshot.get("meta", {}) or {}).get("account_currency"), "symbol_specs": sp}

    with open(overlay_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, allow_unicode=True, sort_keys=False)

    LOGGER.info("Wrote snapshot to %s and overlay to %s", snapshot_out, overlay_out)


if __name__ == "__main__":  # pragma: no cover
    main()
