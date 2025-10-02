# -*- coding: utf-8 -*-
# f09_execution/promote.py   (Promote مدل از staging به prod)
# Status in (Bot-RL-2): Completed

"""
Promote مدل از staging به prod (اتمیک و نسخه‌گذاری شده).
- هیچ حدسی دربارهٔ ساختار مدل‌ها نمی‌زنیم؛ صرفاً فایل .npz را کپی/اتمیک می‌کنیم.
- اگر health-check/criteria دیگری دارید، این‌جا اضافه کنید (الان صرفاً وجود فایل و اندازه بررسی می‌شود).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from f10_utils.config_loader import load_config
from f03_env.utils import paths_from_cfg

LOGGER = logging.getLogger("promote")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _ok_file(p: Path) -> bool:
    """بررسی سادهٔ سلامت فایل (وجود و حداقل اندازه)."""
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def promote_model(symbol: str, cfg_path: str = "f01_config/config.yaml", base_tf: str = "M5") -> Path:
    """
    کپی مدل از staging به prod با نام استاندارد.
    در صورت موفقیت مسیر مقصد را برمی‌گرداند.
    """
    cfg = load_config(cfg_path, enable_env_override=True)
    paths = paths_from_cfg(cfg)

    src = paths["models_dir"] / "staging" / f"{symbol}_{base_tf}_reinforce_linear.npz"
    if not _ok_file(src):
        # -----
        #raise FileNotFoundError(f"Staging model not found or invalid: {src}")
        
        # تلاش برای یافتن آخرین مدل staging با هر TF (Auto-detect)
        stg_dir = paths["models_dir"] / "staging"
        candidates = sorted(
            stg_dir.glob(f"{symbol}_*_reinforce_linear.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            # اگر دقیقِ base_tf موجود نیست، جدیدترین را می‌گیریم و هشدار می‌دهیم
            LOGGER.warning("Requested base_tf %s not found; using latest staging model: %s",
                        base_tf, candidates[0].name)
            src = candidates[0]
            # TF واقعی را از نام فایل استخراج می‌کنیم تا نام مقصد صحیح باشد
            import re
            m = re.match(rf"{symbol}_(?P<tf>[A-Z0-9]+)_reinforce_linear\.npz$", src.name)
            if m:
                base_tf = m.group("tf")
        else:
            raise FileNotFoundError(f"Staging model not found or invalid: {src}")


    dst_dir = paths["models_dir"] / "prod"
    dst_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dst = dst_dir / f"{symbol}_{base_tf}_reinforce_linear_{stamp}.npz"

    # کپی اتمیک: اول tmp سپس rename
    tmp = dst.with_suffix(".npz.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    LOGGER.info("Model promoted to: %s", dst)
    # لینک آخرین نسخه (اختیاری)
    latest = dst_dir / f"{symbol}_{base_tf}_reinforce_linear.npz"
    try:
        if latest.exists():
            latest.unlink()
        shutil.copy2(dst, latest)
        LOGGER.info("Latest symlink updated: %s", latest)
    except Exception:
        LOGGER.warning("Failed to update latest copy (non-fatal).")
    return dst


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Promote model from staging to prod")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("-c", "--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default="M5")
    args = p.parse_args()
    promote_model(symbol=args.symbol, cfg_path=args.config, base_tf=args.base_tf.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
