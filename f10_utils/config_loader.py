# -*- coding: utf-8 -*-
# f10_utils/config_loader.py
"""
ConfigLoader (نسخهٔ پیشرفته برای Bot-RL-1)
-------------------------------------------
امکانات (کامل و توسعه‌پذیر):
- بارگذاری YAML از مسیر پیش‌فرض پروژه یا مسیر دلخواه
- بارگذاری متغیرهای محیطی از .env (در ریشهٔ پروژه) و اوورراید کلیدها
- پشتیبانی merge چند-فایلی: top-level keys: `extends` (Baseها) و `overlays` (اوِررایدها)
- اعتبارسنجی شِمای حداقلی و ارتقایافته (بر اساس Basic_2ok و نیازهای Self-Optimize/Executor)
- aliasing برای سازگاری نام‌های قدیمی/جدید (مثلاً max_dd_max -> max_drawdown_max)
- ابزارهای کمکی مسیرها و نسخه‌گذاری: `save_config_versioned(cfg, prefix="prod_")`
- API: get(key, default), get_all(copy=False), reload(), dump(path), ensure_dirs()

نکته:
- فقط از کتابخانه‌های استاندارد + PyYAML + python-dotenv استفاده شده.
- شِما سبک است تا نیاز به نصب پکیج اضافی نباشد؛ در صورت نیاز می‌توان به Pydantic ارتقا داد.
"""

from __future__ import annotations

import os
import copy
import yaml
import logging

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ============================================================
# ابزارهای کمکی سطح پایین
# ============================================================
# جایگزینِ مقاوم:
def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "f01_config").exists() and (p / "f10_utils").exists():
            return p
    return here.parent  # fallback

def _default_config_path() -> Path:
    """
    مسیر پیش‌فرض فایل کانفیگ.
    بنا به درخواست کاربر، از `config.yaml` استفاده می‌کنیم.
    """
    return _project_root() / "f01_config" / "config.yaml"

def _read_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error at {p}: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"Root of YAML must be a mapping (dict). Got: {type(data)}")
    return data

def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """merge عمیق دیکشنری‌ها با حفظ نوع‌ها و عدم دست‌کاری base."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return copy.deepcopy(overlay)
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _ensure_dir(d: Union[str, Path]) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# ============================================================
# ConfigLoader
# ============================================================
class ConfigLoader:
    """
    لودر کانفیگ پروژه با امکانات Merge/ENV/Validation/Versioning.
    """
    def __init__(self,
                 config_path: Optional[Union[str, Path]] = None,
                 env_prefix: str = "BOT_",
                 enable_env_override: bool = True):
        """
        config_path: مسیر YAML. اگر None باشد از f01_config/config.yaml استفاده می‌شود.
        env_prefix: پیشوند کلیدهای محیطی (مثلاً BOT_). جستجوی بدون پیشوند نیز انجام می‌شود.
        enable_env_override: اگر False، ENV نادیده گرفته می‌شود.
        """
        self.base_dir: Path = _project_root()

        # بارگذاری .env از ریشهٔ پروژه (اگر موجود باشد)
        env_path = self.base_dir / ".env"
        if env_path.exists():
            try:
                load_dotenv(dotenv_path=str(env_path))
            except Exception:
                logger.exception("Failed to load .env from %s", env_path)

        self.config_path: Path = Path(config_path) if config_path else _default_config_path()
        self.env_prefix: str = env_prefix
        self.enable_env_override: bool = bool(enable_env_override)

        self.config: Dict[str, Any] = {}
        self.reload()

    # ---------- بارگذاری اولیه با merge لایه‌ای ----------
    def _load_yaml_layered(self) -> Dict[str, Any]:
        """
        YAML اصلی را می‌خواند و اگر کلیدهای top-level زیر وجود داشت ادغام می‌کند:
          - extends: list[str]  → به ترتیب خوانده و به‌عنوان Base merge می‌کند (اولی کم‌اهمیت‌تر)
          - overlays: list[str] → در انتها روی نتیجه merge می‌کند (بالاترین اولویت)
        مسیرها می‌توانند نسبی به ریشهٔ پروژه یا مطلق باشند.
        """
        root_cfg = _read_yaml_file(self.config_path)

        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (self.base_dir / pp)

        # extends / bases
        merged: Dict[str, Any] = {}
        for section in ("extends", "bases"):
            files = root_cfg.get(section, []) or []
            if not isinstance(files, list):
                raise ValueError(f"'{section}' must be a list of file paths.")
            for f in files:
                merged = _deep_merge(merged, _read_yaml_file(_resolve(f)))

        # سپس YAML اصلی را روی Base بنشانیم
        merged = _deep_merge(merged, {k: v for k, v in root_cfg.items() if k not in ("extends", "bases", "overlays")})

        # overlays
        overlays = root_cfg.get("overlays", []) or []
        if not isinstance(overlays, list):
            raise ValueError("'overlays' must be a list of file paths.")
        for f in overlays:
            merged = _deep_merge(merged, _read_yaml_file(_resolve(f)))

        return merged

    # ---------- ENV name helper ----------
    def _env_name_for_path(self, path: Iterable[str]) -> str:
        # مثال مسیر: ["mt5_credentials","login"] -> BOT_MT5_CREDENTIALS_LOGIN
        return f"{self.env_prefix}{'_'.join(p.upper() for p in path)}"

    # ---------- casting ----------
    def _cast_env_value(self, val_str: str, original_value: Any):
        """
        تلاش برای تبدیل مقدار ENV به نوع مناسب:
        - "true"/"false" → bool
        - int/float
        - لیست comma-separated
        - در غیر اینصورت string
        """
        s = (val_str or "").strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # int
        try:
            return int(s)
        except Exception:
            pass
        # float
        try:
            return float(s)
        except Exception:
            pass
        # list
        if "," in s:
            return [item.strip() for item in s.split(",") if item.strip() != ""]
        return s

    # ---------- اعمال ENV ----------
    def _apply_env_overrides(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        برای هر برگ از ساختار cfg، اگر ENV متناظر وجود داشته باشد، مقدار override می‌شود.
        جستجو با دو نام انجام می‌شود: با پیشوند (BOT_...) و بدون پیشوند (برای راحتی).
        """
        def recurse(node, path):
            if isinstance(node, dict):
                return {k: recurse(v, path + [k]) for k, v in node.items()}
            if not self.enable_env_override:
                return node
            # با پیشوند
            env_name = self._env_name_for_path(path)
            env_val = os.getenv(env_name)
            if env_val is not None:
                try:
                    casted = self._cast_env_value(env_val, node)
                    logger.debug("ENV override %s (%s) = %r", ".".join(path), env_name, casted)
                    return casted
                except Exception:
                    logger.exception("Failed to cast env var %s", env_name)
                    return node
            # بدون پیشوند
            env_no_prefix = "_".join(p.upper() for p in path)
            env_val2 = os.getenv(env_no_prefix)
            if env_val2 is not None:
                try:
                    return self._cast_env_value(env_val2, node)
                except Exception:
                    return node
            return node

        return recurse(cfg, [])

    # ---------- سازگاری نام کلیدها (Aliases) ----------
    def _apply_compatibility_aliases(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        نگاشت برخی نام‌های قدیمی به جدید برای جلوگیری از ناسازگاری.
        مثال‌ها:
          - evaluation.acceptance_gates.max_dd_max  →  max_drawdown_max
          - executor.canary.volume_mult            →  executor.canary_deployment.volume_multiplier
          - executor.slippage_max_pips             →  executor.slippage_cap_pips
          - training.total_steps                   →  training.total_timesteps
        """
        out = copy.deepcopy(cfg)
        try:
            # 1) acceptance gates: max_dd_max → max_drawdown_max
            gates = (((out.get("evaluation") or {}).get("acceptance_gates")) or {})
            if "max_dd_max" in gates and "max_drawdown_max" not in gates:
                gates["max_drawdown_max"] = gates.pop("max_dd_max")
                out.setdefault("evaluation", {})["acceptance_gates"] = gates

            # 2) executor aliases
            ex = out.get("executor") or {}
            # slippage
            if "slippage_max_pips" in ex and "slippage_cap_pips" not in ex:
                ex["slippage_cap_pips"] = ex.pop("slippage_max_pips")
                out["executor"] = ex
            # canary: volume_mult → canary_deployment.volume_multiplier
            if "canary" in ex and isinstance(ex["canary"], dict):
                can = ex["canary"]
                if "volume_mult" in can:
                    out.setdefault("executor", {}).setdefault("canary_deployment", {})["volume_multiplier"] = can["volume_mult"]
            # ممکن است ساختار درست already در canary_deployment باشد—دست نمی‌زنیم.

            # 3) training: total_steps → total_timesteps
            tr = out.get("training") or {}
            if "total_steps" in tr and "total_timesteps" not in tr:
                tr["total_timesteps"] = tr.pop("total_steps")
                out["training"] = tr

        except Exception:
            logger.debug("Alias application skipped", exc_info=True)
        return out

    # ---------- اعتبارسنجی حداقلی/ارتقایافته ----------
    @staticmethod
    def _validate(cfg: Dict[str, Any]) -> None:
        """
        بررسی حضور کلیدهای حیاتی و نوع داده‌ها. در صورت مشکل، ValueError می‌دهد.
        هستهٔ حیاتی (الزامی):
          - evaluation.acceptance_gates: sharpe_min, winrate_min, max_drawdown_max (float)
          - risk: risk_per_trade (float)
        توصیه‌شده/عملیاتی (بهتر است باشند؛ در نبودشان خطا نمی‌دهیم اما هشدار می‌دهیم):
          - executor.canary_deployment.enabled/volume_multiplier
          - self_optimize.schedule و steps و acceptance_gates
          - rl.algorithm, env.action_space.type
          - paths.logs_dir/models_dir/config_versions_dir
        """
        # ---- هستهٔ حیاتی
        eval_ = cfg.get("evaluation") or {}
        gates = eval_.get("acceptance_gates") or {}
        for k in ("sharpe_min", "winrate_min", "max_drawdown_max"):
            if k not in gates:
                raise ValueError(f"Missing evaluation.acceptance_gates.{k}")
            try:
                float(gates[k])
            except Exception:
                raise ValueError(f"evaluation.acceptance_gates.{k} must be numeric")

        risk = cfg.get("risk") or {}
        if "risk_per_trade" not in risk:
            raise ValueError("Missing risk.risk_per_trade")
        try:
            float(risk["risk_per_trade"])
        except Exception:
            raise ValueError("risk.risk_per_trade must be numeric")

        # ---- موارد توصیه‌شده (هشدار در صورت نبود)
        def _warn(msg: str) -> None:
            logger.warning("Config validation warning: %s", msg)

        # executor.canary_deployment
        ex = cfg.get("executor") or {}
        can = ex.get("canary_deployment") or {}
        if not can:
            _warn("executor.canary_deployment is missing (recommended for safe rollout).")
        else:
            if "enabled" not in can:
                _warn("executor.canary_deployment.enabled is missing (default assumed: true/false).")
            if "volume_multiplier" not in can:
                _warn("executor.canary_deployment.volume_multiplier is missing (e.g., 0.5).")
            else:
                try:
                    float(can["volume_multiplier"])
                except Exception:
                    raise ValueError("executor.canary_deployment.volume_multiplier must be numeric")

        # self_optimize
        so = cfg.get("self_optimize") or {}
        if not so:
            _warn("self_optimize block is missing (daily/weekly pipelines won't be configurable).")
        else:
            sch = so.get("schedule") or {}
            steps = so.get("steps") or {}
            if not sch:
                _warn("self_optimize.schedule is missing (e.g., daily_utc/weekly_utc).")
            if not steps:
                _warn("self_optimize.steps is missing (data_refresh/finetune/backtest/deploy).")
            # آستانه‌های پذیرش برای self_optimize: اگر نبود از evaluation می‌گیرند
            so_g = so.get("acceptance_gates")
            if so_g and "max_drawdown_max" not in so_g:
                _warn("self_optimize.acceptance_gates exists but lacks max_drawdown_max; will fall back to evaluation gates.")

        # rl & env
        rl = cfg.get("rl") or {}
        if "algorithm" not in rl:
            _warn("rl.algorithm is missing (default assumed: PPO).")
        env = cfg.get("env") or {}
        act = (env.get("action_space") or {}).get("type")
        if act is None:
            _warn("env.action_space.type is missing (e.g., 'discrete').")

        # paths (در صورت حضور، نوع‌ها چک می‌شود)
        paths = cfg.get("paths") or {}
        for k in ("logs_dir", "models_dir", "config_versions_dir"):
            if k in paths and paths[k] is not None and not isinstance(paths[k], (str, Path)):
                raise ValueError(f"paths.{k} must be a string path")

    # ---------- نقطهٔ ورود بازخوانی ----------
    def reload(self) -> None:
        """
        YAML لایه‌ای را بارگذاری می‌کند → aliasing → اوورراید ENV → اعتبارسنجی → ست به self.config
        """
        raw = self._load_yaml_layered()
        raw = self._apply_compatibility_aliases(raw)  # max_dd_max → max_drawdown_max + aliasهای دیگر
        cfg = self._apply_env_overrides(raw)
        self._validate(cfg)
        self.config = cfg
        logger.info("Config loaded: %s (env prefix=%s)", self.config_path, self.env_prefix)

    # ---------- API عمومی ----------
    def get(self, key: str, default: Any = None) -> Any:
        """دسترسی سریع به کلیدهای سطح-اول، مثلاً get('paths')."""
        return self.config.get(key, default)

    def get_all(self, copy_: bool = False) -> Dict[str, Any]:
        """بازگردانی کل کانفیگ؛ با copy_=True یک کپی سطحی برمی‌گرداند."""
        return dict(self.config) if copy_ else self.config

    # ---------- ابزارها ----------
    def ensure_dirs(self) -> Dict[str, Path]:
        """
        ساخت دایرکتوری‌های مهم در صورت نبودن‌شان. مسیرهای ساخته‌شده را برمی‌گرداند.
        """
        made: Dict[str, Path] = {}
        paths = self.config.get("paths") or {}
        for key in ("logs_dir", "reports_dir", "cache_dir", "models_dir", "config_versions_dir", "tmp_dir"):
            p = paths.get(key)
            if p:
                made[key] = _ensure_dir(self.base_dir / p)
        return made

    def dump(self, path: Union[str, Path], *, sort_keys: bool = False) -> Path:
        """نوشتن self.config به فایل YAML (برای خروجی‌های موقت/دیباگ)."""
        p = Path(path)
        _ensure_dir(p.parent)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=sort_keys)
        return p

# ============================================================
# توابع سطح ماژول (سازگاری/نسخه‌گذاری)
# ============================================================
def load_config(path: Optional[Union[str, Path]] = None,
                env_prefix: str = "BOT_",
                enable_env_override: bool = True) -> Dict[str, Any]:
    """
    کمک‌کاربر برای بارگذاری سریع: برمی‌گرداند dict کانفیگ.
    اگر path ندهید، از مسیر پیش‌فرض (config.yaml) استفاده می‌شود.
    """
    loader = ConfigLoader(config_path=path, env_prefix=env_prefix, enable_env_override=enable_env_override)
    return loader.get_all(copy_=True)

def _infer_versions_dir(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths") or {}
    base = _project_root()
    versions = paths.get("config_versions_dir") or "f01_config/versions"
    return base / versions

def save_config_versioned(cfg: Dict[str, Any],
                          prefix: str = "",
                          tag: Optional[str] = None,
                          *, sort_keys: bool = False) -> Path:
    """
    ذخیرهٔ نسخه‌ای از کانفیگ در پوشهٔ versions (برای بایگانی/رول‌بک)
    نام فایل: {prefix}{utc_ts}{'_tag' if tag}.yaml
    خروجی: مسیر کامل فایل ذخیره‌شده
    """
    # قبل از ذخیره یک اعتبارسنجی انجام دهیم تا فایل معیوب ذخیره نشود
    ConfigLoader._validate(cfg)

    versions_dir = _infer_versions_dir(cfg)
    _ensure_dir(versions_dir)

    ts = _now_utc_stamp()
    suffix = f"_{tag}" if tag else ""
    fname = f"{(prefix or '')}{ts}{suffix}.yaml"
    out_path = versions_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=sort_keys)
    logger.info("Config version saved at %s", out_path)
    return out_path

# ============================================================
# نمونهٔ آماده برای import سریع
# ============================================================
try:
    config: Dict[str, Any] = ConfigLoader().get_all(copy_=True)
except Exception:
    # در صورت خطا، یک dict خالی ارائه می‌کنیم تا importهای قدیمی از کار نیفتند.
    logger.exception("Autoload of ConfigLoader failed; `config` set to {}.")
    config = {}
