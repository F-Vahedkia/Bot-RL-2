## f04_env/trading_env.py

# -*- coding: utf-8 -*-
"""
TradingEnv: محیط RL مبتنی بر داده‌های M1 + فیچرهای MTF
- اکشن گسسته: {-1, 0, +1} = Short/Flat/Long (امکان توسعه به پیوسته/سایز پوزیشن)
- پاداش قابل انتخاب (pnl/logret/atr_norm) از config
- بدون لوک‌اِهد: ورودی‌ها قبلاً shift شده‌اند (data_handler + indicators)
- Split زمانی train/val/test از config
- نرمال‌سازی بر اساس آمار train (scaler ذخیره/بارگذاری)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import re, os, csv
import numpy as np
import pandas as pd

from f10_utils.config_loader import load_config
from .base_env import BaseTradingEnv, StepResult
from .rewards import RewardConfig, build_reward_fn
from f04_env.utils import (
    _project_root,
    read_processed,
    build_slices_from_ratios,
    time_slices, paths_from_cfg, slice_df_by_range,
    FeatureSelect, infer_observation_columns,
    fit_scaler, save_scaler, load_scaler,
    write_split_summary, split_summary_path,  # ← جدید
    resolve_spread_selection
)

from f06_news.filter import NewsGate
from f05_envexe_core.risk import RiskManager

logger = logging.getLogger(__name__)

# [LOGLEVEL:ENV] — اگر f04_env_DEBUG=1 باشد، لاگ‌های DEBUG این ماژول فعال می‌شوند
if str(os.getenv("f04_env_SILENT", "0")) == "1":
    # Persian: حالت سایلنت — فقط WARNING و بالاتر
    logger.setLevel(logging.WARNING)
elif str(os.getenv("f04_env_DEBUG", "0")) == "1":
    # Persian: حالت دیباگ — پرلاگ
    logger.setLevel(logging.DEBUG)
# else: سطح پیش‌فرض (INFO) دست‌نخورده


@dataclass
class EnvConfig:
    symbol: str
    base_tf: str = "M1"
    window_size: int = 128
    normalize: bool = True
    features_whitelist: Optional[List[str]] = None
    features_blacklist: Optional[List[str]] = None
    use_ohlc: bool = True
    use_volume: bool = True
    use_spread: bool = False
    reward_mode: str = "pnl"  # pnl | logret | atr_norm

class TradingEnv(BaseTradingEnv):
    """محیط معاملاتی M1 + فیچرهای MTF برای RL."""
    def __init__(self, cfg: Dict[str, Any], 
                       env_cfg: EnvConfig, 
                       news_gate: Optional[NewsGate] = None):
        """
        سازندهٔ محیط معاملاتی.
        - بارگذاری دیتای پردازش‌شده (Base TF)
        - انتخاب ستون‌های مشاهده (OHLC/Volume/Spread + فیچرها) با امکان whitelist/blacklist
        - اعمال splitهای زمانی (train/val/test) از config
        - نرمال‌سازی (fit روی train → ذخیره در cache، سپس load و استفاده)
        - آماده‌سازی سری‌های پاداش (ret/logret/atr) و ساخت تابع پاداش
        - تعریف فضاهای observation/action در صورت وجود gymnasium/gym
        """
        super().__init__()
        self.cfg = cfg


        asp = self.cfg.get("env", {}).get("action_space", {})
        m = asp.get("mapping", {})
        self._action_levels = tuple(m.get("levels", (-1, 0, 1)))
        self._as_type = asp.get("type", "discrete")
        self._as_dim = len(self._action_levels)
        logger.info(f"[ACTION] type={self._as_type} dim={self._as_dim} levels={self._action_levels}")


        self.env_cfg = env_cfg
        # --- NEW: keep gate + default risk scale ---
        self.news_gate: Optional[NewsGate] = news_gate
        self._news_risk_scale: float = 1.0  # Persian: ضریب کاهش ریسک در حالت reduce
        
        # [RISK:INIT] — attach RiskManager (shared core)
        self._risk = RiskManager.from_config(self.cfg)

        # -----------------------------
        # ۱) مسیرها و بارگذاری دیتای پردازش‌شده
        # -----------------------------
        self.paths = paths_from_cfg(cfg)  # {'processed': ..., 'cache': ...}
        # فرمت را به عهدهٔ util بگذاریم (parquet اگر باشد، وگرنه csv)
        self.df_full = read_processed(
            symbol=env_cfg.symbol,
            base_tf=env_cfg.base_tf,
            cfg=cfg,
            fmt=None,
        )

        # اطمینان از وجود ایندکس زمانی و مرتب‌سازی (util همین کار را می‌کند؛ صرفاً دفاعی)
        self.df_full = self.df_full.sort_index()


        # Compute first valid row across all TF OHLC columns (no data deletion)
        _ohlc = [c for c in self.df_full.columns if re.search(r"_(open|high|low|close|tick_volume|spread)$", c)]
        _first_ok = 0
        if _ohlc:
            _num = self.df_full[_ohlc].apply(pd.to_numeric, errors="coerce")
            _valid = np.isfinite(_num.to_numpy(dtype="float64", copy=False)).all(axis=1)
            _first_ok = int(np.argmax(_valid)) if bool(_valid.any()) else 0
        self._first_valid_all_tf = int(_first_ok)
        logger.info("[DATA] first_valid_all_TF=%d", self._first_valid_all_tf)


        # NaN/Inf diagnostics (one-shot)
        _bad = self.df_full.isna().sum().sort_values(ascending=False)
        _inf = (self.df_full == float("inf")).sum().add((self.df_full == float("-inf")).sum(), fill_value=0)
        _top = (_bad + _inf).sort_values(ascending=False).head(10)
        if int(_top.iloc[0]) > 0:
            logger.warning("[DATA] NaN/Inf columns (top): %s", dict(_top[_top > 0]))


        if self.df_full.index.name != "time":
            raise ValueError("Processed DataFrame must be indexed by 'time'.")

        # -----------------------------
        # تشخیص واقعی وجود سری اسپرد در دادهٔ پردازش‌شده
        # -----------------------------
        _has_spread_series = any(("spread" in c.lower()) and self.df_full[c].notna().any()
                         for c in self.df_full.columns)

        # انتخاب منبع اسپرد بر اساس اولویت: سری بروکر ← کانفیگ/اورراید ← صفر
        sel = resolve_spread_selection(self.cfg, df=self.df_full, has_broker_series=bool(_has_spread_series))
        _use_series = bool(sel["use_series"])
        _spread = float(sel["value"]) if not _use_series else None
        if _use_series and "column" in sel:
            self._spread_col = sel["column"]

        if _use_series:
            logger.info("[COSTS] spread source=%s (series)", sel["source"])
        else:
            logger.info("[COSTS] spread source=%s value=%.4f", sel["source"], _spread)


        # -----------------------------------------
        # ۲) انتخاب ستون‌های مشاهده (Observation)
        # -----------------------------------------
        sel = FeatureSelect(
            include_ohlc=env_cfg.use_ohlc,
            include_volume=env_cfg.use_volume,
            include_spread=env_cfg.use_spread,
            whitelist=env_cfg.features_whitelist,
            blacklist=env_cfg.features_blacklist,
        )
        # افزودن داده های حاصل از اندیکاتورها به داده های حاصل از فایل data_handler
        #self.df_full = merge_features_into_df(self.cfg, self.df_full)

        # خواندن دیتافریم مرج شده و قرار دادن آن 
        self.obs_cols = infer_observation_columns(self.df_full, sel)


        # --- یک‌بار در هر اجرا، نامِ تمام ستون‌های دیتافریم ورودی را ذخیره می‌کنیم
        """
        if not getattr(TradingEnv, "_columns_dumped", False):
            import csv; from pathlib import Path
            d = Path(_project_root()) / "f15_testcheck/_MyFiles";
            d.mkdir(parents=True, exist_ok=True)
            tf = str(getattr(self, "_base_tf", getattr(self, "base_tf", "TF")))
            p = d / f"column_names_{tf}_parquet.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["column"])
                [w.writerow([str(c)]) for c in list(self.df_full.columns)]
            TradingEnv._columns_dumped = True
        """
        # --- پایان نوشتن ستونهای دیتافریم


        # [WARMUP:DIAG] نیاز وارم‌آپ به‌ازای هر ستون مشاهده (بدون حذف دیتا)
        _num = self.df_full[self.obs_cols].apply(pd.to_numeric, errors="coerce")
        _arr = np.isfinite(_num.to_numpy(dtype="float64", copy=False))
        _first = {c: (int(np.argmax(_arr[:, i])) if _arr[:, i].any() else 0) for i, c in enumerate(self.obs_cols)}
        _top = sorted(_first.items(), key=lambda kv: kv[1], reverse=True)[:8]
        logger.info("[DATA] top obs warmup requirements: %s", _top)
        _req_obs = max(_first.values() or [0])
        _prev_obs = int(getattr(self, "_warmup_bars", 0))
        self._warmup_bars = max(_prev_obs, int(_req_obs), int(self._first_valid_all_tf) + int(self.env_cfg.window_size))

        assert len(self.obs_cols) > 0, "هیچ ستون مشاهده‌ای پیدا نشد؛ تنظیمات features/whitelist/blacklist را بررسی کنید."

        # -----------------------------------------
        # ۳) split زمانی (train/val/test) از config
        # -----------------------------------------
        # split: ratios → fixed (fallback) + session alignment
        sp_fixed = time_slices(cfg)  # از تاریخ‌های صریح کانفیگ
        sp_ratio = build_slices_from_ratios(self.df_full.index, cfg)  # از نسبت‌ها
        self.slices = sp_ratio or sp_fixed
        if not self.slices:
            t0, t1 = self.df_full.index[0], self.df_full.index[-1]
            self.slices = {"train": (t0, t1)}  # آخرین پناه
        # warmup خودکار/ثابت
        sp_cfg = ((cfg.get("env") or {}).get("split") or {})
        wm = (sp_cfg.get("warmup") or {})
        mode = (wm.get("mode") or "auto").lower()
        margin = int(wm.get("margin_bars", 32))
        fixed_bars = int(wm.get("fixed_bars", 0))
        
        #self._warmup_bars = (fixed_bars if mode == "fixed" else max(int(self.env_cfg.window_size), margin))
        if mode == "fixed":
            self._warmup_bars = fixed_bars
        else:
            idx = self.df_full.index
            leads = []
            for c in self.df_full.columns:
                fv = self.df_full[c].first_valid_index()
                leads.append(idx.get_loc(fv) if fv is not None else 0)
            need = max(leads + [int(self.env_cfg.window_size), margin])
            self._warmup_bars = int(need)
        
        _prev = int(getattr(self, "_warmup_bars", 0))
        self._warmup_bars = max(_prev, int(getattr(self, "_first_valid_all_tf", 0)))
        if self._warmup_bars != _prev:
            logger.warning("[DATA] warmup increased to %d to cover first_valid_all_TF=%d",
                        self._warmup_bars, int(getattr(self, "_first_valid_all_tf", 0)))

        # ensure warmup also covers aggregation lookback (rl.last_k if present)
        rl_cfg = ((cfg.get("training") or {}).get("rl") or {})
        obs_agg = str(rl_cfg.get("obs_agg", "mean")).lower()
        last_k = int(rl_cfg.get("last_k", 0))
        _required = int(max(int(self.env_cfg.window_size), last_k))
        _prev2 = int(self._warmup_bars)
        self._warmup_bars = max(self._warmup_bars, int(self._first_valid_all_tf) + _required)
        if self._warmup_bars != _prev2:
            logger.warning("[DATA] warmup bumped to %d (first_valid=%d + required=%d)",
                        self._warmup_bars, int(self._first_valid_all_tf), _required)

        self._flat_at_boundary = bool(sp_cfg.get("flat_at_boundary", True))

        # Split summary (aligned to session) + warmup
        try:
            idx = self.df_full.index
            a1, b1 = self.slices.get("train", (idx[0], idx[-1]))
            a2, b2 = self.slices.get("val",   (idx[0], idx[-1]))
            a3, b3 = self.slices.get("test",  (idx[0], idx[-1]))
            n1 = int(idx.searchsorted(b1, "right") - idx.searchsorted(a1, "left"))
            n2 = int(idx.searchsorted(b2, "right") - idx.searchsorted(a2, "left"))
            n3 = int(idx.searchsorted(b3, "right") - idx.searchsorted(a3, "left"))
            logger.info("SPLIT | train=%s..%s(%d) | val=%s..%s(%d) | test=%s..%s(%d) | warmup=%d",
                        a1, b1, n1, a2, b2, n2, a3, b3, n3, int(getattr(self, "_warmup_bars", 0)))
            # ثبت گزارش Split + Warmup در کنار processed
            try:
                write_split_summary(self.df_full.index, self.slices,
                                    int(getattr(self, "_warmup_bars", 0)),
                                    env_cfg.symbol, env_cfg.base_tf, cfg)
            except Exception as ex:
                logger.warning("Failed to write split summary: %s", ex)
            
        except Exception:
            pass


        # -----------------------------------------
        # ۴) نرمال‌سازی: fit روی train → ذخیره/بارگذاری
        # -----------------------------------------
        self.scaler = None
        self.scaler_tag = f"{env_cfg.symbol}_{env_cfg.base_tf}_v1"

        # ========== جلوگیریِ امن از «دوبل نرمال‌سازی» (بدون نیاز به تنظیم جدید)
        # --- کمک‌تابع: تشخیص ستونی که قبلاً نرمال شده است ---
        def _is_pre_normalized(col: str) -> bool:
            """
            ستون‌هایی که از قبل نرمال/استاندارد شده‌اند را تشخیص تقریبی می‌دهد.
            - الگوهای رایج: z_*, *_z, *__norm_*
            - اگر ستون z_* باشد، در لیست fit قرار نمی‌گیرد تا دوباره scale نشود.
            """ 
            c = col.lower()
            return c.startswith("z_") or c.endswith("_z") or "__norm_" in c

        # --- انتخاب ستون‌های قابل‌نرمال‌سازی ---
        self.scale_cols = [c for c in self.obs_cols if not _is_pre_normalized(c)]

        # ۴) نرمال‌سازی: فقط روی self.scale_cols
        self.scaler = None
        self.scaler_tag = f"{env_cfg.symbol}_{env_cfg.base_tf}_v1"
        if env_cfg.normalize and len(self.scale_cols) > 0:
            loaded = load_scaler(self.paths["cache"], self.scaler_tag)
            if loaded is None:
                a, b = self.slices["train"]
                df_train = slice_df_by_range(self.df_full[self.scale_cols], a, b)
                if df_train.empty:
                    df_train = self.df_full[self.scale_cols]
                self.scaler = fit_scaler(df_train, self.scale_cols)
                save_scaler(self.scaler, self.paths["cache"], self.scaler_tag)
            else:
                self.scaler = loaded
        # ==========


        # -------------------------------------------------
        # ۵) سری‌های کمکی برای پاداش: بازده، لگ‌ریت و ATR
        #     (بدون look-ahead؛ در step از اندیس t+1 خوانده می‌شود)
        # -------------------------------------------------
        tf = env_cfg.base_tf.upper()

        # انتخاب مقاومِ ستون‌های OHLC برای TF پایه (در برابر یک/دوداش‌خط زیرخط)
        def _pick(field: str) -> str:
            cand1 = f"{tf}_{field}"      # مثل M1_close
            cand2 = f"{tf}__{field}"     # مثل M1__close (اگر جایی این نامگذاری باشد)
            if cand1 in self.df_full.columns: return cand1
            if cand2 in self.df_full.columns: return cand2
            # fallback: نخستین ستونی که به _field ختم می‌شود (برای ایمن‌سازی)
            for c in self.df_full.columns:
                if c.lower().endswith(f"_{field}"):
                    return c
            raise KeyError(f"ستون {field} برای TF {tf} یافت نشد.")

        open_col  = _pick("open")
        close_col = _pick("close")
        high_col  = _pick("high")
        low_col   = _pick("low")

        # --- ذخیرهٔ نام ستون‌های OHLC برای استفاده در step --- start1
        self._open_col  = open_col;
        self._close_col = close_col
        self._high_col  = high_col
        self._low_col   = low_col

        # --- end1

        # optional spread column (per-step) — do not override selection from resolve_spread_selection
        if getattr(self, "_spread_col", None) is None:
            try:
                self._spread_col = _pick("spread")
            except KeyError:
                self._spread_col = None

        close = self.df_full[close_col].astype("float32")
        # pct_change در اندیس i بازده از i-1→i است؛ در step با t+1 می‌خوانیم تا بازده t→t+1 باشد.
        self.ret = close.pct_change(fill_method=None).fillna(0.0).astype("float32")
        self.logret = np.log(close / close.shift(1)).fillna(0.0).astype("float32")

        # اگر ATR از قبل (به‌صورت فیچر) موجود بود، همان را بردار؛ وگرنه محاسبهٔ ساده
        atr_feat = f"{tf}__atr_14"
        if atr_feat in self.df_full.columns:
            self.atr = self.df_full[atr_feat].astype("float32")
        else:
            prev_close = close.shift(1)
            #tr = np.maximum.reduce([
            #    (self.df_full[high_col] - self.df_full[low_col]).abs().values,
            #    (self.df_full[high_col] - prev_close).abs().values,
            #    (self.df_full[low_col]  - prev_close).abs().values,
            #])
            #tr = pd.Series(tr, index=self.df_full.index)

            # --- FIX: make TR 1D via pandas, not numpy reduce ---
            tr1 = (self.df_full[high_col] - self.df_full[low_col]).abs()
            tr2 = (self.df_full[high_col] - prev_close).abs()
            tr3 = (self.df_full[low_col] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).astype("float32")


            self.atr = tr.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean().astype("float32")


        # -----------------------------
        # ۶) ساخت تابع پاداش
        # -----------------------------
        tr_cfg = (cfg.get("trading") or {})
        costs = (tr_cfg.get("costs") or {})

        # [COSTS:FALLBACK] — revised for 4-step spread
        bt = ((cfg.get("evaluation") or {}).get("backtest") or {})
        # relies on _has_spread_series computed earlier from df_full['spread']
        sel = resolve_spread_selection(cfg, df=self.df_full, has_broker_series=bool(_has_spread_series))
        # فقط هزینه‌های غیرِ اسپرد را در صورت نبود تنظیم کن
        costs.setdefault("commission_per_lot", float(bt.get("commissions_per_lot_usd") or 0.0))
        costs.setdefault("slippage_pts", float(bt.get("slippage_pips_mean") or 0.0))

        # [USD:POINT_VALUE:FALLBACK_FIX]
        _pso = (cfg.get("per_symbol_overrides") or {}).get(self.env_cfg.symbol, {})
        _bt  = ((_pso.get("evaluation") or {}).get("backtest") or {})
        _pv_cfg = _bt.get("point_value", None)
        if _pv_cfg is None:
            _sym = ((cfg.get("symbol_specs") or {}).get(self.env_cfg.symbol) or {})
            _pv_cfg = _sym.get("pip_value_per_lot", None)
        pv = float(_pv_cfg) if _pv_cfg is not None else 0.0
        comm_usd_per_lot = float(costs.get("commission_per_lot", 0.0))
        comm_pips_per_lot = (comm_usd_per_lot / pv) if pv > 0.0 else 0.0

        # اسپرد در RewardConfig از _spread تعیین می‌شود؛ اینجا spread_pts را دست نمی‌زنیم

        rcfg = RewardConfig(
            mode=env_cfg.reward_mode,  # 'pnl' | 'logret' | 'atr_norm'
            #cost_spread_pts=(0.0 if _use_series else float(_spread)),  # سری → از ستون خوانده می‌شود
            #cost_commission_per_lot=(float(costs.get("commission_per_lot", 0.0)) / pv),  # USD→pips
            cost_spread_pts=(0.0 if _use_series else float(_spread)),  # سری → از ستون خوانده می‌شود
            cost_commission_per_lot=float(comm_pips_per_lot),          # USD→pips (safe)
            cost_slippage_pts=float(costs.get("slippage_pts", 0.0)),  # قبلاً pips/pts
            point_value=pv,
        )

        # [COSTS:MOVE_TO_ENV] — unique anchor
        # --- start
        self._cost_spread_pts = float(rcfg.cost_spread_pts)
        self._cost_commission = float(rcfg.cost_commission_per_lot)
        self._cost_slippage_pts = float(rcfg.cost_slippage_pts)
        self._point_value = float(rcfg.point_value)
        rcfg.cost_spread_pts = 0.0
        rcfg.cost_commission_per_lot = 0.0
        rcfg.cost_slippage_pts = 0.0
        # --- end

        # [PIPSIZE:INIT] — تعیین اندازهٔ پپ/پوینت از symbol_specs ---- start
        try:
            _sym_specs = (self.cfg.get("symbol_specs") or {})
            _sym = (_sym_specs.get(self.env_cfg.symbol) or {})
            self._pip_size = float(_sym.get("point", 0.0)) if isinstance(_sym, dict) else 0.0
        except Exception:
            self._pip_size = 0.0
        # --- end

        # [USD:POINT_VALUE] — derive from symbol_specs --- start
        if float(getattr(self, "_point_value", 0.0)) <= 0.0:
            try:
                _specs = (cfg.get("symbol_specs") or {}).get(env_cfg.symbol, {})
                pv = None
                if isinstance(_specs, dict):
                    if _specs.get("pip_value_per_lot") is not None:
                        pv = float(_specs["pip_value_per_lot"])
                    elif _specs.get("trade_tick_value") is not None and _specs.get("trade_tick_size") and self._pip_size > 0:
                        pv = float(_specs["trade_tick_value"]) * float(self._pip_size) / float(_specs["trade_tick_size"])
                self._point_value = float(pv) if pv is not None else 0.0
            except Exception:
                self._point_value = 0.0
            if self._point_value <= 0.0:
                logger.warning("point_value unresolved; USD overlay will be zero until symbol_specs pip_value is set.")
        # --- end

        # --- start(temp)
        logger.info("PV/POINT | symbol=%s pip=%.6g pv=%.6g",
                    env_cfg.symbol, float(getattr(self,"_pip_size",0.0)), float(getattr(self,"_point_value",0.0)))
        # --- end(temp)


        # [FX:INIT] --- start
        self._account_ccy = str(((self.cfg.get("project") or {}).get("base_currency") or "USD"))
        #self._fx_rates = ((self.cfg.get("fx_rates") or {}) or ((self.cfg.get("fx") or {}).get("rates") or {}))
        try: self._contract_size = float(_sym.get("contract_size", 0.0))
        except Exception: self._contract_size = 0.0
        _m = ((self.cfg.get("risk") or {}).get("margin") or {})
        self._leverage = float(_m.get("leverage", 0.0)); self._stopout_pct = float(_m.get("stopout_pct", 0.0))
        # --- end

        # [USD:STATE] --- افزودن state پوزیشن --- start
        self._entry_price = None
        self._realized_pips = 0.0
        self._position_lots = 0.0
        # --- end

        # [STATS:INIT] — ensure episode counters exist
        self._ep_entries = 0; self._ep_exits = 0

        # [STATS:LAST-INIT]
        self._last_ep_entries = 0
        self._last_ep_exits = 0
        self._last_ep_transitions = 0
        self._last_ep_pnl_usd = 0.0
        self._last_ep_act_neg   = 0
        self._last_ep_act_zero  = 0
        self._last_ep_act_pos   = 0
        self._last_ep_pnl_pips  = 0.0  # برای گزارش‌دهی در USD-EVAL
        self._ep_net_pips_trades = 0.0  # جمع خالصِ پِپ معاملات بسته‌شده در اپیزود


        # [POS:STATE] --- استیت پوزیشن و پارامترهای حجم از کانفیگ (min/step/max) --- start2
        self._position_lots = 0.0
        self._entry_price = None
        self._sl_price = None
        self._tp_price = None

        _ps = ((cfg.get("risk") or {}).get("position_sizing") or {})
        self._vol_min = float(_ps.get("min_lot", 0.01))
        self._vol_step = float(_ps.get("lot_step", 0.01))
        self._vol_max = float(_ps.get("max_lot", 10.0))
        # --- end2
        
        # [POS:RISKCFG] --- نظیمات SL/TP و BE/Trailing از کانفیگ --- start3
        self._use_atr_sl = bool(_ps.get("use_atr_for_sl", False))
        self._sl_atr_mult = float(_ps.get("sl_atr_mult", 0.0))
        self._tp_atr_mult = float(_ps.get("tp_atr_mult", 0.0))
        _be = (_ps.get("breakeven") or {})
        self._be_enabled = bool(_be.get("enabled", False))
        self._be_at_R = float(_be.get("at_r_multiple", 1.0))
        _tr = (_ps.get("trailing") or {})
        self._trailing_enabled = bool(_tr.get("enabled", False))
        self._trail_atr_period = int(_tr.get("atr_period", 22))
        self._trail_atr_mult = float(_tr.get("atr_mult", 3.0))
        # --- end3


        series = {"ret": self.ret, "logret": self.logret, "atr": self.atr}

        # [REWARD:MODE] — ذخیرهٔ مد پاداش برای تبدیل واحد در step
        self._reward_mode = str(rcfg.mode).lower()

        self.reward_fn = build_reward_fn(rcfg.mode, series, rcfg)

        # --------------------------------
        # ۷) فضاهای مشاهده/اکشن (اختیاری)
        # --------------------------------
        try:
            from gymnasium import spaces  # type: ignore
            env_block = self.cfg.get("env", {})
            action_cfg = (env_block.get("action_space") or {})
            if action_cfg.get("type") == "discrete":
                n_actions = len(action_cfg.get("discrete_actions", [])) or 3
            else:
                n_actions = 1
            self.action_space = spaces.Discrete(n_actions)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(int(env_cfg.window_size), len(self.obs_cols)),
                dtype=np.float32,
            )
        except Exception:
            # اگر gym/gymnasium نصب نبود، فقط فیلدها مقداردهی نمی‌شوند
            pass

        # -----------------------------
        # ۸) وضعیت داخلی اولیهٔ اپیزود
        # -----------------------------
        self._current_df = self.df_full  # در reset بر اساس split برش می‌دهیم
        self._t0 = 0
        self._t1 = len(self._current_df) - 1
        self._t = 0
        self._pos = 0               # -1/0/+1
        self._last_reward = 0.0
        self._done_hard = False
        
        # [SLIP:DIST-STATE]
        self._slip_hist = []

    # --- کمکی‌ها ---
    def _log_trade_csv(self, reason:str, side:str, entry:float, exit_:float, lots:float,
                       gross_pips:float, cost_pips:float, net_pips:float,
                       spread:float, slippage:float, commission_pips:float,
                       t_idx:int, t_global:int, split:str) -> None:
        """ خروجی یک ترید بسته‌شده در CSV برای دیباگ. مسیر پیش‌فرض را با ENV: F03_TRADE_CSV می‌توان تغییر داد."""
        try:
            default_path = os.path.join(str(_project_root()), "f15_testcheck/_MyFiles", "trades_debug.csv")
            path = os.getenv("F03_TRADE_CSV", default_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            newf = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if newf:
                    w.writerow(["split","t_idx","t_global","reason","side","entry","exit","lots",
                                "gross_pips","cost_pips","net_pips","spread","slippage","commission_pips",
                                "sl_price","tp_price"])
                sl_s = "" if getattr(self, "_sl_price", None) is None else f"{float(self._sl_price):.5f}"
                tp_s = "" if getattr(self, "_tp_price", None) is None else f"{float(self._tp_price):.5f}"
                
                
                # [SLTP:ORDER-GUARD:init] — حراست از ترتیب منطقی SL/TP بر حسب جهت --- start+=
                if (self._sl_price is not None) and (self._tp_price is not None):
                    eps = max(float(getattr(self, "_pip_size", 0.0)) or 0.0, 1e-9)
                    if int(self._pos) > 0 and self._sl_price >= self._tp_price:
                        self._tp_price = float(self._sl_price) + eps
                    if int(self._pos) < 0 and self._sl_price <= self._tp_price:
                        self._tp_price = float(self._sl_price) - eps
                # --- end+=
                
                
                w.writerow([split, int(t_idx), int(t_global), reason, side,
                            f"{entry:.5f}", f"{exit_:.5f}", f"{lots:.3f}",
                            f"{gross_pips:.6f}", f"{cost_pips:.6f}", f"{net_pips:.6f}",
                            f"{spread:.6f}", f"{slippage:.6f}", f"{commission_pips:.6f}",
                            sl_s, tp_s])
        except Exception:
            pass

    def _make_obs(self, t: int) -> np.ndarray:
        ws = self.env_cfg.window_size
        lo = max(0, t - ws + 1)
        frame = self._current_df.iloc[lo:t+1][self.obs_cols]

        if self.env_cfg.normalize and self.scaler is not None and len(frame) > 0:
            # فقط ستون‌هایی که در self.scale_cols هستند scale می‌شوند
            sub = frame[self.scale_cols]
            scaled = self.scaler.transform(sub)
            if hasattr(scaled, "reindex"):  # our DF-based scaler
                scaled = scaled.reindex(columns=self.scale_cols)
                for c in self.scale_cols: frame[c] = scaled[c].to_numpy()
            else:  # numpy/other scaler
                #import numpy as np
                arr = np.asarray(scaled)
                for i, c in enumerate(self.scale_cols): frame[c] = arr[:, i]



        obs = np.zeros((ws, len(self.obs_cols)), dtype=np.float32)
        if len(frame) > 0:
            obs[-len(frame):, :] = frame.astype("float32").values  # فقط وقتی طول>0
        return obs

    def _reset_range(self, split: str = "train") -> None:
        # 1) برش طبق split + warmup برای val/test
        if split in self.slices:
            a, b = self.slices[split]
            if split in ("val","test") and getattr(self, "_warmup_bars", 0) > 0:
                idx = self.df_full.index
                pos = int(idx.searchsorted(a, side="left"))
                pos0 = max(0, pos - int(self._warmup_bars))
                if pos - pos0 < 1: pos0 = max(0, pos - 1)  # حداقل یک کندل warmup برای پایداری اندیکاتورها
                a0 = idx[pos0]
                df = self.df_full.loc[(idx >= a0) & (idx <= b)].copy()
                self._boundary_ts = a  # مرز واقعی ترید
            else:
                df = self.df_full.loc[(self.df_full.index >= a) & (self.df_full.index <= b)].copy()
                self._boundary_ts = None
        else:
            df = self.df_full.copy(); self._boundary_ts = None


        # 2) اگر خالی شد، به کل دیتاست برگرد
        if df.empty:
            logger.warning("Selected split '%s' is empty. Falling back to full dataset.", split)
            df = self.df_full.copy()

        # 3) مرتب‌سازی زمانی و ست‌کردن وضعیت داخلی
        df = df.sort_index()
        if df.empty:
            # این حالت یعنی دیتای پردازش‌شده نداریم؛ به‌جای کرش، پیام شفاف بده
            raise RuntimeError(
                "Current dataframe is empty after split/fallback. "
                "Check processed data and your config.env.split ranges."
            )

        self._current_df = df
        self._t0 = 0
        self._t1 = len(self._current_df) - 1

    def _news_status(self, ts) -> Dict[str, Any]:
        """وضعیت گیت خبری در لحظهٔ ts (UTC)."""
        if self.news_gate is None:
            return {"freeze": False, "reduce_risk": False, "reason": "no_gate", "events": []}
        try:
            ts = pd.to_datetime(ts, utc=True)
            return self.news_gate.status(ts)
        except Exception as ex:
            # Persian: اگر گِیت خطا داد، محیط را از کار نینداز
            ###=== import logging
            logging.warning("News gate error: %s", ex)
            return {"freeze": False, "reduce_risk": False, "reason": "error", "events": []}

    # --- API استاندارد ---
    #def reset(self, seed: Optional[int] = None, split: str = "train") -> Tuple[np.ndarray, Dict[str, Any]]:
    def reset(self, seed=None, split="train", options=None):
        if options and "split" in options:
            split = str(options["split"])
        super().seed(seed)
        self._reset_range(split)
        ws = int(self.env_cfg.window_size)


        ws = int(self.env_cfg.window_size)
        # همیشه warmup را لحاظ کن (حتی در train)
        start_off = max(ws, int(getattr(self, "_warmup_bars", 0)))
        self._t = min(self._t0 + start_off, self._t1)


        if self._t < 0:
            self._t = 0

        if getattr(self, "_flat_at_boundary", True):
            self._pos = 0
            self._pos_age = 0  # Persian: سن پوزیشن برای کندل جاری

        self._done_hard = False
        self._last_reward = 0.0
        obs = self._make_obs(self._t)
        info = {"t": int(self._t), "pos": int(self._pos), "split": split, "len": int(self._t1 + 1)}
        info["warmup_bars"] = int(getattr(self, "_warmup_bars", 0))
        info["boundary_ts"] = getattr(self, "_boundary_ts", None)

        # === start
        # [BOUNDARY:CLOSE:RESET] — اگر اپیزود قبلی با پوزیشن باز تمام شده، همین‌جا ببند
        try:
            if int(getattr(self, "_pos", 0)) != 0 and (self._entry_price is not None) and float(getattr(self, "_pip_size", 0.0)) > 0.0:
                t_last = int(getattr(self, "_last_t_global", -1))
                if t_last >= 0:
                    px = float(self.df_full[self._close_col].iloc[t_last]) if hasattr(self, "_close_col") else float(self._last_close or 0.0)
                    lots_now = float(getattr(self, "_position_lots", 0.0))
                    pip_sz   = float(self._pip_size)
                    entry_px = float(self._entry_price)
                    side     = ("LONG" if self._pos > 0 else "SHORT")

                    pnl_pips = ((px - entry_px) / pip_sz) * (1 if self._pos > 0 else -1) * lots_now

                    # ثبت در جمع خالص تریدهای بسته‌شدهٔ اپیزود قبلی
                    self._ep_net_pips_trades = float(getattr(self, "_ep_net_pips_trades", 0.0)) + float(pnl_pips)
                    self._ep_exits += 1

                    # لاگ کم‌حجم
                    logger.debug("CLOSE(RESET-BOUNDARY) | fill=%.6f", float(px))
                    logger.debug(
                        "TRADE | side=%s entry=%.5f exit=%.5f lots=%.3f gross_pips=%.6f cost_pips=%.6f net_pips=%.6f reason=RESET-BOUNDARY",
                        side, float(entry_px), float(px), float(lots_now), float(pnl_pips), 0.0, float(pnl_pips)
                    )
                    # پاکسازی پوزیشن
                    self._pos = 0
                    self._entry_price = self._sl_price = self._tp_price = None
        except Exception:
            pass
        # === end

        
        # خنثی‌سازی نشت بین اپیزودها
        self._ep_net_pips_trades = 0.0
        self._realized_pips = 0.0

        # شمارنده‌های اپیزودی (پپی)
        self._ep_pnl_pips = 0.0
        self._ep_transitions = 0
        self._ep_pnl_usd = 0.0

        # [STATS:ACT-RESET] — action histogram
        self._ep_act_neg = 0; self._ep_act_zero = 0; self._ep_act_pos = 0

        # [STATS:EP-INIT] — episode counters
        self._ep_entries = 0; self._ep_exits = 0

        return obs, info


    # --- تابع کمکی دیسکریت‌سازی/کلیپ حجم (min + n×step) --- start5
    def _normalize_lots(self, lots: float) -> float:
        mn, st, mx = float(getattr(self,"_vol_min",0.0)), float(getattr(self,"_vol_step",0.01)), float(getattr(self,"_vol_max",1e9))
        if lots <= 0.0: return 0.0
        k = max(0, int(round((lots - mn)/st))); return min(mx, mn + k*st)
    # --- end5

    def step(self, action: int) -> StepResult:
        if self._done_hard:
            # وقتی تمام شد، دیگر قدم‌زدن ممکن نیست
            return StepResult(self._make_obs(self._t), 0.0, True, False, {"msg": "episode done"})


        # نگاشت اکشن 5-سطحی → گیت ورود --- start
        raw_action = int(action)
        levels = getattr(self, "_action_levels", (-1, 0, 1))
        if 0 <= raw_action < len(levels):
            a5 = levels[raw_action]
            cfg_map = ((self.cfg.get("env") or {}).get("action_space") or {}).get("mapping", {})
            gate_extreme = bool(cfg_map.get("gate_from_flat_requires_extreme", True))
            new_pos = 0 if (self._pos == 0 and gate_extreme and abs(a5) < max(abs(x) for x in levels)) else (1 if a5 > 0 else (-1 if a5 < 0 else 0))
        else:
            new_pos = BaseTradingEnv.action_to_position(raw_action)
        # --- end


        # [GUARD:SAME-DIR-REENTRY]
        if new_pos != 0 and int(new_pos) == int(self._pos):
            new_pos = self._pos


        # [STATS:ACT-COUNT] — count action per step
        if new_pos < 0:   self._ep_act_neg  += 1
        elif new_pos > 0: self._ep_act_pos  += 1
        else:             self._ep_act_zero += 1



        # [GUARD:ONE-TRADE-PER-CANDLE]
        if int(getattr(self, "_t_last_transition", -2)) == int(getattr(self, "_t", -1)) and int(new_pos) != int(self._pos):
            new_pos = int(self._pos)  # block second change on the same candle
        # [GUARD:COOLDOWN]
        cd = int(getattr(self, "_pos_cooldown", 0))
        if cd > 0 and int(new_pos) != int(self._pos):
            new_pos = int(self._pos)  # still cooling down → no flip
        self._pos_cooldown = max(0, cd - 1)
        if int(new_pos) != int(self._pos):
            self._pos_cooldown = 2  # start 2-candle cooldown on change



        # [POS:LOTS-ACTION] --- «حجم پیوسته + اسکِیلینگ» بر اساس اکشن (داخل step) --- start6
        prev_lots = float(getattr(self, "_position_lots", 0.0))
        if new_pos != 0:
            if self._pos == 0: self._position_lots = self._normalize_lots(max(self._vol_min, self._position_lots or self._vol_min))
            elif (self._pos > 0 and new_pos > 0) or (self._pos < 0 and new_pos < 0):
                self._position_lots = self._normalize_lots(self._position_lots + self._vol_step)
            else:
                self._position_lots = self._normalize_lots(self._position_lots - self._vol_step); 
                if self._position_lots <= 0.0: new_pos = 0
        else:
            self._position_lots = 0.0
        # --- end6

        # [RISK:CAP-LOTS] — محدودسازی حجم بر اساس قوانین ریسک + کاهش ریسک خبری
        self._position_lots = float(self._risk.cap_lot(float(getattr(self,"_position_lots",0.0))))
        self._position_lots = float(self._position_lots) * float(getattr(self, "_news_risk_scale", 1.0))
        self._position_lots = float(self._risk.cap_lot(float(self._position_lots)))


        # محاسبهٔ پاداش در t → t+1
        # توجه: self.ret/logret/atr از کل df_full هستند؛ ایندکس محلی را به ایندکس سراسری نگاشت می‌کنیم
        idx_global = self._current_df.index[self._t]
        t_global = int(self.df_full.index.get_indexer([idx_global])[0])

        # Persian: برای بستن قطعی در reset (اگر اپیزود با TimeLimit تمام شود)
        self._last_t_global = int(t_global)
        try:
            self._last_close = float(self.df_full[self._close_col].iloc[t_global])
        except Exception:
            self._last_close = None


        #--- بلوک NewsGate
        st = {"freeze": False, "reduce_risk": False, "reason": "no_gate", "events": []}
        if getattr(self, "news_gate", None) is not None:
            ts_utc = pd.to_datetime(idx_global, utc=True)
            st = self.news_gate.status(ts_utc)

            # اگر فریز است، پوزیشن جدید را همان قبلی نگه دار (HOLD اجباری)
            if st.get("freeze") and new_pos != self._pos:
                new_pos = self._pos

            # اگر کاهش ریسک است، ضریب ریسک داخلی را ست کن (در جای محاسبه‌ی سایز استفاده کن)
            self._news_risk_scale = st.get("reduce_scale", 0.5) if st.get("reduce_risk") else 1.0
        else:
            self._news_risk_scale = 1.0

        reward = 0.0
        if t_global + 1 <= len(self.df_full) - 1:
            
            # پاداش باید با پوزیشنِ فعال روی بازه [t, t+1] و لاتِ فعلی هم‌گام باشد
            pos_for_interval  = int(new_pos)
            lots_for_interval = float(self._position_lots) if pos_for_interval != 0 else 0.0

            reward = self.reward_fn(pos_for_interval, t_global + 1)            

            #- [REWARD:CANONICAL_PIPS] — تبدیل بازده نسبی به پپ (pnl/logret)
            if getattr(self, "_pip_size", 0.0) > 0.0 and str(getattr(self, "_reward_mode", "pnl")).lower() in ("pnl", "logret"):
                close_t = float(self.df_full[self._close_col].iloc[t_global])
                #- تبدیل: pips = ret_t1 * close_t / pip_size
                reward  = float(reward) * float(close_t / self._pip_size)
                #- [REWARD:LOTS] — scale reward by position size (lots)
                reward  = float(reward) * float(lots_for_interval)

        # [COSTS:TRANSITION] — unique anchor
        # --- start
        trans_cost_pips = 0.0
        if new_pos != self._pos:
            # همه‌چیز برحسب «پپ»
            raw_spread = (float(self.df_full[self._spread_col].iloc[t_global])
                          if getattr(self, "_spread_col", None) else float(self._cost_spread_pts))
            spread_pts_now = float(raw_spread) if np.isfinite(raw_spread) else float(self._cost_spread_pts)
            spread_pts_now = max(0.0, spread_pts_now)
            slip_pips = abs(float(self._cost_slippage_pts))

            # [SLIP:DIST-UPDATE] افزودن نمونهٔ اسلیپیج در لحظهٔ گذار (داخل بلاک هزینه) --- start
            try:
                self._slip_hist.append(float(slip_pips))
                if len(self._slip_hist) > 512:
                    del self._slip_hist[:len(self._slip_hist)-512]
            except Exception:
                pass
            # --- end

            unit_pips = spread_pts_now + slip_pips  # pips

            # ذخیرهٔ اجزای هزینه در لحظهٔ تغییر پوزیشن (برای ثبت دقیق) === start
            self._last_spread_pips = float(spread_pts_now)
            self._last_slippage_pips = float(slip_pips)
            self._last_commission_pips = float(self._cost_commission)
            # === end

            exit_cost_pips = (unit_pips + float(self._cost_commission)) if self._pos != 0 else 0.0
            entry_cost_pips = (unit_pips + float(self._cost_commission)) if new_pos != 0 else 0.0
            trans_cost_pips = exit_cost_pips + entry_cost_pips

            # [USD:REALIZED] --- به‌روزرسانی Realized و Entry هنگام تغییر پوزیشن --- start
            # [USD:REALIZED] — set last-ep USD for eval overlay
            self._last_ep_pnl_usd = float(getattr(self, "_ep_pnl_usd", 0.0))
            self._last_ep_pnl_pips = float(self._ep_pnl_pips)  # NEW: نگه‌داری پیپس اپیزود

            # ثبت وضعیتِ پیشین قبل از هر تغییر (برای محاسبه و لاگِ دقیقِ close/flip)
            prev_pos = int(self._pos)
            entry_px_before = float(self._entry_price) if (self._entry_price is not None) else None
            px_close = float(self.df_full[self._close_col].iloc[t_global])

            if float(getattr(self, "_pip_size", 0.0)) > 0.0:
                pip_sz = float(self._pip_size)
                # خروج کامل: prev≠0 → new=0
                if prev_pos != 0 and new_pos == 0 and entry_px_before is not None:
                    self._realized_pips += ((px_close - entry_px_before) / pip_sz) * (1 if prev_pos > 0 else -1) * float(prev_lots)
                    self._ep_exits += 1
                    self._entry_price = None

                # ورود تازه: prev=0 → new≠0
                elif prev_pos == 0 and new_pos != 0:
                    self._entry_price = float(px_close)
                    self._ep_entries += 1

                # تعویض علامت (flip): prev≠0, new≠0 و علامت عوض شود
                elif prev_pos != 0 and new_pos != 0 and (prev_pos != new_pos) and entry_px_before is not None:
                    self._realized_pips += ((px_close - entry_px_before) / pip_sz) * (1 if prev_pos > 0 else -1) * float(prev_lots)
                    self._ep_exits += 1
                    self._ep_entries += 1
                    self._entry_price = float(px_close)
            # --- end

            # [COSTS:LOTS] --- هزینهٔ تراکنش متناسب با «لات‌ها» در تغییر پوزیشن --- start7
            exit_lots = float(prev_lots) if self._pos != 0 else 0.0
            entry_lots = float(self._position_lots) if new_pos != 0 else 0.0
            trans_cost_pips = (unit_pips + float(self._cost_commission)) * (exit_lots + entry_lots)
            # --- end7

            # لاگ فقط برای close/flip (نه ورود)
            try:
                if (prev_pos != 0) and (new_pos == 0 or (prev_pos != new_pos)) and (entry_px_before is not None) and float(getattr(self, "_pip_size", 0.0)) > 0.0:
                    pip_sz = float(self._pip_size)
                    gross_pips = ((px_close - entry_px_before) / pip_sz) * (1 if prev_pos > 0 else -1) * float(prev_lots)
                    # خالص پپ این معامله:
                    net_pips = float(gross_pips) - float(trans_cost_pips)
                    # تجمیع در جمع اپیزود:
                    self._ep_net_pips_trades = float(getattr(self, "_ep_net_pips_trades", 0.0)) + float(net_pips)

                    logger.debug(
                        "TRADE | side=%s entry=%.5f exit=%.5f lots=%.3f gross_pips=%.6f cost_pips=%.6f net_pips=%.6f",
                        ("LONG" if prev_pos > 0 else "SHORT"),
                        float(entry_px_before), float(px_close), float(prev_lots),
                        float(gross_pips), float(trans_cost_pips), float(gross_pips - trans_cost_pips)
                    )
                    # فراخوانی CSV بعد از لاگ «TRADE» در مسیر تغییر پوزیشن (خروج/فلیپ) === start
                    self._log_trade_csv(
                        reason=("FLIP" if (prev_pos != 0 and new_pos != 0 and (prev_pos != new_pos)) else "CLOSE"),
                        side=("LONG" if prev_pos > 0 else "SHORT"),
                        entry=float(entry_px_before), exit_=float(px_close), lots=float(prev_lots),
                        gross_pips=float(gross_pips), cost_pips=float(trans_cost_pips), net_pips=float(gross_pips - trans_cost_pips),
                        spread=float(getattr(self,"_last_spread_pips",0.0)), slippage=float(getattr(self,"_last_slippage_pips",0.0)),
                        commission_pips=float(getattr(self,"_last_commission_pips",0.0)),
                        t_idx=int(getattr(self,"_t",0)), t_global=int(getattr(self,"_last_t_global", -1)), split=str(getattr(self, "_split", "?"))
                    )

                    # === end

            except Exception:
                pass


            reward -= float(trans_cost_pips)  # کسر هزینه به «پپ»

            # [SLTP:ENTRY] --- ثبت نقطهٔ ورود و پاکسازی SL/TP در خروج/برگشت --- start8
            if self._pos == 0 and new_pos != 0:
                close_t = float(self.df_full[self._close_col].iloc[t_global]); self._entry_price = float(close_t); self._sl_price = None; self._tp_price = None
            elif new_pos == 0:
                self._entry_price = None; self._sl_price = None; self._tp_price = None
            # --- end8

            # [SLTP:ENTRY:INIT-SLTP] — use RiskManager ATR-based distances (price units) --- start9
            try:
                atr_t = float(self.atr.iloc[t_global]) if (0 <= t_global < len(self.atr)) else 0.0
                dist = float(self._risk.stop_distance_from_atr(atr_t))

                # [SLTP:ENTRY:MIN-DIST] enforce floor above costs --- start-a
                try:
                    pip_sz = float(getattr(self, "_pip_size", 0.0))
                    unit_pips = float(getattr(self, "_last_spread_pips", 0.0)) + float(getattr(self, "_last_slippage_pips", 0.0)) + float(getattr(self, "_last_commission_pips", 0.0))
                    min_price = float(2.0 * unit_pips * pip_sz)  # ≥ 2× total entry cost
                    if pip_sz > 0.0 and unit_pips > 0.0 and dist < min_price:
                        dist = float(min_price)
                except Exception:
                    pass
                # --- end-a


                if self._entry_price is not None and dist > 0.0:
                    if self._pos > 0:
                        self._sl_price = float(self._entry_price) - dist
                        if float(getattr(self, "_tp_atr_mult", 0.0)) > 0.0:
                            self._tp_price = float(self._entry_price) + float(self._tp_atr_mult) * float(atr_t)
                    elif self._pos < 0:
                        self._sl_price = float(self._entry_price) + dist
                        if float(getattr(self, "_tp_atr_mult", 0.0)) > 0.0:
                            self._tp_price = float(self._entry_price) - float(self._tp_atr_mult) * float(atr_t)
            
                    # [SLTP:ORDER-GUARD:init] — حراست از ترتیب منطقی SL/TP بر حسب جهت
                    if (self._sl_price is not None) and (self._tp_price is not None):
                        eps = max(float(getattr(self, "_pip_size", 0.0)) or 0.0, 1e-9)
                        if int(self._pos) > 0 and self._sl_price >= self._tp_price:
                            self._tp_price = float(self._sl_price) + eps
                        if int(self._pos) < 0 and self._sl_price <= self._tp_price:
                            self._tp_price = float(self._sl_price) - eps
           
            
            except Exception:
                # keep graceful if anything missing
                pass
            # --- end9


            # به‌روزرسانی شمارنده‌ها
            self._ep_transitions += 1
        # --- end

        prev_pos = int(self._pos)
        self._pos = int(new_pos)
        self._pos_age = (0 if self._pos != prev_pos else int(getattr(self, "_pos_age", 0)) + 1)  # Persian: سن پوزیشن
        if prev_pos != self._pos:
            self._t_last_transition = int(getattr(self, "_t", -1))


        # حرکت به گام بعد
        self._t += 1
        terminated = (self._t >= self._t1)
        truncated = False
        self._last_reward = float(reward)
        self._ep_pnl_pips += float(reward)

        # [POS:UNREAL] --- محاسبهٔ «سود/زیان تحقق‌نیافته» (پپ) در همان استپ --- start10
        ##if self._entry_price is not None and self._pos != 0 and getattr(self,"_pip_size",0.0)>0.0:
        ##    close_t = float(self.df_full[self._close_col].iloc[t_global])
        ##    unrealized_pips = ((close_t - float(self._entry_price)) / float(self._pip_size)) * (1 if self._pos>0 else -1)
        ##else:
        ##    unrealized_pips = 0.0

        # moved below: unrealized_pips is computed once (authoritative) in [USD:UNREALIZED]
        unrealized_pips = 0.0

        # --- end10

        # ATR در همان تایم جاری (به واحد قیمت)
        atr_t = float(self.atr.iloc[t_global]) if (0 <= t_global < len(self.atr)) else float("nan")

        # --- محاسبهٔ step_pnl_pips پیش از ساخت info ---
        if getattr(self,"_pip_size",0.0)>0.0 and t_global+1 <= len(self.df_full)-1:
            p0 = float(self.df_full[self._close_col].iloc[t_global]); p1 = float(self.df_full[self._close_col].iloc[t_global+1])
            step_pnl_pips = ((p1 - p0)/float(self._pip_size)) * (1 if self._pos>0 else (-1 if self._pos<0 else 0)) * float(getattr(self,"_position_lots",0.0))
        else:
            step_pnl_pips = 0.0
        
        # [SLIP:DIST-PCTS] محاسبهٔ p50/p90 قبل از ساختن info --- start
        p50 = float(np.percentile(self._slip_hist, 50)) if getattr(self, "_slip_hist", None) else 0.0
        p90 = float(np.percentile(self._slip_hist, 90)) if getattr(self, "_slip_hist", None) else 0.0
        # --- end

        # [FX:FACTOR] --- start
        fx_factor = 1.0
        #try:
        #    _fx = (self.cfg.get("fx_rates") or (self.cfg.get("fx") or {}).get("rates") or {})
        #    fx_factor = float(_fx.get(self.env_cfg.symbol, 1.0))
        #except Exception:
        #    fx_factor = 1.0
        # --- end

        # [USD:EP-ACCUM] --- تصحیح انباشت دلاری اپیزود
        self._ep_pnl_usd = float(getattr(self, "_ep_pnl_usd", 0.0)) + (
            float(reward) * float(getattr(self, "_point_value", 0.0)) * float(fx_factor)
        )


        # [STATS:SYNC-LAST] — expose episode stats to callback
        self._last_ep_pnl_usd      = float(self._ep_pnl_usd)
        self._last_ep_transitions  = int(getattr(self, "_ep_transitions", 0))
        self._last_ep_entries      = int(getattr(self, "_ep_entries", 0))
        self._last_ep_exits        = int(getattr(self, "_ep_exits", 0))
        self._last_ep_act_neg      = int(getattr(self, "_ep_act_neg", 0))
        self._last_ep_act_zero     = int(getattr(self, "_ep_act_zero", 0))
        self._last_ep_act_pos      = int(getattr(self, "_ep_act_pos", 0))


        # [USD:UNREALIZED] --- محاسبهٔ Unrealized قبل از ساختن info --- start
        if getattr(self, "_pip_size", 0.0) > 0.0 and self._entry_price is not None and self._pos != 0:
            unrealized_pips = ((float(close_t) - float(self._entry_price)) / float(self._pip_size)) * (1 if self._pos > 0 else -1)
        else:
            unrealized_pips = 0.0
        # --- end

        # [MARGIN:COMPUTE] --- محاسبهٔ مارجین موردنیاز همان استپ --- start
        used_margin_usd = 0.0; notional_usd = 0.0
        if float(getattr(self,"_contract_size",0.0))>0.0 and float(getattr(self,"_position_lots",0.0))>0.0 and float(getattr(self,"_leverage",0.0))>0.0:
            notional_usd = float(close_t) * float(self._contract_size) * float(self._position_lots) * float(fx_factor)
            used_margin_usd = notional_usd / float(self._leverage)
        # --- end

        # [SLTP:BE] --- Breakeven (ریسک‌فری) حرفه‌ای برحسب R --- start11
        if self._be_enabled and self._sl_price is not None and getattr(self,"_pip_size",0.0)>0.0:
            R = abs((float(self._entry_price) - float(self._sl_price)) / float(self._pip_size))
            if R>0 and float(unrealized_pips) >= self._be_at_R * R:
                self._sl_price = float(self._entry_price)
        # --- end11

        # [SLTP:TRAIL] --- Trailing Stop by RiskManager --- start12
        if self._trailing_enabled and np.isfinite(atr_t) and self._entry_price is not None:
            a = max(0, t_global - int(self._trail_atr_period) + 1)
            off = float(self._risk.trailing_offset_from_atr(atr_t))  # atr_mult * atr
            if self._pos > 0:
                hh = float(self.df_full[self._high_col].iloc[a:t_global+1].max())
                trail = hh - off
                self._sl_price = max(float(self._sl_price or self._entry_price), float(trail))
            elif self._pos < 0:
                ll = float(self.df_full[self._low_col].iloc[a:t_global+1].min())
                trail = ll + off
                self._sl_price = min(float(self._sl_price or self._entry_price), float(trail))
        # --- end12


        # [SLTP:CHECK] — بررسی برخورد SL/TP با درنظرگرفتن گپ --- start12a
        if self._pos!=0 and \
            self._entry_price is not None and \
            int(getattr(self,"_pos_age",0))>=1 and \
            t_global+1<=len(self.df_full)-1 and \
            getattr(self,"_pip_size",0.0)>0.0:

            o1=float(self.df_full[self._open_col].iloc[t_global+1])
            h1=float(self.df_full[self._high_col].iloc[t_global+1])
            l1=float(self.df_full[self._low_col].iloc[t_global+1])
            boundary_close = False  # ← پرچم بستن مرزی برای تغییر سطح لاگ            
            
            hp=None
            if self._pos>0:
                if self._sl_price is not None and l1<=float(self._sl_price):
                    hp = o1 if o1 < float(self._sl_price) else float(self._sl_price)
                elif self._tp_price is not None and (o1>=float(self._tp_price) or h1>=float(self._tp_price)):
                    # عبور از TP در لانگ: فقط ریسک‌فری/راتچت (SL ← TP)؛ بستن فوری انجام نشود
                    self._sl_price = float(self._tp_price)
                    logger.info("RATCHET | side=LONG set_sl=%.6f reason=TP-cross", float(self._sl_price))
                    # hp را تعیین نکن تا پوزیشن باز بماند

            else:
                if self._sl_price is not None and h1>=float(self._sl_price):
                    hp = o1 if o1 > float(self._sl_price) else float(self._sl_price)
                elif self._tp_price is not None and (o1<=float(self._tp_price) or l1<=float(self._tp_price)):
                    # عبور از TP در شورت: فقط ریسک‌فری/راتچت (SL ← TP)؛ بستن فوری انجام نشود
                    self._sl_price = float(self._tp_price)
                    logger.info("RATCHET | side=SHORT set_sl=%.6f reason=TP-cross", float(self._sl_price))
                    # hp را تعیین نکن تا پوزیشن باز بماند


            # --- Boundary close: اگر اپیزود در همین استپ تمام می‌شود و هنوز hp تعیین نشده، همین‌جا ببند
            if hp is None and bool(terminated):
                # داده‌ی تیک نداریم؛ محافظه‌کارانه روی close همین بار می‌بندیم
                hp = float(close_t)
                boundary_close = True
                #logger.info("CLOSE(BOUNDARY) | fill=%.6f", float(hp))
                logger.debug("CLOSE(BOUNDARY) | fill=%.6f", float(hp))

            if hp is not None:
                # محاسبهٔ PnL برحسب پپ با «قیمت درست خروج» (hp)
                lots_now = float(getattr(self,"_position_lots",0.0))
                pip_sz   = float(self._pip_size)
                entry_px = float(self._entry_price)
                side     = ("LONG" if self._pos>0 else "SHORT")

                step_pnl_pips = ((float(hp) - entry_px) / pip_sz) * (1 if self._pos>0 else -1) * lots_now

                # ثبت در جمع خالص معاملاتِ بسته‌شدهٔ اپیزود (هزینهٔ خروج در این مسیر لحاظ نمی‌شود)
                self._ep_net_pips_trades = float(getattr(self,"_ep_net_pips_trades",0.0)) + float(step_pnl_pips)

                # لاگ معاملهٔ بسته‌شده با قیمت خروج hp
                _log = logger.debug if (boundary_close or abs(float(step_pnl_pips)) < 1e-9) else logger.info
                _log(
                    "TRADE | side=%s entry=%.5f exit=%.5f lots=%.3f gross_pips=%.6f cost_pips=%.6f net_pips=%.6f%s",
                    side, entry_px, float(hp), lots_now, float(step_pnl_pips), 0.0, float(step_pnl_pips),
                    (" reason=BOUNDARY" if boundary_close else "")
                )
                # فراخوانی CSV در مسیر SL/TP/BOUNDARY (وقتی داخل [SLTP:CHECK] می‌بندیم) === start
                unit_pips = float(getattr(self,"_last_spread_pips",0.0)) + float(getattr(self,"_last_slippage_pips",0.0)) + float(getattr(self,"_last_commission_pips",0.0))
                cost_pips = unit_pips * float(lots_now)
                self._log_trade_csv(
                    reason=("BOUNDARY" if boundary_close else "SLTP"),
                    side=side, entry=float(entry_px), exit_=float(hp), lots=float(lots_now),
                    gross_pips=float(step_pnl_pips), cost_pips=float(cost_pips), net_pips=float(step_pnl_pips - cost_pips),
                    spread=float(getattr(self,"_last_spread_pips",0.0)), slippage=float(getattr(self,"_last_slippage_pips",0.0)),
                    commission_pips=float(getattr(self,"_last_commission_pips",0.0)),
                    t_idx=int(getattr(self,"_t",0)), t_global=int(getattr(self,"_last_t_global", -1)), split=str(getattr(self, "_split", "?"))
                )
                # === end

                # بستن پوزیشن و پاکسازی
                self._pos = 0
                self._entry_price = self._sl_price = self._tp_price = None
                self._ep_exits += 1

        # --- end12a


        obs = self._make_obs(self._t)

        # [STATS:STEP-INIT] init-once for episode counters
        if not hasattr(self, "_ep_entries"): self._ep_entries = 0
        if not hasattr(self, "_ep_exits"):   self._ep_exits   = 0

        info = {
            "t": int(self._t),
            "pos": int(self._pos),

            "atr_price": atr_t,      # ← برای RiskManager/Executor
            "news_gate": st,

            # [COSTS:INFO] — unique anchor
            "slippage_p50": float(p50),
            "slippage_p90": float(p90),
            "pnl_pips": float(step_pnl_pips),
            "pnl_usd": float(step_pnl_pips) * float(getattr(self, "_point_value", 0.0)) * float(fx_factor),

            # --- افزودن فیلدهای USD Overlay به info --- start
            "realized_pips": float(getattr(self, "_realized_pips", 0.0)),
            "realized_usd": float(getattr(self, "_realized_pips", 0.0)) * float(getattr(self, "_point_value", 0.0)) * float(fx_factor),
            "unrealized_pips": float(unrealized_pips),
            "unrealized_usd": float(unrealized_pips) * float(getattr(self, "_point_value", 0.0)) * float(getattr(self, "_position_lots", 0.0)) * float(fx_factor),
            
            "point_value": float(getattr(self, "_point_value", 0.0)),
            # --- end
            
            "transition_cost": float(trans_cost_pips) if "trans_cost_pips" in locals() else 0.0,
            "transition_cost_usd": (
                float(trans_cost_pips)
                * float(getattr(self, "_point_value", 0.0))
                * float(getattr(self, "_position_lots", 0.0))
                * float(fx_factor)
            ),

            "reward": float(reward),
            # --- reward overlay (native & USD) ---
            # پاداش دلاری همان گام (برای لاگ‌های مانیتور SB3)
            "reward_pips": float(reward),
            "reward_usd": (float(reward)
                           * float(getattr(self, "_point_value", 0.0))
                           * float(getattr(self, "_position_lots", 0.0))
                           * float(fx_factor)) 
                        if str(self.env_cfg.reward_mode).lower() == "pnl" else 0.0,

            #"fx_factor": float(fx_factor),
            "used_margin_usd": float(used_margin_usd),
            "notional_usd": float(notional_usd),
            "leverage": float(getattr(self,"_leverage",0.0)),

            # اضافه‌کردن Overlay دلاری (گزارشی)
            "ep_pnl_pips": float(self._ep_pnl_pips),
            "ep_pnl_usd": float(getattr(self, "_ep_pnl_usd", 0.0)),

            # --- افزودن فیلدهای وضعیت به info (گزارش دقیق همان استپ) --- start13
            "entry_price": float(self._entry_price) if self._entry_price is not None else None,
            "sl_price": float(self._sl_price) if self._sl_price is not None else None,
            "tp_price": float(self._tp_price) if self._tp_price is not None else None,
            "position_lots": float(getattr(self, "_position_lots", 0.0)),
            
            "ep_transitions": int(getattr(self, "_ep_transitions", 0)),
            "ep_entries":     int(getattr(self, "_ep_entries", 0)),
            "ep_exits":       int(getattr(self, "_ep_exits", 0)),
            "ep_act_neg":     int(getattr(self, "_ep_act_neg", 0)),
            "ep_act_zero":    int(getattr(self, "_ep_act_zero", 0)),
            "ep_act_pos":     int(getattr(self, "_ep_act_pos", 0)),
            # --- end13
            "ep_net_pips_trades": float(getattr(self, "_ep_net_pips_trades", 0.0)),
        }

        # DEBUG: reward breakdown at step --- start
        logger.debug(
            "REWARD | pips=%.6f unreal_pips=%.6f realized_pips=%.6f cost_pips=%.6f usd=%.6f lots=%.4f",
            float(reward),
            float(unrealized_pips),
            float(getattr(self, "_realized_pips", 0.0)),
            float(trans_cost_pips),
            float(reward) * float(getattr(self, "_point_value", 0.0)) * float(getattr(self, "_position_lots", 0.0)) * float(fx_factor),
            float(getattr(self, "_position_lots", 0.0))
        )
        # --- end

        if isinstance(info, dict):
            info.setdefault("warmup_bars", int(getattr(self, "_warmup_bars", 0)))
            if getattr(self, "_boundary_ts", None) is not None:
                info.setdefault("boundary_ts", getattr(self, "_boundary_ts"))

        # [USD:STEP-EP] — expose ep_pnl_usd for eval-callback
        # محاسبهٔ مستقیم از وضعیت داخلی برای جلوگیری از حالت‌های Unbound/None
        #self._last_ep_pnl_usd = (
        #    float(getattr(self, "_ep_pnl_pips", 0.0))
        #    * float(getattr(self, "_point_value", 0.0))
        #    * float(getattr(self, "_position_lots", 0.0))
        #    * float(fx_factor)
        #)
        self._last_ep_pnl_usd = float(getattr(self, "_ep_pnl_usd", 0.0))

        # افزودن لاگ DEBUG برای reward_usd و هزینه‌ها در هر استپ
        logger.debug("STEP | t=%d pos=%d reward_pips=%.6f reward_usd=%.6f trans_cost_pips=%.6f trans_cost_usd=%.6f ep_pnl_usd=%.6f",
                    int(self._t), int(self._pos), float(reward), float((info or {}).get("reward_usd", 0.0)),
                    float((info or {}).get("transition_cost", 0.0)), float((info or {}).get("transition_cost_usd", 0.0)),
                    float((info or {}).get("ep_pnl_usd", 0.0)))

        if terminated:
            self._done_hard = True
            self._last_ep_pnl_usd = float(getattr(self, "_ep_pnl_usd", 0.0))

            # === start
            # [BOUNDARY:CLOSE — out-of-t+1] اگر اپیزود تمام می‌شود و هنوز پوزیشن باز است، همین‌جا با close_t ببند
            # توضیح فارسی: این مسیر «خارج از شرط وجود کندل بعدی» اجرا می‌شود تا در آخرین بار نیز بستن مرزی انجام شود.
            try:
                if int(getattr(self, "_pos", 0)) != 0 and (self._entry_price is not None):
                    # قیمت بسته‌شدن مرزی را محاسبه کن (close همین بار)
                    close_t = float(self.df_full[self._close_col].iloc[t_global])
                    lots_now = float(getattr(self, "_position_lots", 0.0))
                    pip_sz   = float(self._pip_size)
                    entry_px = float(self._entry_price)
                    side     = ("LONG" if self._pos > 0 else "SHORT")

                    step_pnl_pips = ((close_t - entry_px) / pip_sz) * (1 if self._pos > 0 else -1) * lots_now

                    # جمع معاملات بسته‌شدهٔ اپیزود را به‌روزرسانی کن
                    self._ep_net_pips_trades = float(getattr(self, "_ep_net_pips_trades", 0.0)) + float(step_pnl_pips)

                    # لاگ بستن مرزی کم‌حجم (DEBUG) تا ترمینال شلوغ نشود
                    logger.debug("CLOSE(BOUNDARY) | fill=%.6f", float(close_t))
                    logger.debug(
                        "TRADE | side=%s entry=%.5f exit=%.5f lots=%.3f gross_pips=%.6f cost_pips=%.6f net_pips=%.6f reason=BOUNDARY",
                        side, entry_px, float(close_t), lots_now, float(step_pnl_pips), 0.0, float(step_pnl_pips)
                    )

                    # بستن و پاکسازی
                    self._pos = 0
                    self._entry_price = self._sl_price = self._tp_price = None
                    self._ep_exits += 1
            except Exception:
                # در هر صورت نگذاریم لاگ EPISODE از بین برود
                pass
            # === end


            logger.info("EPISODE | split=%s transitions=%d pnl_pips=%.6f pnl_usd=%.6f",
                        info.get("split","?"), int(self._ep_transitions),
                        float(self._ep_pnl_pips), float(getattr(self, "_ep_pnl_usd", 0.0)))
            # گزارش اختلاف «episode_reward» و «جمع net_pips معاملات»
            logger.info("EPISODE | net_pips_trades=%.6f  vs  episode_reward_pips=%.6f",
                        float(getattr(self, "_ep_net_pips_trades", 0.0)),
                        float(getattr(self, "_ep_pnl_pips", 0.0)))

            # Delta summary: اختلاف پاداش اپیزود با جمع معاملات بسته‌شده + شمارش ورود/خروج
            try:
                delta = float(getattr(self, "_ep_pnl_pips", 0.0)) - float(getattr(self, "_ep_net_pips_trades", 0.0))
            except Exception:
                delta = 0.0
            logger.debug("DELTA | episode_reward_minus_trades=%.6f entries=%d exits=%d",
                        float(delta), int(getattr(self, "_ep_entries", 0)), int(getattr(self, "_ep_exits", 0)))
            
            # [SLIP:DIST-LOG] — گزارش صدکی‌های اسلیپیج
            logger.debug("SLIPPAGE | p50=%.4f p90=%.4f samples=%d",
                        float(info.get("slippage_p50", 0.0)),
                        float(info.get("slippage_p90", 0.0)),
                        int(len(getattr(self, "_slip_hist", []))))
            
        return StepResult(obs, float(reward), bool(terminated), bool(truncated), info)

    # رندرِ متنی ساده (اختیاری)
    def render(self) -> None:
        print(f"t={self._t} pos={self._pos} last_reward={self._last_reward:.6f}")


# --- CLI تست دود ---

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Smoke test for TradingEnv")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c", "--config", default=str(_project_root()/"f01_config"/"config.yaml"))
    p.add_argument("--base-tf", default=None)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--split", default="train", choices=["train","val","test"])
    p.add_argument("--reward", default="pnl", choices=["pnl","logret","atr_norm"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)
    base_tf = (args.base_tf or (cfg.get("features") or {}).get("base_timeframe", "M1")).upper()

    env_cfg = EnvConfig(
        symbol=args.symbol,
        base_tf=base_tf,
        window_size=int(args.window),
        normalize=bool(args.normalize),
        reward_mode=str(args.reward).lower(),
    )

    from f06_news.integration import make_news_gate
    gate = make_news_gate(cfg, symbol=args.symbol)
    env = TradingEnv(cfg, env_cfg, news_gate=gate)

    obs, info = env.reset(split=args.split)

    total_r = 0.0
    for i in range(int(args.steps)):
        a = np.random.choice([0, 1, 2, 3, 4])  # سیاست تصادفی ۵حالته برای تست دود

        step = env.step(a)
        total_r += step.reward
        if step.terminated or step.truncated:
            break
    print(f"SmokeTest finished: steps={i+1}, total_reward={total_r:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
