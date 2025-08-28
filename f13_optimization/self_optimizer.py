"""
self_optimizer.py
ماژول هماهنگ‌کننده‌ی خودبهینه‌سازی روزانه/هفتگی برای Bot-RL-1

وظایف سطح بالا:
- تازه‌سازی داده
- اجرای بک‌تست پایه (baseline)
- جستجوی محدود هایپرپارامترها (اختیاری)
- فاین‌تیون/ادامه‌ی آموزش
- بک‌تست و گیت‌های ریسک/عملکرد
- استیجینگ و انتشار قناری
- لاگ/هشدار/نسخه‌گذاری

نکته: این ماژول صرفاً orchestration است؛ منطق هر گام باید در ماژول‌های مربوطه پیاده شود
(f02_data, f07_training, f08_evaluation, f09_execution, risk/).
"""

from __future__ import annotations
import sys, json, time, shutil, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

# وابستگی به ماژول‌های داخلی پروژه (مطابق ساختار پوشه‌های شما)
# این importها به محض آماده بودن ماژول‌ها به‌درستی resolve خواهند شد.
from f10_utils.config_loader import load_config, save_config_versioned
from f10_utils.logging_cfg import setup_logging
from f02_data.data_handler import refresh_datasets
from f07_training.train import finetune_model, hyperparam_search
from f08_evaluation.backtest import run_backtest
from f08_evaluation.metrics import summarize_metrics
from f09_execution.executor import stage_deploy, promote_to_prod, rollback
from f10_utils.risk_manager import SafetyGates

logger = logging.getLogger("self_optimizer")

@dataclass
class OptimizeParams:
    symbol: str
    horizon_days: int = 7                 # افق داده برای بک‌تست/جستجو
    do_hparam_search: bool = False        # فعال/غیرفعال کردن جستجوی هایپرپارامتر
    max_trials: int = 15                  # سقف تریال‌ها (Optuna/Bayes)
    finetune_steps: int = 50_000          # قدم‌های فاین‌تیون (قابل تغییر بر اساس منابع)
    canary_risk_mult: float = 0.5         # محدودیت حجم در استیجینگ (۵۰٪)
    accept_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_min": 0.8,
        "max_dd_max": 0.12,               # 12%
        "winrate_min": 0.48
    })

@dataclass
class OptimizeArtifacts:
    base_report: Path
    trials_csv: Optional[Path]
    candidate_ckpt: Path
    candidate_report: Path
    picked_ckpt: Path
    picked_config: Path

class SelfOptimizer:
    def __init__(self, cfg_path: str, out_dir: str = "artifacts/optimize"):
        self.cfg_path = Path(cfg_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        setup_logging()  # کنسول + فایل

        self.cfg = load_config(self.cfg_path)
        logger.info("Loaded config from %s", self.cfg_path)

    def run(self, p: OptimizeParams) -> None:
        logger.info("=== Self-Optimization started for %s ===", p.symbol)

        # 1) Data refresh
        ds_path = refresh_datasets(symbol=p.symbol, days=p.horizon_days, cfg=self.cfg)
        logger.info("Datasets refreshed at %s", ds_path)

        # 2) Baseline backtest with current production model/config
        base_rep = run_backtest(symbol=p.symbol, cfg=self.cfg, dataset_path=ds_path, tag="baseline")
        base_metrics = summarize_metrics(base_rep)
        logger.info("Baseline metrics: %s", base_metrics)

        # 3) Optional Hyperparam Search (bounded)
        trials_csv = None
        best_hparams = {}
        if p.do_hparam_search:
            trials_csv, best_hparams = hyperparam_search(
                symbol=p.symbol, cfg=self.cfg, dataset_path=ds_path, max_trials=p.max_trials
            )
            logger.info("Hparam search done. Best: %s", best_hparams)

        # 4) Fine-tune / Continue training (use best_hparams if present)
        cand_ckpt = finetune_model(
            symbol=p.symbol, cfg=self.cfg, dataset_path=ds_path,
            steps=p.finetune_steps, overrides=best_hparams
        )
        logger.info("Candidate model checkpoint: %s", cand_ckpt)

        # 5) Backtest candidate + risk/acceptance gates
        cand_rep = run_backtest(symbol=p.symbol, cfg=self.cfg, dataset_path=ds_path,
                                tag="candidate", model_path=cand_ckpt)
        cand_metrics = summarize_metrics(cand_rep)
        logger.info("Candidate metrics: %s", cand_metrics)

        # Safety/acceptance decision
        gates = SafetyGates(
            sharpe_min=p.accept_thresholds["sharpe_min"],
            max_dd_max=p.accept_thresholds["max_dd_max"],
            winrate_min=p.accept_thresholds["winrate_min"],
        )
        if not gates.accept(cand_metrics, baseline=base_metrics):
            logger.warning("Candidate REJECTED by safety gates. Keeping production.")
            return

        # 6) Staging/Canary deploy with reduced risk
        staged_cfg = stage_deploy(
            symbol=p.symbol, cfg=self.cfg, model_path=cand_ckpt,
            risk_multiplier=p.canary_risk_mult
        )
        logger.info("Staged config deployed (canary). Monitoring live for one session...")

        # 7) Promote or Rollback based on live short window (implemented in executor/monitoring)
        ok = promote_to_prod(staged_cfg)   # یا بعد از مانیتورینگ کوتاه‌مدت
        if not ok:
            logger.error("Canary failed. Rolling back...")
            rollback()
            return

        # 8) Versioning: persist config/model as a new version
        saved_cfg = save_config_versioned(staged_cfg, prefix="prod_")
        logger.info("Promotion DONE. Prod config versioned at %s", saved_cfg)

        logger.info("=== Self-Optimization finished successfully for %s ===", p.symbol)

if __name__ == "__main__":
    # مثال اجرای روزانه برای XAUUSD
    params = OptimizeParams(
        symbol="XAUUSD",
        horizon_days=7,
        do_hparam_search=False,   # در اجراهای نخست خاموش نگه دارید؛ بعداً فعال کنید
        max_trials=20,
        finetune_steps=75_000,
        canary_risk_mult=0.5
    )
    so = SelfOptimizer(cfg_path="f01_config/config.yaml")
    so.run(params)
