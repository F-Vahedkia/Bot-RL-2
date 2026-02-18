├── **f01_config/**
│   ├── **symbols/**
│   │   └── XAUUSD.yaml
│   ├── **versions/**
│   │   ├── prod_20250826T051703Z_dryrun.yaml
│   │   ├── prod_20250930T092716Z.yaml
│   │   ├── prod_20251001T052956Z.yaml
│   │   ├── prod_20251001T151133Z.yaml
│   │   └── prod_20251001T200107Z.yaml
│   ├── base_config.yaml
│   ├── config.20250922_172901.bak.yaml
│   ├── config.yaml
│   ├── extend_config.yaml
│   └── symbol_specs.yaml
├── **f02_data/**
│   ├── **cache/**
│   ├── **news/**
│   │   ├── **raw/**
│   │   ├── calendar.parquet
│   │   └── ff_calendar_thisweek.csv
│   ├── **processed/**
│   ├── **raw/**
│   ├── **test_process/**
│   │   └── **XAUUSD/**
│   │       ├── M1.parquet
│   │       └── M1.split.json
│   ├── _ExamCaller_write_specs_yaml.py
│   ├── __init__.py
│   ├── data_handler.py
│   ├── mt5_connector.py
│   ├── mt5_data_loader.py
│   ├── specs_snapshot.yaml
│   └── symbol_specs_snapshot.py
├── **f03_features/**
│   ├── **indicators/**
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── core.py
│   │   ├── divergences.py
│   │   ├── extras_channel.py
│   │   ├── extras_trend.py
│   │   ├── fibo_pipeline.py
│   │   ├── fibonacci.py
│   │   ├── levels.py
│   │   ├── parser.py
│   │   ├── patterns.py
│   │   ├── sr_advanced.py
│   │   ├── utils.py
│   │   ├── volume.py
│   │   └── zigzag.py
│   ├── **price_action/**
│   │   ├── __init__.py
│   │   ├── breakouts.py
│   │   ├── config_wiring.py
│   │   ├── confluence.py
│   │   ├── imbalance.py
│   │   ├── market_structure.py
│   │   ├── microchannels.py
│   │   ├── mtf_context.py
│   │   ├── regime.py
│   │   ├── registry.py
│   │   ├── registry_adapter.py
│   │   └── zones.py
│   ├── __init__.py
│   ├── feature_bootstrap.py
│   ├── feature_engine.py
│   ├── feature_registry.py
│   └── feature_store.py
├── **f04_env/**
│   ├── __init__.py
│   ├── base_env.py
│   ├── labels.py
│   ├── rewards.py
│   ├── trading_env.py
│   └── utils.py
├── **f05_envexe_core/**
│   ├── __init__.py
│   ├── dataset_deleted.py
│   ├── env_deleted.py
│   ├── reward_deleted.py
│   └── risk.py
├── **f06_news/**
│   ├── **providers/**
│   │   ├── forexfactory_http.py
│   │   └── local_csv.py
│   ├── dataset.py
│   ├── filter.py
│   ├── integration.py
│   ├── normalizer.py
│   ├── provider_base.py
│   ├── runtime_store.py
│   └── schemas.py
├── **f07_training/**
│   ├── __init__.py
│   ├── agent_sb3.py
│   └── agent_sb3_train.py
├── **f08_evaluation/**
│   ├── __init__.py
│   ├── backtest.py
│   └── cluster_zones.py
├── **f09_execution/**
│   ├── broker_adapter_mt5.py
│   ├── executor.py
│   └── promote.py
├── **f10_utils/**
│   ├── __init__.py
│   ├── config_loader.py
│   └── config_ops.py
├── **f11_monitoring/**
├── **f12_models/**
│   ├── **logss/**
│   ├── **prod/**
│   ├── **staging/**
│   ├── **versions/**
│   └── promote_from_staging.py
├── **f13_optimization/**
│   ├── **spaces/**
│   │   └── ppo_space.json
│   ├── hparam_bridge.py
│   ├── hparam_search.py
│   ├── hpo_sb3.py
│   ├── self_optimizer.py
│   └── tune_patterns_rates.py
├── **f14_services/**
└── **f15_testcheck/**
    ├── **broker/**
    │   └── check_mt5_credentials.py
    ├── **data/**
    │   └── check_quick_download.py
    ├── **diagnostics/**
    │   ├── probe_ado_keys.py
    │   ├── probe_engine_adv.py
    │   ├── probe_registry.py
    │   └── test_trace_adv_triplet.py
    ├── **fibo/**
    │   ├── check_fibo_audit.py
    │   ├── check_fibo_cfg_smoke.py
    │   ├── check_fibo_cluster_run.py
    │   └── check_wiring_fib_cluster.py
    ├── **integration/**
    │   ├── __init__.py
    │   ├── run_phaseD_build_store.py
    │   ├── run_spec_phaseC.py
    │   ├── run_spec_phaseC_12x8.py
    │   ├── run_spec_phaseC_adv_probe.py
    │   ├── run_spec_phaseC_mtf.py
    │   ├── test_data_artifacts_and_split_summary.py
    │   ├── test_executor_cli_modes_smoke.py
    │   ├── test_executor_news_canary_wiring.py
    │   ├── test_executor_trading_hours_behavior.py
    │   ├── test_executor_trading_hours_wiring.py
    │   ├── test_feature_engine_full.py
    │   └── test_feature_engine_full_1.py
    ├── **news/**
    │   ├── check_env_newsgate.py
    │   ├── check_news_gate.py
    │   ├── news_build_cache.py
    │   └── news_refresh_daily.py
    ├── **optimization/**
    │   ├── test_executor_dry_smoke.py
    │   └── test_hparam_bridge.py
    ├── **price_action/**
    │   ├── __init__.py
    │   ├── check_fvg.py
    │   ├── check_patterns_smoke_real.py
    │   ├── test_breakouts.py
    │   ├── test_config_wiring.py
    │   ├── test_confluence.py
    │   ├── test_confluence_extras.py
    │   ├── test_imbalance.py
    │   ├── test_market_structure.py
    │   ├── test_microchannels.py
    │   ├── test_mtf_context.py
    │   ├── test_regime.py
    │   ├── test_registry.py
    │   ├── test_registry_adapter.py
    │   └── test_zones.py
    ├── **smoke/**
    │   ├── check_dry_run_config.py
    │   ├── check_long_action.py
    │   ├── check_zero_action.py
    │   └── run_phaseE_quickcheck.py
    ├── **unit/**
    │   ├── __init__.py
    │   ├── check_registry.py
    │   ├── check_registry_all.py
    │   ├── test_adv_indicators.py
    │   ├── test_env_warmup_and_obs.py
    │   ├── test_indicators.py
    │   ├── test_normalization_skip_pre_normalized.py
    │   └── test_parser.py
    ├── __init__.py
    ├── conftest.py
    ├── fetch_mt5_registry.py
    ├── fetch_registry_output.csv
    └── folder_file_list.py