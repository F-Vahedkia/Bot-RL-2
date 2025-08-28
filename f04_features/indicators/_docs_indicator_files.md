ساختار پکیج اندیکاتورها (Bot-RL-1)

این پکیج مطابق تأیید شما به فایل‌های کوچک‌تر شکسته شده است. API و Spec همان فلسفهٔ قبلی را حفظ می‌کند و با pipeline فعلی (config → data_handler → processed → indicators) کاملاً سازگار است. همهٔ کدها کامنت‌گذاری فارسی دارند و برای جلوگیری از look-ahead، قابلیت shift_all در Engine پیاده شده است.




درخت فایل‌ها
f04_features/
  indicators/
    init.py              # CLI و entrypoint اصلی + اکسپورت Engine/Parser
    engine.py            # IndicatorEngine (اعمال Specها، شیفت، dropna)
    parser.py            # Spec parser:  name(args)@TF
    registry.py          # رجیستریِ یکپارچه (core + extras + volume + patterns + levels + divergences)
    utils.py             # کمکی‌ها: کشف TFها، نگهبان NaN/Inf، zscore، TR، dtype، ...
    core.py              # اندیکاتورهای پایه: SMA/EMA/WMA/RSI/ATR/TR/MACD/BB/Keltner/Stoch/CCI/MFI/OBV/ROC/W%R/HA/SAR
    extras_trend.py      # Supertrend, ADX/DI/ADXR, Aroon, KAMA, DEMA, TEMA, HMA, Ichimoku
    extras_channel.py    # Donchian, Chaikin Volatility
    volume.py            # VWAP(روزانه/رولینگ)، ADL، Chaikin Money Flow (CMF)
    patterns.py          # چند الگوی کندلی: Engulfing, Doji, Pinbar (به‌صورت فلگ)
    levels.py            # Pivotهای کلاسیک + S/R مبتنی بر فراکتال + Fibonacci distances
    divergences.py       # واگرایی کلاسیک/مخفی روی RSI و MACD (با pivot و shift(+1))




python -m f04_features.indicators --symbol XAUUSD -c .\f01_config\config.yaml --timeframes M1 M5 M30 H1 --out-mode augment --format parquet