حتماً. این «چک‌لیست اندیکاتورها و Specها» بر اساس رجیستری پکیج اندیکاتورها (core + extras + volume + patterns + levels + divergences) تهیه شده؛ هر موردی که الان در config.yaml > features.indicators استفاده شده با تیک ✅ علامت خورده است.

چک‌لیست اندیکاتورها (Specها + وضعیت استفاده)
Core

✅ rsi(period) — نمونه: rsi(14)@M1

⬜ sma(col,period) — مثل: sma(close,50)@M1

✅ ema(col,period) — مثل: ema(close,20)@M5

⬜ wma(col,period) — مثل: wma(close,20)@M1

✅ macd(fast,slow,signal) — مثل: macd(12,26,9)@M1

✅ bbands(col,period,k) — مثل: bbands(close,20,2)@M1

⬜ keltner(period,m) — مثل: keltner(20,2)@M1

✅ stoch(n,d) — مثل: stoch(14,3)@M1

✅ atr(n) — مثل: atr(14)@M1

⬜ tr() — True Range خام

⬜ cci(n) — مثل: cci(20)@M1

⬜ mfi(n) — مثل: mfi(14)@M1 (نیاز به حجم)

⬜ obv() (نیاز به حجم)

⬜ roc(n) — نرخِ تغییر

⬜ wr(n) — Williams %R

⬜ ha() — Heikin Ashi (چهار ستون ha_)

✅ sar(af_start,af_step,af_max) — مثل: sar(0.02,0.02,0.2)@M1

Extras – Trend

✅ supertrend(period,multiplier) — مثل: supertrend(10,3)@M5

✅ adx(n) — مثل: adx(14)@M30 (خروجی‌های pdi/md i/adx/adxr)

⬜ aroon(n) — خروجی up/down/osc

⬜ kama(col?,n,fast,slow) — مثل: kama(close,10,2,30)@M1

⬜ dema(col?,n) — مثل: dema(close,20)@M1

⬜ tema(col?,n) — مثل: tema(close,20)@M1

⬜ hma(col?,n) — مثل: hma(close,20)@M1

⬜ ichimoku(tenkan,kijun,span_b) — خروجی: tenkan/kijun/spanA/spanB/chikou

Extras – Channel

✅ donchian(n) — مثل: donchian(20)@H1 (up/mid/lo)

⬜ chaikin_vol(n,roc) — نوسان‌گیری چایکین

Volume

✅ vwap — VWAP روزانه؛ نمونه: vwap@M1 (نیاز به حجم)

⬜ vwap_roll(n) — مثل: vwap_roll(100)@M1 (نیاز به حجم)

⬜ adl — Accumulation/Distribution (نیاز به حجم)

✅ cmf(n) — مثل: cmf(20)@M1 (نیاز به حجم)

Patterns (کندل‌پترن‌ها)

⬜ pat_engulf — خروجی فلگ‌های bull/bear

⬜ pat_doji(thresh) — مثل: pat_doji(0.1)@M1

⬜ pat_pin(ratio) — مثل: pat_pin(2.0)@M1

Levels (سطوح)

✅ pivots — Pivot کلاسیک (pivot/r1/s1/r2/s2/r3/s3)

✅ sr(k,lookback) — مثل: sr(2,500)@M1 (فاصله تا مقاومت/حمایت اخیر)

✅ fibo(k) — مثل: fibo(2)@M1 (سطوح 23.6/38.2/50/61.8/78.6 + فاصله تا هر سطح)

Divergences (واگرایی)

✅ div_rsi(period,k,mode) — مثل: div_rsi(14,2,classic)@M1 (خروجی bull/bear با shift+1)

⬜ div_macd(fast,slow,signal,k,mode) — مثل: div_macd(12,26,9,2,classic)@M1



نکات کاربردی

هر Spec می‌تواند با @TF مشخص شود؛ اگر نگذارید، base_timeframe از features.base_timeframe اعمال می‌شود.

برای اندیکاتورهای حجمی (VWAP/ADL/CMF/OBV/MFI) وجود ستون حجم لازم است. اگر در دادهٔ MT5 فقط tick_volume دارید، سیستم به‌صورت خودکار به volume نگاشت می‌کند.

خروجی بعضی اندیکاتورها چندستونه است (مثل macd, bbands, donchian, adx, ichimoku, pivots, sr, fibo, div_*). نام ستون‌ها به‌صورت استاندارد و با پیشوند TF ذخیره می‌شود (مثلاً M5__rsi_14).