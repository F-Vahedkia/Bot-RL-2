# چک‌لیست اندیکاتورها و Specها (Bot-RL-1)

> راهنما: هر خط یک **Spec** قابل‌استفاده در پکیج `f04_features/indicators` است. مواردی که **اکنون** در `config.yaml > features.indicators` فعال‌اند با `[X]` علامت خورده‌اند؛ بقیه `[ ]` هستند.
>
> قالب Spec:  
> `name(args)@TF`  — مثال: `ema(close,20)@M5` ، `rsi(14)@M1`  
> اگر `@TF` حذف شود، از `features.base_timeframe` استفاده می‌شود.

---

## Core
- [X] `rsi(period)` → **استفاده‌شده:** `rsi(14)@M1`
- [ ] `sma(col,period)` — نمونه: `sma(close,50)@M1`
- [X] `ema(col,period)` → **استفاده‌شده:** `ema(close,20)@M5`
- [ ] `wma(col,period)` — نمونه: `wma(close,20)@M1`
- [X] `macd(fast,slow,signal)` → **استفاده‌شده:** `macd(12,26,9)@M1`  
  *(خروجی‌ها: macd, macd_signal, macd_hist)*
- [X] `bbands(col,period,k)` → **استفاده‌شده:** `bbands(close,20,2)@M1`  
  *(خروجی‌ها: mid, up, lo)*
- [ ] `keltner(period,m)` — نمونه: `keltner(20,2)@M1`  
  *(خروجی‌ها: mid, up, lo)*
- [X] `stoch(n,d)` → **استفاده‌شده:** `stoch(14,3)@M1`  
  *(خروجی‌ها: k, d)*
- [X] `atr(n)` → **استفاده‌شده:** `atr(14)@M1`
- [ ] `tr()` — True Range خام
- [ ] `cci(n)` — نمونه: `cci(20)@M1`
- [ ] `mfi(n)` — نمونه: `mfi(14)@M1` *(نیاز به حجم)*
- [ ] `obv()` *(نیاز به حجم)*
- [ ] `roc(n)`
- [ ] `wr(n)` — Williams %R
- [ ] `ha()` — Heikin Ashi *(چهار ستون: ha_open/ha_high/ha_low/ha_close)*
- [X] `sar(af_start,af_step,af_max)` → **استفاده‌شده:** `sar(0.02,0.02,0.2)@M1`

## Extras – Trend
- [X] `supertrend(period,multiplier)` → **استفاده‌شده:** `supertrend(10,3)@M5`
- [X] `adx(n)` → **استفاده‌شده:** `adx(14)@M30`  
  *(خروجی‌ها: pdi, mdi, adx, adxr)*
- [ ] `aroon(n)` *(خروجی‌ها: up, down, osc)*
- [ ] `kama(col?,n,fast,slow)` — نمونه: `kama(close,10,2,30)@M1`
- [ ] `dema(col?,n)` — نمونه: `dema(close,20)@M1`
- [ ] `tema(col?,n)` — نمونه: `tema(close,20)@M1`
- [ ] `hma(col?,n)` — نمونه: `hma(close,20)@M1`
- [ ] `ichimoku(tenkan,kijun,span_b)`  
  *(خروجی‌ها: ichi_tenkan, ichi_kijun, ichi_span_a, ichi_span_b, ichi_chikou)*

## Extras – Channel
- [X] `donchian(n)` → **استفاده‌شده:** `donchian(20)@H1`
  *(خروجی‌ها: donch_up, donch_mid, donch_lo)*
- [ ] `chaikin_vol(n,roc)`

## Volume
- [X] `vwap` → **استفاده‌شده:** `vwap@M1` *(VWAP روزانه؛ نیاز به حجم)*
- [ ] `vwap_roll(n)` — نمونه: `vwap_roll(100)@M1` *(رولینگ؛ نیاز به حجم)*
- [ ] `adl` *(Accum/Distribution Line؛ نیاز به حجم)*
- [X] `cmf(n)` → **استفاده‌شده:** `cmf(20)@M1` *(Chaikin Money Flow؛ نیاز به حجم)*

## Patterns (الگوهای کندلی)
- [ ] `pat_engulf` *(خروجی: bull/bear)*
- [ ] `pat_doji(thresh)` — نمونه: `pat_doji(0.1)@M1`
- [ ] `pat_pin(ratio)` — نمونه: `pat_pin(2.0)@M1`

## Levels (سطوح)
- [X] `pivots` → **استفاده‌شده:** `pivots@M1`
  *(خروجی‌ها: pivot, r1/s1, r2/s2, r3/s3)*
- [X] `sr(k,lookback)` → **استفاده‌شده:** `sr(2,500)@M1`
- [X] `fibo(k)` → **استفاده‌شده:** `fibo(2)@M1`

## Divergences (واگرایی)
- [X] `div_rsi(period,k,mode)` → **استفاده‌شده:** `div_rsi(14,2,classic)@M1` *(خروجی: bull/bear؛ با shift(+1))*
- [ ] `div_macd(fast,slow,signal,k,mode)` — نمونه: `div_macd(12,26,9,2,classic)@M1`

---

### نکات
- همهٔ خروجی‌های چندستونه با پیشوند تایم‌فریم ذخیره می‌شوند؛ مثال: `M5__rsi_14`, `M1__macd_hist_12_26_9`.
- ضدّ **look-ahead**: موتور به‌صورت پیش‌فرض `shift_all=1` را از کانفیگ می‌گیرد.  
- برای اندیکاتورهای حجمی در داده‌های MT5، از ستون `tick_volume` به `volume` نگاشت می‌شود.

