### نحوهٔ اجرا (Smoke Test)

## In powershell
# 1) تست دود با سیاست تصادفی و پاداش PnL (normalize خاموش)
python -m f03_env.trading_env --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M1 --window 128 --steps 512 --split train --reward pnl
python -m f03_env.trading_env --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M5 --window 128 --steps 512 --split train --reward pnl

# 2) تست دود با نرمال‌سازی و پاداش logret
python -m f03_env.trading_env --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M1 --window 128 --steps 512 --split train --reward logret --normalize
python -m f03_env.trading_env --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M5 --window 128 --steps 512 --split train --reward logret --normalize

# 3) تست دود با پاداش atr_norm (به نوسان وابسته)
python -m f03_env.trading_env --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M1 --window 128 --steps 512 --split train --reward atr_norm --normalize
python -m f03_env.trading_env --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M5 --window 128 --steps 512 --split train --reward atr_norm --normalize