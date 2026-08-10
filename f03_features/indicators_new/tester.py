import pandas as pd
import numpy as np
from datetime import datetime
import f03_features.indicators_new.indicators_B_batch as ind

def test_indicators(df):
    """
    تست 12 اندیکاتور با داده واقعی
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم با ستون‌های open, high, low, close, tick_volume
    """
    
    # تغییر نام ستون tick_volume به volume
    df = df.rename(columns={'tick_volume': 'volume'})
    
    print("=" * 80)
    print("Start indicators test")
    print(f"Rows number: {len(df)}")
    print("=" * 80)
    
    results = {}

        # --- 15. Williams %R -----------------------------------------------------
    print("\n[15/23] test of Williams %R...")
    try:
        williamsr_14,r1,r2,r3,r4 = ind.williamsr_batch_df(df, high_col='high', low_col='low', close_col='close',
                                               n=14, min_periods=5,result_col='williamsr_14')
        results['williamsr'] = williamsr_14['williamsr_14']
        results['r1']=r1['r1']
        results['r2']=r2['r2']
        results['r3']=r3['r3']
        results['r4']=r4['r4']

        print(f"  ✓ Williams %R: {williamsr_14['williamsr_14'].notna().sum()} valid values out of {len(williamsr_14)}")
        print(f"    columns: {list(williamsr_14.columns)}")
        print(f"    last value: {williamsr_14['williamsr_14'].iloc[-1]:.2f}")
        print(f"    range: [{williamsr_14['williamsr_14'].min():.2f}, {williamsr_14['williamsr_14'].max():.2f}]")
    except Exception as e:
        print(f"  ✗ error: {e}")
    
    # --- Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    df_result = pd.concat(results, axis=1)  # کلیدها به عنوان MultiIndex
    print(f"\n Result Columns: {list(df_result.columns)}")
    column_number = len(list(df_result.columns))
    print(f"\n Number of created columns = {column_number}")

    print(f"Number of indicators tested: {len(results)}/{column_number}")
    # print(f"Successful indicators: {list(results.keys())}")

    return results


if __name__ == "__main__":

    # --- Load data -----------------------------------------------------
    t1 = datetime.now()
    data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
    t2 = datetime.now()

    # --- Preparing data ------------------------------------------------
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data.set_index("time", inplace=True)
    data = data.astype("float64")

    elapsed = round((t2 - t1).total_seconds(), 2)
    print(f"Time taken to load data: {elapsed} seconds, length_df:{len(data)}")

    if not {"open", "high", "low", "close"}.issubset(data.columns):
        raise ValueError("Data must contain open, high, low, close")

    df = data[-1_000:]
    
    # --- Running Tests -------------------------------------------------

    t1 = datetime.now()
    results = test_indicators(df)
    t2 = datetime.now()

    dt = t2-t1
    second = dt.total_seconds()
    sec_round = round(second, 2)
    print(f"Time taken to run 23 tests: {round(sec_round,2)} seconds, length_df:{len(df)}")
    print("\n✓ The tests ran successfully.")

    pd.concat([df[['high', 'low', 'close']].iloc[-1000:], results.iloc[-1000:]], axis=1).to_csv("williams.csv")

