import numpy as np
import pandas as pd


# =====================================================================
# Data Preprocessing Function (2 Scenario Datasets)
# =====================================================================
def load_and_preprocess_data(file_path: str):
    print('1. Read and process the dataset...')

    # 1. Read CSV file
    df = pd.read_csv(
        file_path, parse_dates=['Date Time'], date_format='%d.%m.%Y %H:%M:%S'
    )

    # Rename & Resample to Hour
    df = df.rename(columns={'Date Time': 'ds'})
    df = df.set_index('ds')
    df = df.resample('h').mean()
    df = df.interpolate(method='time').reset_index()

    # 2. Create Exogenous Time Features (Cyclical Encoding)
    minute_of_day = df['ds'].dt.hour * 60 + df['ds'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df['day_cos'] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df['month_sin'] = np.sin(2 * np.pi * df['ds'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['ds'].dt.month / 12.0)

    time_exog_cols = ['day_sin', 'day_cos', 'month_sin', 'month_cos']

    # 3. Definition of Weather Feature List and Targets
    all_target = [
        'T (degC)',
        'p (mbar)',
        'Tpot (K)',
        'Tdew (degC)',
        'rh (%)',
        'VPmax (mbar)',
        'VPact (mbar)',
        'VPdef (mbar)',
        'sh (g/kg)',
        'H2OC (mmol/mol)',
        'rho (g/m**3)',
        'wv (m/s)',
        'max. wv (m/s)',
        'wd (deg)',
    ]

    target_col_1 = ['T (degC)']
    weather_exog_cols = [col for col in all_target if col not in target_col_1]
    hist_exog_cols_1 = weather_exog_cols + time_exog_cols

    # 4. SCENARIO A: 14-Target Dataset (Full Multivariate for iTransformer & PatchTST)
    df_all = pd.melt(
        df,
        id_vars=['ds'] + time_exog_cols,
        value_vars=all_target,
        var_name='unique_id',
        value_name='y',
    )
    df_all = df_all.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # 5. SCENARIO B: Dataset with 1 Main Target + 13 Exogenous Weather Variables (for GRU, DLinear, N-HiTS)
    df_1 = pd.melt(
        df,
        id_vars=['ds'] + hist_exog_cols_1,
        value_vars=target_col_1,
        var_name='unique_id',
        value_name='y',
    )
    df_1 = df_1.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    print('✅ pra-pemrosesan selesai:')
    print(f'   - df_1  (1 target)   : {df_1.shape} | unique_id: {df_1["unique_id"].nunique()}')
    print(f'   - df_all (14 target)  : {df_all.shape} | unique_id: {df_all["unique_id"].nunique()}')

    return (
        df_1,
        df_all,
        hist_exog_cols_1,
        time_exog_cols,
        target_col_1,
        all_target,
    )