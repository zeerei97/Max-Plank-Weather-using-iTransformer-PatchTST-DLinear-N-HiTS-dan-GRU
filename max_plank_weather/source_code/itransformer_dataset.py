import numpy as np
import pandas as pd


# =====================================================================
# FUNGSI PREPROCESSING DATA
# =====================================================================
def load_and_preprocess_data(file_path: str):
    print('1. Membaca dan memproses dataset multivariate...')

    # 1. Baca file CSV
    df = pd.read_csv(
        file_path, parse_dates=['Date Time'], date_format='%d.%m.%Y %H:%M:%S'
    )

    # Perbaikan 1: Rename tanpa 'inplace=True' agar df tidak menjadi None
    df = df.rename(columns={'Date Time': 'ds'})

    # Perbaikan 2: Set ds sebagai index sebelum resample & interpolate
    df = df.set_index('ds')
    df = df.resample('h').mean()
    df = df.interpolate(method='time').reset_index()

    # 2. Buat Fitur Eksogen Waktu (Cyclical Encoding)
    minute_of_day = df['ds'].dt.hour * 60 + df['ds'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df['day_cos'] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df['month_sin'] = np.sin(2 * np.pi * df['ds'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['ds'].dt.month / 12.0)

    time_exog_cols = ['day_sin', 'day_cos', 'month_sin', 'month_cos']

    # 3. Pisahkan Target Utama dan 13 Fitur Cuaca Lainnya
    target_cols = ['T (degC)']

    weather_exog_cols = [
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

    # Gabungkan semua fitur pendukung ke dalam hist_exog_cols
    hist_exog_cols = weather_exog_cols + time_exog_cols

    # 4. Transformasi Wide Format ke Long Format (pd.melt)
    df_long = pd.melt(
        df,
        id_vars=['ds'] + hist_exog_cols,
        value_vars=target_cols,
        var_name='unique_id',
        value_name='y',
    )

    # Urutkan berdasarkan timestamp
    df_long = df_long.sort_values(by=['ds', 'unique_id']).reset_index(
        drop=True
    )

    return df_long, hist_exog_cols, target_cols