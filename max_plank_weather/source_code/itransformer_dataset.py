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
  df['ds'] = df['Date Time']

  # 2. Buat Fitur Eksogen Waktu (Cyclical Encoding)
  minute_of_day = df['ds'].dt.hour * 60 + df['ds'].dt.minute
  df['day_sin'] = np.sin(2 * np.pi * minute_of_day / 1440.0)
  df['day_cos'] = np.cos(2 * np.pi * minute_of_day / 1440.0)
  df['month_sin'] = np.sin(2 * np.pi * df['ds'].dt.month / 12.0)
  df['month_cos'] = np.cos(2 * np.pi * df['ds'].dt.month / 12.0)

  hist_exog_cols = ['day_sin', 'day_cos', 'month_sin', 'month_cos']

  # 3. Daftar 14 fitur target yang diprediksi
  target_cols = [
      'p (mbar)',
      'T (degC)',
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

  # 4. Imputasi missing values jika ada
  df[target_cols] = df[target_cols].fillna(df[target_cols].mean())

  # 5. Transformasi Wide Format ke Long Format (pd.melt)
  df_long = pd.melt(
      df,
      id_vars=['ds'] + hist_exog_cols,
      value_vars=target_cols,
      var_name='unique_id',
      value_name='y',
  )

  # Urutkan berdasarkan timestamp dan unique_id
  df_long = df_long.sort_values(by=['ds', 'unique_id']).reset_index(drop=True)

  return df_long, hist_exog_cols, target_cols