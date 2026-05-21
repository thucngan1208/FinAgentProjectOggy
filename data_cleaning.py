import numpy as np
import pandas as pd
from scipy import stats

def inject_data_errors(df: pd.DataFrame, missing_pct: float = 0.02,
                        outlier_pct: float = 0.005, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_dirty = df.copy()
    n_total  = df.size
    
    for _ in range(int(n_total * missing_pct)):
        r = rng.integers(0, len(df))
        c = rng.integers(0, len(df.columns))
        df_dirty.iloc[r, c] = np.nan
    
    for _ in range(int(n_total * outlier_pct)):
        r = rng.integers(0, len(df))
        c = rng.integers(0, len(df.columns))
        direction = rng.choice([-1, 1])
        df_dirty.iloc[r, c] *= (10 ** (direction * rng.uniform(0.5, 1.5)))
    
    return df_dirty

def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    if method == 'iqr':
        Q1  = series.quantile(0.25)
        Q3  = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_idx = series.dropna().index[z_scores > threshold]
        return series.index.isin(outlier_idx)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

def clean_price_data(df: pd.DataFrame, outlier_method: str = 'iqr') -> tuple:
    log = {}
    df_clean = df.copy()
    
    n_dups = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    log['duplicates_removed'] = int(n_dups)
    
    outlier_total = 0
    for col in df_clean.columns:
        mask = detect_outliers(df_clean[col].dropna(), method=outlier_method)
        outlier_idx = df_clean[col].dropna().index[mask]
        df_clean.loc[outlier_idx, col] = np.nan
        outlier_total += len(outlier_idx)
    log['outliers_replaced'] = outlier_total
    
    missing_before = df_clean.isna().sum().sum()
    df_clean = df_clean.ffill()
    df_clean = df_clean.bfill()
    missing_after = df_clean.isna().sum().sum()
    
    log['missing_filled'] = int(missing_before)
    log['remaining_missing'] = int(missing_after)
    
    return df_clean, log

def engineer_features(prices: pd.DataFrame) -> dict:
    features = {}
    for ticker in prices.columns:
        S = prices[ticker]
        daily_ret = S.pct_change()
        log_ret   = np.log(S / S.shift(1))
        r7  = S.rolling(7).mean()
        r30 = S.rolling(30).mean()
        vol7  = log_ret.rolling(7).std()  * np.sqrt(252)
        vol30 = log_ret.rolling(30).std() * np.sqrt(252)
        bb_std   = S.rolling(30).std()
        bb_upper = r30 + 2 * bb_std
        bb_lower = r30 - 2 * bb_std
        cum_ret  = (1 + daily_ret).cumprod() - 1
        
        features[ticker] = pd.DataFrame({
            'Close':           S,
            'Daily_Return':    daily_ret,
            'Log_Return':      log_ret,
            'SMA_7d':          r7,
            'SMA_30d':         r30,
            'Volatility_7d':   vol7,
            'Volatility_30d':  vol30,
            'BB_Upper':        bb_upper,
            'BB_Lower':        bb_lower,
            'Cumulative_Return': cum_ret,
        })
    return features
