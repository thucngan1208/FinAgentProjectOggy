import numpy as np
import pandas as pd

def simulate_gbm_prices(S0: float, mu: float, sigma: float, n_days: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1 / 252
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(n_days)
    price_path = S0 * np.exp(np.cumsum(log_returns))
    return price_path

def generate_ohlcv(close_prices: np.ndarray, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(close_prices)
    noise = rng.uniform(0.005, 0.015, n)
    open_  = close_prices * (1 + rng.uniform(-0.008, 0.008, n))
    high   = close_prices * (1 + noise)
    low    = close_prices * (1 - noise)
    volume = rng.lognormal(mean=np.log(1_000_000), sigma=0.5, size=n).astype(int)
    
    return pd.DataFrame({
        'Open':   open_,
        'High':   high,
        'Low':    low,
        'Close':  close_prices,
        'Volume': volume
    })

def generate_macro_indicators(countries: dict, years: range, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    
    for country, params in countries.items():
        gdp_base    = params['gdp_mu']
        infl_base   = params['inflation_mu']
        unemp_base  = params['unemployment_mu']
        
        for yr in years:
            gdp_g  = gdp_base  + rng.normal(0, params['gdp_sigma'])
            infl   = infl_base + rng.normal(0, params['inflation_sigma'])
            unemp  = max(0.5, unemp_base + rng.normal(0, 0.5))
            
            records.append({
                'Country': country,
                'Year': yr,
                'GDP_Growth_Pct':    round(gdp_g, 2),
                'Inflation_Pct':     round(infl, 2),
                'Unemployment_Pct':  round(unemp, 2),
            })
    
    return pd.DataFrame(records)

def collect_data():
    ASSET_PARAMS = {
        'AAPL': {'S0': 150.0, 'mu': 0.25, 'sigma': 0.28, 'sector': 'Technology'},
        'MSFT': {'S0': 300.0, 'mu': 0.22, 'sigma': 0.24, 'sector': 'Technology'},
        'JPM':  {'S0': 140.0, 'mu': 0.12, 'sigma': 0.22, 'sector': 'Financials'},
        'XOM':  {'S0': 100.0, 'mu': 0.08, 'sigma': 0.25, 'sector': 'Energy'},
        'GLD':  {'S0': 180.0, 'mu': 0.05, 'sigma': 0.15, 'sector': 'Commodities'},
    }

    TRADING_DATES = pd.bdate_range(start='2023-01-01', end='2024-12-31')
    N_DAYS = len(TRADING_DATES)

    stock_data = {}
    ohlcv_data = {}

    for i, (ticker, params) in enumerate(ASSET_PARAMS.items()):
        prices = simulate_gbm_prices(
            S0    = params['S0'],
            mu    = params['mu'],
            sigma = params['sigma'],
            n_days= N_DAYS,
            seed  = 42 + i
        )
        stock_data[ticker] = prices
        ohlcv_data[ticker] = generate_ohlcv(prices, seed=100+i)
        ohlcv_data[ticker].index = TRADING_DATES

    prices_df = pd.DataFrame(stock_data, index=TRADING_DATES)
    prices_df.index.name = 'Date'
    
    COUNTRIES = {
        'USA': {'gdp_mu': 2.3, 'gdp_sigma': 1.2, 'inflation_mu': 3.1, 'inflation_sigma': 1.5, 'unemployment_mu': 4.0},
        'VNM': {'gdp_mu': 6.5, 'gdp_sigma': 1.5, 'inflation_mu': 4.0, 'inflation_sigma': 1.8, 'unemployment_mu': 2.5},
        'CHN': {'gdp_mu': 5.8, 'gdp_sigma': 1.0, 'inflation_mu': 2.5, 'inflation_sigma': 0.8, 'unemployment_mu': 5.5},
        'SGP': {'gdp_mu': 3.5, 'gdp_sigma': 1.8, 'inflation_mu': 3.5, 'inflation_sigma': 1.2, 'unemployment_mu': 2.0},
    }

    macro_df = generate_macro_indicators(COUNTRIES, years=range(2015, 2025), seed=42)
    
    return prices_df, ohlcv_data, macro_df, ASSET_PARAMS
