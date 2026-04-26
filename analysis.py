"""
==============================================================================
USDT Demand Pressure in Turkey: A Path-Dependence Test
==============================================================================
Author : Anne-Lise Saive - April 2026

KEY METHODOLOGICAL PRINCIPLE 
─────────────────────────────────────────────
λ and ESS weights are tuned on the AUG 2018 shock (primary event).
Performance is evaluated on the DEC 2021 shock (held-out event).
These two events never touch each other.
This eliminates the data snooping / partial leakage present in v5.

THREE MODELS
────────────
A  Memoryless ARX
   premium ~ AR(1) + FX_shock + FX_vol + CPI_proxy + Trends

B  Salience-weighted path-dependence
   Model A + M(t) + FX×M
   λ and ESS weights tuned on Aug 2018, evaluated on Dec 2021

C  Reactivation asymmetry  [novel — exploratory]
   Model B + Abruptness×M + Trends×M
   Ridge regression (CV alpha). Bootstrap CIs for novel terms.
   Reported honestly: if OOS R² < B, flag as exploratory.

DATA  (all public, no keys needed)
────────────────────────────────────
Binance   USDT/TRY weekly klines
FRED      USD/TRY weekly  (DEXTHUS)
WorldBank Turkey CPI annual (used only to validate FX proxy)
Google    Trends TR: USDT, Tether, dolar kripto
==============================================================================
"""

import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.facecolor'   : '#F8F9FA',
    'figure.facecolor' : 'white',
    'axes.grid'        : True,
    'grid.alpha'       : 0.22,
    'grid.linestyle'   : '--',
})

C = dict(
    fx='#C1121F', premium='#264653',
    trends='#E76F51', memory='#457B9D',
    A='#888888', B='#6A0572', Cm='#2D6A4F',
    shock='#E9C46A', event='#F4A261',
)

START, END   = '2018-01-01', '2024-12-31'
TUNE_SHOCK   = pd.Timestamp('2018-08-13')   # tuning event   — Aug 2018
EVAL_SHOCK   = pd.Timestamp('2021-12-20')   # main evaluation — Dec 2021
ROBUST_SHOCK = pd.Timestamp('2023-06-15')   # robustness test  — Jun 2023
IS_SYNTHETIC = False
ABORT_ON_SYNTHETIC = True   # hard abort if any key source falls back


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_binance(start=START):
    print("  Binance USDT/TRY …", end=' ', flush=True)
    url, start_ms, rows = "https://api.binance.com/api/v3/klines", \
        int(pd.Timestamp(start).timestamp() * 1000), []
    while True:
        r = requests.get(url,
            params=dict(symbol='USDTTRY', interval='1w',
                        startTime=start_ms, limit=1000), timeout=15)
        data = r.json()
        if not data or isinstance(data, dict):
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        start_ms = data[-1][6] + 1
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=[
        'ot','o','h','l','c','v','ct','qv','nt','tbv','tbqv','ig'])
    df.index          = pd.to_datetime(df['ot'], unit='ms').dt.normalize()
    df['usdt_try_px'] = df['c'].astype(float)
    df['usdt_try_vol']= df['qv'].astype(float)
    print(f"OK ({len(df)} weeks)")
    return df[['usdt_try_px', 'usdt_try_vol']]

def fetch_fred():
    print("  FRED USD/TRY …", end=' ', flush=True)
    df = pd.read_csv(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXTHUS",
        index_col=0, parse_dates=True, na_values='.')
    df.columns = ['usd_try_official']
    df = df.resample('W-MON').mean().ffill()
    print(f"OK ({len(df)} weeks)")
    return df

def fetch_worldbank():
    """Annual CPI — used only for proxy validation, not as model input."""
    print("  World Bank CPI (validation only) …", end=' ', flush=True)
    url = ("https://api.worldbank.org/v2/country/TUR/indicator/"
           "FP.CPI.TOTL.ZG?format=json&per_page=100&mrv=30")
    payload = requests.get(url, timeout=15).json()
    rows = [{'date': r['date'], 'cpi_yoy': r['value']}
            for r in payload[1] if r['value'] is not None]
    df = (pd.DataFrame(rows)
          .assign(date=lambda x: pd.to_datetime(x['date'], format='%Y'))
          .set_index('date').sort_index()
          .resample('W-MON').ffill())
    print(f"OK ({len(rows)} annual obs)")
    return df

def fetch_trends():
    """
    Google Trends via direct HTTPS request — bypasses pytrends entirely.
    This avoids the urllib3 >= 2.0 incompatibility that breaks pytrends
    (Retry.__init__() unexpected keyword argument 'method_').

    Falls back to pytrends if direct request fails, then to a fixed
    neutral value (0.5) as last resort — with a clear warning.

    Note: Trends data is normalised search interest (0-100), sample-based,
    not absolute volume. Used here as attention/cue proxy only.
    """
    print("  Google Trends TR …", end=' ', flush=True)

    terms   = ['USDT', 'Tether', 'dolar kripto']
    results = {}

    # ── Attempt 1: direct HTTP to unofficial Trends CSV endpoint ────────────
    import csv, io, time as _time
    session = requests.Session()
    session.headers.update({
        'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36'),
        'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
    })

    # Step 1: get consent cookie
    try:
        session.get('https://trends.google.com/', timeout=10)
    except Exception:
        pass

    for term in terms:
        try:
            # Step 2: fetch explore token
            explore_url = 'https://trends.google.com/trends/api/explore'
            explore_params = {
                'hl': 'tr', 'tz': '-180', 'req': (
                    f'{{"comparisonItem":[{{"keyword":"{term}",'
                    f'"geo":"TR","time":"{START} {END}"}}],'
                    f'"category":0,"property":""}}'
                )
            }
            r = session.get(explore_url, params=explore_params, timeout=12)
            # Response starts with ")]}',\n" — strip it
            text  = r.text[5:] if r.text.startswith(")]}',") else r.text
            token = None
            import json as _json
            data  = _json.loads(text)
            for widget in data.get('widgets', []):
                if widget.get('title') == 'Interest over time':
                    token = widget['token']
                    break
            if token is None:
                raise ValueError("no token")

            # Step 3: fetch CSV with token
            csv_url = 'https://trends.google.com/trends/api/widgetdata/multiline/csv'
            csv_params = {
                'req': (f'{{"time":"{START} {END}","resolution":"WEEK",'
                        f'"locale":"tr","comparisonItem":[{{"geo":{{"country":"TR"}},'
                        f'"complexKeywordsRestriction":{{"keyword":[{{"type":"BROAD",'
                        f'"value":"{term}"}}]}}}}],"requestOptions":{{"property":"",'
                        f'"backend":"IZG","category":0}}}}'),
                'token': token, 'hl': 'tr', 'tz': '-180', 'csv': '1',
            }
            cr = session.get(csv_url, params=csv_params, timeout=12)
            reader = csv.reader(io.StringIO(cr.text))
            rows   = list(reader)
            # Skip header rows — find the date column
            data_rows = [(r[0], r[1]) for r in rows
                         if len(r) >= 2 and r[0].startswith('20')]
            if not data_rows:
                raise ValueError("no data rows")
            dates_t = pd.to_datetime([r[0] for r in data_rows])
            vals_t  = pd.to_numeric([r[1] for r in data_rows],
                                     errors='coerce').fillna(0)
            results[term] = pd.Series(vals_t.values, index=dates_t)
            _time.sleep(0.8)

        except Exception as e:
            results[term] = None

    successful = {k: v for k, v in results.items() if v is not None}

    if successful:
        gt = pd.DataFrame(successful)
        gt.index = pd.to_datetime(gt.index)
        gt = gt.resample('W-MON').mean()
        gt['trends_composite'] = gt.mean(axis=1)
        print(f"OK via direct HTTP ({len(gt)} weeks, "
              f"{len(successful)}/{len(terms)} terms)")
        return gt[['trends_composite']]

    # ── Attempt 2: pytrends with pinned Retry args ───────────────────────────
    try:
        from pytrends.request import TrendReq
        import urllib3
        # Build session without the broken 'method_' kwarg
        retry = urllib3.Retry(total=3, backoff_factor=0.5)
        pt = TrendReq(hl='tr-TR', tz=180, timeout=(15, 30), retries=retry)
        pt.build_payload(terms, geo='TR',
                         timeframe=f'{START} {END}')
        gt = pt.interest_over_time()
        if 'isPartial' in gt.columns:
            gt = gt.drop(columns=['isPartial'])
        gt.index = pd.to_datetime(gt.index)
        gt = gt.resample('W-MON').mean()
        gt['trends_composite'] = gt.mean(axis=1)
        print(f"OK via pytrends ({len(gt)} weeks)")
        return gt[['trends_composite']]
    except Exception as e2:
        pass

    # ── Attempt 3: pip install compatible pytrends version ───────────────────
    try:
        import subprocess
        subprocess.run(['pip', 'install', 'pytrends==4.9.2',
                        'urllib3<2.0', '-q', '--break-system-packages'],
                       check=True, capture_output=True)
        from importlib import reload
        import pytrends.request
        reload(pytrends.request)
        from pytrends.request import TrendReq
        pt = TrendReq(hl='tr-TR', tz=180, timeout=(15, 30),
                      retries=3, backoff_factor=0.5)
        pt.build_payload(terms, geo='TR',
                         timeframe=f'{START} {END}')
        gt = pt.interest_over_time()
        if 'isPartial' in gt.columns:
            gt = gt.drop(columns=['isPartial'])
        gt.index = pd.to_datetime(gt.index)
        gt = gt.resample('W-MON').mean()
        gt['trends_composite'] = gt.mean(axis=1)
        print(f"OK via pytrends 4.9.2 ({len(gt)} weeks)")
        return gt[['trends_composite']]
    except Exception as e3:
        raise RuntimeError(
            f"All Trends fetch methods failed.\n"
            f"  Direct HTTP : {results}\n"
            f"  pytrends    : {e2}\n"
            f"  pip+retry   : {e3}\n"
            f"Fix: pip install 'urllib3<2' pytrends==4.9.2"
        )

def interp_anchors(anchors, dates):
    a_d = pd.to_datetime(list(anchors.keys()))
    a_v = np.array(list(anchors.values()), dtype=float)
    return np.interp(dates.astype(np.int64), a_d.astype(np.int64), a_v)

def synthetic_fallback(dates):
    """
    ═══════════════════════════════════════════════════════
    PROTOTYPE ONLY — NOT EMPIRICAL EVIDENCE
    Anchors are historically documented but data is synthetic.
    Figures produced from this data are watermarked.
    ═══════════════════════════════════════════════════════
    """
    global IS_SYNTHETIC
    IS_SYNTHETIC = True
    np.random.seed(42)
    n = len(dates)

    usd_try = interp_anchors({
        '2018-01-01':3.80,'2018-08-13':7.00,'2018-09-13':6.20,
        '2018-12-31':5.28,'2019-12-31':5.95,'2020-12-31':7.43,
        '2021-09-01':8.80,'2021-12-20':18.36,'2021-12-31':13.32,
        '2022-03-01':14.50,'2022-12-31':18.72,'2023-06-15':23.50,
        '2023-12-31':29.51,'2024-12-31':34.20,
    }, dates) + np.random.normal(0, 0.10, n)

    # Premium spikes anchored conservatively
    # (intentionally smaller than v5 to reduce circularity in synthetic test)
    log_prem = interp_anchors({
        '2018-01-01':0.001,'2018-08-13':0.035,'2018-09-13':0.012,
        '2018-12-31':0.006,'2019-12-31':0.003,'2020-12-31':0.006,
        '2021-09-01':0.008,'2021-12-20':0.030,'2021-12-31':0.014,
        '2022-10-01':0.022,'2022-12-31':0.008,'2023-06-15':0.028,
        '2023-12-31':0.010,'2024-12-31':0.007,
    }, dates) + np.random.normal(0, 0.003, n)

    gt = interp_anchors({
        '2018-01-01':8,'2018-08-13':65,'2018-09-15':30,
        '2018-12-31':13,'2019-12-31':11,'2020-12-31':17,
        '2021-09-01':24,'2021-12-20':75,'2021-12-31':40,
        '2022-10-01':50,'2022-12-31':26,'2023-06-15':58,
        '2023-12-31':33,'2024-12-31':28,
    }, dates)
    gt = np.clip(gt + np.random.normal(0, 3, n), 0, 100)

    cpi = interp_anchors({
        '2018-01-01':10.3,'2018-08-01':17.9,'2018-10-01':25.2,
        '2018-12-31':20.3,'2019-12-31':11.8,'2020-12-31':14.6,
        '2021-12-01':21.3,'2022-03-01':61.1,'2022-10-01':85.5,
        '2022-12-31':64.3,'2023-12-31':64.8,'2024-12-31':47.1,
    }, dates)

    return dict(
        usd_try_official=usd_try,
        usdt_try_px=usd_try * np.exp(log_prem),
        usdt_try_vol=np.maximum(
            interp_anchors({'2018-01-01':0.5,'2018-08-13':6.5,
                '2018-12-31':1.3,'2019-12-31':1.0,'2020-12-31':2.6,
                '2021-12-20':10.0,'2021-12-31':4.5,'2022-10-01':6.0,
                '2023-06-15':7.8,'2024-12-31':5.5}, dates)
            + np.abs(np.random.normal(0, 0.3, n)), 0.1),
        cpi_yoy=cpi,
        trends_composite=gt,
    )

def load_data():
    global IS_SYNTHETIC
    dates = pd.date_range(START, END, freq='W-MON')
    df, failed = pd.DataFrame(index=dates), []

    print("\n── Fetching data ────────────────────────────────────────────")
    for name, fetcher, cols in [
        ('Binance',    fetch_binance,    ['usdt_try_px','usdt_try_vol']),
        ('FRED',       fetch_fred,       ['usd_try_official']),
        ('WorldBank',  fetch_worldbank,  ['cpi_yoy']),
        ('Trends',     fetch_trends,     ['trends_composite']),
    ]:
        try:
            src = fetcher()
            for col in cols:
                if col in src.columns:
                    df[col] = src[col].reindex(dates, method='ffill')
        except Exception as e:
            print(f"  WARNING: {name} failed — {str(e)[:60]}")
            failed.append(name)

    if failed:
        critical_failed_now = [s for s in failed if s in ('Binance', 'FRED')]
        syn = synthetic_fallback(dates) if critical_failed_now else None
        if syn is None:
            # Only non-critical failed — don't mark as synthetic
            IS_SYNTHETIC = False

        for col, key, is_critical in [
            ('usd_try_official', 'usd_try_official', True),
            ('usdt_try_px',      'usdt_try_px',      True),
            ('usdt_try_vol',     'usdt_try_vol',      True),
            ('cpi_yoy',          'cpi_yoy',           False),
            ('trends_composite', 'trends_composite',  False),
        ]:
            if col not in df.columns or df[col].isna().all():
                if is_critical and syn is not None:
                    df[col] = syn[key]
                elif not is_critical:
                    # Neutral fill: 0.5 for Trends (mid-range, uninformative)
                    # cpi_yoy fallback: use 12w FX proxy built later in features
                    df[col] = 50.0 if col == 'trends_composite' else 0.0
                    print(f"  Neutral fill for {col} — "
                          f"{'Trends×M terms will be uninformative' if col=='trends_composite' else 'CPI proxy from FX will be used'}")

    df = df.ffill().bfill()
    df = df[(df.index >= START) & (df.index <= END)]

    # ── Abort logic ───────────────────────────────────────────────────────────
    # Binance and FRED are critical — without them the primary dep. variable
    # and shock series cannot be constructed from real data.
    # World Bank CPI is validation-only, Trends is a cue proxy:
    # both are useful but not fatal if absent.
    critical_failed = [s for s in failed if s in ('Binance', 'FRED')]
    if critical_failed and ABORT_ON_SYNTHETIC:
        print(f"\n  ABORT: Critical sources unavailable: {critical_failed}")
        print("  Binance (USDT/TRY premium) and FRED (USD/TRY) are required.")
        print("  Set ABORT_ON_SYNTHETIC=False to run on synthetic data")
        print("  (results will be watermarked and not publishable).")
        raise SystemExit(1)

    if failed and not critical_failed:
        print(f"\n  NOTE: Non-critical sources unavailable: {failed}")
        print("  Proceeding with real data for critical sources.")
        print("  Trends replaced with neutral 0.5 — cue-reactivation")
        print("  terms in Model C will be uninformative.")

    status = "⚠  SYNTHETIC PROTOTYPE" if IS_SYNTHETIC else "✓  REAL DATA"
    print(f"\n  Status : {status}")
    print(f"  Rows   : {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df):
    # Primary dep var — level
    with np.errstate(divide='ignore', invalid='ignore'):
        df['log_premium'] = np.log(
            df['usdt_try_px'] / df['usd_try_official'].replace(0, np.nan)
        ).fillna(0)

    # FX shock features
    df['fx_ret_1w']  = df['usd_try_official'].pct_change(1).fillna(0)
    df['fx_ret_4w']  = df['usd_try_official'].pct_change(4).fillna(0)
    df['fx_vol_4w']  = (df['fx_ret_1w'].rolling(4, min_periods=2)
                         .std().fillna(0))

    df['cpi_proxy'] = df['usd_try_official'].pct_change(12).fillna(0)
    mu_cpi = df['cpi_proxy'].mean()
    sd_cpi = df['cpi_proxy'].std() + 1e-9
    df['cpi_proxy_norm'] = (df['cpi_proxy'] - mu_cpi) / sd_cpi

    # Realised abruptness from data (not hand-scored)
    df['fx_abruptness'] = (
        df['fx_ret_1w'].abs() /
        (df['fx_ret_4w'].abs().rolling(4, min_periods=1).mean() + 1e-6)
    ).clip(0, 10)

    # ── Extra features for Model C variants ──────────────────────────────────
    # Squared FX shock — captures nonlinear (convex) response to large moves
    # Used in C2 and C3: prior encoding amplifies extreme shocks most
    df['fx_ret_4w_sq'] = df['fx_ret_4w'] ** 2

    # Current-moment ESS proxy (continuous, computed each week from real data)
    # Operationalises similarity to past high-ESS events along the same
    # three dimensions used to score catalogue shocks:
    #   magnitude  → |fx_ret_4w| normalised
    #   abruptness → fx_abruptness normalised
    #   relevance  → cpi_proxy_norm (household inflation pressure)
    fx_abs_norm  = (df['fx_ret_4w'].abs() /
                    (df['fx_ret_4w'].abs().max() + 1e-9))
    abr_norm     = df['fx_abruptness'] / (df['fx_abruptness'].max() + 1e-9)
    rel_norm     = (df['cpi_proxy_norm'] - df['cpi_proxy_norm'].min()) / \
                   (df['cpi_proxy_norm'].max() - df['cpi_proxy_norm'].min() + 1e-9)
    # Weights match ESS catalogue weights (w_mag=0.4, w_abr=0.3, w_rel=0.3)
    # giving a continuous weekly ESS-analogue scored the same way
    df['current_ess'] = 0.4 * fx_abs_norm + 0.3 * abr_norm + 0.3 * rel_norm

    df['trends_norm'] = (df['trends_composite'] / 100.0).clip(0, 1)

    # ── AR terms  ─────────────────────────────────────────────────
    # Level model: AR(1) of log_premium
    df['ar1_level'] = df['log_premium'].shift(1).fillna(0)
    # Difference model: AR(1) of d_premium (not of level)
    df['d_premium'] = df['log_premium'].diff().fillna(0)
    df['ar1_diff']  = df['d_premium'].shift(1).fillna(0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SHOCK CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

SHOCKS = pd.DataFrame([
    dict(date='2018-08-13', short='Aug18', type='primary',
         magnitude=0.40, abruptness_days=14,  personal_relevance=0.80,
         description='US tariff escalation. Lira −40% in 14 days. '
                     'CBRT hikes to 24% (13 Sep 2018).'),
    dict(date='2021-12-20', short='Dec21', type='secondary',
         magnitude=0.44, abruptness_days=60,  personal_relevance=0.70,
         description='Erdoğan rate cuts against inflation. '
                     'Lira −44% across Q4 2021.'),
    dict(date='2022-10-01', short='Oct22', type='secondary',
         magnitude=0.30, abruptness_days=180, personal_relevance=0.90,
         description='CPI 85.5% — 24-year high. Gradual but historic.'),
    dict(date='2023-06-15', short='Jun23', type='secondary',
         magnitude=0.25, abruptness_days=30,  personal_relevance=0.60,
         description='Post-election FM change. Orthodox policy reversal.'),
])
SHOCKS['date']            = pd.to_datetime(SHOCKS['date'])
max_abrupt                = (1.0 / SHOCKS['abruptness_days']).max()
SHOCKS['abruptness_norm'] = 1.0 / SHOCKS['abruptness_days'] / max_abrupt


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MEMORY KERNEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_ESS(shocks, w_mag, w_abr, w_rel):
    return (w_mag * shocks['magnitude']
          + w_abr * shocks['abruptness_norm']
          + w_rel * shocks['personal_relevance'])

def compute_M(dates, shocks, lam, w_mag, w_abr, w_rel,
              exclude_shock_date=None):
    """
    M(t) = Σ_{τ<t} ESS(τ) · exp(−λ · Δweeks)

    If exclude_shock_date is set, that shock's ESS contribution is
    omitted entirely — producing M_prior(t), the memory trace from
    PRIOR shocks only.

    Why this matters for the reactivation test:
    Without exclusion, Model B during the Dec 2021 evaluation window
    includes Dec 2021's own ESS in M(t) once dw > 0 — so the model
    partly benefits from knowing the magnitude of the very shock it
    is trying to predict. This conflates contemporaneous shock signal
    with prior memory. M_prior fixes this: during the Dec 2021 window,
    M(t) only reflects the 2018 primary shock (and any others before
    Dec 2021), giving a clean test of the claim:
      'Prior memory alone amplifies the response to a new shock.'

    During training (all dates before eval shock), exclusion has no
    effect, the excluded shock hasn't happened yet so dw <= 0 anyway.
    The difference only appears in the evaluation window itself.
    """
    ess       = compute_ESS(shocks, w_mag, w_abr, w_rel).values
    excl_date = pd.Timestamp(exclude_shock_date) if exclude_shock_date else None
    M         = np.zeros(len(dates))
    for i, (_, s) in enumerate(shocks.iterrows()):
        if excl_date is not None and s['date'] == excl_date:
            continue                        # exclude evaluated shock
        dw = (dates - s['date']).days.values / 7.0
        M += np.where(dw > 0, ess[i] * np.exp(-lam * dw), 0.0)
    return M


# ══════════════════════════════════════════════════════════════════════════════
# 5.  REGRESSION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def ols_fit(X, y, mask):
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return coef

def ridge_fit(X, y, mask, alpha=0.01):
    """Ridge with alpha. Intercept (last col) not penalised."""
    Xm, ym = X[mask], y[mask]
    p   = Xm.shape[1]
    reg = alpha * np.eye(p)
    reg[-1, -1] = 0.0
    return np.linalg.solve(Xm.T @ Xm + reg, Xm.T @ ym)

def select_alpha(X, y, mask_tr, alphas=None):
    """5-fold CV within training set to select ridge alpha."""
    if alphas is None:
        alphas = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
    idx = np.where(mask_tr)[0]
    np.random.seed(42)
    np.random.shuffle(idx)
    folds, best_a, best_mse = np.array_split(idx, 5), alphas[0], np.inf
    for a in alphas:
        mses = []
        for fold in folds:
            val = fold
            tr  = np.concatenate([f for f in folds if f is not fold])
            mk_t = np.zeros(len(y), bool); mk_t[tr]  = True
            mk_v = np.zeros(len(y), bool); mk_v[val] = True
            try:
                c = ridge_fit(X, y, mk_t, a)
                mses.append(((y[mk_v] - X[mk_v] @ c)**2).mean())
            except Exception:
                mses.append(np.inf)
        mse = np.mean(mses)
        if mse < best_mse:
            best_mse, best_a = mse, a
    return best_a

def r2_score(y, yhat, mask):
    if mask.sum() < 2:
        return np.nan
    ss_r = ((y[mask] - yhat[mask])**2).sum()
    ss_t = ((y[mask] - y[mask].mean())**2).sum()
    return float(1 - ss_r / (ss_t + 1e-12))

def fit_predict(X, y, mask_tr, mask_te, use_ridge=False, alpha=0.01):
    ok_tr = mask_tr & np.isfinite(X).all(1) & np.isfinite(y)
    ok_te = mask_te & np.isfinite(X).all(1) & np.isfinite(y)
    if ok_tr.sum() < 5:
        return np.full(len(y), np.nan), np.nan, np.nan, None
    coef = ridge_fit(X, y, ok_tr, alpha) if use_ridge else ols_fit(X, y, ok_tr)
    yhat = np.full(len(y), np.nan)
    yhat[ok_tr | ok_te] = X[ok_tr | ok_te] @ coef
    return yhat, r2_score(y, yhat, ok_tr), r2_score(y, yhat, ok_te), coef

def compute_vif(X, names):
    vifs = {}
    for i in range(X.shape[1]):
        xi = X[:, i]
        Xr = np.delete(X, i, axis=1)
        c, *_ = np.linalg.lstsq(Xr, xi, rcond=None)
        r2 = 1 - ((xi - Xr @ c)**2).sum() / ((xi - xi.mean())**2).sum()
        vifs[names[i]] = 1 / (1 - r2 + 1e-10)
    return vifs

def true_bootstrap_ci(X, y, mask_tr, coef_fn, n_boot=500, seed=42):
    """True bootstrap with replacement — preserves duplicates."""
    rng    = np.random.default_rng(seed)
    idx_tr = np.where(mask_tr & np.isfinite(y) & np.isfinite(X).all(1))[0]
    boots  = []
    for _ in range(n_boot):
        samp = rng.choice(idx_tr, size=len(idx_tr), replace=True)
        Xb, yb = X[samp], y[samp]
        try:
            boots.append(coef_fn(Xb, yb))
        except Exception:
            continue
    boots = np.array(boots)
    return (np.percentile(boots, 2.5, axis=0),
            np.percentile(boots, 97.5, axis=0),
            boots)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVENT-CENTRED MASKS
# ══════════════════════════════════════════════════════════════════════════════

def event_masks(dates, shock_date, train_pre_weeks=2, test_post_weeks=24):
    """
    Train: all data strictly before (shock_date - train_pre_weeks).
    Test:  shock_date to shock_date + test_post_weeks.
    The gap of train_pre_weeks prevents last-minute leakage.
    """
    idx    = np.argmin(np.abs(dates - shock_date))
    tr_end = max(0, idx - train_pre_weeks)
    te_end = min(len(dates), idx + test_post_weeks)
    mask_tr = np.zeros(len(dates), bool)
    mask_te = np.zeros(len(dates), bool)
    mask_tr[:tr_end] = True
    mask_te[idx:te_end] = True
    return mask_tr, mask_te


# ══════════════════════════════════════════════════════════════════════════════
# 7.  OPTIMISATION 
# ══════════════════════════════════════════════════════════════════════════════

def build_XB(df, M):
    """Exact Model B feature matrix."""
    return np.column_stack([
        df['ar1_level'].values,
        df['fx_ret_4w'].values,
        df['fx_vol_4w'].values,
        df['cpi_proxy_norm'].values,
        df['trends_norm'].values,
        M,
        df['fx_ret_4w'].values * M,
        np.ones(len(df))
    ])

def tune_oos_loss(params, df, shocks, tune_shock, test_window=16):
    """OOS MSE of Model B evaluated on TUNE_SHOCK window only."""
    log_lam, *lw = params
    lam = np.clip(np.exp(log_lam), 1e-4, 0.5)
    w   = np.exp(lw); w /= w.sum()
    M   = compute_M(df.index, shocks, lam, *w)
    XB  = build_XB(df, M)
    y   = df['log_premium'].values
    mask_tr, mask_te = event_masks(df.index, tune_shock,
                                   test_post_weeks=test_window)
    ok  = np.isfinite(XB).all(1) & np.isfinite(y)
    mt, me = ok & mask_tr, ok & mask_te
    if mt.sum() < 10 or me.sum() < 3:
        return 1e6
    coef  = ols_fit(XB, y, mt)
    resid = y[me] - XB[me] @ coef
    return float((resid**2).mean())

def optimise(df, shocks, tune_shock=TUNE_SHOCK):
    from scipy.optimize import differential_evolution
    print(f"\n── Optimising λ + ESS on TUNE event: {tune_shock.date()} ──")
    print(f"   EVAL event (held out): {EVAL_SHOCK.date()} — never touched here")
    bounds = [(-7, -0.3), (-3, 3), (-3, 3), (-3, 3)]
    res = differential_evolution(
        tune_oos_loss, bounds,
        args=(df, shocks, tune_shock),
        seed=42, maxiter=500, tol=1e-7,
        popsize=15, workers=1, polish=True
    )
    log_lam, *lw = res.x
    lam = np.exp(log_lam)
    w   = np.exp(lw); w /= w.sum()
    hl  = np.log(2) / lam / 52
    print(f"  λ_opt    = {lam:.5f}  (half-life {hl:.2f} yr)")
    print(f"  w_mag    = {w[0]:.3f}")
    print(f"  w_abr    = {w[1]:.3f}")
    print(f"  w_rel    = {w[2]:.3f}")
    print(f"  Tune OOS MSE = {res.fun:.7f}")
    print(f"  Dominant ESS dimension: "
          f"{['magnitude','abruptness','relevance'][int(np.argmax(w))]}")
    return lam, w[0], w[1], w[2]


# ══════════════════════════════════════════════════════════════════════════════
# 8.  LAMBDA SENSITIVITY  (ESS weights fixed at tuned values)
# ══════════════════════════════════════════════════════════════════════════════

def lambda_sensitivity(df, shocks, w_mag, w_abr, w_rel,
                       tune_shock=TUNE_SHOCK, test_window=16):
    lam_grid = np.logspace(-3, -0.3, 60)
    losses   = []
    for lam in lam_grid:
        M  = compute_M(df.index, shocks, lam, w_mag, w_abr, w_rel)
        XB = build_XB(df, M)
        y  = df['log_premium'].values
        mask_tr, mask_te = event_masks(df.index, tune_shock,
                                       test_post_weeks=test_window)
        ok  = np.isfinite(XB).all(1) & np.isfinite(y)
        mt, me = ok & mask_tr, ok & mask_te
        if mt.sum() < 5 or me.sum() < 2:
            losses.append(np.nan); continue
        coef  = ols_fit(XB, y, mt)
        resid = y[me] - XB[me] @ coef
        losses.append(float((resid**2).mean()))
    return lam_grid, np.array(losses)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  THREE MODELS  — evaluated on held-out EVAL_SHOCK
# ══════════════════════════════════════════════════════════════════════════════

def run_models(df, shocks, lam, w_mag, w_abr, w_rel,
               eval_shock=EVAL_SHOCK, test_window=24):
    """
    PRIMARY SPEC: levels (log_premium). Lead with this in reporting.
    SECONDARY SPEC: first differences. Robustness check only.

    Four models — hierarchy matters:
      A_trend  Null comparator: AR(1) + smooth time trend
               Tests whether M(t) beats trend, not just complexity.
      A        Memoryless ARX baseline
      B        Path-dependent  ← PRIMARY CLAIM
      C        Reactivation asymmetry ← EXPLORATORY ONLY, do not lead with

    Tune event: TUNE_SHOCK (Dec 2021)
    Eval event: EVAL_SHOCK (Jun 2023) — never touched in tuning.
    M_prior excludes the eval shock's own ESS during the test window.
    """
    M_full  = compute_M(df.index, shocks, lam, w_mag, w_abr, w_rel)
    M_prior = compute_M(df.index, shocks, lam, w_mag, w_abr, w_rel,
                        exclude_shock_date=eval_shock)
    mask_tr, mask_te = event_masks(df.index, eval_shock,
                                   test_post_weeks=test_window)
    # Train rows: M_full (eval shock hasn't happened, identical to M_prior)
    # Test rows:  M_prior (exclude eval shock's own ESS — clean reactivation)
    M = np.where(mask_te, M_prior, M_full)

    # Smooth time trend — normalised 0→1 over full sample
    t_trend = np.linspace(0, 1, len(df))

    results = {}
    for dep, ar_col, label, is_primary in [
        ('log_premium', 'ar1_level', 'levels',      True),   # PRIMARY
        ('d_premium',   'ar1_diff',  'differences', False),  # robustness
    ]:
        y  = df[dep].values
        ok = np.isfinite(y)
        mt, me = ok & mask_tr, ok & mask_te

        # ── Model A_trend — null comparator ─────────────────────────────────
        # AR(1) + time trend only.
        # If Model B does not beat A_trend, M(t) is not adding signal
        # beyond what a simple secular drift would capture.
        XA_trend = np.column_stack([
            df[ar_col].values,
            t_trend,
            np.ones(len(df))
        ])
        yAt, r2At_tr, r2At_te, _ = fit_predict(XA_trend, y, mt, me)

        # ── Model A — memoryless ARX ─────────────────────────────────────────
        XA = np.column_stack([
            df[ar_col].values,
            df['fx_ret_4w'].values,
            df['fx_vol_4w'].values,
            df['cpi_proxy_norm'].values,
            df['trends_norm'].values,
            np.ones(len(df))
        ])
        yA, r2A_tr, r2A_te, cA = fit_predict(XA, y, mt, me)

        # ── Model B — path-dependent  [PRIMARY CLAIM] ────────────────────────
        XB = np.column_stack([
            df[ar_col].values,
            df['fx_ret_4w'].values,
            df['fx_vol_4w'].values,
            df['cpi_proxy_norm'].values,
            df['trends_norm'].values,
            M,
            df['fx_ret_4w'].values * M,
            np.ones(len(df))
        ])
        yB, r2B_tr, r2B_te, cB = fit_predict(XB, y, mt, me)

        # ── Model C1 — original sensitisation [v8 baseline for comparison] ───
        # Abruptness×M + Trends×M.
        # Tests: does prior fear make ANY current signal hit harder?
        # This is general sensitisation, not specifically reactivation.
        # Included here for comparison against C2 and C3.
        XC1 = np.column_stack([
            df[ar_col].values,
            df['fx_ret_4w'].values,
            df['fx_vol_4w'].values,
            df['cpi_proxy_norm'].values,
            df['trends_norm'].values,
            M,
            df['fx_ret_4w'].values     * M,
            df['fx_abruptness'].values * M,   # general sensitisation term 1
            df['trends_norm'].values   * M,   # general sensitisation term 2
            np.ones(len(df))
        ])
        alpha_C1 = select_alpha(XC1, y, mt)
        yC1, r2C1_tr, r2C1_te, cC1 = fit_predict(XC1, y, mt, me,
                                                    use_ridge=True, alpha=alpha_C1)
        c_names_C1 = ['AR(1)','FX_4w','FX_vol','CPI_proxy','Trends',
                      'M','FX×M',
                      'Abruptness×M  [C1: general sensitisation]',
                      'Trends×M      [C1: general sensitisation]',
                      'Intercept']
        vifs_C1 = compute_vif(XC1[mt], c_names_C1)
        def coef_fn_C1(Xb, yb):
            p = Xb.shape[1]; reg = alpha_C1*np.eye(p); reg[-1,-1]=0
            return np.linalg.solve(Xb.T@Xb+reg, Xb.T@yb)
        ci_lo_C1, ci_hi_C1, _ = true_bootstrap_ci(XC1, y, mt, coef_fn_C1, n_boot=400)

        # ── Model C2 — threshold-sensitisation [encoding-strength gated] ─────
        # Abruptness×M + FX²×M.
        # Theory: prior high-ESS encoding lowers the threshold so later
        # shocks produce responses that are (a) more abrupt and (b) nonlinearly
        # amplified — larger shocks overshoot more than smaller ones.
        # Amplification is gated by prior encoding STRENGTH, not cue similarity.
        # This is the "more abrupt and drastic" prediction.
        XC2 = np.column_stack([
            df[ar_col].values,
            df['fx_ret_4w'].values,
            df['fx_vol_4w'].values,
            df['cpi_proxy_norm'].values,
            df['trends_norm'].values,
            M,
            df['fx_ret_4w'].values     * M,
            df['fx_abruptness'].values * M,   # response more abrupt when M high
            df['fx_ret_4w_sq'].values  * M,   # nonlinear: extreme shocks overshoot
            np.ones(len(df))
        ])
        alpha_C2 = select_alpha(XC2, y, mt)
        yC2, r2C2_tr, r2C2_te, cC2 = fit_predict(XC2, y, mt, me,
                                                    use_ridge=True, alpha=alpha_C2)
        c_names_C2 = ['AR(1)','FX_4w','FX_vol','CPI_proxy','Trends',
                      'M','FX×M',
                      'Abruptness×M  [C2: abrupt response when M high]',
                      'FX²×M         [C2: nonlinear overshoot when M high]',
                      'Intercept']
        vifs_C2 = compute_vif(XC2[mt], c_names_C2)
        def coef_fn_C2(Xb, yb):
            p = Xb.shape[1]; reg = alpha_C2*np.eye(p); reg[-1,-1]=0
            return np.linalg.solve(Xb.T@Xb+reg, Xb.T@yb)
        ci_lo_C2, ci_hi_C2, _ = true_bootstrap_ci(XC2, y, mt, coef_fn_C2, n_boot=400)

        # ── Model C3 — similarity-gated reactivation ──────────────────────────
        # current_ESS × M(t).
        # Theory: reactivation fires when the CURRENT event resembles the
        # original high-ESS shock along the same three dimensions (magnitude,
        # abruptness, household relevance). Low-ESS current events (gradual,
        # small, abstract) do not reactivate the trace even when M(t) is high.
        # current_ess is computed weekly from real FX/CPI data using the same
        # 0.4/0.3/0.3 weights as the catalogue ESS scores.
        XC3 = np.column_stack([
            df[ar_col].values,
            df['fx_ret_4w'].values,
            df['fx_vol_4w'].values,
            df['cpi_proxy_norm'].values,
            df['trends_norm'].values,
            M,
            df['fx_ret_4w'].values    * M,
            df['current_ess'].values  * M,   # similarity × memory = reactivation
            np.ones(len(df))
        ])
        alpha_C3 = select_alpha(XC3, y, mt)
        yC3, r2C3_tr, r2C3_te, cC3 = fit_predict(XC3, y, mt, me,
                                                    use_ridge=True, alpha=alpha_C3)
        c_names_C3 = ['AR(1)','FX_4w','FX_vol','CPI_proxy','Trends',
                      'M','FX×M',
                      'CurrentESS×M  [C3: similarity-gated reactivation]',
                      'Intercept']
        vifs_C3 = compute_vif(XC3[mt], c_names_C3)
        def coef_fn_C3(Xb, yb):
            p = Xb.shape[1]; reg = alpha_C3*np.eye(p); reg[-1,-1]=0
            return np.linalg.solve(Xb.T@Xb+reg, Xb.T@yb)
        ci_lo_C3, ci_hi_C3, _ = true_bootstrap_ci(XC3, y, mt, coef_fn_C3, n_boot=400)

        results[label] = dict(
            y=y, M=M, mask_tr=mask_tr, mask_te=mask_te,
            is_primary=is_primary,
            yAt=yAt, r2At_tr=r2At_tr, r2At_te=r2At_te,
            yA=yA,   r2A_tr=r2A_tr,   r2A_te=r2A_te,
            yB=yB,   r2B_tr=r2B_tr,   r2B_te=r2B_te,
            # C1 — original sensitisation (v8 baseline)
            yC1=yC1, r2C1_tr=r2C1_tr, r2C1_te=r2C1_te,
            cC1=cC1, c_names_C1=c_names_C1, vifs_C1=vifs_C1,
            ci_lo_C1=ci_lo_C1, ci_hi_C1=ci_hi_C1, alpha_C1=alpha_C1,
            # C2 — threshold-sensitisation (abruptness + nonlinear)
            yC2=yC2, r2C2_tr=r2C2_tr, r2C2_te=r2C2_te,
            cC2=cC2, c_names_C2=c_names_C2, vifs_C2=vifs_C2,
            ci_lo_C2=ci_lo_C2, ci_hi_C2=ci_hi_C2, alpha_C2=alpha_C2,
            # C3 — similarity-gated reactivation (current_ESS × M)
            yC3=yC3, r2C3_tr=r2C3_tr, r2C3_te=r2C3_te,
            cC3=cC3, c_names_C3=c_names_C3, vifs_C3=vifs_C3,
            ci_lo_C3=ci_lo_C3, ci_hi_C3=ci_hi_C3, alpha_C3=alpha_C3,
            cA=cA, cB=cB,
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PLACEBO TEST — for both B-A and C-B
# ══════════════════════════════════════════════════════════════════════════════

def placebo_test(df, shocks, lam, w_mag, w_abr, w_rel,
                 alpha_C1=0.01, alpha_C2=0.01, alpha_C3=0.01,
                 eval_shock=EVAL_SHOCK,
                 n_placebo=50, test_window=16, seed=42):
    """
    Placebo test for B-A and all three C variants (C1, C2, C3) vs B.
    Each C uses its own CV-selected alpha for fair comparison.
    M_prior logic applied consistently.
    """
    np.random.seed(seed)
    dates     = df.index
    shock_idx = set(np.argmin(np.abs(dates - s['date']))
                    for _, s in shocks.iterrows())
    buffer    = 52
    valid     = [i for i in range(buffer, len(dates)-test_window-4)
                 if all(abs(i-s) > 8 for s in shock_idx)]
    chosen    = np.random.choice(valid, size=min(n_placebo, len(valid)),
                                 replace=False)

    y       = df['log_premium'].values
    XA_base = np.column_stack([
        df['ar1_level'].values, df['fx_ret_4w'].values,
        df['fx_vol_4w'].values, df['cpi_proxy_norm'].values,
        df['trends_norm'].values, np.ones(len(df))
    ])

    def gains_at(test_date, exclude_date=None):
        M_p = compute_M(dates, shocks, lam, w_mag, w_abr, w_rel,
                        exclude_shock_date=exclude_date)
        XB = np.column_stack([
            df['ar1_level'].values, df['fx_ret_4w'].values,
            df['fx_vol_4w'].values, df['cpi_proxy_norm'].values,
            df['trends_norm'].values, M_p,
            df['fx_ret_4w'].values * M_p, np.ones(len(df))
        ])
        # C1: general sensitisation
        XC1 = np.column_stack([
            df['ar1_level'].values, df['fx_ret_4w'].values,
            df['fx_vol_4w'].values, df['cpi_proxy_norm'].values,
            df['trends_norm'].values, M_p,
            df['fx_ret_4w'].values * M_p,
            df['fx_abruptness'].values * M_p,
            df['trends_norm'].values   * M_p,
            np.ones(len(df))
        ])
        # C2: threshold-sensitisation
        XC2 = np.column_stack([
            df['ar1_level'].values, df['fx_ret_4w'].values,
            df['fx_vol_4w'].values, df['cpi_proxy_norm'].values,
            df['trends_norm'].values, M_p,
            df['fx_ret_4w'].values     * M_p,
            df['fx_abruptness'].values * M_p,
            df['fx_ret_4w_sq'].values  * M_p,
            np.ones(len(df))
        ])
        # C3: similarity-gated
        XC3 = np.column_stack([
            df['ar1_level'].values, df['fx_ret_4w'].values,
            df['fx_vol_4w'].values, df['cpi_proxy_norm'].values,
            df['trends_norm'].values, M_p,
            df['fx_ret_4w'].values   * M_p,
            df['current_ess'].values * M_p,
            np.ones(len(df))
        ])
        mt, me = event_masks(dates, test_date, test_post_weeks=test_window)
        ok = np.isfinite(y)
        _, _, r2A,  _ = fit_predict(XA_base, y, ok&mt, ok&me)
        _, _, r2B,  _ = fit_predict(XB,  y, ok&mt, ok&me)
        _, _, r2C1, _ = fit_predict(XC1, y, ok&mt, ok&me, use_ridge=True, alpha=alpha_C1)
        _, _, r2C2, _ = fit_predict(XC2, y, ok&mt, ok&me, use_ridge=True, alpha=alpha_C2)
        _, _, r2C3, _ = fit_predict(XC3, y, ok&mt, ok&me, use_ridge=True, alpha=alpha_C3)
        if not all(np.isfinite([r2A, r2B, r2C1, r2C2, r2C3])):
            return None
        return dict(BA=r2B-r2A, C1B=r2C1-r2B, C2B=r2C2-r2B, C3B=r2C3-r2B)

    real = gains_at(eval_shock, exclude_date=eval_shock)

    plac_BA, plac_C1B, plac_C2B, plac_C3B = [], [], [], []
    for ci in chosen:
        g = gains_at(dates[ci], exclude_date=dates[ci])
        if g is not None:
            plac_BA.append(g['BA'])
            plac_C1B.append(g['C1B'])
            plac_C2B.append(g['C2B'])
            plac_C3B.append(g['C3B'])

    plac_BA  = np.array(plac_BA)
    plac_C1B = np.array(plac_C1B)
    plac_C2B = np.array(plac_C2B)
    plac_C3B = np.array(plac_C3B)

    real_BA  = real['BA']  if real else np.nan
    real_C1B = real['C1B'] if real else np.nan
    real_C2B = real['C2B'] if real else np.nan
    real_C3B = real['C3B'] if real else np.nan

    def pct_rank(plac, real_val):
        if len(plac) > 0 and np.isfinite(real_val):
            return np.sum(plac < real_val) / len(plac) * 100
        return np.nan

    print(f"\n── Placebo test ({len(plac_BA)} random dates) ──────────────────")
    print(f"  {'Claim':<40}  {'Real gain':>10}  {'Plac mean':>10}  {'Pctile':>8}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*8}")
    for claim, rg, plac in [
        ('B  over A  (path-dependence)',  real_BA,  plac_BA),
        ('C1 over B  (general sensitisation)', real_C1B, plac_C1B),
        ('C2 over B  (threshold-sensitisation)', real_C2B, plac_C2B),
        ('C3 over B  (similarity-gated)',  real_C3B, plac_C3B),
    ]:
        rg_s  = f"{rg:+.4f}" if np.isfinite(rg) else "   n/a"
        pm_s  = f"{np.nanmean(plac):+.4f}" if len(plac) > 0 else "   n/a"
        pct   = pct_rank(plac, rg)
        pct_s = f"{pct:.0f}th" if np.isfinite(pct) else "  n/a"
        print(f"  {claim:<40}  {rg_s:>10}  {pm_s:>10}  {pct_s:>8}")

    return (plac_BA, plac_C1B, plac_C2B, plac_C3B,
            real_BA, real_C1B, real_C2B, real_C3B)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  EVENT WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def event_window(series, shock_date, dates, pre=8, post=24):
    idx = np.argmin(np.abs(dates - shock_date))
    if idx - pre < 0 or idx + post >= len(series):
        return None, None
    w = series[idx-pre : idx+post].copy()
    b = np.nanmean(w[:pre])
    if not np.isfinite(b) or np.abs(b) < 1e-9:
        return None, None
    return np.arange(-pre, post), (w - b) / np.abs(b) * 100


# ══════════════════════════════════════════════════════════════════════════════
# 12.  FOUR-PANEL FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(df, res_lev, shocks, lam, w_mag, w_abr, w_rel,
                 lam_grid, lam_losses,
                 plac_BA, plac_C1B, plac_C2B, plac_C3B,
                 real_BA, real_C1B, real_C2B, real_C3B,
                 outpath):

    dates = df.index
    M     = res_lev['M']
    ess   = compute_ESS(shocks, w_mag, w_abr, w_rel)
    hl    = np.log(2) / lam / 52

    fig = plt.figure(figsize=(16, 17))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.36)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Watermark
    for ax in [ax1, ax2, ax3]:
        if IS_SYNTHETIC:
            ax.text(0.5, 0.5, 'SYNTHETIC — NOT EMPIRICAL',
                    transform=ax.transAxes, fontsize=15, color='red',
                    alpha=0.15, ha='center', va='center',
                    rotation=30, fontweight='bold', zorder=10)

    def vshock(ax, a1=0.13, a2=0.07):
        for _, s in shocks.iterrows():
            col = C['shock'] if s['type']=='primary' else C['event']
            ax.axvline(s['date'], color=col, lw=1.8, ls='--', alpha=0.9, zorder=3)
            ax.axvspan(s['date']-pd.Timedelta(weeks=2),
                       s['date']+pd.Timedelta(weeks=2),
                       alpha=a1 if s['type']=='primary' else a2,
                       color=col, zorder=2)
        # Mark the tuning / eval separation clearly
        ax.axvline(TUNE_SHOCK, color='blue', lw=1.2, ls=':', alpha=0.5)
        ax.axvline(EVAL_SHOCK, color='green', lw=1.2, ls=':', alpha=0.5)

    # ── Panel 1: USD/TRY + M(t) + premium ───────────────────────────────────
    ax1b = ax1.twinx()
    ax1.plot(dates, df['usd_try_official'],
             color=C['fx'], lw=2, label='USD/TRY (FRED)', zorder=4)
    ax1.fill_between(dates, df['usd_try_official'],
                     df['usd_try_official'].min(),
                     alpha=0.06, color=C['fx'])
    prem_scaled = (df['log_premium'] * 100 /
                   (df['log_premium'].abs().max() + 1e-9) *
                   df['usd_try_official'].max() * 0.14)
    ax1.fill_between(dates, prem_scaled + df['usd_try_official'].min(),
                     df['usd_try_official'].min(),
                     where=df['log_premium']>0, alpha=0.28,
                     color=C['premium'], label='USDT/TRY premium (scaled)')
    ax1b.plot(dates, M, color=C['memory'], lw=1.5, ls='-.',
              alpha=0.85, label=f'Memory M(t), λ={lam:.4f}')
    ax1b.set_ylabel('M(t)', color=C['memory'], fontsize=9)
    ax1b.tick_params(axis='y', colors=C['memory'])
    vshock(ax1)

    for i, (_, s) in enumerate(shocks.iterrows()):
        yfx = float(np.interp(s['date'].value, dates.astype(np.int64),
                              df['usd_try_official'].values))
        ax1.annotate(
            f"{s['short']}\nESS={ess.iloc[i]:.2f}",
            xy=(s['date'], yfx),
            xytext=(s['date']+pd.Timedelta(weeks=10+20*(i%2)),
                    yfx + 3*(1+i%2)),
            fontsize=8, color='#222',
            arrowprops=dict(arrowstyle='->', color='#666', lw=0.9),
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec='#ccc', alpha=0.9)
        )

    # Label tuning/eval events
    for d, label, col in [(TUNE_SHOCK, 'TUNE\n(Aug18)', 'blue'),
                           (EVAL_SHOCK, 'EVAL\n(Dec21)', 'green')]:
        ax1.text(d + pd.Timedelta(weeks=2),
                 df['usd_try_official'].max() * 0.97,
                 label, fontsize=7.5, color=col, fontweight='bold')

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=8.5, loc='upper left')
    ax1.set_ylabel('TRY per USD', fontsize=10, color=C['fx'])
    ax1.tick_params(axis='y', colors=C['fx'])
    ax1.set_title(
        f'Panel 1 — USD/TRY, USDT premium, and memory kernel M(t)\n'
        f'λ={lam:.4f}  (half-life {hl:.1f} yr)  |  '
        f'ESS weights: mag={w_mag:.2f}, abr={w_abr:.2f}, rel={w_rel:.2f}  |  '
        f'Blue dotted = tuning event, Green dotted = held-out eval event',
        fontsize=10, fontweight='bold'
    )

    # ── Panel 2: Event window — best C variant highlighted ────────────────────
    y_lev = res_lev['y']
    lags, ewO  = event_window(y_lev*100,               EVAL_SHOCK, dates)
    _,    ewA  = event_window(res_lev['yA']*100,       EVAL_SHOCK, dates)
    _,    ewB  = event_window(res_lev['yB']*100,       EVAL_SHOCK, dates)
    _,    ewC1 = event_window(res_lev['yC1']*100,      EVAL_SHOCK, dates)
    _,    ewC2 = event_window(res_lev['yC2']*100,      EVAL_SHOCK, dates)
    _,    ewC3 = event_window(res_lev['yC3']*100,      EVAL_SHOCK, dates)

    # Pick the best C variant by OOS R² for emphasis
    c_oos = {'C1': res_lev['r2C1_te'], 'C2': res_lev['r2C2_te'],
             'C3': res_lev['r2C3_te']}
    best_c_tag = max(c_oos, key=lambda k: c_oos[k] if np.isfinite(c_oos[k]) else -999)
    ew_best = {'C1': ewC1, 'C2': ewC2, 'C3': ewC3}[best_c_tag]
    best_r2  = c_oos[best_c_tag]
    best_lbl = {'C1':'General sensitisation','C2':'Threshold-sensitisation',
                'C3':'Similarity-gated'}[best_c_tag]

    if lags is not None:
        ax2.plot(lags, ewO,  color=C['premium'], lw=2.2, label='Observed', zorder=5)
        ax2.plot(lags, ewA,  color=C['A'],  lw=1.6, ls='--',
                 label=f"A Memoryless  OOS={res_lev['r2A_te']:.3f}")
        ax2.plot(lags, ewB,  color=C['B'],  lw=1.8,
                 label=f"B Path-dep     OOS={res_lev['r2B_te']:.3f}")
        ax2.plot(lags, ewC1, color=C['Cm'], lw=1.4, ls=':',
                 label=f"C1 Sensitise   OOS={res_lev['r2C1_te']:.3f}", alpha=0.7)
        ax2.plot(lags, ewC2, color='#E9C46A', lw=1.4, ls=':',
                 label=f"C2 Threshold   OOS={res_lev['r2C2_te']:.3f}", alpha=0.7)
        ax2.plot(lags, ewC3, color='#E76F51', lw=1.4, ls=':',
                 label=f"C3 Similarity  OOS={res_lev['r2C3_te']:.3f}", alpha=0.7)
        # Shade best C surplus over B
        ewB_a    = np.array(ewB, float)
        ewBest_a = np.array(ew_best, float)
        ax2.fill_between(lags, ewBest_a, ewB_a,
                         where=np.isfinite(ewBest_a)&np.isfinite(ewB_a)&(ewBest_a>ewB_a),
                         alpha=0.15, color=C['Cm'],
                         label=f'Best C ({best_c_tag}) surplus over B')
        ax2.axvline(0, color='black', lw=1.2)
        ax2.axhline(0, color='black', lw=0.5)
        ax2.set_xlabel(f'Weeks relative to {EVAL_SHOCK.date()} shock', fontsize=10)
        ax2.set_ylabel('% change from pre-shock baseline', fontsize=10)
        note = (f"Best C: {best_c_tag} ({best_lbl}, OOS={best_r2:.3f})"
                if np.isfinite(best_r2) else "")
        ax2.set_title(
            f'Panel 2 — Event Window: {EVAL_SHOCK.date()} (held-out eval)\n'
            f'{note}',
            fontsize=10, fontweight='bold'
        )
        ax2.legend(fontsize=7.5, loc='upper left')

    # ── Panel 3: Lambda sensitivity + dual placebo ───────────────────────────
    ax3a = ax3
    ax3b = ax3.inset_axes([0.52, 0.54, 0.45, 0.42])

    valid  = np.isfinite(lam_losses)
    ax3a.plot(lam_grid[valid], lam_losses[valid], color=C['B'], lw=2)
    ax3a.axvline(lam, color=C['shock'], lw=2, ls='--',
                 label=f'λ_opt={lam:.4f}')
    ax3a.set_xscale('log')
    ax3a.set_xlabel('λ (log scale)', fontsize=10)
    ax3a.set_ylabel('OOS MSE on Aug 2018 tune window', fontsize=9)
    ax3a.set_title(
        'Panel 3 — λ Sensitivity & Placebo Test\n'
        'B−A and three C variants vs B',
        fontsize=10, fontweight='bold'
    )
    ax3a.legend(fontsize=8.5)

    # Placebo inset — B-A plus best C variant
    if len(plac_BA) > 0:
        all_vals = np.concatenate([plac_BA, plac_C1B, plac_C2B, plac_C3B])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) > 0:
            bins = np.linspace(np.nanmin(all_vals)-0.05,
                               np.nanmax(all_vals)+0.05, 14)
            ax3b.hist(plac_BA,  bins=bins, color=C['B'],    alpha=0.55,
                      edgecolor='white', lw=0.8, label='B−A placebos')
            ax3b.hist(plac_C1B, bins=bins, color=C['Cm'],   alpha=0.30,
                      edgecolor='white', lw=0.8, label='C1−B placebos')
            ax3b.hist(plac_C2B, bins=bins, color='#E9C46A', alpha=0.30,
                      edgecolor='white', lw=0.8, label='C2−B placebos')
            ax3b.hist(plac_C3B, bins=bins, color='#E76F51', alpha=0.30,
                      edgecolor='white', lw=0.8, label='C3−B placebos')
            for rv, col, lbl in [
                (real_BA,  C['B'],    f'B−A {real_BA:+.3f}'),
                (real_C1B, C['Cm'],   f'C1 {real_C1B:+.3f}'),
                (real_C2B, '#E9C46A', f'C2 {real_C2B:+.3f}'),
                (real_C3B, '#E76F51', f'C3 {real_C3B:+.3f}'),
            ]:
                if np.isfinite(rv):
                    ax3b.axvline(rv, color=col, lw=1.8, ls='--', label=lbl)
            ax3b.axvline(0, color='black', lw=0.7)
            ax3b.set_xlabel('OOS R² gain', fontsize=7.5)
            ax3b.set_ylabel('Count', fontsize=7.5)
            ax3b.tick_params(labelsize=7)
            ax3b.legend(fontsize=6, loc='upper left')

    data_label = ("SYNTHETIC — PROTOTYPE ONLY"
                  if IS_SYNTHETIC else
                  "Data: Binance · FRED · World Bank · Google Trends")
    fig.suptitle(
        f'USDT/TRY Premium Path-Dependence — Turkey 2018–2024\n'
        f'Tuned on {TUNE_SHOCK.date()} → Evaluated on {EVAL_SHOCK.date()}  |  '
        f'C1=sensitisation · C2=threshold · C3=similarity  |  {data_label}\n'
        f'Anne-Lise Saive',
        fontsize=11, fontweight='bold', y=0.997, linespacing=1.6
    )
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓  Figure → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# 13.  PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results, lam, w_mag, w_abr, w_rel, shocks,
                  plac_BA, plac_C1B, plac_C2B, plac_C3B,
                  real_BA, real_C1B, real_C2B, real_C3B,
                  eval_shock=None, title_suffix=""):
    if eval_shock is None:
        eval_shock = EVAL_SHOCK
    ess = compute_ESS(shocks, w_mag, w_abr, w_rel)
    hl  = np.log(2) / lam / 52

    print(f"\n{'='*66}")
    if title_suffix:
        print(title_suffix)
    print(f"FITTED ESS SCORES  (optimised on TUNE: {TUNE_SHOCK.date()})")
    print(f"{'='*66}")
    for i, (_, s) in enumerate(shocks.iterrows()):
        print(f"\n  {s['short']:6s}  ESS={ess.iloc[i]:.3f}  [{s['type']}]")
        print(f"    {s['description']}")
    print(f"\n  λ = {lam:.5f}  →  half-life {hl:.1f} years")

    # ── Lead with PRIMARY spec (levels) ─────────────────────────────────────
    for spec, label, primary_note in [
        ('levels',      'LEVELS  ← PRIMARY SPEC',  '← report this'),
        ('differences', 'FIRST DIFFERENCES  (robustness only)', ''),
    ]:
        r = results[spec]
        print(f"\n{'='*66}")
        print(f"MODEL PERFORMANCE — {label}")
        print(f"  Tune: {TUNE_SHOCK.date()}  |  Eval: {eval_shock.date()}  (clean OOS)")
        print(f"{'='*66}")

        rows = [
            ('A_trend  AR(1) + time trend  [null]',
             r['r2At_tr'], r['r2At_te']),
            ('A        Memoryless ARX',
             r['r2A_tr'],  r['r2A_te']),
            ('B        Path-dependent  ← PRIMARY CLAIM',
             r['r2B_tr'],  r['r2B_te']),
            ('C1       General sensitisation  [Abrupt×M, Trends×M]',
             r['r2C1_tr'], r['r2C1_te']),
            ('C2       Threshold-sensitisation [Abrupt×M, FX²×M]',
             r['r2C2_tr'], r['r2C2_te']),
            ('C3       Similarity-gated  [CurrentESS×M]',
             r['r2C3_tr'], r['r2C3_te']),
        ]
        print(f"\n  {'Model':<52}  {'R²_IS':>8}  {'R²_OOS':>9}")
        print(f"  {'-'*52}  {'-'*8}  {'-'*9}")
        for nm, is_, oos in rows:
            oos_s = f"{oos:>9.4f}" if np.isfinite(oos) else f"{'n/a':>9}"
            print(f"  {nm:<52}  {is_:>8.4f}  {oos_s}")

        dAt  = r['r2A_te']   - r['r2At_te']
        dB   = r['r2B_te']   - r['r2A_te']
        dBt  = r['r2B_te']   - r['r2At_te']
        dC1  = r['r2C1_te']  - r['r2B_te']
        dC2  = r['r2C2_te']  - r['r2B_te']
        dC3  = r['r2C3_te']  - r['r2B_te']
        best_C = max(r['r2C1_te'], r['r2C2_te'], r['r2C3_te'])
        best_C_name = ['C1','C2','C3'][[r['r2C1_te'], r['r2C2_te'],
                                         r['r2C3_te']].index(best_C)]

        def yn(v): return '✓' if v > 0 else '✗'
        print(f"\n  A  over A_trend : {dAt:+.4f}  {yn(dAt)} macro controls add signal")
        print(f"  B  over A_trend : {dBt:+.4f}  {yn(dBt)} M(t) beats trend null")
        print(f"  B  over A       : {dB:+.4f}  {yn(dB)} path-dependence adds signal")
        print(f"  C1 over B       : {dC1:+.4f}  {yn(dC1)} general sensitisation")
        print(f"  C2 over B       : {dC2:+.4f}  {yn(dC2)} threshold-sensitisation")
        print(f"  C3 over B       : {dC3:+.4f}  {yn(dC3)} similarity-gated reactivation")
        print(f"\n  Best exploratory variant: {best_C_name}  (OOS R²={best_C:.4f})")

    # ── C variant coefficients — levels only ──────────────────────────────────
    r = results['levels']
    for tag, cname, coef_key, names_key, vifs_key, cilo_key, cihi_key, alpha_key, theory in [
        ('C1', 'General sensitisation',
         'cC1','c_names_C1','vifs_C1','ci_lo_C1','ci_hi_C1','alpha_C1',
         'Abruptness×M + Trends×M  |  Does prior fear make ANY signal hit harder?'),
        ('C2', 'Threshold-sensitisation',
         'cC2','c_names_C2','vifs_C2','ci_lo_C2','ci_hi_C2','alpha_C2',
         'Abruptness×M + FX²×M    |  Does prior fear make responses more abrupt and nonlinear?'),
        ('C3', 'Similarity-gated reactivation',
         'cC3','c_names_C3','vifs_C3','ci_lo_C3','ci_hi_C3','alpha_C3',
         'CurrentESS×M            |  Does cue similarity to 2018 trigger reactivation?'),
    ]:
        print(f"\n{'='*66}")
        print(f"MODEL {tag} — EXPLORATORY  (ridge α={r.get(alpha_key, np.nan):.4f})")
        print(f"Theory: {theory}")
        print(f"{'='*66}")
        coef   = r[coef_key]
        ci_lo  = r[cilo_key]
        ci_hi  = r[cihi_key]
        vifs   = r[vifs_key]
        names  = r[names_key]
        print(f"\n  {'Term':<47}  {'Coef':>8}  {'CI_lo':>7}  {'CI_hi':>7}  {'VIF':>6}  {'Sig':>4}")
        print(f"  {'-'*47}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*4}")
        for i, nm in enumerate(names):
            sig   = "YES" if (ci_lo[i] > 0 or ci_hi[i] < 0) else " no"
            vif_v = vifs.get(nm, np.nan)
            vif_s = f"{vif_v:.1f}" if np.isfinite(vif_v) else "  n/a"
            flag  = "  ★" if tag in nm or 'novel' in nm.lower() else ""
            print(f"  {nm:<47}  {coef[i]:>8.4f}  {ci_lo[i]:>7.4f}  "
                  f"{ci_hi[i]:>7.4f}  {vif_s:>6}  {sig:>4}{flag}")

    # Placebo
    print(f"\n{'='*66}")
    print("PLACEBO TEST RESULTS")
    print(f"{'='*66}")
    for claim, rg, plac in [
        ('B−A  path-dependence',              real_BA,  plac_BA),
        ('C1−B general sensitisation',        real_C1B, plac_C1B),
        ('C2−B threshold-sensitisation',      real_C2B, plac_C2B),
        ('C3−B similarity-gated reactivation',real_C3B, plac_C3B),
    ]:
        if len(plac) > 0 and np.isfinite(rg):
            pct = np.sum(plac < rg) / len(plac) * 100
            print(f"\n  {claim}")
            print(f"    Real {eval_shock.date()} gain : {rg:+.4f}")
            print(f"    Placebo mean              : {np.nanmean(plac):+.4f}")
            print(f"    Placebo std               : {np.nanstd(plac):.4f}")
            print(f"    Percentile                : {pct:.0f}th")
            if pct >= 85:
                print(f"    → Strong signal: real gain exceeds {pct:.0f}% of placebos")
            elif pct >= 70:
                print(f"    → Moderate signal")
            else:
                print(f"    → Weak / not distinguishable from placebo")

    gap_months = (eval_shock - TUNE_SHOCK).days // 30
    status     = "⚠  SYNTHETIC" if IS_SYNTHETIC else "✓  REAL DATA"
    syn_warn   = ("WARNING: All results are from synthetic data.\n"
                  "Run locally with real data to produce publishable output."
                  if IS_SYNTHETIC else
                  "Results are from real Binance / FRED / Trends data.")
    print(f"""
{'='*66}
HONEST SUMMARY  [{status}]
{'='*66}
Parameter selection : {TUNE_SHOCK.date()} (TUNE event)
Performance eval    : {eval_shock.date()} (EVAL event — never touched in tuning)
Clean OOS evaluation. Events are {gap_months} months apart.

{syn_warn}

PRIMARY CLAIM  : Model B — salience-weighted path-dependence.
EXPLORATORY C1 : General sensitisation     — Abruptness×M + Trends×M
EXPLORATORY C2 : Threshold-sensitisation   — Abruptness×M + FX²×M
EXPLORATORY C3 : Similarity-gated          — CurrentESS×M
NULL COMPARATOR: A_trend — if B does not beat trend, M(t) adds no signal.

Compare C1/C2/C3 OOS R² to identify which reactivation mechanism
best fits the data. On real data the winner guides the next research step.

Public data : test whether adoption is path-dependent.
Tether data : test where the memory lives, how thresholds differ
              across cohorts, and how adoption cascades through networks.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 14.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from scipy.optimize import differential_evolution

    print("="*66)
    print("STABLECOIN ADOPTION IS NOT MEMORYLESS — Turkey 2018–2024")
    print("Three C variants — C1 sensitisation · C2 threshold · C3 similarity")
    print(f"TUNE   event: {TUNE_SHOCK.date()}  (Aug 2018 lira crisis — primary encoding shock)")
    print(f"EVAL   event: {EVAL_SHOCK.date()}  (Dec 2021 rate shock — main held-out test)")
    print(f"ROBUST event: {ROBUST_SHOCK.date()}  (Jun 2023 devaluation — robustness test)")
    print("="*66)

    df = load_data()
    df = build_features(df)

    lam, w_mag, w_abr, w_rel = optimise(df, SHOCKS, tune_shock=TUNE_SHOCK)

    print("\n── Lambda sensitivity ─────────────────────────────────────")
    lam_grid, lam_losses = lambda_sensitivity(df, SHOCKS, w_mag, w_abr, w_rel,
                                              tune_shock=TUNE_SHOCK)

    print(f"\n── Running MAIN evaluation on {EVAL_SHOCK.date()} ─────────────")
    results = run_models(df, SHOCKS, lam, w_mag, w_abr, w_rel,
                         eval_shock=EVAL_SHOCK)

    r_lev = results['levels']
    (plac_BA, plac_C1B, plac_C2B, plac_C3B,
     real_BA, real_C1B, real_C2B, real_C3B) = placebo_test(
        df, SHOCKS, lam, w_mag, w_abr, w_rel,
        alpha_C1=r_lev.get('alpha_C1', 0.01),
        alpha_C2=r_lev.get('alpha_C2', 0.01),
        alpha_C3=r_lev.get('alpha_C3', 0.01),
        eval_shock=EVAL_SHOCK,
    )

    make_figures(df, r_lev, SHOCKS,
                 lam, w_mag, w_abr, w_rel,
                 lam_grid, lam_losses,
                 plac_BA, plac_C1B, plac_C2B, plac_C3B,
                 real_BA, real_C1B, real_C2B, real_C3B,
                 outpath='turkey_results.png')

    print_results(results, lam, w_mag, w_abr, w_rel, SHOCKS,
                  plac_BA, plac_C1B, plac_C2B, plac_C3B,
                  real_BA, real_C1B, real_C2B, real_C3B,
                  eval_shock=EVAL_SHOCK)

    print(f"\n── Running ROBUSTNESS evaluation on {ROBUST_SHOCK.date()} ─────")
    results_robust = run_models(df, SHOCKS, lam, w_mag, w_abr, w_rel,
                                eval_shock=ROBUST_SHOCK)
    print_results(results_robust, lam, w_mag, w_abr, w_rel, SHOCKS,
                  np.array([]), np.array([]), np.array([]), np.array([]),
                  np.nan, np.nan, np.nan, np.nan,
                  eval_shock=ROBUST_SHOCK,
                  title_suffix="SECOND HELD-OUT ROBUSTNESS")
