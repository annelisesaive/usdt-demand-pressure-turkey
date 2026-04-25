# USDT Demand Pressure in Turkey: A Path-Dependence Test

**Anne-Lise Saive, April 2026**

A self-initiated empirical project testing whether USDT/TRY demand pressure
in Turkey responds to monetary shocks in a path-dependent way where earlier
high-salience shocks amplify the response to later ones.

TDLR; with public data, no. The premium series is dominated
by short-horizon market mechanics rather than slow behavioral
accumulation. The honest null is the primary finding. A weaker
exploratory signal appears in first differences but does not survive as a
clean test of the hypothesis. Wallet-level data is what would actually
allow to test this hypothesis best.

![Main result figure](figures/turkey_v9_results.png)

---

## What this is testing

The hypothesis comes from computational models of salience-weighted
episodic memory: high-intensity past events lower the threshold for
future behavioral responses to related cues. In this context, if a country
experiences a severe currency shock, accumulated "vulnerability" should
amplify population-level demand for stablecoin protection during later
shocks of comparable or smaller magnitude.

The cleanest place to test this empirically is Turkey 2018–2024, which
has three well-separated shock events:

- **Aug 2018**: US tariffs trigger a ~40% lira drop in days
- **Dec 2021**: Erdoğan rate-cut episode, ~44% lira drop over months
- **Jun 2023**: post-election devaluation, ~25–30% drop over six weeks

These give one tuning event, one held-out evaluation event, and one
robustness event with no refitting.

## What I'm actually measuring (and what I'm not)

The dependent variable is the **USDT/TRY premium over the official
USD/TRY rate** reflecting the spread Turkish buyers pay above market for 
fast stablecoin access. The premium captures local demand pressure and
urgency.

It is *not* adoption at the user-level concept (new wallets,
retained users, transaction counts) and isn't observable in public
market data. The premium is a downstream proxy for adoption-related
demand, not a direct measurement of it.

## Data

All public, all reproducible without API keys.

- **Binance** weekly USDT/TRY klines
- **FRED** official weekly USD/TRY (series `DEXTHUS`)
- **World Bank** Turkey annual CPI, used only to sanity-check the FX-based inflation proxy
- **Google Trends** Turkey-localized search interest in `USDT`, `Tether`, and `dolar kripto`

A 12-week rolling FX change replaces forward-filled annual CPI as the
inflation proxy because it moves at weekly frequency, which the modeling
needs.

## Methodology

The shape of the test:

```
Tune on Aug 2018  →  Evaluate OOS on Dec 2021  →  Robustness on Jun 2023
```

The tuning and evaluation events never touch. Memory-kernel parameters
(decay rate λ, salience weights) are fit on Aug 2018 and frozen before
the Dec 2021 evaluation. Jun 2023 is a second held-out window with no
refitting at all.

### The Emotional Salience Score

Each shock event is scored on three axes:

- **Magnitude** size of the FX move
- **Abruptness** speed of the move
- **Relevance** proxied by household exposure (search interest spike + FX volatility regime)

The three weights are jointly optimized with the memory decay λ on the
Aug 2018 window using differential evolution. The resulting kernel
`M(t) = Σ ESS_i · exp(-λ(t - t_i))` summarizes accumulated prior
vulnerability at time `t`, excluding any contribution from the event
currently being evaluated.

### Three nested models

**Model A — memoryless ARX baseline**
```
premium ~ AR(1) + FX_shock + FX_vol + CPI_proxy + Trends
```

**Model B — salience-weighted path-dependence (the primary claim)**
```
A + M(t) + FX × M
```

**Model C — reactivation asymmetry (explicitly exploratory)**
```
B + Abruptness × M + Trends × M
```
Model C uses ridge regression with CV-selected alpha to handle the
collinearity introduced by the interaction terms. A and B remain OLS.

### Validation design

- **Tune/evaluate separation.** λ and weights frozen after Aug 2018.
- **Event-centered out-of-sample windows.** Train on the 2 weeks pre-shock, test on 24 weeks post-shock.
- **AR(1) terms.** For the differences specification, the AR term is a lag of `d_premium`, not the level — a common leakage trap.
- **Trend-null comparator (Model A+trend).** Catches any spurious gain Model B might claim from a smooth time trend.
- **Placebo tests on 50 random non-shock dates.** Reports both B−A and C−B OOS R² gains, giving percentile ranks for each novel claim.
- **Bootstrap with replacement.** 500 resamples; duplicates preserved.
- **Hard abort on synthetic data.** If Binance or FRED falls back to anything synthetic, the script terminates rather than silently mixing.

## Results

### Levels and the honest null

Predicting log(premium) levels OOS produces deeply negative R² across all
three models, in both the Dec 2021 and Jun 2023 windows. None of the
nested specifications meaningfully outperforms the AR(1) baseline.

**Reading:** the premium series is non-stationary at the resolution of
weekly public data. Levels-based forecasting fails for reasons that have
nothing to do with whether path-dependence is real or not. It fails
because the target itself is too noisy and too event-driven for the
specification to be identified at this sample size and frequency.

### First differences show exploratory positive signal

Modeling week-to-week change in the premium rather than its level shifts
the picture. Model C achieves positive OOS R² (+0.09) on the Jun 2023
window, with the `Trends × M` interaction's bootstrap confidence interval
excluding zero. The placebo distribution puts this gain in a high
percentile of random-date controls.

**Reading:** there is a weak signal that search-interest spikes may
predict premium changes more strongly when accumulated prior
vulnerability is already elevated. But this is one event, in one country,
on a robustness window, in a non-primary specification. It warrants
follow-up; it does not confirm the hypothesis.

### What the fitted decay rate suggests

The optimized memory kernel converges on **λ ≈ 0.68 per week**, a
half-life of roughly two weeks. This is much faster than what a
slow behavioral-accumulation interpretation of the hypothesis would
predict.

**Reading:** if a path-dependence effect exists in the premium, it
behaves like short-horizon market memory rather than population-level
vulnerability accumulation. The hypothesis is about user-level
adoption behavior, but the premium responds at a timescale dominated
by market mechanics. This is consistent with the premium being a
proxy *downstream* of adoption rather than a direct measurement of it.

## Limitations

- **Four shock events in one country.** Identification of a salience-weighted memory effect requires more events and more cross-sectional variation than this design provides. With public data, that's the ceiling.
- **The premium is not adoption.** It measures urgency and local demand pressure, both of which are influenced by adoption but also by liquidity, arbitrage frictions, and regulatory frictions that the model doesn't separate.
- **Google Trends is a coarse attention proxy.** Search interest captures public attention but not the specific cohort that's actually moving USDT.
- **No user-level cohort structure.** The hypothesis is fundamentally about how prior exposure changes individual decision-making thresholds. That cannot be tested in market-aggregated data.
- **Single specification family.** A more flexible non-linear model (e.g., a regime-switching specification on volatility) might recover signal the linear models miss.

## What would actually test this

The hypothesis is about adoption being path-dependent at the level of
individual user behavior. Public market data can only test a downstream
proxy. The clean version of this test would need:

- **Wallet-level activity data** new wallet creation, retention, transaction frequency
- **Geographic resolution** separating Turkish, Argentine, Nigerian, and Lebanese cohorts
- **More shock events** 15+ across multiple countries gives the design statistical power
- **Cohort segmentation by prior exposure** testing whether users who lived through 2018 respond differently to 2021 than users who joined after

This is precisely the data Tether has and public researchers don't.
That's the asymmetry that makes this kind of question genuinely
testable internally.

## Reproducing

```bash
git clone https://github.com/annelisesaive/usdt-demand-pressure-turkey
cd usdt-demand-pressure-turkey
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

The script will fetch from Binance, FRED, World Bank, and Google Trends
on first run. No API keys are needed for any of these sources. If any
critical source returns synthetic-fallback data, the script will abort
with an explicit error rather than silently mix sources.

Outputs land in `figures/`:

- `turkey_v9_results.png` main three-panel figure
- `lambda_sensitivity.png` λ sensitivity sweep
- `placebo_distribution.png` null distribution of B−A and C−B gains

## Files

- `analysis.py` full pipeline (data fetch, feature engineering, three-model comparison, placebo, bootstrap)
- `requirements.txt` pinned versions
- `figures/` output plots
- `LICENSE` MIT

## Citation

If this is useful for your own work:

```
Saive, A.-L. (2026). USDT Demand Pressure in Turkey: A Path-Dependence
Test. https://github.com/annelisesaive/usdt-demand-pressure-turkey
```
