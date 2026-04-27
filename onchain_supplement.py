"""
==============================================================================
On-chain Supplement: USDT Net Flow to Turkish Exchanges
==============================================================================
Author : Anne-Lise Saive (April 2026)

Plots weekly TRC-20 USDT net flow to Turkey-domiciled CEX-attributed wallets
across the event windows used in the on-chain supplement. Source data is
exported from dune_query.sql to data/onchain_flows.csv.

The supplement is a venue-level proxy. It does not identify wallet cohorts or
individual Turkish users, and the sign of CEX net flow is ambiguous. For this
reason, the path-dependence check compares absolute net-flow response
magnitudes rather than interpreting inflow as mechanically bullish or bearish.
==============================================================================
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#F8F9FA',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linestyle': '--',
})

C = dict(
    flow='#264653',
    rolling='#C1121F',
    shock='#F4A261',
    zero='#999999',
)

SHOCKS = [
    ('2021-12-20', 'Dec 2021', 'Rate-cut crisis'),
    ('2023-06-15', 'Jun 2023', 'Post-election devaluation'),
]

PRE_WEEKS = 8
POST_WEEKS = 16
TEST_HORIZON_WEEKS = 8
DATA_PATH = 'data/onchain_flows.csv'


def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['week'])
    required = {'week', 'inflow_usdt', 'outflow_usdt', 'net_flow_usdt'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    # Remove timezone info to make tz-naive for comparison
    df['week'] = df['week'].dt.tz_localize(None)
    df = df.set_index('week').sort_index()
    df['net_flow_usdt_m'] = df['net_flow_usdt'] / 1e6

    if 'gross_flow_usdt' not in df.columns:
        df['gross_flow_usdt'] = df['inflow_usdt'] + df['outflow_usdt']

    if 'abs_net_share' not in df.columns:
        denom = df['gross_flow_usdt'].replace(0, np.nan)
        df['abs_net_share'] = df['net_flow_usdt'].abs() / denom

    df['inflow_usdt_m'] = df['inflow_usdt'] / 1e6
    df['outflow_usdt_m'] = df['outflow_usdt'] / 1e6
    df['gross_flow_usdt_m'] = df['gross_flow_usdt'] / 1e6
    return df


def post_shock_abs_flow(df: pd.DataFrame, shock_date: str,
                        horizon_weeks: int = TEST_HORIZON_WEEKS) -> pd.Series:
    shock = pd.Timestamp(shock_date)
    expected = horizon_weeks + 1
    window = df.loc[df.index >= shock, 'net_flow_usdt'].abs().head(expected) / 1e6
    if len(window) < expected:
        raise ValueError(
            f"Need {expected} weekly observations on or after {shock.date()}, "
            f"found {len(window)}"
        )
    return window.iloc[:expected].reset_index(drop=True)


def path_dependence_test(df: pd.DataFrame,
                         horizon_weeks: int = TEST_HORIZON_WEEKS,
                         n_boot: int = 10000,
                         seed: int = 42) -> dict:
    dec = post_shock_abs_flow(df, '2021-12-20', horizon_weeks)
    jun = post_shock_abs_flow(df, '2023-06-15', horizon_weeks)
    paired_diff = jun.to_numpy() - dec.to_numpy()

    observed = float(paired_diff.sum())
    dec_integral = float(dec.sum())
    jun_integral = float(jun.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(paired_diff), size=(n_boot, len(paired_diff)))
    boot = paired_diff[idx].sum(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    signs = np.array(np.meshgrid(*[[-1, 1]] * len(paired_diff))).T.reshape(-1, len(paired_diff))
    perm = (signs * paired_diff).sum(axis=1)
    p_one_sided = float(np.mean(perm >= observed))
    p_two_sided = float(np.mean(np.abs(perm) >= abs(observed)))

    return {
        'horizon_weeks': horizon_weeks,
        'dec_integral_m': dec_integral,
        'jun_integral_m': jun_integral,
        'diff_m': observed,
        'ci_lo_m': float(ci_lo),
        'ci_hi_m': float(ci_hi),
        'p_one_sided': p_one_sided,
        'p_two_sided': p_two_sided,
        'n_weeks': len(paired_diff),
    }


def print_test_result(result: dict) -> None:
    print("\n── On-chain path-dependence check ───────────────────────────")
    print("  Scope     : venue-level CEX wallet flow proxy")
    print(f"  Metric    : sum |net_flow_usdt| over first {result['n_weeks']} post-shock weeks")
    print(f"  Dec 2021  : {result['dec_integral_m']:.2f}M USDT")
    print(f"  Jun 2023  : {result['jun_integral_m']:.2f}M USDT")
    print(f"  Jun-Dec   : {result['diff_m']:+.2f}M USDT")
    print(f"  Bootstrap : 95% CI [{result['ci_lo_m']:+.2f}, {result['ci_hi_m']:+.2f}]M")
    print(f"  Perm test : one-sided p={result['p_one_sided']:.3f}, "
          f"two-sided p={result['p_two_sided']:.3f}")
    print("  Note      : sign is ambiguous, so the test uses response magnitude.")


def plot_supplement(df: pd.DataFrame,
                    test_result: dict | None = None,
                    outpath: str = 'figures/onchain_supplement.png') -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, (date_str, label, desc) in zip(axes, SHOCKS):
        shock = pd.Timestamp(date_str)
        start = shock - pd.Timedelta(weeks=PRE_WEEKS)
        end = shock + pd.Timedelta(weeks=POST_WEEKS)
        win = df.loc[start:end].copy()

        if win.empty:
            ax.text(0.5, 0.5, f'No data in {label} window',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#666')
            ax.set_title(f'{label} - {desc}', fontsize=11, fontweight='bold')
            continue

        weeks_from_shock = ((win.index - shock).days / 7).values
        flow = win['net_flow_usdt_m'].values
        rolling = pd.Series(flow).rolling(4, min_periods=1).mean().values

        ax.bar(weeks_from_shock, flow, width=0.7, color=C['flow'],
             alpha=0.55, edgecolor='none', label='Weekly net flow')
        ax.plot(weeks_from_shock, rolling, color=C['rolling'],
                lw=2, label='4-week rolling mean')
        ax.axvline(0, color=C['shock'], lw=1.6, alpha=0.85,
                   ls='--', label='Shock date')
        ax.axhline(0, color=C['zero'], lw=0.7, alpha=0.5)

        ax.set_title(f'{label} - {desc}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Weeks from shock', fontsize=9)
        ax.set_xlim(-PRE_WEEKS - 0.5, POST_WEEKS + 0.5)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel('Net USDT inflow, millions', fontsize=9)
    axes[0].legend(loc='upper left', fontsize=8, framealpha=0.92,
                   frameon=True)

    title = ('On-chain supplement: TRC-20 USDT flow involving '
             'Turkey-domiciled CEX-attributed wallets')
    fig.suptitle(title, fontsize=11.5, fontweight='bold', y=1.02)
    fig.text(0.5, -0.04, 'Source: Dune | stablecoins_tron.transfers | cex.addresses',
             ha='center', fontsize=8, style='italic', color='#555')

    fig.text(
        0.5, -0.08,
        (f'Path-dependence check: |net flow| first {test_result["n_weeks"]} post-shock weeks, '
         f'Jun-Dec={test_result["diff_m"]:+.1f}M USDT, '
         f'95% CI [{test_result["ci_lo_m"]:+.1f}, {test_result["ci_hi_m"]:+.1f}], '
         f'one-sided p={test_result["p_one_sided"]:.3f}'
         if test_result else
         'Path-dependence check unavailable for this export.'),
        ha='center',
        fontsize=8,
        color='#555'
    )
    fig.text(
        0.5, -0.12,
        'Coverage starts July 2019; August 2018 shock is not covered by the on-chain supplement. '
        'Venue-level flow proxy; sign is ambiguous and not user-level demand.',
        ha='center',
        fontsize=8,
        color='#555'
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DATA_PATH,
                        help='Path to CSV exported from dune_query.sql')
    parser.add_argument('--out', default='figures/onchain_supplement.png',
                        help='Output figure path')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            "Run dune_query.sql on Dune, export the result as CSV, then rerun."
        )

    df = load_flows(args.csv)
    test_result = path_dependence_test(df)
    print_test_result(test_result)
    plot_supplement(df, test_result, args.out)


if __name__ == '__main__':
    main()
