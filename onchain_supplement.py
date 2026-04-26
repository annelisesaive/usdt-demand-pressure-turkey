"""
==============================================================================
On-chain Supplement: USDT Net Flow to Turkish Exchanges
==============================================================================
Author : Anne-Lise Saive (April 2026)

Plots weekly TRC-20 USDT net flow to Turkey-domiciled CEX wallets across the
three event windows used in the main analysis. Source data is exported from
dune_query.sql to data/onchain_flows.csv.
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
    ('2018-08-13', 'Aug 2018', 'US tariff escalation'),
    ('2021-12-20', 'Dec 2021', 'Rate-cut crisis'),
    ('2023-06-15', 'Jun 2023', 'Post-election devaluation'),
]

PRE_WEEKS = 8
POST_WEEKS = 16
DATA_PATH = 'data/onchain_flows.csv'


def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['week'])
    required = {'week', 'net_flow_usdt'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    # Remove timezone info to make tz-naive for comparison
    df['week'] = df['week'].dt.tz_localize(None)
    df = df.set_index('week').sort_index()
    df['net_flow_usdt_m'] = df['net_flow_usdt'] / 1e6
    return df


def plot_supplement(df: pd.DataFrame,
                    outpath: str = 'figures/onchain_supplement.png') -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

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
               alpha=0.55, edgecolor='none', label='Weekly net inflow')
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

    title = ('On-chain supplement: TRC-20 USDT net inflow to '
             'Turkey-domiciled CEX wallets')
    fig.suptitle(title, fontsize=11.5, fontweight='bold', y=1.02)
    fig.text(0.5, -0.04, 'Source: Dune | stablecoins_tron.transfers | cex.addresses',
             ha='center', fontsize=8, style='italic', color='#555')

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
    plot_supplement(df, args.out)


if __name__ == '__main__':
    main()
