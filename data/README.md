# On-chain data export

The on-chain supplement uses a Dune query to aggregate weekly TRC-20 USDT flows involving wallets attributed to Turkey-domiciled centralized exchanges, currently Paribu and BtcTurk.

## Coverage

The available on-chain export starts in July 2019. It covers the December 2021 and June 2023 event windows, but not the August 2018 tuning shock.

## Reproducing the CSV

Run `dune_query.sql` on Dune, then fetch the latest query result:

```bash
mkdir -p data
curl -fSL \
  -H "X-Dune-Api-Key: $DUNE_API_KEY" \
  "https://api.dune.com/api/v1/query/7377063/results/csv?limit=1000" \
  -o data/onchain_flows.csv
```

Then run:

```bash
python onchain_supplement.py
```

The script plots the December 2021 and June 2023 windows and prints a
path-dependence check comparing the absolute net-flow integral over the first
nine post-shock weekly buckets. The reported statistic is the Jun-minus-Dec
difference, with a bootstrap confidence interval and paired sign-flip
permutation p-value.

## Limitations

These flows are not equivalent to Turkish user-level demand. They measure
transfer activity involving exchange-attributed wallets, where one wallet can
represent a single large account, many users, exchange treasury activity, or
operational wallet management.

The sign of net flow is also ambiguous. USDT inflow to a Turkish CEX can mean
users are depositing USDT to sell for TRY, or that the venue is topping up
wallets to support TRY-to-USDT demand. The supplement therefore treats
absolute net-flow amplitude as the response metric.

The analysis should be interpreted as corroborating venue-level evidence, not
as causal evidence of adoption or a direct wallet-cohort test of behavioral
path-dependence.
