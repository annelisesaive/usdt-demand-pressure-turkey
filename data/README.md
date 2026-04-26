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

## Limitations

These flows are not equivalent to Turkish user-level demand. They measure transfer activity involving exchange-attributed wallets. They may include customer deposits and withdrawals, exchange treasury movements, hot-wallet rebalancing, OTC activity, and operational wallet management.

The analysis should therefore be interpreted as a complementary exchange-flow proxy, not as causal evidence of adoption or behavioral path-dependence.
