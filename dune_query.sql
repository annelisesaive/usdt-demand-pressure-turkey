-- Weekly USDT net flow to Turkey-domiciled CEX wallets.
-- Data available from July 2019 onwards.

WITH turkish_cex AS (
    SELECT address, cex_name
    FROM cex.addresses
    WHERE LOWER(cex_name) LIKE '%paribu%'
       OR LOWER(cex_name) LIKE '%btcturk%'
),

transfers AS (
    SELECT
        DATE_TRUNC('week', block_time) AS week,
        "from" AS from_address,
        "to" AS to_address,
        amount
    FROM stablecoins.transfers
    WHERE token_symbol = 'USDT'
      AND block_time >= DATE '2019-07-01'
      AND block_time < DATE '2025-01-01'
      AND (
          "from" IN (SELECT address FROM turkish_cex)
       OR "to" IN (SELECT address FROM turkish_cex)
      )
),

weekly AS (
    SELECT
        week,
        SUM(CASE WHEN to_address IN (SELECT address FROM turkish_cex) THEN amount ELSE 0 END) AS inflow_usdt,
        SUM(CASE WHEN from_address IN (SELECT address FROM turkish_cex) THEN amount ELSE 0 END) AS outflow_usdt
    FROM transfers
    GROUP BY 1
),

address_audit AS (
    SELECT COUNT(*) AS matched_cex_addresses FROM turkish_cex
)

SELECT
    weekly.week,
    COALESCE(weekly.inflow_usdt, 0) AS inflow_usdt,
    COALESCE(weekly.outflow_usdt, 0) AS outflow_usdt,
    COALESCE(weekly.inflow_usdt, 0) - COALESCE(weekly.outflow_usdt, 0) AS net_flow_usdt,
    address_audit.matched_cex_addresses
FROM weekly
CROSS JOIN address_audit
ORDER BY 1;
