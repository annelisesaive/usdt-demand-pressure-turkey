
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

-- Weekly TRC-20 USDT gross and net flow involving Turkey-domiciled CEX wallets.
-- Coverage starts in July 2019; the on-chain supplement therefore covers
-- the 2021 and 2023 shock windows, but not the August 2018 tuning shock.

WITH turkish_cex AS (
    SELECT
        address,
        cex_name
    FROM cex.addresses
    WHERE LOWER(blockchain) = 'tron'
      AND (
          LOWER(cex_name) LIKE '%paribu%'
          OR LOWER(cex_name) LIKE '%btcturk%'
      )
),

transfers AS (
    SELECT
        DATE_TRUNC('week', block_time) AS week,
        "from" AS from_address,
        "to" AS to_address,
        amount
    FROM stablecoins.transfers
    WHERE LOWER(blockchain) = 'tron'
      AND token_symbol = 'USDT'
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
        SUM(
            CASE
                WHEN to_address IN (SELECT address FROM turkish_cex)
                THEN amount ELSE 0
            END
        ) AS inflow_usdt,
        SUM(
            CASE
                WHEN from_address IN (SELECT address FROM turkish_cex)
                THEN amount ELSE 0
            END
        ) AS outflow_usdt
    FROM transfers
    GROUP BY 1
),

address_audit AS (
    SELECT COUNT(*) AS matched_cex_addresses
    FROM turkish_cex
)

SELECT
    weekly.week,
    COALESCE(weekly.inflow_usdt, 0) AS inflow_usdt,
    COALESCE(weekly.outflow_usdt, 0) AS outflow_usdt,
    COALESCE(weekly.inflow_usdt, 0) + COALESCE(weekly.outflow_usdt, 0) AS gross_flow_usdt,
    COALESCE(weekly.inflow_usdt, 0) - COALESCE(weekly.outflow_usdt, 0) AS net_flow_usdt,
    CASE
        WHEN COALESCE(weekly.inflow_usdt, 0) + COALESCE(weekly.outflow_usdt, 0) = 0
        THEN NULL
        ELSE ABS(COALESCE(weekly.inflow_usdt, 0) - COALESCE(weekly.outflow_usdt, 0))
             / (COALESCE(weekly.inflow_usdt, 0) + COALESCE(weekly.outflow_usdt, 0))
    END AS abs_net_share,
    address_audit.matched_cex_addresses
FROM weekly
CROSS JOIN address_audit
ORDER BY 1;
SELECT
    weekly.week,
    COALESCE(weekly.inflow_usdt, 0) AS inflow_usdt,
    COALESCE(weekly.outflow_usdt, 0) AS outflow_usdt,
    COALESCE(weekly.inflow_usdt, 0) - COALESCE(weekly.outflow_usdt, 0) AS net_flow_usdt,
    address_audit.matched_cex_addresses
FROM weekly
CROSS JOIN address_audit
ORDER BY 1;
