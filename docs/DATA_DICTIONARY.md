# Data dictionary

**Real warehouse schema: NOT YET INSPECTED.** Do not fill from memory. After running the schema cell in the capstone notebook, record tables, columns, types, grain and date coverage here.

## Pipeline input contract (after mapping in `configs/data_config.yaml`)
| Column | Meaning |
|---|---|
| page_id | content item identifier (never published; pseudonymised in outputs) |
| date | calendar day |
| clicks | search clicks that day |
| impressions | search impressions that day |

## Derived features (all point-in-time, window = 28 days)
clicks_recent, clicks_prior, impressions_recent, impressions_prior, ctr_recent, ctr_prior, clicks_delta_pct, impressions_delta_pct, active_days_recent.

## Label
`label_decline = 1` if clicks in the next 28 days are at least 30% below the last 28 days (pages with at least 10 recent clicks).
