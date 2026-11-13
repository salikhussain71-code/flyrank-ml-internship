# Architecture

```
warehouse (Hugging Face, gated)
   -> DuckDB over hf:// (aggregate to page-day: page_id, date, clicks, impressions)
   -> data.quality: report + clean_daily (exclusions recorded)
   -> evaluation.splits: held-out pages (hash-based) + later period
   -> features.builder: two 28-day windows ending at decision date (point-in-time)
   -> labels.builder: decline in next 28 days (strictly after decision date)
   -> models.baseline (transparent) and models.train (scikit-learn)
   -> evaluation.metrics / leakage: same-split comparison, guards raise errors
   -> recommendations.engine: ranked, reason codes, pseudonymous IDs
   -> visualization.plots -> reports/ -> scripts/build_paper.py -> paper/index.html
```
`pipeline.run_pipeline` wires these together. `data_source="flyrank_real"` is the only value that counts as a FlyRank result.
