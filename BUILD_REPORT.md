# Build report

Built: 2026-11-13. Environment: Python 3.12.3, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, no network.

## What was built
Full project structure, 11 notebooks (10 assignments + capstone), library (`src/flyrank_ml`), 3 configs, 5 scripts, static paper (DRAFT), 8 docs, tests, CI workflow.

## Tests actually run
`python scripts/run_tests.py` -> **Ran 49 tests, OK** (unit: config, data quality, features, labels, leakage, splits, metrics, baseline, model training, recommendations; integration: end-to-end pipeline on SYNTHETIC data with output-file checks, determinism; repo checks: required files, notebook JSON, Python syntax via `ast`, secret scan, raw-data scan, paper state, tracker, markdown links, notebook outputs cleared).
`python scripts/run_pipeline.py --synthetic-demo` ran (synthetic only; its numbers are meaningless and must never be cited).
`python scripts/run_pipeline.py` without data correctly returned BLOCKED.

## NOT verified (be aware)
- **DuckDB / Hugging Face code path** (`connect_duckdb`, `inspect_schema`, `load_daily_from_warehouse`): `duckdb` and `huggingface_hub` could not be installed here (no network), so this path has never been executed. Expect to debug it on first real run (for example the exact `CREATE SECRET` syntax or `hf://` glob form).
- Notebooks were validated as JSON and Python syntax only. They were not executed in Jupyter (no jupyter available).
- No real FlyRank data was touched. No assignment briefs 01 to 10 were provided, so those notebooks are scaffolds.
- Real column names, dataset repo id and the final lane are unknown.
- The GitHub Pages URL pattern in the guide follows GitHub's standard format but is unverified until you deploy.

## Status
See `PROJECT_STATUS.md`. Nothing here is DATA-VALIDATED, RESULTS-GENERATED, PAPER-DEPLOYED or SUBMITTED.
