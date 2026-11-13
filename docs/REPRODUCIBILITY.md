# Reproducibility

1. Same Python packages (`requirements.txt`).
2. Same warehouse release (record it in `docs/EXPERIMENTS.md`).
3. Same `configs/*.yaml` and seed (42).
4. `python scripts/run_pipeline.py` regenerates `reports/`.
5. `python scripts/build_paper.py` regenerates `paper/index.html`.
6. `python scripts/run_tests.py` verifies the code (tests use synthetic data only).

The pipeline is deterministic given identical data (tested).
