# Project status

Last updated: 2026-11-13

| Item | Status |
|---|---|
| Project structure, source code, configs, scripts | TESTED (synthetic data and repository checks only) |
| Assignment notebooks 01 to 10 | BUILT (scaffolds; official briefs not provided) |
| Capstone notebook | BUILT (gated; needs token and schema mapping) |
| Real FlyRank data validation | NOT STARTED (BLOCKED: needs HF token and schema inspection) |
| Experiment results | NOT STARTED |
| Research paper | DRAFT only (template, no results) |
| Paper deployment | NOT STARTED (`submission/paper_url.txt` holds a TODO line) |
| FlyRank submission | NOT STARTED |

Status vocabulary: NOT STARTED, BUILT, TESTED, DATA-VALIDATED, RESULTS-GENERATED, PAPER-DRAFTED, PAPER-DEPLOYED, SUBMITTED, UNDER REVIEW, ACCEPTED, CERTIFICATE ISSUED.

## Blocked on you
1. Hugging Face read token (local `.env`).
2. Warehouse dataset repo id / path (from the FlyRank resource card) and real column names, to fill `configs/data_config.yaml`.
3. Official briefs for assignments 01 to 10 (see `docs/ASSIGNMENT_TRACKER.md`).
