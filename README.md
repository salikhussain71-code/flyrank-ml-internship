# FlyRank ML Internship: Google Search Ranking & Discoverability Capstone

Owner: Salik Hussain (GitHub: `salikhussain71-code`). Planned repo: https://github.com/salikhussain71-code/flyrank-ml-internship

## 1. Project overview
A reproducible Python project containing every assignment notebook, the capstone workflow, tests, and a static research-paper website.

**Current status: BUILT and TESTED on synthetic data only. No FlyRank data has been used yet. No results exist. The paper is a DRAFT.** See `PROJECT_STATUS.md` and `docs/ASSIGNMENT_TRACKER.md`.

## 2. FlyRank internship context
Machine Learning track requirements (as supplied): at least 5 ML assignments plus the mandatory ML capstone. Other assignments and the General AI Fluency track are optional. FlyRank reviewers decide acceptance and certificate issuance. Nothing here guarantees a certificate.

## 3. Research question and lane
Provisional lane (candidate only): **Refresh / Content Opportunity Scoring**. Draft question: *which pages are at risk of a material click decline over the next 28 days, so reviewers can prioritise content review?* The lane is final only after the real warehouse schema is inspected.

## 4. Project structure
- `work/assignments/` one notebook per assignment (01 to 10)
- `work/capstone/` capstone notebook
- `src/flyrank_ml/` library code (data, features, labels, models, evaluation, recommendations, visualization)
- `tests/` unit and integration tests
- `configs/` data, experiment and model configuration
- `scripts/` setup, validation, pipeline, tests, paper build
- `paper/` static research paper website
- `reports/` generated figures, tables, metrics, data-quality (empty until real results)
- `docs/` architecture, methodology, security, tracker, guides
- `submission/paper_url.txt` the one-line paper URL (mandatory)

## 5. Environment setup (Windows, beginner)
1. Install Python 3.11 or newer from https://www.python.org/downloads/ . Tick **Add python.exe to PATH**.
2. Extract `flyrank-ml-internship.zip` (right-click, Extract All).
3. Open the extracted folder, click the address bar, type `cmd`, press Enter.
4. Create an environment and install packages:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Hugging Face access
1. Request access to the gated FlyRank warehouse from the dataset page linked in your FlyRank capstone resources.
2. On huggingface.co: Settings > Access Tokens > create a **Read** token.
3. Run `python scripts/setup_environment.py`. Open the new `.env` file in Notepad and paste the token after `HF_TOKEN=`. Save.
4. Never paste the token into a notebook or share `.env`. It is git-ignored.

## 7. Check the setup
```
python scripts/validate_environment.py
```

## 8. Run notebooks
```
jupyter notebook
```
Open `work/capstone/flyrank_google_search_discoverability_capstone.ipynb`. Cells print `BLOCKED` until the token and `configs/data_config.yaml` are set.

## 9. Run the pipeline
- Real data (after you map the schema in `configs/data_config.yaml`): `python scripts/run_pipeline.py`
- Synthetic demo (NOT a FlyRank result; writes to `reports/_demo_synthetic/`): `python scripts/run_pipeline.py --synthetic-demo`

## 10. Run tests
```
python scripts/run_tests.py
```

## 11. Reproduce experiments
Same data release + `configs/*.yaml` + seed 42 gives the same metrics. Details: `docs/REPRODUCIBILITY.md`.

## 12. View the paper
Run `python scripts/build_paper.py`, then open `paper/index.html` in a browser.

## 13. Enable GitHub Pages
See `docs/GITHUB_UPLOAD_GUIDE.md`.

## 14. Complete the submission
Put the live paper URL in `submission/paper_url.txt` (one line). Submit your **repo URL** on the FlyRank capstone page (Deliverable links).

## 15. Limitations and honest framing
Scores are decision-support estimates, not confirmed outcomes. No causal claim about refreshing and no claim about how Google ranks pages. Only FlyRank can accept the work or issue a certificate.
