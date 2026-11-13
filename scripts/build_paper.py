"""Build paper/index.html from reports/. Stays a clearly-marked DRAFT unless real results exist."""
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from flyrank_ml.utils.io import load_json  # noqa: E402

REPO_URL = "https://github.com/salikhussain71-code/flyrank-ml-internship"


def e(x) -> str:
    return html.escape(str(x))


def fmt(v) -> str:
    return "n/a" if v != v else f"{v:.3f}"


def load_real_results():
    p = ROOT / "reports" / "metrics" / "results.json"
    if not p.exists():
        return None
    r = load_json(p)
    return r if r.get("is_flyrank_real_data") is True else None


def metrics_table(r) -> str:
    keys = ["roc_auc", "average_precision", "precision_at_10", "precision_at_50", "ndcg_at_10", "lift_at_10"]
    rows = "".join(
        f"<tr><td>{e(k)}</td><td>{fmt(r['baseline_metrics'].get(k, float('nan')))}</td>"
        f"<td>{fmt(r['model_metrics'].get(k, float('nan')))}</td></tr>" for k in keys)
    return f"<table><thead><tr><th>Metric</th><th>Baseline</th><th>Model</th></tr></thead><tbody>{rows}</tbody></table>"


def recs_table() -> str:
    p = ROOT / "reports" / "tables" / "ranked_recommendations.csv"
    df = pd.read_csv(p).head(10)
    cols = ["rank", "page_pseudonym", "action", "reason_codes", "score"]
    head = "".join(f"<th>{e(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{e(round(v, 3) if isinstance(v, float) else v)}</td>" for v in row) + "</tr>"
                   for row in df[cols].itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build():
    r = load_real_results()
    draft = r is None
    banner = ('<div class="banner">DRAFT. No real FlyRank experiment results exist yet. This page is a template and is '
              'not a finished research paper.</div>') if draft else ""
    if draft:
        abstract = ("<p>Research question: pending (provisional lane: Refresh / Content Opportunity Scoring). "
                    "Method: pending real-data schema inspection. Result: none yet. "
                    "No numbers are reported until experiments run on the real warehouse. "
                    "This abstract will be replaced by five sentences summarising question, method and result.</p>")
        data = "<p>PENDING: dataset release, tables, date windows, exclusions and reasons will be filled from the real inspection.</p>"
        results = "<p>PENDING: model vs baseline comparison on the same validation split.</p>"
        recs = "<p>PENDING: ranked action playbook with reason codes.</p>"
        fig = ""
    else:
        m, b = r["model_metrics"], r["baseline_metrics"]
        d = r["roc_auc_diff_bootstrap_model_minus_baseline"]
        abstract = (f"<p>We ask which pages are at risk of a material click decline so reviewers can prioritise content review. "
                    f"We build point-in-time features from two 28-day windows and a future-window label ({e(r['label'])}). "
                    f"We compare a transparent baseline with a gradient-boosted model on pages held out of training and a later period. "
                    f"On that split the model reached ROC AUC {fmt(m['roc_auc'])} versus {fmt(b['roc_auc'])} for the baseline "
                    f"(bootstrap 95% interval for the difference: {fmt(d['ci_low'])} to {fmt(d['ci_high'])}). "
                    f"These are directional decision-support results, not causal claims.</p>")
        ex = "".join(f"<li>{e(k)}: {e(v)}</li>" for k, v in r["exclusions"].items())
        data = (f"<p>Built on the FlyRank warehouse. Feature cutoff for training {e(r['cutoffs']['train_cutoff'])}, for validation "
                f"{e(r['cutoffs']['valid_cutoff'])}. Training pages: {r['n_train']}; validation pages: {r['n_valid']}. Exclusions:</p><ul>{ex}</ul>")
        results = metrics_table(r) + '<img src="assets/images/model_vs_baseline.png" alt="Model versus baseline metrics">'
        recs = recs_table() + '<img src="assets/images/action_counts.png" alt="Recommended action counts">'
        fig = "ok"
        for name in ("model_vs_baseline.png", "action_counts.png"):
            src = ROOT / "reports" / "figures" / name
            if src.exists():
                shutil.copy(src, ROOT / "paper" / "assets" / "images" / name)
    method = (f"<p>Features use only data on or before the decision date. The label uses only data after it. "
              f"The split holds out whole pages and uses a later period for validation. Leakage checks raise errors on temporal overlap, "
              f"page overlap and near-perfect feature/label correlation.</p>"
              + (f"<p>Baseline: {e(r['baseline_description'])}</p>" if not draft else "<p>Baseline: pending.</p>"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search Ranking and Discoverability: Content Opportunity Scoring</title>
<link rel="stylesheet" href="assets/css/style.css"></head>
<body><main>
{banner}
<h1>Search Ranking and Discoverability: Content Opportunity Scoring</h1>
<p class="by">Salik Hussain. FlyRank AI Internship, Machine Learning track capstone.</p>
<h2>Abstract</h2>{abstract}
<h2>Introduction and problem statement</h2>
<p>Content teams cannot review every page. This work supports one decision: which pages to review first for a refresh, a metadata change, or monitoring.</p>
<h2>Data</h2>{data}
<h2>Methodology</h2>{method}
<h2>Results</h2>{results}
<h2>Limitations and honest framing</h2>
<ul><li><strong>Observed:</strong> patterns in historical search metrics.</li>
<li><strong>Directional:</strong> model scores that rank pages by estimated decline risk.</li>
<li><strong>Decision-support:</strong> recommendations are prompts for human review. They do not prove how Google ranks pages and do not show that any refresh causes improvement.</li></ul>
<h2>Ranked recommendations</h2>{recs}
<h2>Reproducibility</h2>
<p>Code, notebooks and configuration: <a href="{REPO_URL}">{REPO_URL}</a>. Notebooks are in <code>work/</code>.</p>
<h2>Acknowledgments and data credit</h2>
<p>Built on the FlyRank ML Internship dataset (<a href="https://flyrank.ai" target="_blank" rel="noopener">https://flyrank.ai</a>).</p>
</main></body></html>
""", draft


if __name__ == "__main__":
    page, is_draft = build()
    (ROOT / "paper" / "index.html").write_text(page, encoding="utf-8")
    print("Wrote paper/index.html as", "DRAFT" if is_draft else "RESULTS-BASED (still review before deploying)")
