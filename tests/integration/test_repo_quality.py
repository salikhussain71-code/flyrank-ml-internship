"""Repository hygiene: notebooks, syntax, secrets, paper state, docs links, structure."""
import _bootstrap  # noqa: F401
import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}

REQUIRED = [
    "README.md", "BUILD_REPORT.md", "PROJECT_STATUS.md", "CONTRIBUTING.md", "LICENSE", ".gitignore", ".env.example",
    "requirements.txt", "pyproject.toml", "submission/paper_url.txt", "paper/index.html", "paper/README.md",
    ".github/workflows/tests.yml", "configs/data_config.yaml", "configs/experiment_config.yaml",
    "configs/model_config.yaml", "tests/conftest.py", "scripts/setup_environment.py", "scripts/validate_environment.py",
    "scripts/run_pipeline.py", "scripts/run_tests.py", "scripts/build_paper.py",
    "work/capstone/flyrank_google_search_discoverability_capstone.ipynb",
    "docs/PROJECT_ARCHITECTURE.md", "docs/DATA_DICTIONARY.md", "docs/METHODOLOGY.md", "docs/EXPERIMENTS.md",
    "docs/REPRODUCIBILITY.md", "docs/SECURITY.md", "docs/ASSIGNMENT_TRACKER.md", "docs/GITHUB_UPLOAD_GUIDE.md",
] + [f"work/assignments/{n}.ipynb" for n in [
    "01_run_starter_notebooks", "02_research_question_and_lane", "03_frame_lane_as_ml_task",
    "04_search_intelligence_data_contract", "05_baseline_action_score_top10", "06_capstone_modeling_lane",
    "07_validation_research_claim_audit", "08_content_action_playbook", "09_ship_the_paper", "10_tell_the_story"]]


def files(suffixes):
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in suffixes and not (set(p.parts) & SKIP_DIRS):
            yield p


class TestRepo(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [r for r in REQUIRED if not (ROOT / r).exists()]
        self.assertEqual(missing, [])

    def test_notebooks_valid_json(self):
        nbs = list(files({".ipynb"}))
        self.assertGreaterEqual(len(nbs), 11)
        for p in nbs:
            nb = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4, p)
            self.assertTrue(nb["cells"], p)
            for c in nb["cells"]:
                self.assertIn(c["cell_type"], ("markdown", "code"))
                if c["cell_type"] == "code":
                    src = "".join(c["source"])
                    ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!"))), str(p))

    def test_python_syntax(self):
        for p in files({".py"}):
            ast.parse(p.read_text(encoding="utf-8"), str(p))

    def test_no_secrets_or_env_file(self):
        self.assertFalse((ROOT / ".env").exists())
        patterns = [re.compile(r"hf_[A-Za-z0-9]{30,}"), re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
                    re.compile(r"ghp_[A-Za-z0-9]{30,}"), re.compile(r"github_pat_[A-Za-z0-9_]{30,}")]
        for p in files({".py", ".md", ".ipynb", ".yaml", ".yml", ".txt", ".html", ".json", ".toml", ".example"}):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in patterns:
                self.assertIsNone(pat.search(text), f"possible secret in {p}")

    def test_no_raw_data_files(self):
        bad = [p for p in ROOT.rglob("*") if p.suffix in {".parquet", ".duckdb", ".feather"} and not (set(p.parts) & SKIP_DIRS)]
        self.assertEqual(bad, [])

    def test_paper_url_single_line(self):
        lines = [l for l in (ROOT / "submission" / "paper_url.txt").read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)

    def test_paper_is_draft_without_real_results(self):
        res = ROOT / "reports" / "metrics" / "results.json"
        html = (ROOT / "paper" / "index.html").read_text(encoding="utf-8")
        real = res.exists() and json.loads(res.read_text()).get("is_flyrank_real_data") is True
        if not real:
            self.assertIn("DRAFT", html)
        for needle in ["Abstract", "Introduction", "Data", "Methodology", "Results", "Limitations", "Ranked recommendations",
                       "Reproducibility", "Built on the FlyRank ML Internship dataset", "https://flyrank.ai"]:
            self.assertIn(needle, html)

    def test_tracker_lists_ten_assignments_and_capstone(self):
        t = (ROOT / "docs" / "ASSIGNMENT_TRACKER.md").read_text(encoding="utf-8")
        for i in range(1, 11):
            self.assertRegex(t, rf"\|\s*{i}\s*\|")
        self.assertIn("Capstone", t)

    def test_markdown_relative_links_resolve(self):
        link = re.compile(r"\]\(((?!https?:|#|mailto:)[^)\s]+)\)")
        for p in files({".md"}):
            for m in link.finditer(p.read_text(encoding="utf-8")):
                target = m.group(1).split("#")[0]
                if target:
                    self.assertTrue((p.parent / target).exists(), f"{p}: broken link {target}")

    def test_notebooks_contain_no_outputs_with_data(self):
        for p in files({".ipynb"}):
            nb = json.loads(p.read_text(encoding="utf-8"))
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    self.assertEqual(c.get("outputs", []), [], f"{p} has stored outputs; clear before publishing")
