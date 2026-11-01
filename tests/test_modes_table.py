"""The README's mode tables are exactly what the committed benchmark results give."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_mode_tables_match_the_results():
    spec = importlib.util.spec_from_file_location("modes_table", ROOT / "benchmarks/modes_table.py")
    modes_table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modes_table)
    readme = (ROOT / "README.md").read_text("utf-8")
    assert "<!-- modes:en -->" in readme and "<!-- modes:ru -->" in readme
    assert modes_table.render(readme) == readme, (
        "run: python benchmarks/modes_table.py --write README.md"
    )
