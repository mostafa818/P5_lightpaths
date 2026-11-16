"""Keeps docs/test_matrix.md honest: every test it names must exist, and each section must meet the plan's minimum."""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "src" / "tests"
MATRIX = ROOT / "docs" / "test_matrix.md"
MINIMUM = {"Normal": 5, "Edge": 3, "Impossible": 2}


def sections():
    out, current = {}, None
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## (\w+)", line)
        if m:
            current = m.group(1)
            out[current] = []
        elif current and line.startswith("- "):
            out[current].append(line)
    return out


def defined_tests(filename):
    tree = ast.parse((TESTS / filename).read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_every_named_test_exists():
    refs = re.findall(r"`(test_\w+\.py)::(test_\w+)`", MATRIX.read_text(encoding="utf-8"))
    assert len(refs) >= 20
    for filename, name in refs:
        assert (TESTS / filename).exists(), filename
        assert name in defined_tests(filename), f"{filename}::{name} is named in docs/test_matrix.md but does not exist"


@pytest.mark.parametrize("section,minimum", MINIMUM.items())
def test_each_section_meets_the_plans_minimum(section, minimum):
    bullets = sections()[section.split()[0]]
    assert len(bullets) >= minimum, f"{section}: {len(bullets)} requirements, plan needs {minimum}"
    assert all(re.search(r"`test_\w+\.py::test_\w+`", b) for b in bullets), "every requirement must name a test"
