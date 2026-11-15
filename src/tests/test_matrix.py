"""Keeps docs/test_matrix.md honest: every test it names must exist, and each section must meet the plan's minimum."""
import ast, os, re
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(ROOT, "src", "tests")
MATRIX = os.path.join(ROOT, "docs", "test_matrix.md")
MINIMUM = {"Normal": 5, "Edge": 3, "Impossible": 2}


def sections():
    text = open(MATRIX, encoding="utf-8").read()
    out, current = {}, None
    for line in text.splitlines():
        m = re.match(r"## (\w+)", line)
        if m:
            current = m.group(1); out[current] = []
        elif current and line.startswith("- "):
            out[current].append(line)
    return out


def defined_tests(filename):
    tree = ast.parse(open(os.path.join(TESTS, filename), encoding="utf-8").read())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_every_named_test_exists():
    refs = re.findall(r"`(test_\w+\.py)::(test_\w+)`", open(MATRIX, encoding="utf-8").read())
    assert len(refs) >= 20
    for filename, name in refs:
        assert os.path.exists(os.path.join(TESTS, filename)), filename
        assert name in defined_tests(filename), f"{filename}::{name} is named in docs/test_matrix.md but does not exist"


@pytest.mark.parametrize("section,minimum", MINIMUM.items())
def test_each_section_meets_the_plans_minimum(section, minimum):
    bullets = sections()[section.split()[0]]
    assert len(bullets) >= minimum, f"{section}: {len(bullets)} requirements, plan needs {minimum}"
    assert all(re.search(r"`test_\w+\.py::test_\w+`", b) for b in bullets), "every requirement must name a test"
