import yaml
from common import Request, Route, Stats, Result, link_id


def test_config_loads():
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["rwa"]["num_wavelengths"] in (4, 8)


def test_contract_types():
    assert link_id("B", "A") == link_id("A", "B")
    r = Route(("A", "B"), frozenset({link_id("A", "B")}), 10.0)
    assert Request(0, "A", "B", 1.0).src == "A" and r.length_km == 10.0
    assert Result(None).stats.backtracks == 0
