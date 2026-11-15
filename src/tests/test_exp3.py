"""Experiment 3 tests (Member A): statistics on hand-made data, registry, and the committed raw results."""
import json, os
import numpy as np, pandas as pd, pytest

from eval import exp3_nn as e3
from eval.run_experiments import EXPERIMENTS

ROOT = e3.ROOT
RAW = os.path.join(ROOT, "outputs", "experiments", "exp3_nn_raw.csv")


def toy_frame(K=20, W=4):
    """3 hours, 2 seeds: last-hour served share .8/.8/.6; NN seed0 .9/.7/.8, seed1 .9/.7/.6 (blocking 1 - served)."""
    rows = []
    lh = [0.8, 0.8, 0.6]
    nn = {0: [0.9, 0.7, 0.8], 1: [0.9, 0.7, 0.6]}
    for h in range(3):
        rows.append(dict(source="last_hour", seed=-1, hour=504 + h, K=K, W=W, served_share=lh[h],
                         blocking_rate=1 - lh[h], topk_hit_rate=0.9, n_blocked=0))
        for s in (0, 1):
            rows.append(dict(source="nn", seed=s, hour=504 + h, K=K, W=W, served_share=nn[s][h],
                             blocking_rate=1 - nn[s][h], topk_hit_rate=0.9, n_blocked=0))
    return pd.DataFrame(rows)


def test_paired_hourly_averages_seeds_per_hour():
    d = e3.paired_hourly(toy_frame(), 20, 4, "served_share")
    assert list(d.index) == [504, 505, 506]
    assert d.to_numpy() == pytest.approx([0.9 - 0.8, 0.7 - 0.8, 0.7 - 0.6])      # seed-averaged NN: .9, .7, .7


def test_block_bootstrap_ci_properties():
    assert e3.block_bootstrap_ci(np.full(48, 0.25)) == (0.25, 0.25)              # constant series: no uncertainty
    x = np.random.default_rng(1).normal(0.1, 1.0, 168)
    a, b = e3.block_bootstrap_ci(x), e3.block_bootstrap_ci(x)
    assert a == b and a[0] < x.mean() < a[1]                                        # seeded -> reproducible, brackets the mean
    assert e3.block_bootstrap_ci(np.array([1.0, 2.0, 3.0]), block=24)[0] <= 2.0     # block longer than the series is handled


def test_summary_table_numbers_and_better_worse_counts():
    table, per_seed = e3.summarise_setting(toy_frame(), 20, 4)
    assert per_seed["served_share"] == pytest.approx([(0.1 - 0.1 + 0.2) / 3, (0.1 - 0.1 + 0.0) / 3])
    served_row = next(l for l in table.splitlines() if l.startswith("| served share"))
    assert "0.733" in served_row                       # last-hour mean (.8+.8+.6)/3
    assert "+0.0333 ± " in served_row                  # mean paired difference (.1 - .1 + .1)/3
    assert served_row.rstrip(" |").endswith("2 / 0 / 1")     # hours NN better / equal / worse
    blocking_row = next(l for l in table.splitlines() if l.startswith("| blocking rate"))
    assert blocking_row.rstrip(" |").endswith("2 / 0 / 1")   # here blocking = 1 - served, so lower blocking is better in the same hours
    flipped = toy_frame(); flipped["blocking_rate"] = flipped["served_share"]   # blocking now rises with served share
    table2, _ = e3.summarise_setting(flipped, 20, 4)
    assert next(l for l in table2.splitlines() if l.startswith("| blocking rate")).rstrip(" |").endswith("1 / 0 / 2")
    hit_row = next(l for l in table.splitlines() if l.startswith("| top-K hit rate"))
    assert hit_row.rstrip(" |").endswith("0 / 3 / 0")        # all equal


def test_write_outputs_to_temp_folder(tmp_path):
    df = pd.concat([toy_frame(20, 4), toy_frame(20, 8)], ignore_index=True)
    e3.write_outputs(df, tmp_path, json_path=tmp_path / "nn_effect.json")
    assert (tmp_path / "exp3_nn_summary.md").exists() and (tmp_path / "exp3_nn.png").stat().st_size > 2000
    data = json.loads((tmp_path / "nn_effect.json").read_text())
    assert [s["W"] for s in data["settings"]] == [4, 8] and len(data["runs"]) == len(df)


def test_write_outputs_can_skip_the_json(tmp_path):
    df = pd.concat([toy_frame(20, 4), toy_frame(20, 8)], ignore_index=True)
    e3.write_outputs(df, tmp_path, json_path=tmp_path / "nn_effect.json", write_json=False)
    assert (tmp_path / "exp3_nn_summary.md").exists() and not (tmp_path / "nn_effect.json").exists()
    text = (tmp_path / "exp3_nn_summary.md").read_text()
    assert "only 7 blocks" in text and "four distinct comparisons" in text


def test_registered_in_run_experiments():
    assert EXPERIMENTS["exp3"] == "eval.exp3_nn"


@pytest.mark.skipif(not os.path.exists(RAW), reason="run `python -m eval.run_experiments exp3` first")
def test_committed_raw_results_are_complete_and_consistent():
    df = pd.read_csv(RAW)
    assert len(df) == len(e3.SETTINGS) * len(e3.HOURS) * (1 + len(e3.SEEDS))          # 2 x 168 x 6
    for (K, W), g in df.groupby(["K", "W"]):
        assert sorted(g[g.source == "nn"].seed.unique()) == e3.SEEDS
        assert (g[g.source == "last_hour"].seed == -1).all() and g[g.source == "last_hour"].hour.nunique() == 168
        assert g[["served_share", "blocking_rate", "topk_hit_rate"]].apply(lambda c: c.between(0, 1 + 1e-9).all()).all()   # tolerance: float sums
    w8 = df[(df.W == 8) & (df.source == "nn")]
    assert (w8.blocking_rate == 0).mean() > 0.95       # at W=8 almost nothing is blocked


# ---------- Stage 8 additions (Member A): the resumable run and the CLI ---------------------------------
def _small_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(e3, "SETTINGS", [(10, 8)])
    monkeypatch.setattr(e3, "SEEDS", [0])
    monkeypatch.setattr(e3, "HOURS", [510, 511])
    monkeypatch.setattr(e3, "out_dir", lambda: tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(e3, "ROOT", tmp_path)                     # so main() never touches the real outputs/nn_effect.json


needs_abilene = pytest.mark.skipif(not (os.path.exists(os.path.join(ROOT, "data", "processed", "hourly.npz"))
                                        and os.path.exists(os.path.join(ROOT, "outputs", "nn", "mlp_seed0.joblib"))
                                        and os.path.exists(os.path.join(ROOT, "data", "raw", "topology", "abilene_sndlib_native.txt"))),
                                   reason="needs processed data, NN models and the Abilene topology file")


@needs_abilene
def test_run_writes_chunks_and_resumes_without_recomputing(monkeypatch, tmp_path):
    _small_experiment(monkeypatch, tmp_path)
    assert e3.run(max_seconds=-1) is None                          # stops before the first chunk: nothing written
    assert not list((tmp_path / ".exp3_chunks").glob("*.csv"))
    df = e3.run()
    assert len(df) == 2 * 2 and set(df.source) == {"last_hour", "nn"}
    assert len(list((tmp_path / ".exp3_chunks").glob("*.csv"))) == 2
    monkeypatch.setattr(e3, "run_planner", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed a finished chunk")))
    pd.testing.assert_frame_equal(e3.run(), df, check_dtype=False)   # second run reads the chunks only


@needs_abilene
def test_main_full_and_reuse_paths(monkeypatch, tmp_path, capsys):
    _small_experiment(monkeypatch, tmp_path)
    e3.main()
    assert (tmp_path / "exp3_nn_raw.csv").exists() and (tmp_path / "exp3_nn_summary.md").exists()
    assert (tmp_path / "outputs" / "nn_effect.json").exists()
    before = (tmp_path / "outputs" / "nn_effect.json").read_bytes()
    (tmp_path / "exp3_nn_summary.md").unlink()
    e3.main(reuse=True)                                            # rebuilds the summary, leaves the existing json alone
    assert (tmp_path / "exp3_nn_summary.md").exists() and (tmp_path / "outputs" / "nn_effect.json").read_bytes() == before
    assert "exp3 done" in capsys.readouterr().out


@needs_abilene
def test_nn_effect_cli_writes_json(monkeypatch, tmp_path):
    import sys
    from eval import nn_effect
    out = tmp_path / "ne.json"
    monkeypatch.setattr(sys, "argv", ["nn_effect", "--K", "10", "--W", "8", "--seeds", "0", "--out", str(out)])
    nn_effect.main()
    data = json.loads(out.read_text())
    assert data["settings"][0]["W"] == 8 and len(data["runs"]) == 2 * 168
