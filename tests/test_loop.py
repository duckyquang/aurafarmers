from sim import policies
from sim.loop import World
from sim.metrics import compute


def test_bot_world_runs_and_progresses(tmp_path):
    w = World(seed=51, cond="A", out_dir=tmp_path, n_agents=6,
              policy_factory=policies.honest_bot)
    w.run(ticks=30)
    m = compute(w.log_path, burn_in=0, n_agents=6)
    assert m["progress_value"] > 0
    assert (tmp_path / "log.jsonl").exists()


def test_budget_enforced(tmp_path):
    w = World(seed=51, cond="A", out_dir=tmp_path, n_agents=1,
              policy_factory=policies.honest_bot)
    ag = w.agents[0]
    cost = w.charge(ag, {"kind": "intervene",
                         "targets": {"F01.L01.V01": 1.0},
                         "measure": ["F01.L02.V01"], "n": 300})
    assert cost is None                     # 300*5 = 1500 > 1000 budget


def test_locked_layers_rejected(tmp_path):
    w = World(seed=51, cond="A", out_dir=tmp_path, n_agents=1,
              policy_factory=policies.honest_bot)
    assert not w.experiment_allowed(["F01.L09.V01"])
    assert w.experiment_allowed(["F01.L03.V01"])


def test_identical_seed_same_truth_across_conditions(tmp_path):
    wa = World(seed=99, cond="A", out_dir=tmp_path / "A", n_agents=2,
               policy_factory=policies.honest_bot)
    wb = World(seed=99, cond="B", out_dir=tmp_path / "B", n_agents=2,
               policy_factory=policies.honest_bot)
    assert wa.truth.edges == wb.truth.edges
    assert [a["persona"] for a in wa.agents] == [a["persona"] for a in wb.agents]
