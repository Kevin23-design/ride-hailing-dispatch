"""核心正确性检查：ε-约束精确前沿 vs 穷举；Q 更新手算；β=0 的 Q-KM 与 KM-即时收入一致。"""
import itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dispatch.moo import DispatchProblem, exact_front, nondominated
from dispatch.sim import Config
from dispatch.rl import QTable, q_update, make_q_km, region_long_term_value, km_fare


def brute_front(prob):
    edges = [(int(d), j) for j in range(prob.n_var) for d in prob.cand[j] if d >= 0]
    pts = set()
    def rec(i, used_d, used_o, g, p):
        pts.add((round(-g, 6), round(p, 6)))
        for k in range(i, len(edges)):
            d, o = edges[k]
            if d not in used_d and o not in used_o:
                rec(k + 1, used_d | {d}, used_o | {o}, g + prob.fare[o], p + prob.pickup[d, o])
    rec(0, frozenset(), frozenset(), 0.0, 0.0)
    return nondominated(np.array(sorted(pts)))


def test_exact_front():
    for s in range(5):
        rng = np.random.default_rng(s)
        nd, no = 4, 6
        pick = np.round(rng.uniform(0, 3.5, (nd, no)), 2)
        trip = np.round(rng.uniform(1, 15, no), 2)
        prob = DispatchProblem(dict(fare=trip, trip_km=trip, pickup=pick), K=3)
        Z, info = exact_front(prob)
        assert info["verified"]
        B = brute_front(prob)
        assert len(Z) == len(B) and np.allclose(np.sort(Z, 0), np.sort(B, 0), atol=1e-6), (Z, B)


def test_q_update():
    cfg = Config(step_min=10)
    qt = QTable(cfg); W = cfg.WAIT; g = cfg.gamma
    qt.Q[5, 3, 7], qt.Q[5, 3, 9], qt.Q[5, 3, W] = 30.0, 99.0, 10.0
    td = q_update(qt, cfg, 2, 0, 3, 20.0, 5, 3, np.array([7]))
    assert abs(td - (20 + g ** 3 * 30)) < 1e-9
    assert abs(qt.V[5, 3] - 30) < 1e-9
    assert abs(q_update(qt, cfg, cfg.T - 1, 0, W, 7.0, cfg.T + 2, 1, np.array([5])) - 7.0) < 1e-9


def test_beta0_equals_km():
    cfg = Config(step_min=10)
    rng = np.random.default_rng(0)
    LT = region_long_term_value(cfg, rng.random((cfg.T + 1, cfg.n_zones)) * 50)
    ctx = dict(t=3, fare=rng.uniform(5, 40, 8), drv_zone=rng.integers(0, 64, 5),
               dest_zone=rng.integers(0, 64, 8), feas=rng.random((5, 8)) < 0.6)
    assert sorted(make_q_km(LT, 0.0)(ctx)) == sorted(km_fare(ctx))


if __name__ == "__main__":
    test_exact_front(); test_q_update(); test_beta0_equals_km(); print("all tests passed")
