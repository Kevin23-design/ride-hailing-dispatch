"""拓展 D：价值目标的“乐观度”连续谱。

期望分位数 TD：τ=0.5 为策略评估（PE），τ→1 逼近对后继回报取上界；另含 max 目标 Q-learning。
每个 (车队, 目标, 种子) 训练 300 个模拟日（训练时 β=0.5），在 10 个测试日上扫 β 评估收入，
并沿司机真实轨迹回溯折扣回报，测量价值表的估计偏差 (V − G)/G。
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from dispatch.data import load_all, ROOT
from dispatch.sim import Config, sample_episode, run_episode, morning_drivers
from dispatch.rl import make_q_km, region_long_term_value, train_q_km, mc_bias, km_fare

FLEETS = [40, 60, 100]
TARGETS = [("expectile", 0.5), ("expectile", 0.6), ("expectile", 0.7), ("expectile", 0.8),
           ("expectile", 0.9), ("expectile", 0.95), ("max", None)]
SEEDS = [0, 1, 2]
BETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
D = load_all()
MORNING = morning_drivers(D["ck_train"])
cfg = Config(step_min=10)


def job(args):
    fleet, (target, tau), seed = args
    qt, _, _ = train_q_km(cfg, D["trips_train"], MORNING, fleet, n_episodes=300, seed=seed,
                          target=target, expectile=tau if tau is not None else 0.5)
    LT = region_long_term_value(cfg, qt.V)
    tests = [sample_episode(D["trips_test"], MORNING, fleet, seed=50_000 + k) for k in range(10)]
    out = []
    for b in BETAS:
        revs, vs, gs, ans = [], [], [], []
        for e in tests:
            m, tr = run_episode(cfg, e, make_q_km(LT, b) if b > 0 else km_fare, collect=True)
            v, g = mc_bias(cfg, qt.V, tr)
            revs.append(m["revenue"]); vs.append(v); gs.append(g); ans.append(m["answer_rate"])
        out.append(dict(fleet=fleet, target=target, tau=tau, seed=seed, beta=b, revenue=float(np.mean(revs)),
                        rev_days=revs, v_mean=float(np.mean(vs)), g_mean=float(np.mean(gs)),
                        answer=float(np.mean(ans))))
    return out


if __name__ == "__main__":
    jobs = [(f, t, s) for f in FLEETS for t in TARGETS for s in SEEDS]
    with Pool(int(os.environ.get("PROCS", 44))) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump([r for rr in res for r in rr], open(ROOT / "results" / "exp6_expectile.json", "w"))
    print("done", len(res))
