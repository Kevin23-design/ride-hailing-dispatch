"""拓展 E：批次级帕累托权衡在全天（系统级）是否仍成立？

边权 w = fare − λ·PER_KM·pickup（λ 把 1 km 接驾折算成 λ km 送客的车费），λ 扫描；
基础策略为 KM 即时收入与 PE-KM（表格期望目标，β=0.6）。Δt=10，三种车队，10 个测试日。
记录全天收入 / 总送客 km / 总接驾 km / 应答率。
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from dispatch.data import load_all, ROOT, PER_KM
from dispatch.sim import Config, sample_episode, run_episode, morning_drivers
from dispatch.rl import km_max, greedy_nearest, region_long_term_value, train_q_km

FLEETS = [40, 60, 100]
LAMBDAS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8]
STEPS = [5, 10, 20]
D = load_all()
MORNING = morning_drivers(D["ck_train"])


def policy(lam, LT=None, beta=0.0):
    def pol(ctx):
        w = ctx["fare"][None, :] - lam * PER_KM * ctx["pickup"]
        if LT is not None:
            w = w + beta * LT[ctx["t"], ctx["drv_zone"][:, None], ctx["dest_zone"][None, :]]
        return km_max(np.where(ctx["feas"], w, 0.0))
    return pol


def job(args):
    step, fleet = args
    cfg = Config(step_min=step)
    tests = [sample_episode(D["trips_test"], MORNING, fleet, seed=50_000 + k) for k in range(10)]
    qt, _, _ = train_q_km(cfg, D["trips_train"], MORNING, fleet, n_episodes=300, seed=0, target="expect")
    LT = region_long_term_value(cfg, qt.V)
    rows = []

    def ev(name, lam, pol):
        ms = [run_episode(cfg, e, pol)[0] for e in tests]
        rows.append(dict(step=step, fleet=fleet, base=name, lam=lam,
                         revenue=[m["revenue"] for m in ms], trip_km=[m["trip_km"] for m in ms],
                         pickup_km=[m["avg_pickup"] * m["served"] for m in ms],
                         answer=[m["answer_rate"] for m in ms], served=[m["served"] for m in ms]))
    ev("贪心最近", np.nan, greedy_nearest)
    for lam in LAMBDAS:
        ev("KM", lam, policy(lam))
        ev("PE-KM", lam, policy(lam, LT, 0.6))
    return rows


if __name__ == "__main__":
    jobs = [(s, f) for s in STEPS for f in FLEETS]
    with Pool(len(jobs)) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump([r for rr in res for r in rr], open(ROOT / "results" / "exp7_system_front.json", "w"))
    print("done")
