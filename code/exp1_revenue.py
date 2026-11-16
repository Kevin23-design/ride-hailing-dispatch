"""实验一：多批次间隔 Δt 下 贪心 / KM-即时收入 / Q-KM 的全天收入对比。

每个 (Δt, 车队, 训练种子) 独立训练表格 Q-KM；β 在 3 个验证日上选择；
在 10 个由测试集订单池生成的模拟日上评估（各 Δt / 策略共用同一批测试日，配对比较）。
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dispatch.data import load_all, ROOT
from dispatch.sim import Config, sample_episode, run_episode, morning_drivers
from dispatch.rl import greedy_nearest, km_fare, make_q_km, region_long_term_value, train_q_km

STEPS = [2, 5, 10, 15, 20, 30]
FLEETS = [40, 60, 100]
SEEDS = [0, 1, 2, 3, 4]
BETAS = [0.2, 0.4, 0.5, 0.6, 0.8]
N_TEST, N_VAL = 10, 3
N_EPISODES = 300

D = load_all()
MORNING = morning_drivers(D["ck_train"])


def test_days(fleet):
    return [sample_episode(D["trips_test"], MORNING, fleet, seed=50_000 + k) for k in range(N_TEST)]


def val_days(fleet):
    return [sample_episode(D["trips_train"], MORNING, fleet, seed=10_000 + k) for k in range(N_VAL)]


def job(args):
    step, fleet, seed = args
    cfg = Config(step_min=step)
    rows = []
    tests = test_days(fleet)
    if seed == 0:
        for name, pol in [("贪心最近", greedy_nearest), ("KM-即时收入", km_fare)]:
            for k, e in enumerate(tests):
                m, _ = run_episode(cfg, e, pol)
                rows.append(dict(step=step, fleet=fleet, seed=-1, policy=name, day=k, beta=np.nan, **m))
    vals = val_days(fleet)
    qt, curve, td = train_q_km(cfg, D["trips_train"], MORNING, fleet, n_episodes=N_EPISODES, seed=seed,
                               eval_eps=vals if (step == 10 and fleet == 60) else None)
    LT = region_long_term_value(cfg, qt.V)
    val_rev = {b: np.mean([run_episode(cfg, e, make_q_km(LT, b))[0]["revenue"] for e in vals]) for b in BETAS}
    beta = max(val_rev, key=val_rev.get)
    pol = make_q_km(LT, beta)
    for k, e in enumerate(tests):
        m, _ = run_episode(cfg, e, pol)
        rows.append(dict(step=step, fleet=fleet, seed=seed, policy="Q-KM", day=k, beta=beta, **m))
    extra = dict(step=step, fleet=fleet, seed=seed, val_rev=val_rev, td=td.tolist(),
                 curve=curve.tolist() if len(curve) else [])
    return rows, extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jobs = [(s, f, sd) for s in STEPS for f in FLEETS for sd in SEEDS]
    if a.quick:
        jobs = [(10, 60, 0)]
        N_EPISODES = 300
    jobs.sort(key=lambda j: j[0])          # Δt 小的最慢，先调度
    out = ROOT / "results"
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    extras = [e for _, e in res]
    tag = "_quick" if a.quick else ""
    pd.DataFrame(rows).to_csv(out / f"exp1{tag}.csv", index=False)
    json.dump(extras, open(out / f"exp1_extra{tag}.json", "w"))
    df = pd.DataFrame(rows)
    print(df.groupby(["step", "fleet", "policy"])[["revenue", "answer_rate", "avg_pickup"]].mean().round(3))
