"""拓展 C：事后最优（全信息）上下界与“短视代价”分解。

对每个 (Δt, 车队, 测试日)：
  UB = LP(接驾时间松弛为 0，不限等待)        —— 任意策略（含知道未来的离线策略）收入上界；
  LB = MILP(接驾按 R 保守计，等待 ≤ 200 min)  —— 一个可执行的事后计划，回放仿真器验证。
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dispatch.data import load_all, ROOT
from dispatch.sim import Config, sample_episode, run_episode, morning_drivers
from dispatch.hindsight import hindsight, replay_policy

STEPS = [int(x) for x in os.environ.get("STEPS", "10,30").split(",")]
FLEETS = [40, 60, 100]
N_TEST = int(os.environ.get("N_TEST", 5))
D = load_all()
MORNING = morning_drivers(D["ck_train"])


ROWS = ROOT / "results" / "exp5_rows"


def pin_one_core():
    """每个工作进程绑定到一个核，HiGHS 只能用单线程。"""
    from multiprocessing import current_process
    i = (current_process()._identity or (1,))[0] - 1
    try:
        os.sched_setaffinity(0, {i % os.cpu_count()})
    except AttributeError:
        pass


def job(args):
    step, fleet, k = args
    out = ROWS / f"{step}_{fleet}_{k}.json"
    if out.exists():
        return json.load(open(out))
    cfg = Config(step_min=step)
    e = sample_episode(D["trips_test"], MORNING, fleet, seed=50_000 + k)
    ub = hindsight(cfg, e, pk_lb=0.0)
    lb = hindsight(cfg, e, pk_lb=3.0, integral=True, time_limit=400, max_wait=200, return_plan=True)
    m, _ = run_episode(cfg, e, replay_policy(lb["plan"]))
    assert abs(m["revenue"] - lb["value"]) < 1e-6 * max(1, lb["value"]), (m["revenue"], lb["value"])
    row = dict(step=step, fleet=fleet, day=k, ub=float(ub["value"]), lb=float(lb["value"]),
               lb_bound=float(lb["bound"]), lb_status=int(lb["status"]), ub_served=float(ub["n_served"]),
               lb_served=int(m["served"]), lb_answer=float(m["answer_rate"]), lb_pickup=float(m["avg_pickup"]),
               n_edges=int(ub["n_edges"]))
    json.dump(row, open(out, "w"))          # 每完成一个就落盘
    return row


if __name__ == "__main__":
    jobs = [(s, f, k) for s in STEPS for f in FLEETS for k in range(N_TEST)]
    jobs.sort(key=lambda j: j[0])
    ROWS.mkdir(exist_ok=True)
    with Pool(int(os.environ.get("PROCS", 30)), initializer=pin_one_core) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / f"exp5_hindsight{os.environ.get('TAG', '')}.csv", index=False)
    print(df.groupby(["step", "fleet"])[["lb", "ub", "lb_status"]].mean().round(1))
