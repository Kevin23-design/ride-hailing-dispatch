"""拓展 D 补充：高估“落在哪里”。β=1 的 Q-KM 派单时，统计可行边上的三项：
即时车费、出发价值 γ^Δ V(t+Δ,k)、等待价值 γ V(t+1,i)，以及被判为负权（拒派）的可行边比例。
Δt=10，5 个测试日，3 个训练种子。"""
import json
import os
from multiprocessing import Pool

import numpy as np

from dispatch.data import load_all, ROOT
from dispatch.sim import Config, sample_episode, run_episode, morning_drivers
from dispatch.rl import make_q_km, region_long_term_value, train_q_km

D = load_all()
MORNING = morning_drivers(D["ck_train"])
cfg = Config(step_min=10)
TARGETS = [("expectile", 0.5), ("expectile", 0.7), ("expectile", 0.8), ("expectile", 0.9),
           ("expectile", 0.95), ("max", None)]


def job(args):
    fleet, (target, tau), seed = args
    qt, _, _ = train_q_km(cfg, D["trips_train"], MORNING, fleet, n_episodes=300, seed=seed, target=target,
                          expectile=tau if tau is not None else 0.5)
    base = make_q_km(region_long_term_value(cfg, qt.V), 1.0)
    st = []

    def pol(ctx):
        F = ctx["feas"]
        if F.any():
            t, i, k = ctx["t"], ctx["drv_zone"][:, None], ctx["dest_zone"][None, :]
            go = cfg.gamma ** cfg.DELTA[i, k] * qt.V[np.minimum(t + cfg.DELTA[i, k], cfg.T), k]
            stay = np.broadcast_to(cfg.gamma * qt.V[t + 1, i], F.shape)
            fare = np.broadcast_to(ctx["fare"][None, :], F.shape)
            st.append([F.sum(), ((fare + go - stay)[F] <= 0).sum(), fare[F].sum(), go[F].sum(), stay[F].sum()])
        return base(ctx)
    for k in range(5):
        run_episode(cfg, sample_episode(D["trips_test"], MORNING, fleet, seed=50_000 + k), pol)
    s = np.sum(st, 0)
    return dict(fleet=fleet, target=target, tau=tau, seed=seed, refuse=s[1] / s[0], fare=s[2] / s[0],
                go=s[3] / s[0], stay=s[4] / s[0])


if __name__ == "__main__":
    jobs = [(f, t, s) for f in [40, 60, 100] for t in TARGETS for s in [0, 1, 2]]
    with Pool(int(os.environ.get("PROCS", 10))) as p:
        rows = p.map(job, jobs, chunksize=1)
    json.dump(rows, open(ROOT / "results" / "exp6b_bias.json", "w"))
    import pandas as pd
    df = pd.DataFrame(rows)
    df["name"] = df.tau.fillna("max").astype(str)
    df["adv"] = df.go - df.stay
    print(df.groupby(["fleet", "name"])[["refuse", "fare", "go", "stay", "adv"]].mean().round(3).to_string())
