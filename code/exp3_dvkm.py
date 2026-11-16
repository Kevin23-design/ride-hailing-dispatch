"""实验三（拓展 A）：DV-KM 深度价值网络 vs 表格 Q-KM（max 目标）/ 表格策略评估 PE-KM（期望目标）。

用法：python exp3_dvkm.py --fleet 60 --seed 0 --gpu 0 [--ablation]
流程（广义策略迭代）：
  第 0 轮：表格 Q-KM(β=0.5)+ε=0.2 随机 KM 在训练日上采集经验 -> 训练 V_θ
  第 1..R 轮：DV-KM(β=0.5)+ε=0.1 采集新经验，与旧数据合并 -> 热启动继续训练
评估：β∈{0,0.2,...,1.0}，10 个测试模拟日（与实验一相同），Δt=10。
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch

from dispatch.data import load_all, ROOT
from dispatch.sim import Config, sample_episode, run_episode, morning_drivers
from dispatch.rl import (km_fare, greedy_nearest, random_km, make_q_km, region_long_term_value, train_q_km)
from dispatch.dvkm import ValueNet, make_dv_km, train_value_net, transitions_to_array

D = load_all()
MORNING = morning_drivers(D["ck_train"])
CFG = Config(step_min=10)
BETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_TEST = 10

_G = {}


def _init(state, coarse):
    torch.set_num_threads(1)
    if state is not None:
        net = ValueNet(coarse=coarse)
        net.load_state_dict(state)
        _G["net"] = net.eval()


def _collect(args):
    kind, fleet, day_seed, eps, beta, LT = args
    rng = np.random.default_rng(day_seed)
    base = make_q_km(LT, beta) if kind == "tab" else make_dv_km(_G["net"], CFG, beta)
    pol = lambda ctx: random_km(ctx, rng) if rng.random() < eps else base(ctx)
    ep = sample_episode(D["trips_train"], MORNING, fleet, seed=day_seed)
    _, tr = run_episode(CFG, ep, pol, collect=True)
    return transitions_to_array(tr)


def _eval(args):
    fleet, k, beta = args
    ep = sample_episode(D["trips_test"], MORNING, fleet, seed=50_000 + k)
    m, _ = run_episode(CFG, ep, make_dv_km(_G["net"], CFG, beta))
    return m


def pool(state, coarse, procs):
    return get_context("spawn").Pool(procs, initializer=_init, initargs=(state, coarse))


def eval_dv(net, coarse, fleet, procs):
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    with pool(state, coarse, procs) as p:
        ms = p.map(_eval, [(fleet, k, b) for b in BETAS for k in range(N_TEST)])
    return [dict(beta=b, day=k, **ms[i * N_TEST + k]) for i, b in enumerate(BETAS) for k in range(N_TEST)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--days0", type=int, default=150)
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    tag = f"f{a.fleet}_s{a.seed}"
    out = ROOT / "results" / "exp3"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)
    res = dict(fleet=a.fleet, seed=a.seed, rows=[], loss={})

    # ---------- 表格方法 ----------
    tests = [sample_episode(D["trips_test"], MORNING, a.fleet, seed=50_000 + k) for k in range(N_TEST)]
    tabs = {}
    for name, target in [("Tab-Q-KM", "max"), ("Tab-PE-KM", "expect")]:
        qt, _, _ = train_q_km(CFG, D["trips_train"], MORNING, a.fleet, n_episodes=300, seed=a.seed, target=target)
        LT = region_long_term_value(CFG, qt.V)
        tabs[name] = LT
        for b in BETAS:
            for k, e in enumerate(tests):
                m, _ = run_episode(CFG, e, make_q_km(LT, b))
                res["rows"].append(dict(method=name, beta=b, day=k, **m))
        log(f"{name} done")
    for name, pol in [("贪心最近", greedy_nearest), ("KM-即时收入", km_fare)]:
        for k, e in enumerate(tests):
            m, _ = run_episode(CFG, e, pol)
            res["rows"].append(dict(method=name, beta=np.nan, day=k, **m))

    # ---------- DV-KM：广义策略迭代 ----------
    day_base = 700_000 + 10_000 * a.seed
    with pool(None, True, a.procs) as p:
        chunks = p.map(_collect, [("tab", a.fleet, day_base + d, 0.2, 0.5, tabs["Tab-Q-KM"]) for d in range(a.days0)])
    data = np.concatenate(chunks)
    log(f"round0 data {len(data)}")
    net, hist = train_value_net(data, CFG.gamma_min, epochs=a.epochs, device=dev, seed=a.seed)
    res["loss"]["r0"] = hist
    for r in range(1, a.rounds + 1):
        state = {k: v.cpu() for k, v in net.state_dict().items()}
        with pool(state, True, a.procs) as p:
            chunks = p.map(_collect, [("dv", a.fleet, day_base + 1000 * r + d, 0.1, 0.5, None) for d in range(a.days)])
        data = np.concatenate([data] + chunks)
        net, hist = train_value_net(data, CFG.gamma_min, epochs=max(10, a.epochs // 2), device=dev,
                                    net=net, seed=a.seed)
        res["loss"][f"r{r}"] = hist
        log(f"round{r} data {len(data)} loss {hist[-1]:.3f}")
    for row in eval_dv(net, True, a.fleet, a.procs):
        res["rows"].append(dict(method="DV-KM", **row))
    torch.save(net.state_dict(), out / f"dv_{tag}.pt")
    log("DV-KM eval done")

    # ---------- 价值图（供可视化）----------
    xs, ys = np.linspace(0.2, 25, 60), np.linspace(0.2, 31, 72)
    XX, YY = np.meshgrid(xs, ys)
    vmaps = {}
    with torch.no_grad():
        for hh in [8, 13, 18, 22]:
            tt = torch.full((XX.size,), float(hh * 60))
            vmaps[hh] = net(tt, torch.tensor(XX.ravel(), dtype=torch.float32),
                            torch.tensor(YY.ravel(), dtype=torch.float32)).numpy().reshape(XX.shape).tolist()
    res["vmap"] = vmaps

    # ---------- 消融：同一份数据训练不同结构 ----------
    if a.ablation:
        for name, coarse, smooth in [("DV-KM 无平滑正则", True, 0.0), ("DV-KM 无粗编码", False, 1.0)]:
            n2, hist = train_value_net(data, CFG.gamma_min, coarse=coarse, smooth=smooth,
                                       epochs=a.epochs, device=dev, seed=a.seed)
            res["loss"][name] = hist
            for row in eval_dv(n2, coarse, a.fleet, a.procs):
                res["rows"].append(dict(method=name, **row))
            log(f"{name} done")
        n3, hist = train_value_net(data, CFG.gamma_min, epochs=a.epochs, device=dev, seed=a.seed)
        for row in eval_dv(n3, True, a.fleet, a.procs):
            res["rows"].append(dict(method="DV-KM 同数据重训", **row))

    json.dump(res, open(out / f"exp3_{tag}.json", "w"), default=float)
    log("all done")


if __name__ == "__main__":
    main()
