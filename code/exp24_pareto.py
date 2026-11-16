"""实验二 + 实验四：双目标（max 送客距离, min 接驾距离）帕累托前沿。

20 个批次（8/13/18/23 点 × 5 个随机批次，40 司机 × 80 订单，距离量化到 10 m）。
每批：ε-约束精确前沿（真值）；NSGA-II / NSGA-III / KM 混合初始化 NSGA-II 各 5 个种子；加权和 KM（41 个 λ）。
指标以精确前沿的理想点/最差点归一化：HV 比（HV/HV*）、IGD、IGD+、Spacing、解数。
"""
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dispatch.data import load_all, ROOT
from dispatch.moo import (DispatchProblem, make_batch, round_batch, run_nsga, das_dennis, km_weighted_sum,
                          hypervolume, igd, igd_plus, spacing, exact_front, supported_mask, nondominated,
                          knee_point)

HOURS = [8, 13, 18, 23]
BATCH_SEEDS = [0, 1, 2, 3, 4]
NSGA_SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = np.concatenate([[0.0], np.geomspace(0.05, 60, 40)])
REF2 = das_dennis(2, 99)
D = load_all()


def km_seed_genes(prob):
    _, genes = km_weighted_sum(prob, LAMBDAS, return_genes=True)
    _, idx = np.unique(genes, axis=0, return_index=True)
    genes = genes[np.sort(idx)]
    return genes[np.linspace(0, len(genes) - 1, min(12, len(genes)), dtype=int)]


def job(args):
    hour, bs = args
    batch = round_batch(make_batch(D["trips_train"], D["ck_train"], hour, seed=1000 * hour + bs))
    prob = DispatchProblem(batch)
    Z, xinfo = exact_front(prob)
    ideal, nadir = Z.min(0), Z.max(0)
    H = hypervolume(Z, ideal, nadir)
    sup = supported_mask(Z)
    rows, curves, fronts = [], [], {"exact": Z, "supported": Z[sup]}

    def score(name, s, F):
        F = nondominated(F)
        on_front = (np.abs(F[:, None] - Z[None]).max(-1) < 1e-6).any(1)
        rows.append(dict(hour=hour, batch=bs, method=name, seed=s, hv_ratio=hypervolume(F, ideal, nadir) / H,
                         igd=igd(F, Z, ideal, nadir), igd_plus=igd_plus(F, Z, ideal, nadir),
                         spacing=spacing(F, ideal, nadir) if len(F) > 2 else np.nan, n_points=len(F),
                         frac_exact=on_front.mean(), max_trip=-F[:, 0].min()))
        return F

    Fk = score("加权和 KM", 0, km_weighted_sum(prob, LAMBDAS))
    fronts["加权和 KM"] = Fk
    seeds = km_seed_genes(prob)
    for name, algo, init in [("NSGA-II", "nsga2", None), ("NSGA-III", "nsga3", None),
                             ("NSGA-II+KM 初始化", "nsga2", seeds)]:
        for s in NSGA_SEEDS:
            _, F, hist = run_nsga(prob, algo, pop_size=100, n_gen=300, seed=s, ref_dirs=REF2,
                                  log_every=10, initial_genes=init)
            F = score(name, s, F)
            if s == 0:
                fronts[name] = F
            curves.append(dict(hour=hour, batch=bs, method=name, seed=s,
                               gen=[g for g, _ in hist], hv=[hypervolume(h, ideal, nadir) / H for _, h in hist]))
    info = dict(hour=hour, batch=bs, n_exact=len(Z), n_supported=int(sup.sum()), **xinfo,
                n_feasible_orders=int((prob.n_cand > 0).sum()), knee=Z[knee_point(Z)].tolist())
    return rows, curves, info, fronts


if __name__ == "__main__":
    jobs = [(h, b) for h in HOURS for b in BATCH_SEEDS]
    with Pool(min(len(jobs), os.cpu_count())) as p:
        res = p.map(job, jobs, chunksize=1)
    out = ROOT / "results"
    pd.DataFrame([r for rr, _, _, _ in res for r in rr]).to_csv(out / "exp24_metrics.csv", index=False)
    json.dump([c for _, cc, _, _ in res for c in cc], open(out / "exp24_curves.json", "w"))
    pd.DataFrame([i for _, _, i, _ in res]).to_csv(out / "exp24_info.csv", index=False)
    pickle.dump({(i["hour"], i["batch"]): f for _, _, i, f in res}, open(out / "exp24_fronts.pkl", "wb"))
    df = pd.read_csv(out / "exp24_metrics.csv")
    print(df.groupby("method")[["hv_ratio", "igd", "igd_plus", "spacing", "n_points", "frac_exact"]].mean().round(4))
    print(pd.read_csv(out / "exp24_info.csv"))
