"""事后（全信息）离线上界：时间扩展网络上的线性规划。

给定一天的全部订单与司机初始位置（事后已知），在批次网格 τ_t 上求“最多能赚多少”：
  节点 (j,t)：订单 j 在批次 t 被派（m_j ≤ τ_t ≤ m_j + 耐心）；
  边 司机 d → (j,t)：|u_d − o_j|₁ ≤ R；
  边 (j,t) → (k,t')：|dest_j − o_k|₁ ≤ R 且 τ_t + (pk_lb + ℓ_j)/v ≤ τ_t'。
pk_lb = 0 时（接驾时间松弛为 0）为任意在线/离线策略收入的**上界**（LP 松弛进一步放大，仍为上界）；
pk_lb = R 时每条链都可执行，整数解为可行的事后计划（下界）。
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from .data import PICKUP_RADIUS
from .sim import KM_PER_MIN


def hindsight(cfg, episode, pk_lb=0.0, integral=False, time_limit=300, max_wait=None, return_plan=False):
    orders, drv = episode
    m, o, dd, L, fare = orders["minute"], orders["oxy"], orders["dxy"], orders["trip_km"], orders["fare"]
    taus = np.array([cfg.tau(t) for t in range(cfg.T)])
    nodes_j, nodes_t = [], []
    for j in range(len(m)):
        ts = np.where((taus >= m[j] - 1e-9) & (taus - m[j] <= cfg.patience_min + 1e-9))[0]
        nodes_j += [j] * len(ts)
        nodes_t += list(ts)
    nj, nt = np.array(nodes_j), np.array(nodes_t)
    N = len(nj)
    ntau = taus[nt]
    free_lb = ntau + (pk_lb + L[nj]) / KM_PER_MIN
    order_idx = np.argsort(ntau, kind="stable")
    sorted_tau = ntau[order_idx]
    src, dst = [], []
    for u in range(N):
        lo = np.searchsorted(sorted_tau, free_lb[u] - 1e-9)
        cand = order_idx[lo:]
        if max_wait is not None:
            cand = cand[ntau[cand] <= free_lb[u] + max_wait]
        if len(cand) == 0:
            continue
        ok = np.abs(dd[nj[u]] - o[nj[cand]]).sum(-1) <= PICKUP_RADIUS + 1e-9
        cand = cand[ok]
        src += [u] * len(cand)
        dst += list(cand)
    E1s, E1d = np.array(src, int), np.array(dst, int)
    # 司机起点边
    ds, dn = [], []
    for d in range(len(drv)):
        ok = np.where(np.abs(drv[d] - o[nj]).sum(-1) <= PICKUP_RADIUS + 1e-9)[0]
        ds += [d] * len(ok)
        dn += list(ok)
    E0s, E0d = np.array(ds, int), np.array(dn, int)
    n1, n0 = len(E1s), len(E0s)
    nv = n0 + n1
    c = -np.concatenate([fare[nj[E0d]], fare[nj[E1d]]])
    # 流守恒：out(u) − in(u) ≤ 0
    rows = np.concatenate([E1s, E0d, E1d])
    cols = np.concatenate([n0 + np.arange(n1), np.arange(n0), n0 + np.arange(n1)])
    vals = np.concatenate([np.ones(n1), -np.ones(n0), -np.ones(n1)])
    A_flow = sp.csr_matrix((vals, (rows, cols)), shape=(N, nv))
    # 每个订单至多被服务一次：Σ_t in(j,t) ≤ 1
    in_node = np.concatenate([E0d, E1d])
    A_ord = sp.csr_matrix((np.ones(nv), (nj[in_node], np.arange(nv))), shape=(len(m), nv))
    # 每个司机至多一条链
    A_drv = sp.csr_matrix((np.ones(n0), (E0s, np.arange(n0))), shape=(len(drv), nv))
    A = sp.vstack([A_flow, A_ord, A_drv]).tocsr()
    ub = np.concatenate([np.zeros(N), np.ones(len(m)), np.ones(len(drv))])
    if integral:
        r = milp(c, constraints=LinearConstraint(A, -np.inf, ub), bounds=Bounds(0, 1),
                 integrality=np.ones(nv), options=dict(time_limit=time_limit, mip_rel_gap=1e-3))
        val = -r.fun if r.x is not None else np.nan
        bound = -getattr(r, "mip_dual_bound", np.nan) if r.x is not None else np.nan
    else:
        r = linprog(c, A_ub=A, b_ub=ub, bounds=(0, 1), method="highs")
        val, bound = -r.fun, -r.fun
    served = (-c @ r.x) if r.x is not None else np.nan
    x = r.x
    n_served = float(np.concatenate([np.ones(n0), np.ones(n1)]) @ x) if x is not None else np.nan
    out = dict(value=val, bound=bound, n_nodes=N, n_edges=nv, n_served=n_served, status=r.status)
    if return_plan and x is not None:
        # 由整数解还原每个司机的订单链 [(订单, 批次), ...]
        nxt = {E1s[i]: E1d[i] for i in np.where(x[n0:] > 0.5)[0]}
        plan = {}
        for i in np.where(x[:n0] > 0.5)[0]:
            d, u, chain = E0s[i], E0d[i], []
            while u is not None:
                chain.append((int(nj[u]), int(nt[u])))
                u = nxt.get(u)
            plan[int(d)] = chain
        out["plan"] = plan
    return out


def replay_policy(plan):
    """把事后计划变成仿真器策略：批次 t 时，空闲司机若计划在 t 接某单就派该单。用于验证计划可执行。"""
    want = {}
    for d, chain in plan.items():
        for j, t in chain:
            want[(t, j)] = d

    def pol(ctx):
        pairs = []
        idle, act = ctx["idle_ids"], ctx["order_ids"]
        pos = {d: a for a, d in enumerate(idle)}
        for b, j in enumerate(act):
            d = want.get((ctx["t"], j))
            if d is not None and d in pos and ctx["feas"][pos[d], b]:
                pairs.append((pos[d], b))
        return pairs
    return pol
