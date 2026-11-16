"""基线策略与表格 Q-KM（平台级 Q-learning，KDD'18 风格），参数化 Δt。"""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .sim import Config, run_episode, sample_episode

ALPHA_MIN = 0.05


def km_max(W):
    W = np.where(W > 0, W, 0.0)
    r, c = linear_sum_assignment(-W)
    keep = W[r, c] > 0
    return list(zip(r[keep], c[keep]))


def greedy_nearest(ctx):
    used, pairs = set(), []
    P, F = ctx["pickup"], ctx["feas"]
    for b in range(P.shape[1]):
        for a in np.argsort(P[:, b]):
            if not F[a, b]:
                break
            if a not in used:
                used.add(a)
                pairs.append((a, b))
                break
    return pairs


def km_fare(ctx):
    return km_max(np.where(ctx["feas"], ctx["fare"][None, :], 0.0))


def random_km(ctx, rng):
    return km_max(np.where(ctx["feas"], rng.random(ctx["feas"].shape) + 0.01, 0.0))


class QTable:
    def __init__(self, cfg: Config):
        T, Z = cfg.T, cfg.n_zones
        self.Q = np.zeros((T, Z, Z + 1))
        self.N = np.zeros((T, Z, Z + 1), np.int32)
        self.V = np.zeros((T + 1, Z))
        self.NV = np.zeros((T + 1, Z), np.int32)


def q_update(qt, cfg, t, z, a, r, t2, z2, avail, target="max", expectile=0.5):
    """target='max'：Q-learning（max 仅在 A(s')∪{WAIT} 上取）；
    target='expect'：表格策略评估 V(s)←V(s)+α(r+γ^Δ V(s')−V(s))（SARSA 式，无 max 偏差）。"""
    g = cfg.gamma
    if target == "expectile":
        # 期望分位数 TD（IQL 式）：τ=0.5 与 'expect' 完全一致，τ→1 趋向对后继回报取上确界（乐观）
        nv = qt.V[t2, z2] if t2 < cfg.T else 0.0
        y = r + g ** (t2 - t) * nv
        qt.NV[t, z] += 1
        alpha = max(1 / qt.NV[t, z], ALPHA_MIN)
        td = y - qt.V[t, z]
        # 非对称权重归一化到最大为 1（等价于对期望分位损失做步长重标定），保证 α·w ≤ 1 不发散；τ=0.5 时 w≡1
        wmax = max(expectile, 1 - expectile)
        w = (expectile if td > 0 else 1 - expectile) / wmax
        qt.V[t, z] += alpha * w * td
        return td
    if target == "expect":
        nv = qt.V[t2, z2] if t2 < cfg.T else 0.0
        y = r + g ** (t2 - t) * nv
        qt.NV[t, z] += 1
        alpha = max(1 / qt.NV[t, z], ALPHA_MIN)
        td = y - qt.V[t, z]
        qt.V[t, z] += alpha * td
        return td
    nv = 0.0
    if t2 < cfg.T:
        av = qt.Q[t2, z2]
        nv = av[cfg.WAIT]
        if len(avail):
            nv = max(nv, av[avail].max())
        qt.NV[t2, z2] += 1
        qt.V[t2, z2] += (nv - qt.V[t2, z2]) / qt.NV[t2, z2]
    y = r + g ** (t2 - t) * nv
    qt.N[t, z, a] += 1
    alpha = max(1 / qt.N[t, z, a], ALPHA_MIN)
    td = y - qt.Q[t, z, a]
    qt.Q[t, z, a] += alpha * td
    return td


def region_long_term_value(cfg, V):
    T = cfg.T
    lt = np.empty((T, cfg.n_zones, cfg.n_zones))
    dest = np.arange(cfg.n_zones)[None, :]
    disc = cfg.gamma ** cfg.DELTA
    for t in range(T):
        arr = np.minimum(t + cfg.DELTA, T)
        lt[t] = disc * V[arr, dest] - cfg.gamma * V[t + 1, :, None]
    return lt


def make_q_km(LT, beta=0.5):
    def q_km(ctx):
        lt = LT[ctx["t"], ctx["drv_zone"][:, None], ctx["dest_zone"][None, :]]
        return km_max(np.where(ctx["feas"], ctx["fare"][None, :] + beta * lt, 0.0))
    return q_km


def train_q_km(cfg, trips_train, morning, n_drivers, n_episodes=300, beta=0.5,
               eps_start=0.5, eps_end=0.05, seed=0, target="max", eval_eps=None, eval_every=25,
               verbose=False, expectile=0.5):
    qt = QTable(cfg)
    rng = np.random.default_rng(seed)
    curve, td_log = [], []
    t0 = time.time()
    for ep in range(n_episodes):
        eps = eps_start + (eps_end - eps_start) * ep / max(1, n_episodes - 1)
        q_pol = make_q_km(region_long_term_value(cfg, qt.V), beta)
        pol = lambda ctx: random_km(ctx, rng) if rng.random() < eps else q_pol(ctx)
        episode = sample_episode(trips_train, morning, n_drivers, seed=seed * 100_000 + ep,
                                 orders_per_day=cfg.orders_per_day)
        _, trans = run_episode(cfg, episode, pol, collect=True)
        trans.sort(key=lambda x: -x[0])
        td = [abs(q_update(qt, cfg, *tr[:7], target=target, expectile=expectile)) for tr in trans]
        td_log.append(float(np.mean(td)))
        if eval_eps is not None and (ep % eval_every == 0 or ep == n_episodes - 1):
            p = make_q_km(region_long_term_value(cfg, qt.V), beta)
            rev = np.mean([run_episode(cfg, e, p)[0]["revenue"] for e in eval_eps])
            curve.append((ep, rev))
            if verbose:
                print(f"ep {ep} eps={eps:.2f} |TD|={td_log[-1]:.2f} val={rev:.0f} {time.time()-t0:.0f}s", flush=True)
    return qt, np.array(curve), np.array(td_log)


def mc_bias(cfg, V, trans):
    """沿每个司机的真实轨迹回溯折扣回报 G，返回 (mean(V(t,z)), mean(G))，
    即价值表对“实际执行策略”回报的估计偏差。"""
    by = {}
    for tr in trans:
        by.setdefault(tr[13], []).append(tr)
    v_sum = g_sum = 0.0
    n = 0
    for recs in by.values():
        recs.sort(key=lambda x: x[0])
        G = 0.0
        for tr in reversed(recs):
            t, z, r, t2 = tr[0], tr[1], tr[3], tr[4]
            G = r + (cfg.gamma ** (t2 - t)) * (G if t2 < cfg.T else 0.0)
            v_sum += V[t, z]
            g_sum += G
            n += 1
    return v_sum / n, g_sum / n


def mc_state_returns(cfg, trans_list):
    """按状态 (t,z) 汇总沿真实轨迹回溯的折扣回报：返回 (sum_G, count) 两个 (T, Z) 数组。"""
    S = np.zeros((cfg.T, cfg.n_zones))
    N = np.zeros((cfg.T, cfg.n_zones))
    for trans in trans_list:
        by = {}
        for tr in trans:
            by.setdefault(tr[13], []).append(tr)
        for recs in by.values():
            recs.sort(key=lambda x: x[0])
            G = 0.0
            for tr in reversed(recs):
                t, z, r, t2 = tr[0], tr[1], tr[3], tr[4]
                G = r + (cfg.gamma ** (t2 - t)) * (G if t2 < cfg.T else 0.0)
                S[t, z] += G
                N[t, z] += 1
    return S, N


def bias_decomposition(V, S, N, min_n=30):
    """把价值误差 e(t,z)=V−Ĝ 分解为“同一时间片内的共同平移”与“区域间相对扭曲”。
    返回：总相对偏差、平移分量占比、区域间扭曲的 RMS（相对 Ĝ 均值）、时间片内 V 与 Ĝ 的 Spearman 均值。"""
    from scipy.stats import spearmanr
    ok = N >= min_n
    G = np.where(ok, S / np.maximum(N, 1), np.nan)
    E = np.where(ok, V[:G.shape[0]] - G, np.nan)
    w = np.where(ok, N, 0)
    gbar = np.nansum(G * w) / w.sum()
    total = np.nansum(E * w) / w.sum()
    level = np.array([np.nansum(E[t] * w[t]) / w[t].sum() if w[t].sum() else np.nan for t in range(len(E))])
    resid = E - level[:, None]
    distort = np.sqrt(np.nansum(resid ** 2 * w) / w.sum())
    rhos = []
    for t in range(len(E)):
        m = ok[t]
        if m.sum() >= 8:
            r = spearmanr(V[t][m], G[t][m]).correlation
            if np.isfinite(r):
                rhos.append(r)
    return dict(rel_bias=total / gbar, distort=distort / gbar, spearman=float(np.mean(rhos)),
                n_states=int(ok.sum()))
