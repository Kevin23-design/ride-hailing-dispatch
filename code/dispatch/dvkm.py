"""DV-KM：连续时空深度价值网络 + KM（参考 Tang et al. KDD'19 CVNet）。

V_θ(τ, x, y)：
  - 时间：归一化时刻 + 多频周期编码；
  - 空间：多尺度层次粗编码（hierarchical coarse coding，4/8/16/32 网格的嵌入拼接）+ 归一化坐标；
  - 训练目标：半马尔可夫 TD(0) 策略评估  y = r + γ^{Δτ} V_θ⁻(τ', x', y')（不取 max，避免过估计），
    目标网络 θ⁻ 以 Polyak 平均更新；
  - 平滑正则：随机空间扰动下 |V(s) − V(s+δ)| 的平方惩罚（有限差分 Lipschitz 正则）。
派单边权：w = fare + β[γ^{Δ_ij} V(τ+Δ_ij, 目的地) − γ^{Δt} V(τ+Δt, 司机当前位置)]，Δ_ij 为该司机-订单对的真实忙碌分钟数。
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .data import W_KM, H_KM
from .sim import DAY_START, DAY_END, KM_PER_MIN
from .rl import km_max

GRIDS = (4, 8, 16, 32)


class ValueNet(nn.Module):
    def __init__(self, coarse=True, emb=24, hidden=256, n_freq=4):
        super().__init__()
        self.coarse, self.n_freq = coarse, n_freq
        if coarse:
            # 每个尺度的空间网格 × 3 小时段，嵌入表
            self.embs = nn.ModuleList([nn.Embedding(g * g * 6, emb) for g in GRIDS])
        d_in = 2 + 1 + 2 * n_freq + (emb * len(GRIDS) if coarse else 0)
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, tau, x, y):
        tn = (tau - DAY_START) / (DAY_END - DAY_START)
        xn, yn = x / W_KM, y / H_KM
        k = torch.arange(1, self.n_freq + 1, device=tau.device, dtype=tau.dtype)
        ang = 2 * math.pi * tn[:, None] * k[None]
        feats = [xn[:, None], yn[:, None], tn[:, None], torch.sin(ang), torch.cos(ang)]
        if self.coarse:
            seg = torch.clamp((tn * 6).long(), 0, 5)
            for g, E in zip(GRIDS, self.embs):
                c = torch.clamp((xn * g).long(), 0, g - 1)
                r = torch.clamp((yn * g).long(), 0, g - 1)
                feats.append(E((seg * g + r) * g + c))
        return self.mlp(torch.cat(feats, -1)).squeeze(-1)


def transitions_to_array(trans):
    """[t,z,a,r,t2,z2,avail,x,y,tau,x2,y2,tau2] -> float32 数组 (x,y,tau,r,x2,y2,tau2)。"""
    a = np.array([[tr[7], tr[8], tr[9], tr[3], tr[10], tr[11], tr[12]] for tr in trans], np.float32)
    return a


def train_value_net(data, gamma_min, coarse=True, smooth=1.0, epochs=30, batch=8192, lr=1e-3,
                    polyak=0.01, device="cuda", net=None, seed=0, log=None):
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net = (net or ValueNet(coarse=coarse)).to(dev)
    tgt = ValueNet(coarse=coarse).to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.tensor(data, device=dev)
    n = len(X)
    lg = math.log(gamma_min)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            b = X[perm[i:i + batch]]
            x, y, tau, r, x2, y2, tau2 = b.T
            with torch.no_grad():
                nv = tgt(tau2, x2, y2)
                nv = torch.where(tau2 >= DAY_END, torch.zeros_like(nv), nv)
                target = r + torch.exp(lg * (tau2 - tau)) * nv
            v = net(tau, x, y)
            loss = nn.functional.smooth_l1_loss(v, target, beta=5.0)
            if smooth > 0:
                dx = torch.randn_like(x) * 0.5
                dy = torch.randn_like(y) * 0.5
                v2 = net(tau, (x + dx).clamp(0, W_KM), (y + dy).clamp(0, H_KM))
                loss = loss + smooth * ((v - v2) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, q in zip(net.parameters(), tgt.parameters()):
                    q.mul_(1 - polyak).add_(polyak * p)
            tot += loss.item() * len(b)
        hist.append(tot / n)
        if log:
            log(f"  epoch {ep} loss {hist[-1]:.4f}")
    return net.cpu().eval(), hist


def make_dv_km(net, cfg, beta=0.5):
    g = cfg.gamma_min

    @torch.no_grad()
    def policy(ctx):
        P, feas = ctx["pickup"], ctx["feas"]
        n, m = P.shape
        tau = ctx["tau"]
        dur = (P + ctx["trip_km"][None, :]) / KM_PER_MIN                  # (n, m) 忙碌分钟
        arr = tau + dur
        dxy = np.broadcast_to(ctx["dest_xy"][None], (n, m, 2))
        ii, jj = np.nonzero(feas)
        if len(ii) == 0:
            return []
        t_in = torch.tensor(arr[ii, jj], dtype=torch.float32)
        v_after = net(t_in, torch.tensor(dxy[ii, jj, 0], dtype=torch.float32),
                      torch.tensor(dxy[ii, jj, 1], dtype=torch.float32)).numpy()
        v_after = np.where(arr[ii, jj] >= DAY_END, 0.0, v_after)
        tw = torch.full((n,), float(tau + cfg.step_min))
        dxy0 = ctx["drv_xy"]
        v_wait = net(tw, torch.tensor(dxy0[:, 0], dtype=torch.float32),
                     torch.tensor(dxy0[:, 1], dtype=torch.float32)).numpy()
        if tau + cfg.step_min >= DAY_END:
            v_wait[:] = 0
        W = np.zeros((n, m))
        W[ii, jj] = ctx["fare"][jj] + beta * (g ** dur[ii, jj] * v_after - g ** cfg.step_min * v_wait[ii])
        return km_max(W)

    return policy
