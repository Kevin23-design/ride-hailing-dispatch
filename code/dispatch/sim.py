"""全天派单仿真器（参数化批次间隔 Δt）。

相对 notebook 版本的改动：
1. 时间片长度 step_min 可变（= 批次间隔 = MDP 时间片）；
2. 第 t 个批次在 τ_t = DAY_START + (t+1)·Δt 时刻执行，只包含已到达订单（去掉“未到先派”的前视）；
3. 司机忙碌时长按连续分钟计（接驾 + 送客）/ 车速，不再向上取整到时间片，
   保证 Δt 只改变“攒批”行为而不改变物理行驶时间；
4. 订单耐心按分钟（默认 20 min），超时取消。
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .data import BASE_FARE, PER_KM, PICKUP_RADIUS, W_KM, H_KM

DAY_START, DAY_END = 6 * 60, 24 * 60
SPEED_KMH = 25.0
KM_PER_MIN = SPEED_KMH / 60


@dataclass
class Config:
    step_min: float = 10.0
    grid: int = 8
    patience_min: float = 20.0
    gamma10: float = 0.92           # 每 10 分钟的折扣
    orders_per_day: int = 1500

    def __post_init__(self):
        self.T = int(round((DAY_END - DAY_START) / self.step_min))
        self.n_zones = self.grid * self.grid
        self.WAIT = self.n_zones
        self.gamma = self.gamma10 ** (self.step_min / 10)          # 每个时间片的折扣
        self.gamma_min = self.gamma10 ** (1 / 10)                   # 每分钟折扣
        centers = np.array([((c + 0.5) * W_KM / self.grid, (r + 0.5) * H_KM / self.grid)
                            for r in range(self.grid) for c in range(self.grid)])
        dist = np.abs(centers[:, None] - centers[None]).sum(-1)
        self.centers = centers
        self.DELTA = np.maximum(1, np.ceil(dist / KM_PER_MIN / self.step_min)).astype(int)

    def tau(self, t):
        """第 t 个批次的执行时刻（分钟）。"""
        return DAY_START + (t + 1) * self.step_min

    def zone_of(self, x, y):
        col = np.clip((np.asarray(x) / W_KM * self.grid).astype(int), 0, self.grid - 1)
        row = np.clip((np.asarray(y) / H_KM * self.grid).astype(int), 0, self.grid - 1)
        return row * self.grid + col

    def slice_free(self, free_min, t):
        """司机在 free_min 空闲，返回其参与的第一个批次编号（> t）。"""
        s = int(math.ceil((free_min - DAY_START) / self.step_min - 1 - 1e-9))
        return max(t + 1, s)


def morning_drivers(ck_train):
    return ck_train.loc[ck_train.minute.between(360, 479), ["x", "y"]].values


def sample_episode(trip_pool, morning, n_drivers, seed, orders_per_day=1500):
    """一天订单：按历史时间分布有放回抽样 + 小幅时空扰动；司机初始位置取早高峰签到热点。"""
    rng = np.random.default_rng(seed)
    pool = trip_pool[(trip_pool.minute >= DAY_START) & (trip_pool.minute < DAY_END)]
    od = pool.iloc[rng.integers(0, len(pool), orders_per_day)]
    minute = np.clip(od.minute.values + rng.uniform(-5, 5, orders_per_day), DAY_START, DAY_END - 1e-3)
    jit = lambda v, hi: np.clip(v + rng.normal(0, 0.3, len(v)), 0, hi)
    ox, oy = jit(od.ox.values, W_KM), jit(od.oy.values, H_KM)
    dx, dy = jit(od.dx.values, W_KM), jit(od.dy.values, H_KM)
    order = np.argsort(minute, kind="stable")
    trip_km = np.abs(ox - dx) + np.abs(oy - dy)
    orders = dict(minute=minute[order], oxy=np.c_[ox, oy][order], dxy=np.c_[dx, dy][order],
                  trip_km=trip_km[order], fare=(BASE_FARE + PER_KM * trip_km)[order])
    drv = morning[rng.integers(0, len(morning), n_drivers)] + rng.normal(0, 0.5, (n_drivers, 2))
    drv = np.clip(drv, 0, [W_KM, H_KM])
    return orders, drv


def run_episode(cfg: Config, episode, policy, collect=False):
    """按批次推进一天。

    policy(ctx) -> [(空闲司机局部下标, 订单局部下标), ...]
    collect=True 时返回经验列表，每条为
      [t, z, a, r, t2, z2, avail, x, y, tau, x2, y2, tau2, driver]
    avail = 司机下一次参与批次时 3 km 内订单的目的地区域集合 A(s')（下一次空闲时回填）。
    """
    orders, drv0 = episode
    dzone = cfg.zone_of(orders["dxy"][:, 0], orders["dxy"][:, 1])
    pos = drv0.copy()
    zone = cfg.zone_of(pos[:, 0], pos[:, 1])
    free_min = np.full(len(pos), float(DAY_START))
    active = np.zeros(0, int)
    ptr, n_orders = 0, len(orders["minute"])
    revenue = pickup_km = trip_sum = wait_sum = 0.0
    served = 0
    hourly = np.zeros(24)
    trans, pending = [], {}
    no_avail = np.zeros(0, int)
    for t in range(cfg.T):
        tau = cfg.tau(t)
        end = ptr
        while end < n_orders and orders["minute"][end] <= tau:
            end += 1
        active = np.concatenate([active, np.arange(ptr, end)])
        ptr = end
        active = active[tau - orders["minute"][active] <= cfg.patience_min]
        idle = np.where(free_min <= tau + 1e-9)[0]
        pairs, ctx = [], None
        if len(idle) and len(active):
            P = np.abs(pos[idle][:, None] - orders["oxy"][active][None]).sum(-1)
            ctx = dict(t=t, tau=tau, cfg=cfg, pickup=P, feas=P <= PICKUP_RADIUS,
                       fare=orders["fare"][active], trip_km=orders["trip_km"][active],
                       drv_zone=zone[idle], dest_zone=dzone[active],
                       drv_xy=pos[idle], dest_xy=orders["dxy"][active],
                       idle_ids=idle, order_ids=active)
            pairs = policy(ctx)
        if collect:
            for a, d in enumerate(idle):
                if d in pending:
                    pending.pop(d)[6] = (np.unique(ctx["dest_zone"][ctx["feas"][a]])
                                         if ctx is not None else no_avail)
        busy, done = set(), []
        for a, b in pairs:
            d, o = idle[a], active[b]
            pk = abs(pos[d, 0] - orders["oxy"][o, 0]) + abs(pos[d, 1] - orders["oxy"][o, 1])
            assert pk <= PICKUP_RADIUS + 1e-9, "派单违反接驾半径约束"
            dur_min = (pk + orders["trip_km"][o]) / KM_PER_MIN
            f = orders["fare"][o]
            t_free = tau + dur_min
            t2 = cfg.slice_free(t_free, t)
            if collect:
                rec = [t, zone[d], dzone[o], f, t2, dzone[o], no_avail,
                       pos[d, 0], pos[d, 1], tau, orders["dxy"][o, 0], orders["dxy"][o, 1], t_free, d]
                trans.append(rec)
                pending[d] = rec
            revenue += f
            pickup_km += pk
            trip_sum += orders["trip_km"][o]
            wait_sum += tau - orders["minute"][o] + pk / KM_PER_MIN
            hourly[int(min(tau, DAY_END - 1) // 60)] += f
            served += 1
            pos[d] = orders["dxy"][o]
            zone[d] = dzone[o]
            free_min[d] = t_free
            busy.add(d)
            done.append(b)
        if collect:
            for d in idle:
                if d not in busy:
                    rec = [t, zone[d], cfg.WAIT, 0.0, t + 1, zone[d], no_avail,
                           pos[d, 0], pos[d, 1], tau, pos[d, 0], pos[d, 1], tau + cfg.step_min, d]
                    trans.append(rec)
                    pending[d] = rec
        # 未被派的空闲司机原地等待到下一批次
        if len(done):
            active = np.delete(active, done)
    metrics = dict(revenue=revenue, trip_km=trip_sum, answer_rate=served / n_orders,
                   avg_pickup=pickup_km / max(served, 1), avg_response=wait_sum / max(served, 1),
                   served=served)
    metrics.update({f"h{h}": hourly[h] for h in range(6, 24)})
    return metrics, trans
