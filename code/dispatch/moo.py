"""多目标派单：批次构造、NSGA-II/III、加权和 KM、指标、ε-约束精确前沿。"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import PICKUP_RADIUS

def make_batch(trip_pool, ck_train, hour, n_orders=80, n_drivers=40, seed=0):
    """从历史订单中抽取某小时的一个派单批次；司机分布在签到热点附近（模拟巡游出租车）"""
    rng = np.random.default_rng(seed)
    pool = trip_pool[trip_pool.minute // 60 == hour]
    od = pool.sample(n_orders, replace=len(pool) < n_orders, random_state=seed)
    hot = ck_train[ck_train.minute // 60 == hour][["x", "y"]].values
    drv = hot[rng.integers(0, len(hot), n_drivers)] + rng.normal(0, 1.0, (n_drivers, 2))
    o_xy = od[["ox", "oy"]].values
    pickup = np.abs(drv[:, None, :] - o_xy[None, :, :]).sum(-1)   # (n, m) 接驾曼哈顿距离
    return dict(drv=drv, o_xy=o_xy, d_xy=od[["dx", "dy"]].values,
                fare=od["fare"].values, trip_km=od["trip_km"].values, pickup=pickup)


class DispatchProblem:
    """把一个批次包装成多目标优化问题：基因 -> 匹配 -> 目标向量"""

    def __init__(self, batch, K=6, radius=PICKUP_RADIUS, n_obj=2, revenue="trip_km"):
        # revenue="trip_km"：收入用送客距离表示（作业要求）；"fare"：车费口径
        self.fare, self.pickup = batch[revenue], batch["pickup"]
        self.n_drivers, self.n_var = self.pickup.shape
        self.K, self.n_obj = K, n_obj
        order_view = self.pickup.T                              # (m, n)
        cand = np.argsort(order_view, axis=1)[:, :K]           # 每个订单最近的 K 个司机
        ok = np.take_along_axis(order_view, cand, 1) <= radius
        self.cand = np.where(ok, cand, -1)                      # -1 表示该候选不可行
        self.n_cand = ok.sum(1)                                 # 每个订单的可行候选数

    def decode(self, gene):
        """基因 -> [(driver, order), ...]，冲突时顺延到下一个候选司机"""
        used = np.zeros(self.n_drivers, bool)
        pairs = []
        for j, g in enumerate(gene):
            c = self.n_cand[j]
            if g == 0 or c == 0:
                continue
            for s in range(c):
                d = self.cand[j, (g - 1 + s) % c]
                if not used[d]:
                    used[d] = True
                    pairs.append((d, j))
                    break
        return pairs

    def evaluate_one(self, gene):
        pairs = self.decode(gene)
        if not pairs:
            return np.zeros(self.n_obj)
        d, j = np.array(pairs).T
        f = [-self.fare[j].sum(), self.pickup[d, j].sum()]
        if self.n_obj == 3:
            f.append(-len(pairs))                               # 第三目标：最大化服务订单数
        return np.array(f, float)

    def evaluate(self, pop):
        return np.array([self.evaluate_one(g) for g in pop])

    def random_population(self, size, rng):
        pop = rng.integers(0, self.K + 1, (size, self.n_var))
        pop[0] = 1                                              # 种子个体：全部派给最近司机
        pop[1] = 0                                              # 种子个体：全部不派（f=0 的端点）
        return pop



def fast_non_dominated_sort(F):
    """返回 rank 数组（0 = 第一层前沿）"""
    N = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt                          # dom[a, b] = a 支配 b
    n_dominated_by = dom.sum(0)            # 每个个体被多少个体支配
    rank = np.full(N, -1)
    current = np.where(n_dominated_by == 0)[0]
    r = 0
    while len(current):
        rank[current] = r
        n_dominated_by = n_dominated_by - dom[current].sum(0)
        n_dominated_by[rank >= 0] = -1     # 已分层的个体不再参与
        current = np.where(n_dominated_by == 0)[0]
        r += 1
    return rank


def crowding_distance(F):
    N, M = F.shape
    cd = np.zeros(N)
    if N <= 2:
        return np.full(N, np.inf)
    for k in range(M):
        idx = np.argsort(F[:, k])
        span = F[idx[-1], k] - F[idx[0], k]
        cd[idx[0]] = cd[idx[-1]] = np.inf
        if span > 0:
            cd[idx[1:-1]] += (F[idx[2:], k] - F[idx[:-2], k]) / span
    return cd


def tournament(rank, cd, n, rng):
    a, b = rng.integers(0, len(rank), (2, n))
    better_a = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
    return np.where(better_a, a, b)


def variation(parents, K, rng, pc=0.9, pm=None):
    """均匀交叉 + 随机重置变异"""
    n, L = parents.shape
    pm = 1.0 / L if pm is None else pm
    p1, p2 = parents[: n // 2], parents[n // 2: n // 2 * 2]
    mask = (rng.random(p1.shape) < 0.5) & (rng.random((len(p1), 1)) < pc)
    c1, c2 = np.where(mask, p2, p1), np.where(mask, p1, p2)
    kids = np.vstack([c1, c2])
    mut = rng.random(kids.shape) < pm
    kids[mut] = rng.integers(0, K + 1, mut.sum())
    return kids


def nsga2_survival(F, N):
    rank = fast_non_dominated_sort(F)
    chosen = []
    for r in range(rank.max() + 1):
        front = np.where(rank == r)[0]
        if len(chosen) + len(front) <= N:
            chosen.extend(front)
        else:
            cd = crowding_distance(F[front])
            chosen.extend(front[np.argsort(-cd)[: N - len(chosen)]])
            break
    return np.array(chosen)


def run_nsga(problem, algo="nsga2", pop_size=100, n_gen=300, seed=0, ref_dirs=None, log_every=10, initial_genes=None):
    """通用主循环：algo ∈ {'nsga2', 'nsga3'}；返回最终种群、目标值和每代前沿历史"""
    rng = np.random.default_rng(seed)
    pop = problem.random_population(pop_size, rng)
    if initial_genes is not None:
        # 保留两个模板个体，用少量 KM 解替换部分随机个体。
        n_seeds = min(len(initial_genes), pop_size - 2)
        pop[2:2 + n_seeds] = initial_genes[:n_seeds]
    F = problem.evaluate(pop)
    history = []
    for gen in range(n_gen):
        if algo == "nsga2":
            rank = fast_non_dominated_sort(F)
            cd = np.zeros(len(F))
            for r in np.unique(rank):
                cd[rank == r] = crowding_distance(F[rank == r])
            parents = pop[tournament(rank, cd, pop_size, rng)]
        else:                                                 # NSGA-III 原文：随机配对
            parents = pop[rng.integers(0, pop_size, pop_size)]
        kids = variation(parents, problem.K, rng)
        pop_all, F_all = np.vstack([pop, kids]), np.vstack([F, problem.evaluate(kids)])
        if algo == "nsga2":
            keep = nsga2_survival(F_all, pop_size)
        else:
            keep = nsga3_survival(F_all, pop_size, ref_dirs, rng)
        pop, F = pop_all[keep], F_all[keep]
        if gen % log_every == 0 or gen == n_gen - 1:
            history.append((gen, F[fast_non_dominated_sort(F) == 0].copy()))
    front = fast_non_dominated_sort(F) == 0
    return pop[front], F[front], history

def das_dennis(n_obj, n_partitions):
    """Das-Dennis 均匀参考方向"""
    def rec(left, depth):
        if depth == n_obj - 1:
            return [[left]]
        return [[i] + rest for i in range(left + 1) for rest in rec(left - i, depth + 1)]
    return np.array(rec(n_partitions, 0), float) / n_partitions


def normalize(F, ideal):
    """NSGA-III 自适应归一化：极值点 -> 超平面截距；失败时退化为 nadir 归一化"""
    Fp = F - ideal
    M = F.shape[1]
    w = np.eye(M) + 1e-6
    asf = (Fp[:, None, :] / w[None, :, :]).max(-1)        # (N, M)
    extreme = Fp[asf.argmin(0)]                            # 每个轴的极值点
    try:
        b = np.linalg.solve(extreme, np.ones(M))
        intercept = 1.0 / b
        if np.any(intercept <= 1e-6) or not np.all(np.isfinite(intercept)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercept = Fp.max(0)
    intercept = np.where(intercept <= 1e-6, Fp.max(0) + 1e-6, intercept)
    return Fp / intercept


def associate(Fn, ref_dirs):
    """返回每个个体最近的参考方向编号及垂直距离"""
    u = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    proj = np.einsum("nk,rk->nr", Fn, u)                  # (N, R) 投影长度
    perp = np.sqrt(np.maximum((Fn ** 2).sum(1, keepdims=True) - proj ** 2, 0))
    niche = perp.argmin(1)
    return niche, perp[np.arange(len(Fn)), niche]


def nsga3_survival(F, N, ref_dirs, rng):
    rank = fast_non_dominated_sort(F)
    chosen, last = [], None
    for r in range(rank.max() + 1):
        front = np.where(rank == r)[0]
        if len(chosen) + len(front) <= N:
            chosen.extend(front)
            if len(chosen) == N:
                return np.array(chosen)
        else:
            last = front
            break
    if last is None:
        return np.array(chosen)
    cand = np.array(chosen + list(last))
    Fn = normalize(F[cand], F[cand].min(0))
    niche, dist = associate(Fn, ref_dirs)
    k = len(chosen)
    niche_count = np.bincount(niche[:k], minlength=len(ref_dirs))
    pool = list(range(k, len(cand)))                       # 最后一层在 cand 中的下标
    excluded = np.zeros(len(ref_dirs), bool)
    while len(chosen) < N:
        counts = np.where(excluded, np.inf, niche_count)
        j_min = rng.choice(np.where(counts == counts.min())[0])
        members = [p for p in pool if niche[p] == j_min]
        if not members:
            excluded[j_min] = True
            continue
        pick = members[int(np.argmin(dist[members]))] if niche_count[j_min] == 0 else rng.choice(members)
        chosen.append(cand[pick])
        pool.remove(pick)
        niche_count[j_min] += 1
    return np.array(chosen)



def hv2d(F, ref):
    F = F[(F < ref).all(1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    hv, best_f2 = 0.0, ref[1]
    for f1, f2 in F:
        if f2 < best_f2:
            hv += (ref[0] - f1) * (best_f2 - f2)
            best_f2 = f2
    return hv


def hv3d(F, ref):
    F = F[(F < ref).all(1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    for i in range(len(F)):
        top = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        hv += hv2d(F[: i + 1, :2], ref[:2]) * (top - F[i, 2])
    return hv


def hypervolume(F, ideal, nadir):
    Fn = (F - ideal) / np.maximum(nadir - ideal, 1e-12)
    ref = np.full(F.shape[1], 1.1)
    return hv2d(Fn, ref) if F.shape[1] == 2 else hv3d(Fn, ref)


def km_weighted_sum(problem, lambdas, return_genes=False):
    """加权和 + KM：每个 λ 返回一个目标向量（只在与 NSGA 相同的候选边上匹配）"""
    feas = np.zeros_like(problem.pickup, bool)
    for j in range(problem.n_var):
        c = problem.cand[j][problem.cand[j] >= 0]
        feas[c, j] = True
    out, genes = [], []
    for lam in lambdas:
        W = np.where(feas, problem.fare[None, :] - lam * problem.pickup, 0.0)
        W = np.maximum(W, 0.0)                         # 负收益的边等价于"不派"
        r, c = linear_sum_assignment(-W)
        keep = W[r, c] > 0
        d, j = r[keep], c[keep]
        f = [-problem.fare[j].sum(), problem.pickup[d, j].sum()]
        if problem.n_obj == 3:
            f.append(-keep.sum())
        out.append(f)
        if return_genes:
            # KM 的匹配已经一一对应，将司机编号还原为各订单的候选名次。
            gene = np.zeros(problem.n_var, dtype=int)
            gene[j] = np.argmax(problem.cand[j] == d[:, None], axis=1) + 1
            genes.append(gene)
    return (np.array(out), np.array(genes)) if return_genes else np.array(out)


def knee_point(F):
    """离"两端点连线"最远的点：边际收益突变处，常作为折中解推荐给决策者"""
    Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
    a, b = Fn[Fn[:, 0].argmin()], Fn[Fn[:, 0].argmax()]
    v, w = b - a, Fn - a
    d = np.abs(v[0] * w[:, 1] - v[1] * w[:, 0]) / (np.linalg.norm(v) + 1e-12)
    return int(d.argmax())



def spacing(F, ideal, nadir):
    """Spacing 指标：归一化后每个解到最近邻的距离的变异系数，越小越均匀"""
    Fn = np.unique((F - ideal) / np.maximum(nadir - ideal, 1e-12), axis=0)
    D = np.abs(Fn[:, None] - Fn[None]).sum(-1)
    np.fill_diagonal(D, np.inf)
    d = D.min(1)
    return d.std() / d.mean()




# ---------------------------------------------------------------------------
# 指标：IGD / IGD+ / 非支配过滤
# ---------------------------------------------------------------------------
def nondominated(F):
    F = np.unique(np.round(F, 9), axis=0)
    return F[fast_non_dominated_sort(F) == 0]


def igd(A, Z, ideal, nadir):
    s = np.maximum(nadir - ideal, 1e-12)
    An, Zn = (A - ideal) / s, (Z - ideal) / s
    return float(np.linalg.norm(Zn[:, None] - An[None], axis=-1).min(1).mean())


def igd_plus(A, Z, ideal, nadir):
    s = np.maximum(nadir - ideal, 1e-12)
    An, Zn = (A - ideal) / s, (Z - ideal) / s
    d = np.sqrt((np.maximum(An[None] - Zn[:, None], 0) ** 2).sum(-1))
    return float(d.min(1).mean())


def supported_mask(F):
    """2 目标最小化前沿上的支撑解（位于下凸包上）。F 需为非支配集合。"""
    order = np.argsort(F[:, 0])
    P = F[order]
    hull = []
    for i, p in enumerate(P):
        while len(hull) >= 2:
            a, b = P[hull[-2]], P[hull[-1]]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross <= 1e-12:      # b 不在下凸包上
                hull.pop()
            else:
                break
        hull.append(i)
    mask = np.zeros(len(F), bool)
    mask[order[hull]] = True
    return mask


# ---------------------------------------------------------------------------
# ε-约束精确帕累托前沿（HiGHS MILP）
# ---------------------------------------------------------------------------
def round_batch(batch, res=0.01):
    """把距离量化到 res km（默认 10 m），使 ε-约束可用整数字典序目标精确求解。"""
    b = dict(batch)
    for k in ("pickup", "trip_km"):
        b[k] = np.round(batch[k] / res) * res
    return b


def exact_front(problem, res=0.01, max_points=20000, time_limit=120):
    """在与 NSGA 相同的候选边集合上精确枚举 (max 送客距离, min 接驾距离) 的全部非支配点。

    距离已量化为 res 的整数倍，记整数化 G=g/res、P=p/res，M=ΣP_max+1。
    单次求解 max M·G·x − P·x  s.t. 匹配约束, P·x ≤ ε 即为字典序（先最大收入、再最小接驾），
    得到的点必为帕累托最优；令 ε ← P* − 1 迭代至收入为 0，即得全部非支配点（含非支撑解）。
    返回 F=(−收入, 接驾)。
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix

    edges = [(int(d), j) for j in range(problem.n_var) for d in problem.cand[j] if d >= 0]
    if not edges:
        return np.zeros((1, 2)), 0
    d_idx = np.array([e[0] for e in edges])
    o_idx = np.array([e[1] for e in edges])
    G = np.round(problem.fare[o_idx] / res)
    P = np.round(problem.pickup[d_idx, o_idx] / res)
    nE = len(edges)
    rows = np.r_[d_idx, problem.n_drivers + o_idx]
    A_match = coo_matrix((np.ones(2 * nE), (rows, np.r_[np.arange(nE), np.arange(nE)])),
                         shape=(problem.n_drivers + problem.n_var, nE)).tocsr()
    match = LinearConstraint(A_match, -np.inf, 1)
    integ, bnd = np.ones(nE), Bounds(0, 1)
    opts = dict(mip_rel_gap=0, time_limit=time_limit)
    M = float(np.sort(P)[::-1][:min(problem.n_drivers, problem.n_var)].sum() + 1)
    n_solve = [0]

    def lex(eps):
        """max M·G − P（字典序）在 P·x ≤ eps 下；返回 (G, P) 整数值或 None。"""
        cons = [match] + ([LinearConstraint(P[None], -np.inf, eps)] if np.isfinite(eps) else [])
        r = milp(-(M * G - P), constraints=cons, integrality=integ, bounds=bnd, options=opts)
        n_solve[0] += 1
        if r.x is None:
            return None
        x = np.round(r.x)
        return float(G @ x), float(P @ x)

    def h(eps):
        """h(ε) = max G·x s.t. P·x ≤ ε（纯整数小系数目标，数值可靠，用于验证）。"""
        if eps < 0:
            return 0.0
        cons = [match, LinearConstraint(P[None], -np.inf, eps)]
        r = milp(-G, constraints=cons, integrality=integ, bounds=bnd, options=opts)
        n_solve[0] += 1
        return float(G @ np.round(r.x))

    pts, eps = {}, np.inf
    while len(pts) < max_points:
        sol = lex(eps)
        if sol is None or sol[0] <= 0:
            break
        pts[sol[1]] = sol[0]
        eps = sol[1] - 1
    pts[0.0] = 0.0
    # 验证与修复：前沿点集合 {(g_i, p_i)} 精确 ⇔ 对每个点 h(p_i)=g_i，且相邻点间 h(p_{i+1}−1)=g_i。
    n_repair = 0
    for _ in range(50):
        F = nondominated(np.array([(-g, p) for p, g in pts.items()]))
        F = F[np.argsort(F[:, 1])]
        bad = False
        for i in range(len(F)):
            g_i, p_i = -F[i, 0], F[i, 1]
            hv = h(p_i)
            if hv > g_i + 0.5:                      # 该点收入非最优：补求
                sol = lex(p_i); pts[sol[1]] = sol[0]; bad = True; n_repair += 1
            if i + 1 < len(F):
                hv2 = h(F[i + 1, 1] - 1)
                if hv2 > g_i + 0.5:                 # 相邻点之间漏点：补求
                    sol = lex(F[i + 1, 1] - 1); pts[sol[1]] = sol[0]; bad = True; n_repair += 1
        if not bad:
            break
    F = nondominated(np.array([(-g * res, p * res) for p, g in pts.items()]))
    return F, dict(n_solve=n_solve[0], n_repair=n_repair, verified=not bad)
