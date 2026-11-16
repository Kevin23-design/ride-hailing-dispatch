"""从 results/ 生成报告全部图（PDF 矢量）与 LaTeX 表格片段（latex/tables/*.tex）。"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, TAB = ROOT / "results", ROOT / "latex" / "fig", ROOT / "latex" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "Hiragino Sans GB"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
    "axes.linewidth": 0.6, "xtick.color": "#55554f", "ytick.color": "#55554f",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": "#e4e3de", "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# 固定实体 -> 颜色（经 validate_palette.js 校验的默认分类色板，顺序固定、不循环）
C = {"贪心最近": "#2a78d6", "KM-即时收入": "#eb6834", "Q-KM": "#1baf7a",
     "NSGA-II": "#2a78d6", "NSGA-III": "#eb6834", "加权和 KM": "#1baf7a", "NSGA-II+KM 初始化": "#eda100",
     "Tab-Q-KM": "#1baf7a", "Tab-PE-KM": "#6250d6", "DV-KM": "#e87ba4", "精确前沿": "#3d3d3a"}
MK = {"贪心最近": "o", "KM-即时收入": "s", "Q-KM": "^", "NSGA-II": "o", "NSGA-III": "^",
      "加权和 KM": "s", "NSGA-II+KM 初始化": "D", "Tab-Q-KM": "^", "Tab-PE-KM": "v", "DV-KM": "D"}
INK2 = "#55554f"
W1, W2 = 3.35, 7.0      # 单栏 / 双栏宽度（英寸）


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def stars(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


# ======================================================================
# 实验一
# ======================================================================
def exp1():
    df = pd.read_csv(RES / "exp1.csv")
    # Q-KM：先对 5 个训练种子取均值，再按测试日与基线配对
    q = df[df.policy == "Q-KM"].groupby(["step", "fleet", "day"]).mean(numeric_only=True).reset_index()
    q["policy"] = "Q-KM"
    base = df[df.policy != "Q-KM"]
    daily = pd.concat([base, q], ignore_index=True)
    steps, fleets = sorted(daily.step.unique()), sorted(daily.fleet.unique())
    pols = ["贪心最近", "KM-即时收入", "Q-KM"]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.05), sharex=True)
    for ax, f in zip(axes, fleets):
        for p in pols:
            g = daily[(daily.fleet == f) & (daily.policy == p)].groupby("step").revenue
            m = g.mean().reindex(steps) / 1000
            e = g.apply(ci95).reindex(steps) / 1000
            ax.fill_between(steps, m - e, m + e, color=C[p], alpha=0.15, lw=0)
            ax.plot(steps, m, marker=MK[p], ms=4, color=C[p], label=p, mec="white", mew=0.6)
        ax.set_title(f"车队 {f} 辆")
        ax.set_xlabel("批次间隔 Δt / 分钟")
        ax.set_xticks(steps)
    axes[0].set_ylabel("全天总收入 / 千元")
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "exp1_revenue_dt.pdf")
    plt.close(fig)

    # 表：Q-KM 相对两基线的收入增益 + Wilcoxon 配对检验（n=10 个测试日）
    lines = []
    for s in steps:
        cells = [f"{s:g}"]
        for f in fleets:
            sub = daily[(daily.step == s) & (daily.fleet == f)].pivot(index="day", columns="policy", values="revenue")
            for b in ["KM-即时收入", "贪心最近"]:
                gain = (sub["Q-KM"] / sub[b] - 1) * 100
                pv = stats.wilcoxon(sub["Q-KM"], sub[b]).pvalue
                cells.append(f"{gain.mean():+.1f}{stars(pv)}")
        lines.append(" & ".join(cells) + r" \\")
    (TAB / "exp1_gain.tex").write_text("\n".join(lines) + "\n")

    # 表：Δt=10 / 60 辆的详细指标
    sub = daily[(daily.step == 10) & (daily.fleet == 60)]
    rows = []
    for p in pols:
        g = sub[sub.policy == p]
        rows.append(f"{p} & {g.revenue.mean():,.0f} $\\pm$ {ci95(g.revenue):,.0f} & {g.trip_km.mean():,.0f} & "
                    f"{g.answer_rate.mean()*100:.1f} & {g.avg_pickup.mean():.2f} & {g.avg_response.mean():.1f}" + r" \\")
    (TAB / "exp1_detail.tex").write_text("\n".join(rows) + "\n")

    # 选出的 β 分布
    bsel = df[df.policy == "Q-KM"].groupby(["step", "fleet", "seed"]).beta.first()
    print("β 选择计数:\n", bsel.groupby(level=[0, 1]).agg(lambda x: x.value_counts().to_dict()))

    # 图：分时段收入 + 学习曲线
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.95), gridspec_kw=dict(width_ratios=[1.5, 1]))
    hours = list(range(6, 24))
    ax = axes[0]
    for p in pols:
        g = sub[sub.policy == p]
        ax.plot(np.array(hours) + 0.5, [g[f"h{h}"].mean() / 1000 for h in hours], marker=MK[p], ms=3.5,
                color=C[p], label=p, mec="white", mew=0.5)
    ax.set(xlabel="时段（时）", ylabel="小时收入 / 千元", title="分时段收入（Δt = 10 min，60 辆）", xticks=range(6, 25, 2))
    ax.set_ylim(0, None)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0)
    ax = axes[1]
    extra = json.load(open(RES / "exp1_extra.json"))
    cur = [np.array(e["curve"]) for e in extra if e["step"] == 10 and e["fleet"] == 60 and e["curve"]]
    A = np.stack(cur)
    ax.fill_between(A[0, :, 0], A[:, :, 1].min(0) / 1000, A[:, :, 1].max(0) / 1000, color=C["Q-KM"], alpha=0.18, lw=0)
    ax.plot(A[0, :, 0], A[:, :, 1].mean(0) / 1000, color=C["Q-KM"], label="Q-KM（5 种子均值与极差）")
    ax.set(xlabel="训练回合（天）", ylabel="验证日收入 / 千元", title="Q-KM 学习曲线（β = 0.5）")
    ax.legend(loc="lower right")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "exp1_hourly_curve.pdf")
    plt.close(fig)

    # 正文引用的若干数
    hourly_gain = {h: (sub[sub.policy == "Q-KM"][f"h{h}"].mean() / sub[sub.policy == "KM-即时收入"][f"h{h}"].mean() - 1) * 100
                   for h in hours}
    print("分时段增益 (Q vs KM) %:", {h: round(v, 1) for h, v in hourly_gain.items()})
    print(daily.groupby(["step", "fleet", "policy"])[["revenue", "answer_rate", "avg_response"]].mean().round(2).to_string())


# ======================================================================
# 实验二 / 四
# ======================================================================
METHODS = ["NSGA-II", "NSGA-III", "加权和 KM", "NSGA-II+KM 初始化"]


def nemenyi_cd(k, n, q=2.569):            # α = 0.05，k = 4
    return q * np.sqrt(k * (k + 1) / (6 * n))


def exp24():
    met = pd.read_csv(RES / "exp24_metrics.csv")
    info = pd.read_csv(RES / "exp24_info.csv")
    fronts = pickle.load(open(RES / "exp24_fronts.pkl", "rb"))
    curves = json.load(open(RES / "exp24_curves.json"))

    # 代表批次前沿（18 点，第 0 批）
    key = (18, 0)
    fr = fronts[key]
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    Z = fr["exact"][np.argsort(fr["exact"][:, 1])]
    ax.plot(Z[:, 1], -Z[:, 0], color=C["精确前沿"], lw=0.9, label=f"精确前沿（{len(Z)} 点）", zorder=1)
    S = fr["supported"]
    ax.scatter(S[:, 1], -S[:, 0], s=22, facecolors="none", edgecolors=C["精确前沿"], lw=0.8,
               label=f"其中支撑解（{len(S)} 点）", zorder=2)
    for m in METHODS:
        F = fr[m]
        ax.scatter(F[:, 1], -F[:, 0], s=9 if m != "加权和 KM" else 14, marker=MK[m], color=C[m],
                   edgecolors="white", linewidths=0.3, label=m, zorder=3 if m != "加权和 KM" else 4)
    ax.set(xlabel="总接驾距离 / km（越小越好）", ylabel="总送客距离 / km（越大越好）")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.53), handletextpad=0.3, fontsize=6.5, labelspacing=0.3)
    # 局部放大：高收入端
    ins = ax.inset_axes([0.56, 0.06, 0.42, 0.22])
    sel = Z[:, 1] > Z[:, 1].max() * 0.6
    ins.plot(Z[sel, 1], -Z[sel, 0], color=C["精确前沿"], lw=0.8)
    for m in METHODS:
        F = fr[m]
        s2 = F[:, 1] > Z[:, 1].max() * 0.6
        ins.scatter(F[s2, 1], -F[s2, 0], s=6, marker=MK[m], color=C[m], linewidths=0)
    ins.tick_params(labelsize=5.5)
    ins.grid(alpha=0.5)
    fig.savefig(FIG / "exp2_front.pdf")
    plt.close(fig)

    # HV 收敛曲线
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for m in ["NSGA-II", "NSGA-III", "NSGA-II+KM 初始化"]:
        cs = [c for c in curves if c["method"] == m]
        gen = np.array(cs[0]["gen"])
        H = np.array([c["hv"] for c in cs])
        mu = H.mean(0)
        lo, hi = np.percentile(H, 10, 0), np.percentile(H, 90, 0)
        ax.fill_between(gen, lo, hi, color=C[m], alpha=0.15, lw=0)
        ax.plot(gen, mu, color=C[m], label=m)
    ws = met[met.method == "加权和 KM"].hv_ratio.mean()
    ax.axhline(ws, color=C["加权和 KM"], ls="--", lw=1.1, label="加权和 KM")
    ax.set(xlabel="代数", ylabel="HV / HV$^*$", ylim=(0.85, 1.0))
    ax.legend(loc="lower right", ncol=2)
    fig.savefig(FIG / "exp2_convergence.pdf")
    plt.close(fig)

    # 各时段 HV 比箱线图（按批次取 5 种子均值）
    bm = met.groupby(["hour", "batch", "method"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    hours = sorted(bm.hour.unique())
    wdt = 0.18
    for i, m in enumerate(METHODS):
        data = [bm[(bm.hour == h) & (bm.method == m)].igd_plus.values for h in hours]
        pos = np.arange(len(hours)) + (i - 1.5) * (wdt + 0.02)
        bp = ax.boxplot(data, positions=pos, widths=wdt, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1), whiskerprops=dict(color=C[m], lw=0.8),
                        capprops=dict(color=C[m], lw=0.8), boxprops=dict(lw=0))
        for b in bp["boxes"]:
            b.set_facecolor(C[m])
        ax.plot([], [], color=C[m], lw=5, label=m)
    ax.set_xticks(range(len(hours)), [f"{h}:00" for h in hours])
    ax.set(ylabel="IGD$^+$", xlabel="批次时段")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=0.8, columnspacing=0.8, fontsize=6.3)
    fig.savefig(FIG / "exp4_igd_hour.pdf")
    plt.close(fig)

    # 表：方法 × 指标，Friedman + Nemenyi（以 20 个批次为区组，指标 IGD+）
    piv = bm.pivot(index=["hour", "batch"], columns="method", values="igd_plus")[METHODS]
    ranks = piv.rank(axis=1).mean()
    fr_p = stats.friedmanchisquare(*[piv[m] for m in METHODS]).pvalue
    cd = nemenyi_cd(len(METHODS), len(piv))
    rows = []
    for m in METHODS:
        g = met[met.method == m]
        rows.append(f"{m} & {g.hv_ratio.mean():.4f} & {g.igd.mean():.4f} & {g.igd_plus.mean():.4f} & "
                    f"{g.spacing.mean():.3f} & {g.n_points.mean():.1f} & {g.frac_exact.mean()*100:.1f} & {ranks[m]:.2f}" + r" \\")
    (TAB / "exp24_methods.tex").write_text("\n".join(rows) + "\n")
    w = stats.wilcoxon(piv["NSGA-II+KM 初始化"], piv["NSGA-II"]).pvalue
    w2 = stats.wilcoxon(piv["NSGA-II"], piv["加权和 KM"]).pvalue
    w3 = stats.wilcoxon(piv["NSGA-II"], piv["NSGA-III"]).pvalue
    sup_frac = (info.n_supported / info.n_exact)
    ws_sup = met[met.method == "加权和 KM"].n_points.mean()
    txt = dict(friedman_p=fr_p, cd=cd, ranks=ranks.round(3).to_dict(), wil_init_vs_n2=w, wil_n2_vs_ws=w2,
               wil_n2_vs_n3=w3, n_exact_mean=info.n_exact.mean(), n_exact_min=int(info.n_exact.min()),
               n_exact_max=int(info.n_exact.max()), sup_frac_mean=sup_frac.mean(), sup_frac_min=sup_frac.min(),
               sup_frac_max=sup_frac.max(), n_sup_mean=info.n_supported.mean(), ws_points=ws_sup)
    json.dump(txt, open(RES / "exp24_stats.json", "w"), indent=1, default=float)
    print(json.dumps(txt, indent=1, default=float, ensure_ascii=False))
    print(met.groupby(["hour", "method"])[["hv_ratio", "igd_plus"]].mean().round(4).unstack())




# ======================================================================
# 实验三：DV-KM
# ======================================================================
def exp3():
    import glob
    rows, vmaps, losses = [], {}, {}
    for f in sorted(glob.glob(str(RES / "exp3" / "exp3_*.json"))):
        r = json.load(open(f))
        for x in r["rows"]:
            rows.append(dict(fleet=r["fleet"], seed=r["seed"], **x))
        vmaps[(r["fleet"], r["seed"])] = r["vmap"]
        losses[(r["fleet"], r["seed"])] = r["loss"]
    df = pd.DataFrame(rows)
    fleets = sorted(df.fleet.unique())
    betas = sorted(df.beta.dropna().unique())
    meths = ["Tab-Q-KM", "Tab-PE-KM", "DV-KM"]
    lab = {"Tab-Q-KM": "表格 Q-KM（max 目标）", "Tab-PE-KM": "表格 PE-KM（期望目标）", "DV-KM": "DV-KM（深度价值网络）"}

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.05))
    for ax, f in zip(axes, fleets):
        for b, ls in [("KM-即时收入", "--"), ("贪心最近", ":")]:
            v = df[(df.fleet == f) & (df.method == b)].revenue.mean() / 1000
            ax.axhline(v, color=C[b], ls=ls, lw=1.0, label=b)
        for m in meths:
            # 每个种子先对 10 个测试日取均值，误差带为 3 个训练种子的极差
            g = df[(df.fleet == f) & (df.method == m)].groupby(["beta", "seed"]).revenue.mean().unstack() / 1000
            ax.fill_between(betas, g.min(1), g.max(1), color=C[m], alpha=0.15, lw=0)
            ax.plot(betas, g.mean(1), marker=MK[m], ms=3.8, color=C[m], label=lab[m], mec="white", mew=0.5)
        ax.set(title=f"车队 {f} 辆", xlabel="长期价值权重 β", xticks=betas)
    axes[0].set_ylabel("全天总收入 / 千元")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.13), columnspacing=1.2)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "exp3_beta.pdf")
    plt.close(fig)

    # 表：每种方法在验证意义上的最佳 β 与 β=1 时的收入（均值 ± 种子标准差）
    lines = []
    for m in meths + ["KM-即时收入", "贪心最近"]:
        cells = [lab.get(m, m).split("（")[0]]
        for f in fleets:
            g = df[(df.fleet == f) & (df.method == m)]
            if m in meths:
                bymean = g.groupby("beta").revenue.mean()
                bb = bymean.idxmax()
                s1 = g[g.beta == 1.0].groupby("seed").revenue.mean()
                sb = g[g.beta == bb].groupby("seed").revenue.mean()
                cells += [f"{sb.mean():,.0f}\\,({bb:.1f})", f"{s1.mean():,.0f}$\\pm${s1.std(ddof=1):,.0f}"]
            else:
                cells += [f"{g.revenue.mean():,.0f}", "—"]
        lines.append(" & ".join(cells) + r" \\")
    (TAB / "exp3_main.tex").write_text("\n".join(lines) + "\n")

    # 消融（60 辆，种子 0，同一份经验数据）
    ab = df[(df.fleet == 60) & (df.seed == 0) & df.method.str.startswith("DV-KM")]
    ab = ab.groupby(["method", "beta"]).revenue.mean().unstack()
    order = ["DV-KM", "DV-KM 同数据重训", "DV-KM 无粗编码", "DV-KM 无平滑正则"]
    lines = []
    for m in order:
        r = ab.loc[m]
        lines.append(f"{m} & " + " & ".join(f"{r[b]:,.0f}" for b in [0.2, 0.4, 0.6, 0.8, 1.0]) + r" \\")
    (TAB / "exp3_ablation.tex").write_text("\n".join(lines) + "\n")
    print(ab.round(0).to_string())

    # DV 价值图（60 辆，种子 0）与签到点
    sys.path.insert(0, str(ROOT / "code"))
    from dispatch.data import W_KM, H_KM, load_all
    ck = load_all()["ck_train"].sample(4000, random_state=0)
    vm = vmaps[(60, 0)]
    vmax = np.percentile(np.concatenate([np.ravel(vm[k]) for k in vm]), 99)
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.15), sharey=True)
    for ax, hh in zip(axes, ["8", "13", "18", "22"]):
        im = ax.imshow(np.array(vm[hh]), origin="lower", extent=[0, W_KM, 0, H_KM], cmap="Blues",
                       vmin=0, vmax=vmax, aspect="equal")
        ax.scatter(ck.x, ck.y, s=0.15, c="#3d3d3a", alpha=0.25, lw=0)
        ax.set(title=f"$V_\\theta$(τ, x, y) @ {hh}:00", xlabel="x / km")
        ax.grid(False)
    axes[0].set_ylabel("y / km")
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("状态价值 / 元", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(FIG / "exp3_vmap.pdf")
    plt.close(fig)


# ======================================================================
# 拓展 C/D/E
# ======================================================================
FL = {40: "#2a78d6", 60: "#eb6834", 100: "#1baf7a"}
FMK = {40: "o", 60: "s", 100: "^"}


def exp5():
    """事后最优上下界 vs 在线策略：表格 + 汇总（同一批测试日配对）。"""
    import glob
    h = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(RES / "exp5_rows" / "*.json"))])
    h.to_csv(RES / "exp5_hindsight.csv", index=False)
    e1 = pd.read_csv(RES / "exp1.csv")
    q = e1[e1.policy == "Q-KM"].groupby(["step", "fleet", "day"]).revenue.mean().reset_index()
    q["policy"] = "Q-KM"
    on = pd.concat([e1[e1.policy != "Q-KM"][["step", "fleet", "day", "policy", "revenue"]], q])
    on = on.pivot_table(index=["step", "fleet", "day"], columns="policy", values="revenue").reset_index()
    m = h.merge(on, on=["step", "fleet", "day"])
    m["q_lb"] = m["Q-KM"] / m.lb
    m["km_lb"] = m["KM-即时收入"] / m.lb
    m["gr_lb"] = m["贪心最近"] / m.lb
    m["closed"] = (m["Q-KM"] - m["KM-即时收入"]) / (m.lb - m["KM-即时收入"])
    m["width"] = m.ub / m.lb - 1
    t = m.groupby(["step", "fleet"]).agg(n=("day", "size"), lb=("lb", "mean"), ub=("ub", "mean"), q=("Q-KM", "mean"),
                                         km=("KM-即时收入", "mean"), q_lb=("q_lb", "mean"), km_lb=("km_lb", "mean"),
                                         gr_lb=("gr_lb", "mean"), closed=("closed", "mean"), width=("width", "mean"),
                                         ub_srv=("ub_served", "mean"), lb_srv=("lb_served", "mean")).reset_index()
    t.to_csv(RES / "exp5_summary.csv", index=False)
    with open(TAB / "exp5.tex", "w") as fo:
        for s_ in sorted(t.step.unique()):
            cells = []
            for f in sorted(t.fleet.unique()):
                r = t[(t.step == s_) & (t.fleet == f)]
                if len(r) == 0:
                    cells += ["--"] * 4
                    continue
                r = r.iloc[0]
                if (h[(h.step == s_) & (h.fleet == f)].lb_status != 0).any():   # MILP 超时：只报 UB，比例以 UB 为分母
                    cells += [f"[--, {r.ub/1000:.1f}]", f"{100*r.km/r.ub:.0f}$^\\dagger$", f"{100*r.q/r.ub:.0f}$^\\dagger$", "--"]
                else:
                    cells += [f"[{r.lb/1000:.1f}, {r.ub/1000:.1f}]", f"{100*r.km_lb:.0f}", f"{100*r.q_lb:.0f}", f"{100*r.closed:.0f}"]
            fo.write(f"{s_} & " + " & ".join(cells) + " \\\\\n")
    print(t.round(3).to_string())
    print("lb_status", h.lb_status.value_counts().to_dict(), "mip gap max", ((h.lb_bound - h.lb) / h.lb).max())
    return m


def exp6():
    """乐观度谱：(a) 60 辆收入增益热力图；(b) β=1 收入；(c) 价值高估；(d) 拒派比例。"""
    d = pd.DataFrame(json.load(open(RES / "exp6_expectile.json")))
    d["name"] = np.where(d.target == "max", "max", d.tau.map(lambda v: f"{v:g}"))
    order = [f"{v:g}" for v in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]] + ["max"]
    d["bias"] = (d.v_mean - d.g_mean) / d.g_mean
    fleets, betas = sorted(d.fleet.unique()), sorted(d.beta.unique())
    agg = d.groupby(["fleet", "name", "beta"]).agg(revenue=("revenue", "mean"), bias=("bias", "mean")).reset_index()
    km = agg[agg.beta == 0].groupby("fleet").revenue.mean()
    b6 = pd.DataFrame(json.load(open(RES / "exp6b_bias.json")))
    b6["name"] = np.where(b6.target == "max", "max", b6.tau.map(lambda v: f"{v:g}"))
    b6 = b6.groupby(["fleet", "name"])[["refuse", "go", "stay", "fare"]].mean().reset_index()
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.05), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    ax = axes[0]
    f = 60
    M = np.array([[agg[(agg.fleet == f) & (agg.name == n) & (agg.beta == bb)].revenue.iloc[0] / km[f] - 1
                   for bb in betas[1:]] for n in order]) * 100
    vmax = np.abs(M).max()
    cmap = LinearSegmentedColormap.from_list("div", ["#b8412c", "#f1f0ec", "#2a78d6"])
    ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "#3d3d3a")
    ax.set_xticks(range(len(betas) - 1), [f"{x:g}" for x in betas[1:]])
    ax.set_yticks(range(len(order)), order)
    ax.set(title="(a) 60 辆：相对 KM 增益 / %", xlabel="β", ylabel="期望分位 τ")
    ax.grid(False)
    x = np.arange(len(order))
    for f in fleets:
        kw = dict(marker=FMK[f], ms=3.6, color=FL[f], label=f"{f} 辆", mec="white", mew=0.5)
        g = agg[(agg.fleet == f) & (agg.beta == 1.0)].set_index("name").loc[order]
        axes[1].plot(x, 100 * (g.revenue / km[f] - 1), **kw)
        g = agg[(agg.fleet == f) & (agg.beta == 0.6)].set_index("name").loc[order]
        axes[2].plot(x, 100 * g.bias, **kw)
        g = b6[b6.fleet == f].set_index("name").reindex(order)
        ok = g.refuse.notna().values
        axes[3].plot(x[ok], 100 * g.refuse.values[ok], **kw)
    for ax, t in [(axes[1], "(b) β=1 收入相对 KM / %"), (axes[2], "(c) 价值高估 (V−G)/G / %"),
                  (axes[3], "(d) β=1 拒派的可行边 / %")]:
        ax.set_xticks(x, [o.lstrip("0") if o != "max" else "max" for o in order], fontsize=6)
        ax.axvspan(len(order) - 1.5, len(order) - 0.5, color="#e4e3de", alpha=0.6, lw=0)
        ax.set(title=t, xlabel="τ")
    fig.subplots_adjust(wspace=0.38)
    axes[1].axhline(0, color="#8a8a85", lw=0.7)
    axes[1].legend(fontsize=6, loc="lower left")
    fig.savefig(FIG / "exp6_expectile.pdf")
    plt.close(fig)
    best = agg[agg.beta > 0].groupby(["fleet", "name"]).revenue.max().unstack()[order]
    print("best gain %\n", (best.T / km - 1).T.mul(100).round(2).to_string())
    print("bias %\n", agg[agg.beta == 0.6].pivot(index="fleet", columns="name", values="bias")[order].mul(100).round(1).to_string())
    print("beta1 rel %\n", (agg[agg.beta == 1.0].pivot(index="fleet", columns="name", values="revenue")[order].T / km - 1).T.mul(100).round(1).to_string())
    b6["adv"] = b6.go - b6.stay
    b6["ratio"] = b6.go / b6.stay
    print(b6.round(3).to_string())
    lab = {"0.5": "PE ($\\tau{=}0.5$)", "0.95": "$\\tau{=}0.95$", "max": "max"}
    with open(TAB / "exp6b.tex", "w") as fo:
        for f in fleets:
            for i, n in enumerate(["0.5", "0.95", "max"]):
                r = b6[(b6.fleet == f) & (b6.name == n)].iloc[0]
                fo.write(f"{f if i == 0 else ''} & {lab[n]} & {r.go:.1f} & {r.stay:.1f} & {r.adv:.1f} & {r.ratio:.2f} \\\\\n")
            if f != fleets[-1]:
                fo.write("\\midrule\n")
    return d


def exp7():
    """批次级 vs 系统级：λ 扫描下全天（总送客 km, 总接驾 km），并与批次级精确前沿归一化对比。"""
    d = pd.DataFrame(json.load(open(RES / "exp7_system_front.json")))
    for c in ["revenue", "trip_km", "pickup_km", "answer"]:
        d[c + "_m"] = d[c].map(np.mean)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.25))
    ax = axes[0]
    f = 60
    for base, mk, ls, col in [("KM", "s", "-", C["KM-即时收入"]), ("PE-KM", "v", "--", C["Tab-PE-KM"])]:
        g = d[(d.fleet == f) & (d.step == 10) & (d.base == base)].sort_values("lam")
        ax.plot(g.pickup_km_m / 1000, g.revenue_m / 1000, marker=mk, ms=3.6, color=col, ls=ls,
                label=base + " + λ·接驾", mec="white", mew=0.5)
        if base == "KM":
            for _, r in g.iterrows():
                if r.lam in (0, 1, 2, 3, 5):
                    ax.annotate(f"λ={r.lam:g}", (r.pickup_km_m / 1000, r.revenue_m / 1000), fontsize=5.8,
                                color=INK2, xytext=(4, -8), textcoords="offset points")
    g = d[(d.fleet == f) & (d.step == 10) & (d.base == "贪心最近")].iloc[0]
    ax.scatter(g.pickup_km_m / 1000, g.revenue_m / 1000, marker="o", s=24, color=C["贪心最近"], zorder=3,
               label="贪心最近", edgecolors="white", linewidths=0.5)
    ax.set(title="(a) 全天系统级权衡（60 辆，Δt=10）", xlabel="全天总接驾 / 千 km", ylabel="全天收入 / 千元")
    ax.legend(fontsize=6.3, loc="lower right")
    # (b) 归一化：批次级精确前沿 vs 系统级
    ax = axes[1]
    F = pickle.load(open(RES / "exp24_fronts.pkl", "rb"))
    xs = np.linspace(0, 1, 201)
    curves = []
    for fr in F.values():
        Z = fr["exact"]
        G, P = -Z[:, 0], Z[:, 1]
        i = np.argmax(G)
        curves.append([G[P <= x * P[i] + 1e-9].max() / G[i] if (P <= x * P[i] + 1e-9).any() else 0 for x in xs])
    curves = 100 * np.array(curves)
    ax.fill_between(100 * xs, np.percentile(curves, 10, 0), np.percentile(curves, 90, 0), color=C["精确前沿"],
                    alpha=0.13, lw=0)
    ax.plot(100 * xs, curves.mean(0), color=C["精确前沿"], lw=1.2, label="批次级精确前沿（20 批次均值，10–90% 分位）")
    for f in [40, 60, 100]:
        g = d[(d.fleet == f) & (d.step == 10) & (d.base == "KM")].sort_values("lam")
        z = g[g.lam == 0].iloc[0]
        ax.plot(100 * g.pickup_km_m / z.pickup_km_m, 100 * g.trip_km_m / z.trip_km_m, marker=FMK[f], ms=3.6,
                color=FL[f], label=f"系统级（全天）KM + λ，{f} 辆", mec="white", mew=0.5)
    ax.set(title="(b) 归一化对比：送客距离 vs 接驾距离", xlabel="接驾距离 / 最大收入解的 %",
           ylabel="送客距离 / 最大收入解的 %", xlim=(0, 105), ylim=(0, 104))
    ax.legend(fontsize=6, loc="lower right")
    fig.savefig(FIG / "exp7_system_front.pdf")
    plt.close(fig)
    tab = d[d.base != "贪心最近"].groupby(["step", "fleet", "base"]).apply(
        lambda g: pd.Series(dict(lam_best=g.loc[g.revenue_m.idxmax(), "lam"],
                                 gain=g.revenue_m.max() / g[g.lam == 0].revenue_m.iloc[0] - 1,
                                 pk_cut=1 - g.loc[g.revenue_m.idxmax(), "pickup_km_m"] / g[g.lam == 0].pickup_km_m.iloc[0])))
    print(tab.round(3).to_string())
    b = curves.mean(0)
    print("batch: trip% at pickup 80%/65%:", b[160], b[130])
    return d, tab

if __name__ == "__main__":
    which = sys.argv[1:] or ["exp1", "exp24", "exp3"]
    for w in which:
        globals()[w]()
