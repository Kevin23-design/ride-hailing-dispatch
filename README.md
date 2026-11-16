# 实验报告：网约车派单（张子扬 10245102456）

- `latex/main.pdf`：实验报告（9 页，XeLaTeX 编译：`latex/build.sh`）
- `code/dispatch/`：数据、仿真器（参数化批次间隔 Δt）、Q-KM、NSGA-II/III、ε-约束精确前沿、DV-KM
- `code/exp1_revenue.py`：实验一，多批次间隔收入对比（Δt × 车队 × 种子 = 90 个训练任务）
- `code/exp24_pareto.py`：实验二 + 拓展 A，帕累托前沿与精确前沿（20 个批次）
- `code/exp3_dvkm.py`：拓展 B，DV-KM（需 PyTorch + GPU，`--fleet 60 --seed 0 --gpu 0 --ablation`）
- `code/plot_all.py`：从 `results/` 生成全部图与表格片段
- `code/tests/test_core.py`：精确前沿与穷举一致、Q 更新手算、β=0 等价 KM

运行环境：本地 `~/miniforge3/envs/py312`（scipy 1.17）；服务器 `/root/zzy/report`（`/opt/conda/bin/python`，torch 2.0）。
实验二/四的最终结果在本地用 scipy 1.17 的 HiGHS 生成（全部前沿均通过逐点验证）。

## 补充实验（拓展 C/D/E）

| 脚本 | 内容 | 输出 |
|---|---|---|
| `code/exp5_hindsight.py` | 事后最优上下界（LP 上界 + 可执行 MILP 下界，回放验证）；默认 Δt∈{10,30}、5 个测试日，单核/进程，每个任务完成即写 `results/exp5_rows/*.json` | `results/exp5_rows/`, `exp5_summary.csv` |
| `code/exp6_expectile.py` | 期望分位（τ=0.5–0.95）与 max 目标的乐观度谱，β 扫描 + 价值高估 | `results/exp6_expectile.json` |
| `code/exp6b_bias.py` | β=1 时可行边上“出发 / 等待”价值项与拒派比例 | `results/exp6b_bias.json` |
| `code/exp7_system_front.py` | 接驾惩罚 λ 扫描的全天系统级权衡 | `results/exp7_system_front.json` |

出图：`python code/plot_all.py exp5 exp6 exp7`。

## 数据

使用 Foursquare NYC 签到数据 `dataset_TSMC2014_NYC.csv`（Yang et al., 2014），未随仓库提供；请自行下载后放到 `data/` 目录（见 `code/dispatch/data.py`）。
