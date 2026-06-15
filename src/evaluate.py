"""评估与基准测试模块。

在相同输入（同一查询、同一候选块、同一 Token 预算）下对比四种算法：
    - Naive Top-K
    - Exact DP (0-1 Knapsack)
    - Greedy (value/weight)
    - Greedy Submodular

统计核心指标：
    - 总相关性得分 (total relevance value)
    - 总 Token 数 (total tokens used)
    - 算法执行时间 (elapsed seconds)
    - 相对最优解的近似比 (approx ratio vs DP)
    - 选中块内部平均冗余 (mean pairwise similarity，越低越多样)

结果以表格打印，并在 matplotlib 可用时绘制对比图保存到磁盘。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import algorithms
from .dataset import RAGSample
from .pipeline import ContextSelectionPipeline, PreparedContext

# 参与对比的算法（键为内部名，值为展示名）
ALGORITHMS = {
    "topk": "Naive Top-K",
    "dp": "Exact DP",
    "greedy": "Greedy ratio",
    "submodular": "Greedy Submodular",
}


@dataclass
class AlgoMetrics:
    """单个算法在单个样本上的指标。"""

    algorithm: str
    total_value: float
    total_weight: int
    elapsed_seconds: float
    approx_ratio: float          # total_value / DP 最优值
    mean_redundancy: float       # 选中块两两平均相似度
    num_selected: int


def _mean_redundancy(indices: Sequence[int], sim: np.ndarray) -> float:
    """计算选中块两两平均相似度（多样性指标，越低越好）。"""
    idx = list(indices)
    if len(idx) < 2:
        return 0.0
    total, count = 0.0, 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            total += float(sim[idx[a], idx[b]])
            count += 1
    return total / count if count else 0.0


def _redundant_pairs(indices: Sequence[int], sim: np.ndarray,
                     threshold: float = 0.7) -> int:
    """统计选中块中相似度超过阈值的「近似重复对」数量。

    与逐块标签不同，它衡量的是「同时选入了两个高度相似块」这一真正的
    冗余问题——保留近重复对中的任意一个都不算冗余。
    """
    idx = list(indices)
    pairs = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if float(sim[idx[a], idx[b]]) > threshold:
                pairs += 1
    return pairs


def evaluate_sample(
    prepared: PreparedContext,
    capacity: int,
    redundancy_lambda: float = 0.5,
) -> Dict[str, AlgoMetrics]:
    """在单个预处理样本上运行所有算法并收集指标。"""
    values = list(prepared.values)
    weights = prepared.weights
    sim = prepared.similarity_matrix

    # 先跑 DP 得到最优值，作为近似比的分母
    dp_result = algorithms.dp_knapsack(values, weights, capacity)
    optimal = dp_result.total_value or 1e-12

    metrics: Dict[str, AlgoMetrics] = {}
    for key in ALGORITHMS:
        res = algorithms.run_algorithm(
            key, values, weights, capacity,
            similarity_matrix=sim, redundancy_lambda=redundancy_lambda,
        )
        metrics[key] = AlgoMetrics(
            algorithm=ALGORITHMS[key],
            total_value=res.total_value,
            total_weight=res.total_weight,
            elapsed_seconds=res.elapsed_seconds,
            approx_ratio=res.total_value / optimal,
            mean_redundancy=_mean_redundancy(res.selected_indices, sim),
            num_selected=len(res.selected_indices),
        )
    return metrics


def aggregate(
    per_sample: List[Dict[str, AlgoMetrics]],
) -> Dict[str, AlgoMetrics]:
    """对多个样本的指标取平均，得到总体表现。"""
    agg: Dict[str, AlgoMetrics] = {}
    for key, name in ALGORITHMS.items():
        rows = [m[key] for m in per_sample]
        n = len(rows) or 1
        agg[key] = AlgoMetrics(
            algorithm=name,
            total_value=sum(r.total_value for r in rows) / n,
            total_weight=sum(r.total_weight for r in rows) / n,
            elapsed_seconds=sum(r.elapsed_seconds for r in rows) / n,
            approx_ratio=sum(r.approx_ratio for r in rows) / n,
            mean_redundancy=sum(r.mean_redundancy for r in rows) / n,
            num_selected=sum(r.num_selected for r in rows) / n,
        )
    return agg


def print_metrics_table(agg: Dict[str, AlgoMetrics], capacity: int) -> None:
    """以表格形式打印各算法的平均指标。"""
    header = (
        f"{'算法':<20}{'总相关性':>10}{'总Token':>10}"
        f"{'近似比':>10}{'冗余度':>10}{'选中块':>8}{'耗时(ms)':>12}"
    )
    print("\n" + "=" * len(header))
    print(f"基准测试结果（Token 预算 W = {capacity}）")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key in ALGORITHMS:
        m = agg[key]
        print(
            f"{m.algorithm:<20}"
            f"{m.total_value:>10.4f}"
            f"{m.total_weight:>10.1f}"
            f"{m.approx_ratio:>10.3f}"
            f"{m.mean_redundancy:>10.3f}"
            f"{m.num_selected:>8.1f}"
            f"{m.elapsed_seconds * 1000:>12.3f}"
        )
    print("=" * len(header))
    print(
        "说明：近似比 = 算法总相关性 / DP 最优值（越接近 1 越好）；"
        "冗余度 = 选中块两两平均相似度（越低越多样）。\n"
    )


def plot_metrics(
    agg: Dict[str, AlgoMetrics],
    capacity: int,
    out_path: str = "benchmark_results.png",
) -> Optional[str]:
    """绘制各算法对比柱状图并保存。matplotlib 不可用时返回 None。"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无显示环境也可保存
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[evaluate] matplotlib 不可用，跳过绘图: {exc}")
        return None

    names = [agg[k].algorithm for k in ALGORITHMS]
    values = [agg[k].total_value for k in ALGORITHMS]
    tokens = [agg[k].total_weight for k in ALGORITHMS]
    times_ms = [agg[k].elapsed_seconds * 1000 for k in ALGORITHMS]
    redundancy = [agg[k].mean_redundancy for k in ALGORITHMS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"RAG Context Selection Benchmark (W={capacity})", fontsize=14)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    def bar(ax, data, title, ylabel):
        ax.bar(names, data, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(data):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)

    bar(axes[0, 0], values, "Total Relevance (higher better)", "value")
    bar(axes[0, 1], tokens, "Total Tokens Used", "tokens")
    bar(axes[1, 0], times_ms, "Runtime (lower better)", "ms")
    bar(axes[1, 1], redundancy, "Mean Redundancy (lower better)", "cos sim")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[evaluate] 对比图已保存至: {out_path}")
    return out_path


def run_benchmark(
    samples: List[RAGSample],
    capacity: int = 128,
    redundancy_lambda: float = 0.5,
    pipeline: Optional[ContextSelectionPipeline] = None,
    plot: bool = True,
    plot_path: str = "benchmark_results.png",
) -> Dict[str, AlgoMetrics]:
    """完整基准测试流程：预处理每个样本 -> 评估 -> 聚合 -> 打印/绘图。"""
    pipeline = pipeline or ContextSelectionPipeline()
    print(f"嵌入后端: {pipeline.embedder.backend}")
    print(f"Token 后端: {pipeline.counter.backend}")
    print(f"样本数: {len(samples)}，Token 预算 W = {capacity}")

    per_sample: List[Dict[str, AlgoMetrics]] = []
    for s in samples:
        prepared = pipeline.prepare(s.query, s.chunks)
        per_sample.append(
            evaluate_sample(prepared, capacity, redundancy_lambda)
        )

    agg = aggregate(per_sample)
    print_metrics_table(agg, capacity)
    if plot:
        plot_metrics(agg, capacity, plot_path)
    return agg


# ===========================================================================
# 直观演示 1：并排选块对比（看清各算法到底选了哪些块）
# ===========================================================================
_LABEL_CN = {
    "relevant": "高相关",
    "redundant": "冗余",
    "distractor": "干扰",
    "weak": "弱相关",
    "": "-",
}


def compare_selections(
    sample: RAGSample,
    capacity: int,
    redundancy_lambda: float = 0.5,
    pipeline: Optional[ContextSelectionPipeline] = None,
    text_width: int = 46,
) -> None:
    """对同一查询并排展示四种算法各自选中的文本块。

    打印一张表：每行是一个候选块（含人工标注的块类型、相关性 v_i、
    Token 重量 w_i），四列分别用 ✓ 标记该块是否被对应算法选中。
    这样可以直接「看见」朴素 Top-K 把冗余块也选进去、而次模优化主动
    避开冗余块的行为差异。
    """
    pipeline = pipeline or ContextSelectionPipeline()
    prepared = pipeline.prepare(sample.query, sample.chunks)
    labels = sample.labels or [""] * len(sample.chunks)
    sim = prepared.similarity_matrix

    # 运行四种算法，记录各自选中的下标集合与结果
    selected_sets: Dict[str, set] = {}
    results: Dict[str, algorithms.SelectionResult] = {}
    for key in ALGORITHMS:
        res = algorithms.run_algorithm(
            key, list(prepared.values), prepared.weights, capacity,
            similarity_matrix=prepared.similarity_matrix,
            redundancy_lambda=redundancy_lambda,
        )
        selected_sets[key] = set(res.selected_indices)
        results[key] = res

    print("\n" + "#" * 100)
    print(f"查询: {sample.query}")
    print(f"Token 预算 W = {capacity}    冗余惩罚 λ = {redundancy_lambda}")
    print("#" * 100)

    # 表头
    algo_cols = "".join(f"{ALGORITHMS[k]:^16}" for k in ALGORITHMS)
    header = f"{'#':>2} {'类型':<8}{'v_i':>6}{'w_i':>5}  {'文本块':<{text_width}}{algo_cols}"
    print(header)
    print("-" * len(header))

    for i, chunk in enumerate(sample.chunks):
        text = chunk if len(chunk) <= text_width else chunk[: text_width - 1] + "…"
        label_cn = _LABEL_CN.get(labels[i], labels[i])
        marks = "".join(
            f"{'✓' if i in selected_sets[k] else '·':^16}" for k in ALGORITHMS
        )
        print(
            f"{i:>2} {label_cn:<8}{prepared.values[i]:>6.3f}"
            f"{prepared.weights[i]:>5}  {text:<{text_width}}{marks}"
        )

    print("-" * len(header))

    # 汇总行：总相关性 / 总 Token / 选中块内部平均相似度（真实冗余度）
    def footer_row(title: str, fmt, getter) -> None:
        prefix = f"{title:<16}{'':>5}  {'':<{text_width}}"
        cells = "".join(f"{fmt(getter(k)):^16}" for k in ALGORITHMS)
        print(prefix + cells)

    footer_row("总相关性 ↑", lambda v: f"{v:.3f}",
               lambda k: results[k].total_value)
    footer_row("总Token", lambda v: f"{v}",
               lambda k: results[k].total_weight)
    footer_row("平均冗余 ↓", lambda v: f"{v:.3f}",
               lambda k: _mean_redundancy(results[k].selected_indices, sim))
    footer_row("近似重复对 ↓", lambda v: f"{v}",
               lambda k: _redundant_pairs(results[k].selected_indices, sim))
    print("=" * len(header))
    print("说明：✓=被选中。「平均冗余」是选中块两两平均余弦相似度，"
          "「近似重复对」是选中块中相似度>0.7 的对数；二者越低越说明"
          "上下文越多样、信息重复越少。\n")


# ===========================================================================
# 直观演示 2：Token 预算扫描曲线（相关性 vs W）
# ===========================================================================
def capacity_sweep(
    samples: List[RAGSample],
    capacities: Sequence[int],
    redundancy_lambda: float = 0.5,
    pipeline: Optional[ContextSelectionPipeline] = None,
    out_path: str = "capacity_sweep.png",
) -> Dict[str, List[float]]:
    """扫描不同 Token 预算 W，绘制各算法「平均总相关性 vs W」曲线。

    这是背包问题最经典直观的分析视角：随着预算放宽，各算法能取得的
    信息价值如何增长、谁更快逼近最优（DP）。
    """
    pipeline = pipeline or ContextSelectionPipeline()
    print(f"嵌入后端: {pipeline.embedder.backend} | Token 后端: {pipeline.counter.backend}")
    print(f"扫描预算: {list(capacities)}")

    # 预处理一次，复用嵌入结果
    prepared_list = [pipeline.prepare(s.query, s.chunks) for s in samples]

    curves: Dict[str, List[float]] = {k: [] for k in ALGORITHMS}
    for cap in capacities:
        sums: Dict[str, float] = {k: 0.0 for k in ALGORITHMS}
        for prepared in prepared_list:
            for k in ALGORITHMS:
                res = algorithms.run_algorithm(
                    k, list(prepared.values), prepared.weights, cap,
                    similarity_matrix=prepared.similarity_matrix,
                    redundancy_lambda=redundancy_lambda,
                )
                sums[k] += res.total_value
        n = len(prepared_list) or 1
        for k in ALGORITHMS:
            curves[k].append(sums[k] / n)

    # 文本打印
    print("\n预算 W 下的平均总相关性：")
    head = f"{'W':>6}" + "".join(f"{ALGORITHMS[k]:>20}" for k in ALGORITHMS)
    print(head)
    print("-" * len(head))
    for j, cap in enumerate(capacities):
        row = f"{cap:>6}" + "".join(f"{curves[k][j]:>20.4f}" for k in ALGORITHMS)
        print(row)

    # 绘图
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        markers = {"topk": "o", "dp": "s", "greedy": "^", "submodular": "D"}
        plt.figure(figsize=(9, 6))
        for k in ALGORITHMS:
            plt.plot(list(capacities), curves[k], marker=markers[k],
                     label=ALGORITHMS[k])
        plt.xlabel("Token budget W")
        plt.ylabel("Average total relevance")
        plt.title("Relevance vs Token Budget (capacity sweep)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\n[evaluate] 预算扫描曲线已保存至: {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[evaluate] matplotlib 不可用，跳过绘图: {exc}")

    return curves


# ===========================================================================
# 直观演示 3：次模优化贪心过程逐步动画
# ===========================================================================
def trace_submodular(
    sample: RAGSample,
    capacity: int,
    redundancy_lambda: float = 0.9,
    min_relevance: float = 0.1,
    pipeline: Optional[ContextSelectionPipeline] = None,
    delay: float = 0.0,
    text_width: int = 40,
) -> None:
    """逐步演示贪心次模优化：展示每一轮如何计算边际增益、惩罚冗余、选块。

    每一步打印当前所有可用候选的：
        v_i（原始相关性）、max_sim（与已选集合最大相似度）、
        penalty = λ·max_sim（冗余惩罚）、gain = v_i − penalty（边际增益）、
        density = gain / w_i（性价比，实际选取依据）
    并高亮本轮被选中的块。delay>0 时逐步停顿，形成动画效果。
    """
    pipeline = pipeline or ContextSelectionPipeline()
    prepared = pipeline.prepare(sample.query, sample.chunks)
    labels = sample.labels or [""] * len(sample.chunks)

    res = algorithms.greedy_submodular(
        list(prepared.values), prepared.weights, capacity,
        prepared.similarity_matrix,
        redundancy_lambda=redundancy_lambda,
        min_relevance=min_relevance,
        record_trace=True,
    )
    trace = res.extra.get("trace", [])
    dropped = res.extra.get("dropped_below_threshold", [])

    print("\n" + "#" * 92)
    print("次模优化贪心过程逐步演示 (Greedy Submodular, step-by-step)")
    print("#" * 92)
    print(f"查询      : {sample.query}")
    print(f"Token 预算: W = {capacity}")
    print(f"冗余惩罚  : λ = {redundancy_lambda}    相关性门槛 min_relevance = {min_relevance}")
    print(f"目标函数  : gain(i) = v_i − λ·max_sim(i, 已选集合)，按 gain/w_i 贪心选取")
    if dropped:
        names = ", ".join(f"#{i}(v={prepared.values[i]:.3f})" for i in dropped)
        print(f"门槛过滤  : 以下块相关性过低被预先排除 -> {names}")
    print("#" * 92)

    for step_no, step in enumerate(trace, start=1):
        if delay:
            time.sleep(delay)
        sel = step["selected_so_far"]
        sel_str = "{空}" if not sel else "{" + ", ".join(f"#{i}" for i in sel) + "}"
        print(f"\n──── 第 {step_no} 步 ──── 已选集合 S = {sel_str}，已用 Token = {step['used']}/{capacity}")
        head = (f"  {'块':>3} {'类型':<8}{'v_i':>7}{'maxSim':>8}"
                f"{'惩罚λ·s':>9}{'增益gain':>10}{'密度g/w':>10}  {'可行':<4} 文本")
        print(head)
        print("  " + "-" * (len(head) + text_width - 2))
        # 候选按密度降序展示，便于看出谁会被选
        cands = sorted(step["candidates"], key=lambda c: c["density"], reverse=True)
        for c in cands:
            i = c["index"]
            mark = "►" if i == step.get("chosen") else " "
            feas = "是" if c["feasible"] else "否"
            note = ""
            if not c["positive"]:
                note = "  ← 增益≤0(冗余被罚没)"
            elif not c["feasible"]:
                note = "  ← 超预算放不下"
            text = sample.chunks[i]
            text = text if len(text) <= text_width else text[: text_width - 1] + "…"
            label_cn = _LABEL_CN.get(labels[i], labels[i])
            print(f"{mark} {i:>3} {label_cn:<8}{c['value']:>7.3f}{c['max_sim']:>8.3f}"
                  f"{c['penalty']:>9.3f}{c['gain']:>10.3f}{c['density']:>10.4f}"
                  f"  {feas:<4} {text}{note}")
        chosen = step.get("chosen")
        if chosen is None:
            print(f"  => {step['reason']}")
        else:
            print(f"  => 选中 #{chosen}（密度 {step['chosen_density']:.4f} 最高）"
                  f"，加入上下文")

    print("\n" + "=" * 92)
    print(f"最终选择: {res.selected_indices}  "
          f"总相关性 = {res.total_value:.3f}  总 Token = {res.total_weight}/{capacity}")
    print("=" * 92)
    print("观察要点：每选入一个块后，与它高度相似的候选其 max_sim 上升、"
          "惩罚增大、增益下降，从而被「挤出」选择——这正是次模优化抑制冗余的机制。\n")
