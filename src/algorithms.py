"""上下文选择算法模块。

将「在 Token 预算 W 下挑选信息价值最高的文本块子集」建模为组合优化
问题，并实现四种基准策略：

1. naive_top_k         朴素 Top-K：按相关性 v_i 降序选取直到超预算。
2. dp_knapsack         动态规划：精确求解 0-1 背包，得到全局最优子集。
3. greedy_ratio        贪心近似：按价值密度 v_i / w_i 降序选取。
4. greedy_submodular   贪心次模优化：背包 + 集合覆盖，惩罚语义冗余。

统一输入：
    values  : List[float]  每个文本块的相关性得分 v_i
    weights : List[int]    每个文本块的 Token 数 w_i
    capacity: int          Token 预算 W
所有算法统一返回 SelectionResult。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class SelectionResult:
    """一次选择的结果。

    属性
    ----
    selected_indices : 被选中文本块在原列表中的下标（保持选择顺序）
    total_value      : 所选文本块的相关性得分之和（核心质量指标）
    total_weight     : 所选文本块的 Token 数之和（需 <= capacity）
    algorithm        : 算法名称
    elapsed_seconds  : 算法执行时间（秒）
    """

    selected_indices: List[int]
    total_value: float
    total_weight: int
    algorithm: str
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def _summarize(
    indices: Sequence[int],
    values: Sequence[float],
    weights: Sequence[int],
    algorithm: str,
    elapsed: float,
    extra: Optional[dict] = None,
) -> SelectionResult:
    """根据选中下标汇总出 SelectionResult。"""
    idx = list(indices)
    total_value = float(sum(values[i] for i in idx))
    total_weight = int(sum(weights[i] for i in idx))
    return SelectionResult(
        selected_indices=idx,
        total_value=total_value,
        total_weight=total_weight,
        algorithm=algorithm,
        elapsed_seconds=elapsed,
        extra=extra or {},
    )


# ===========================================================================
# 1. 朴素 Top-K 基线
# ===========================================================================
def naive_top_k(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
) -> SelectionResult:
    """按相关性得分从高到低贪心选取，直到无法再放入预算。

    这是最常见的 RAG 基线：只看相关性、不考虑性价比与冗余。
    """
    start = time.perf_counter()
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)

    selected: List[int] = []
    used = 0
    for i in order:
        if used + weights[i] <= capacity:
            selected.append(i)
            used += weights[i]
        # 注意：不提前 break，允许跳过过大的块继续尝试后面的小块
    elapsed = time.perf_counter() - start
    return _summarize(selected, values, weights, "Naive Top-K", elapsed)


# ===========================================================================
# 2. 动态规划：精确 0-1 背包
# ===========================================================================
def dp_knapsack(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
) -> SelectionResult:
    """用动态规划精确求解 0-1 背包，得到相关性总和最大的子集。

    状态：dp[w] 表示容量恰好不超过 w 时可获得的最大相关性总和。
    时间复杂度 O(n * W)，空间 O(n * W)（保留 keep 表用于回溯选择）。

    适用于中等规模（n、W 不太大）以获得最优解作为对照基准。
    """
    start = time.perf_counter()
    n = len(values)
    W = int(capacity)

    if n == 0 or W <= 0:
        elapsed = time.perf_counter() - start
        return _summarize([], values, weights, "Exact DP (0-1 Knapsack)", elapsed)

    # dp[w]：当前考虑过的物品下，容量 w 的最大价值
    dp = [0.0] * (W + 1)
    # keep[i][w]：第 i 个物品在容量 w 下是否被选中（用于回溯）
    keep = [[False] * (W + 1) for _ in range(n)]

    for i in range(n):
        wi = int(weights[i])
        vi = float(values[i])
        # 逆序遍历容量，保证每件物品只被选一次（0-1 背包）
        for w in range(W, wi - 1, -1):
            cand = dp[w - wi] + vi
            if cand > dp[w]:
                dp[w] = cand
                keep[i][w] = True

    # 回溯找出被选中的物品
    selected: List[int] = []
    w = W
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected.append(i)
            w -= int(weights[i])
    selected.reverse()

    elapsed = time.perf_counter() - start
    return _summarize(
        selected, values, weights, "Exact DP (0-1 Knapsack)", elapsed,
        extra={"optimal_value": dp[W]},
    )


# ===========================================================================
# 3. 贪心近似：按价值密度 v_i / w_i
# ===========================================================================
def greedy_ratio(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
) -> SelectionResult:
    """按性价比（价值/重量）降序贪心选取，低延迟近似背包解。

    这是分数背包最优策略在 0-1 背包上的经典近似，速度极快。
    """
    start = time.perf_counter()

    def ratio(i: int) -> float:
        w = weights[i]
        return values[i] / w if w > 0 else float("inf")

    order = sorted(range(len(values)), key=ratio, reverse=True)

    selected: List[int] = []
    used = 0
    for i in order:
        if used + weights[i] <= capacity:
            selected.append(i)
            used += weights[i]
    elapsed = time.perf_counter() - start
    return _summarize(selected, values, weights, "Greedy (value/weight)", elapsed)


# ===========================================================================
# 4. 贪心次模优化：背包 + 集合覆盖（惩罚冗余）
# ===========================================================================
def greedy_submodular(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
    similarity_matrix: np.ndarray,
    redundancy_lambda: float = 0.5,
    min_relevance: float = 0.1,
    record_trace: bool = False,
) -> SelectionResult:
    """贪心次模优化：在选块时动态惩罚与已选集合高度相似的候选。

    每一步选择「边际增益密度」最大的文本块。边际增益定义为：

        gain(i) = v_i - lambda * max_{j in S} sim(i, j)

    其中 S 为当前已选集合。该目标是单调次模的（facility-location 型），
    贪心可获得 (1 - 1/e) 的近似保证。为兼顾 Token 预算，按
    gain(i) / w_i 选取性价比最高且能放入预算的块。

    参数
    ----
    similarity_matrix : (n, n) 文本块两两余弦相似度矩阵
    redundancy_lambda : 冗余惩罚强度，越大越偏向多样性
    min_relevance     : 检索相关性门槛。相关性低于该阈值的候选块被视为
                        与查询无关的噪声，直接排除——避免 lambda 过大时
                        算法为「多样性」而选入实质无关的干扰块。
    record_trace      : 若为 True，则在 result.extra["trace"] 中记录每一步
                        的详细计算（各候选的价值、冗余惩罚、边际增益、密度
                        及最终选择），供逐步动画演示使用。
    """
    start = time.perf_counter()
    n = len(values)
    selected: List[int] = []
    used = 0
    # 仅保留通过相关性门槛的候选，杜绝引入无关噪声块
    remaining = {i for i in range(n) if values[i] >= min_relevance}
    dropped = [i for i in range(n) if values[i] < min_relevance]

    # max_sim_to_selected[i]：候选 i 与已选集合的最大相似度，增量维护
    max_sim = np.zeros(n, dtype=np.float32)
    trace: List[dict] = []

    while remaining:
        best_i = -1
        best_density = -float("inf")
        step_candidates: List[dict] = []
        for i in list(remaining):
            wi = weights[i]
            penalty = redundancy_lambda * float(max_sim[i])
            gain = values[i] - penalty
            density = (gain / wi) if wi > 0 else float("inf")
            feasible = (used + wi <= capacity)
            positive = gain > 0
            if record_trace:
                step_candidates.append({
                    "index": i, "value": float(values[i]), "weight": wi,
                    "max_sim": float(max_sim[i]), "penalty": penalty,
                    "gain": gain, "density": density,
                    "feasible": feasible, "positive": positive,
                })
            if not feasible or not positive:
                continue
            if density > best_density:
                best_density = density
                best_i = i

        if best_i == -1:
            if record_trace:
                trace.append({
                    "selected_so_far": list(selected), "used": used,
                    "candidates": step_candidates, "chosen": None,
                    "reason": "无可行且有正边际增益的候选，停止",
                })
            break  # 无可行且有正收益的候选

        if record_trace:
            trace.append({
                "selected_so_far": list(selected), "used": used,
                "candidates": step_candidates, "chosen": best_i,
                "chosen_density": best_density, "reason": "选取边际增益密度最高者",
            })

        selected.append(best_i)
        used += weights[best_i]
        remaining.discard(best_i)

        # 增量更新其余候选与已选集合的最大相似度
        row = similarity_matrix[best_i]
        for j in remaining:
            if row[j] > max_sim[j]:
                max_sim[j] = row[j]

    elapsed = time.perf_counter() - start
    return _summarize(
        selected, values, weights, "Greedy Submodular", elapsed,
        extra={
            "redundancy_lambda": redundancy_lambda,
            "min_relevance": min_relevance,
            "dropped_below_threshold": dropped,
            "trace": trace,
        },
    )


# ===========================================================================
# 统一调度入口
# ===========================================================================
def run_algorithm(
    name: str,
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
    similarity_matrix: Optional[np.ndarray] = None,
    redundancy_lambda: float = 0.5,
) -> SelectionResult:
    """根据名称调度对应算法，便于评估脚本统一调用。"""
    name = name.lower()
    if name in ("topk", "naive", "naive_top_k"):
        return naive_top_k(values, weights, capacity)
    if name in ("dp", "knapsack", "dp_knapsack"):
        return dp_knapsack(values, weights, capacity)
    if name in ("greedy", "ratio", "greedy_ratio"):
        return greedy_ratio(values, weights, capacity)
    if name in ("submodular", "greedy_submodular", "diverse"):
        if similarity_matrix is None:
            raise ValueError("次模优化算法需要提供 similarity_matrix")
        return greedy_submodular(
            values, weights, capacity, similarity_matrix, redundancy_lambda
        )
    raise ValueError(f"未知算法: {name}")
