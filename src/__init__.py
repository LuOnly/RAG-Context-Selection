"""RAG 上下文选择优化（LLM Context Selection Optimization）仿真系统。

本包将「从大量检索候选文本块中挑选信息价值最高、冗余最低的子集」
建模为组合优化问题（0-1 背包 + 集合覆盖 / 次模优化），并提供四种
基准选择算法用于对比实验。

模块结构：
    tokenizer  : Token 计数（重量 w_i）
    embeddings : 嵌入与余弦相似度（相关性得分 v_i）
    algorithms : 四种文本块选择算法
    dataset    : 测试语料加载 / 模拟
    pipeline   : 端到端管道
    evaluate   : 基准测试与可视化
"""

__all__ = [
    "tokenizer",
    "embeddings",
    "algorithms",
    "dataset",
    "pipeline",
    "evaluate",
]
