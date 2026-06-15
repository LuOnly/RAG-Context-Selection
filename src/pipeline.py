"""端到端管道模块。

把整个上下文选择流程串成一个函数：

    查询 Q + 候选文本块  ->  计算相关性 v_i  ->  计算 Token 重量 w_i
        ->  执行选择算法（背包/贪心/次模）  ->  拼接出最终提示词上下文

对外暴露 ContextSelectionPipeline 类，便于在评估脚本中复用同一份
（已缓存的）相关性得分、Token 重量与相似度矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import algorithms
from .embeddings import EmbeddingModel
from .tokenizer import TokenCounter


@dataclass
class PreparedContext:
    """预处理后的上下文：相关性、重量、相似度矩阵都已就绪。"""

    query: str
    chunks: List[str]
    values: np.ndarray          # 相关性得分 v_i
    weights: List[int]          # Token 重量 w_i
    similarity_matrix: np.ndarray  # 文本块两两相似度（供次模优化）


class ContextSelectionPipeline:
    """RAG 上下文选择端到端管道。"""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.embedder = embedding_model or EmbeddingModel()
        self.counter = token_counter or TokenCounter()

    # ------------------------------------------------------------------
    def prepare(self, query: str, chunks: Sequence[str]) -> PreparedContext:
        """计算相关性得分 v_i、Token 重量 w_i 与相似度矩阵。

        这一步与具体算法无关，可被多种算法复用，避免重复计算嵌入。
        """
        chunks = list(chunks)
        values = self.embedder.relevance_scores(query, chunks)
        weights = self.counter.count_batch(chunks)
        similarity = self.embedder.pairwise_similarity(chunks)
        return PreparedContext(
            query=query,
            chunks=chunks,
            values=values,
            weights=weights,
            similarity_matrix=similarity,
        )

    # ------------------------------------------------------------------
    def select(
        self,
        prepared: PreparedContext,
        algorithm: str = "dp",
        capacity: int = 256,
        redundancy_lambda: float = 0.5,
    ) -> algorithms.SelectionResult:
        """在预处理结果上运行指定的选择算法。"""
        return algorithms.run_algorithm(
            algorithm,
            list(prepared.values),
            prepared.weights,
            capacity,
            similarity_matrix=prepared.similarity_matrix,
            redundancy_lambda=redundancy_lambda,
        )

    # ------------------------------------------------------------------
    def assemble_prompt(
        self,
        prepared: PreparedContext,
        result: algorithms.SelectionResult,
    ) -> str:
        """根据选择结果拼接最终提示词上下文。

        被选中的文本块按其在原列表中的顺序（即检索顺序）拼接，
        并附上查询，形成可直接喂给 LLM 的提示词。
        """
        ordered = sorted(result.selected_indices)
        context_blocks = [f"[{rank + 1}] {prepared.chunks[i]}"
                          for rank, i in enumerate(ordered)]
        context = "\n".join(context_blocks)
        prompt = (
            "请根据以下检索到的上下文回答问题。\n\n"
            f"### 上下文\n{context}\n\n"
            f"### 问题\n{prepared.query}\n\n"
            "### 回答\n"
        )
        return prompt

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        chunks: Sequence[str],
        algorithm: str = "dp",
        capacity: int = 256,
        redundancy_lambda: float = 0.5,
    ) -> dict:
        """一站式执行：预处理 -> 选择 -> 拼接提示词。

        返回包含选择结果、最终提示词与预处理数据的字典。
        """
        prepared = self.prepare(query, chunks)
        result = self.select(prepared, algorithm, capacity, redundancy_lambda)
        prompt = self.assemble_prompt(prepared, result)
        return {
            "prepared": prepared,
            "result": result,
            "prompt": prompt,
        }


if __name__ == "__main__":
    from .dataset import built_in_samples

    pipeline = ContextSelectionPipeline()
    sample = built_in_samples()[0]
    out = pipeline.run(sample.query, sample.chunks, algorithm="dp", capacity=80)
    res = out["result"]
    print(f"算法: {res.algorithm}")
    print(f"选中块: {res.selected_indices}")
    print(f"总相关性: {res.total_value:.4f}  总Token: {res.total_weight}")
    print("\n生成的提示词:\n")
    print(out["prompt"])
