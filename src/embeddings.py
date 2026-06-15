"""嵌入与相关性计算模块。

负责：
1. 将查询 Q 与候选文本块 c_i 编码为向量；
2. 计算文本块与查询的余弦相似度，作为相关性得分 v_i；
3. 计算文本块两两之间的相似度矩阵，供次模优化算法惩罚冗余。

后端优先级（自动降级，保证始终可运行）：
    1) sentence-transformers (all-MiniLM-L6-v2) —— 论文推荐的标准嵌入模型
    2) scikit-learn TF-IDF                     —— 轻量回退
    3) 纯 numpy 哈希 TF-IDF                      —— 零额外依赖的最终回退
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 后端探测
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    _SKLEARN_AVAILABLE = False


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """对矩阵按行做 L2 归一化，便于用点积直接得到余弦相似度。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return matrix / norms


class EmbeddingModel:
    """统一的嵌入接口，封装三种后端并对外暴露一致的 API。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.backend = "hashing-tfidf"
        self._st_model = None
        self._tfidf = None

        if _ST_AVAILABLE:
            try:
                self._st_model = SentenceTransformer(model_name)
                self.backend = f"sentence-transformers({model_name})"
                return
            except Exception:
                self._st_model = None

        if _SKLEARN_AVAILABLE:
            # TF-IDF 需要先 fit；这里延迟到 encode 时根据语料构建
            self.backend = "sklearn-tfidf"

    # ------------------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """将一组文本编码为 (N, dim) 的 L2 归一化矩阵。"""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self._st_model is not None:
            vecs = self._st_model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
            return vecs.astype(np.float32)

        if self.backend == "sklearn-tfidf":
            return self._encode_sklearn(texts)

        return self._encode_hashing(texts)

    # ------------------------------------------------------------------
    def _encode_sklearn(self, texts: List[str]) -> np.ndarray:
        """使用 sklearn 的 TF-IDF 进行编码（每次按当前语料 fit）。"""
        vectorizer = TfidfVectorizer()
        try:
            mat = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        except ValueError:
            # 语料为空词表等极端情况，退回哈希方案
            return self._encode_hashing(texts)
        return _l2_normalize(mat)

    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())

    def _encode_hashing(self, texts: List[str], dim: int = 512) -> np.ndarray:
        """纯 numpy 实现的哈希 TF-IDF（无任何第三方依赖）。

        通过特征哈希把词映射到固定维度，再乘以 IDF 权重并归一化。
        """
        n = len(texts)
        tokenized = [self._tokenize(t) for t in texts]

        # 计算文档频率 df 以得到 IDF
        df: dict[int, int] = {}
        for toks in tokenized:
            for h in {hash(tok) % dim for tok in toks}:
                df[h] = df.get(h, 0) + 1

        mat = np.zeros((n, dim), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            if not toks:
                continue
            tf: dict[int, int] = {}
            for tok in toks:
                h = hash(tok) % dim
                tf[h] = tf.get(h, 0) + 1
            for h, freq in tf.items():
                idf = math.log((1 + n) / (1 + df.get(h, 0))) + 1.0
                mat[i, h] = (freq / len(toks)) * idf
        return _l2_normalize(mat)

    # ------------------------------------------------------------------
    def relevance_scores(self, query: str, chunks: Sequence[str]) -> np.ndarray:
        """计算每个文本块与查询的余弦相似度，作为相关性得分 v_i。

        注意：为保证 TF-IDF 后端共享同一词表，查询与文本块一起编码。
        返回的相似度被裁剪到 [0, 1]，避免出现负权重影响背包求解。
        """
        all_vecs = self.encode([query, *chunks])
        q_vec = all_vecs[0:1]
        c_vecs = all_vecs[1:]
        sims = (c_vecs @ q_vec.T).ravel()
        return np.clip(sims, 0.0, 1.0)

    def pairwise_similarity(self, chunks: Sequence[str]) -> np.ndarray:
        """计算文本块两两余弦相似度矩阵，供次模优化惩罚冗余。"""
        vecs = self.encode(chunks)
        sim = vecs @ vecs.T
        return np.clip(sim, 0.0, 1.0)


if __name__ == "__main__":
    model = EmbeddingModel()
    print(f"当前嵌入后端: {model.backend}")
    q = "What is reinforcement learning?"
    cs = [
        "Reinforcement learning trains agents via rewards.",
        "The capital of France is Paris.",
        "RL uses a reward signal to optimize a policy.",
    ]
    print("相关性得分:", model.relevance_scores(q, cs))
