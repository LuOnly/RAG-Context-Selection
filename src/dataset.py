"""测试数据集模块。

提供三种获取 RAG 候选语料的方式：

1. built_in_samples()  —— 内置的若干 (查询, 候选文本块) 示例，零依赖、
   可离线运行，方便快速验证整套系统。
2. load_ms_marco()     —— 通过 HuggingFace ``datasets`` 加载 MS MARCO QA
   子集（需联网与 datasets 库），自动整理为统一结构。
3. build_wikipedia_corpus() —— 基于给定维基百科段落构建自定义 RAG 语料。

统一数据结构 RAGSample：
    query  : str
    chunks : List[str]   候选文本块（部分相关、部分干扰/冗余）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGSample:
    """单条 RAG 测试样本：一个查询 + 一组候选文本块。

    labels 为可选的「人工标注块类型」，仅用于让对比演示更直观地展示
    不同算法是否选入了冗余/干扰块。取值约定：
        "relevant"   高相关
        "redundant"  相关但与其它块语义重复（冗余）
        "distractor" 不相关干扰
        "weak"       弱相关
    真实数据集（如 MS MARCO）通常没有此标注，留空即可。
    """

    query: str
    chunks: List[str]
    labels: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# 1. 内置示例（默认数据源，零依赖）
# ---------------------------------------------------------------------------
def built_in_samples() -> List[RAGSample]:
    """返回内置的演示样本。

    每个样本刻意混入：高度相关块、相关但冗余块、以及不相关干扰块，
    以便突出不同算法（尤其是次模优化）在多样性上的差异。
    """
    samples: List[RAGSample] = []

    # 样本一：强化学习
    samples.append(
        RAGSample(
            query="How does reinforcement learning train an agent?",
            chunks=[
                # 高相关
                "Reinforcement learning trains an agent by letting it interact "
                "with an environment and learn from reward signals over time.",
                # 与上句冗余（语义重复）
                "In reinforcement learning, an agent learns a policy by maximizing "
                "cumulative reward received through interaction with its environment.",
                # 相关但角度不同（值得作为多样性补充）
                "Key components of RL include states, actions, rewards, and a policy "
                "that maps states to actions.",
                # 相关：具体算法
                "Q-learning is a value-based RL algorithm that updates action values "
                "using the Bellman equation.",
                # 不相关干扰
                "The Eiffel Tower is one of the most visited monuments in Paris.",
                "Photosynthesis converts sunlight into chemical energy in plants.",
                # 弱相关
                "Deep learning uses multilayer neural networks to learn representations.",
            ],
            labels=["relevant", "redundant", "relevant", "relevant",
                    "distractor", "distractor", "weak"],
        )
    )

    # 样本二：HTTP 协议
    samples.append(
        RAGSample(
            query="What is the difference between HTTP and HTTPS?",
            chunks=[
                "HTTPS is HTTP layered over TLS/SSL, encrypting data between the "
                "browser and the server to ensure confidentiality and integrity.",
                "HTTP transmits data in plaintext, while HTTPS encrypts the traffic "
                "using TLS so that attackers cannot read it.",  # 与上句冗余
                "HTTPS uses port 443 by default whereas HTTP uses port 80.",
                "A TLS handshake exchanges certificates and negotiates session keys "
                "before encrypted communication begins.",
                "Bananas are a good source of potassium and dietary fiber.",  # 干扰
                "DNS translates human-readable domain names into IP addresses.",  # 弱相关
            ],
            labels=["relevant", "redundant", "relevant", "relevant",
                    "distractor", "weak"],
        )
    )

    # 样本三：气候变化
    samples.append(
        RAGSample(
            query="What are the main causes of global climate change?",
            chunks=[
                "Burning fossil fuels releases carbon dioxide, the primary greenhouse "
                "gas driving global warming.",
                "Greenhouse gas emissions from coal, oil, and gas combustion are the "
                "leading cause of modern climate change.",  # 冗余
                "Deforestation reduces the planet's capacity to absorb CO2, "
                "amplifying warming.",
                "Methane from agriculture and livestock is a potent greenhouse gas.",
                "The recipe calls for two cups of flour and a pinch of salt.",  # 干扰
                "Industrial processes and cement production also emit greenhouse gases.",
            ],
            labels=["relevant", "redundant", "relevant", "relevant",
                    "distractor", "relevant"],
        )
    )

    return samples


# ---------------------------------------------------------------------------
# 2. MS MARCO QA 子集（可选，需联网 + datasets 库）
# ---------------------------------------------------------------------------
def load_ms_marco(
    num_samples: int = 5,
    split: str = "validation",
) -> List[RAGSample]:
    """加载 MS MARCO QA 子集并整理为 RAGSample 列表。

    MS MARCO 的每条记录包含一个 query 与多个 passages，天然契合
    「一个查询 + 一组候选文本块」的 RAG 设定。

    若 datasets 不可用或下载失败，将抛出异常，调用方应自行回退。
    """
    from datasets import load_dataset  # 延迟导入，避免硬依赖

    ds = load_dataset("ms_marco", "v1.1", split=split, streaming=False)
    samples: List[RAGSample] = []
    for row in ds:
        query = row["query"]
        passages = row.get("passages", {})
        texts = passages.get("passage_text", [])
        if query and texts:
            samples.append(RAGSample(query=query, chunks=list(texts)))
        if len(samples) >= num_samples:
            break
    return samples


# ---------------------------------------------------------------------------
# 3. 自定义维基百科语料
# ---------------------------------------------------------------------------
def build_wikipedia_corpus(query: str, passages: List[str]) -> RAGSample:
    """用给定的维基百科段落构建一个自定义 RAG 样本。"""
    return RAGSample(query=query, chunks=list(passages))


# ---------------------------------------------------------------------------
# 统一加载入口
# ---------------------------------------------------------------------------
def load_dataset_samples(
    source: str = "builtin",
    num_samples: int = 5,
) -> List[RAGSample]:
    """统一数据加载接口。

    source:
        "builtin"   使用内置示例（默认，零依赖）
        "ms_marco"  尝试加载 MS MARCO，失败则自动回退到内置示例
    """
    if source == "ms_marco":
        try:
            return load_ms_marco(num_samples=num_samples)
        except Exception as exc:  # 联网/依赖失败时回退
            print(f"[dataset] 加载 MS MARCO 失败，回退到内置样本: {exc}")
            return built_in_samples()
    return built_in_samples()


if __name__ == "__main__":
    for s in built_in_samples():
        print("Q:", s.query)
        print(f"  候选块数: {len(s.chunks)}")
