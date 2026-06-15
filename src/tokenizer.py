"""Token 计数模块。

在背包建模中，每个文本块 c_i 的「重量」w_i 即其 Token 数量。
本模块优先使用 OpenAI 的 tiktoken 库进行精确计数；当该库不可用
（例如离线环境或 Python 版本过新尚无 wheel）时，自动回退到基于
字符/单词的启发式估计，从而保证整套系统始终可运行。
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# 尝试加载 tiktoken。失败则使用回退方案，并记录原因供日志显示。
# ---------------------------------------------------------------------------
try:
    import tiktoken  # type: ignore

    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """统一的 Token 计数器。

    参数
    ----
    encoding_name : str
        tiktoken 的编码名称。``cl100k_base`` 对应 GPT-3.5/4 系列。
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoder = None
        self.backend = "fallback-heuristic"

        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
                self.backend = f"tiktoken({encoding_name})"
            except Exception:
                # 编码名无效或下载失败时退回启发式
                self._encoder = None
                self.backend = "fallback-heuristic"

    # ------------------------------------------------------------------
    def count(self, text: str) -> int:
        """返回单段文本的 Token 数量（重量 w_i）。"""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return self._heuristic_count(text)

    def count_batch(self, texts: List[str]) -> List[int]:
        """批量计数。"""
        return [self.count(t) for t in texts]

    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_count(text: str) -> int:
        """启发式 Token 估计（tiktoken 不可用时使用）。

        经验法则：英文中约 4 个字符 ≈ 1 个 Token，同时单词数也是一个
        合理下界。这里取两者的折中，使估计在常见文本上接近真实值。
        """
        char_estimate = len(text) / 4.0
        word_estimate = len(re.findall(r"\w+", text))
        # 取二者均值，至少为 1
        return max(1, int(round((char_estimate + word_estimate) / 2.0)))


# 模块级默认实例，方便直接调用
_default_counter: TokenCounter | None = None


def get_default_counter() -> TokenCounter:
    """获取（惰性创建）全局默认 Token 计数器。"""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter()
    return _default_counter


def count_tokens(text: str) -> int:
    """便捷函数：使用默认计数器统计 Token 数。"""
    return get_default_counter().count(text)


if __name__ == "__main__":
    counter = TokenCounter()
    print(f"当前 Token 计数后端: {counter.backend}")
    demo = "Retrieval-Augmented Generation selects the best context chunks."
    print(f"示例文本 Token 数: {counter.count(demo)}")
