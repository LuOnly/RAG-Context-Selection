# RAG-Context-Selection

cs240 project



\# RAG 上下文选择优化（LLM Context Selection Optimization）仿真系统



将检索增强生成（RAG）中的\*\*上下文选择\*\*问题建模为\*\*组合优化问题\*\*

（0-1 背包 + 集合覆盖 / 次模优化），在严格的 Token 预算 `W` 下，从大量

检索候选文本块中挑选出\*\*信息价值最高、语义冗余最低\*\*的子集，并对比四种

经典选择算法的性能。



> 动机：Liu 等人 \*"Lost in the Middle"\* 指出，长上下文中关键信息被淹没时

> LLM 性能会显著下降。因此在固定上下文窗口下高效选块至关重要。



\---



\## 1. 数学建模



每个候选文本块 `c\_i` 具有：

\- 相关性得分 `v\_i`：与查询 `Q` 的\*\*余弦相似度\*\*（价值）

\- Token 长度 `w\_i`：\*\*重量\*\*



\*\*0-1 背包目标\*\*：在 `Σ w\_i ≤ W` 约束下最大化 `Σ v\_i`。



\*\*扩展（次模优化）\*\*：引入集合覆盖思想，对与已选块高度相似的候选块施加

冗余惩罚，提升信息多样性。



\---



\## 2. 实现的四种算法



| 算法 | 模块函数 | 思想 | 复杂度 |

|------|----------|------|--------|

| 朴素 Top-K | `naive\_top\_k` | 按 `v\_i` 降序选取直到超预算 | `O(n log n)` |

| 精确动态规划 | `dp\_knapsack` | 精确求解 0-1 背包，全局最优 | `O(n·W)` |

| 贪心近似 | `greedy\_ratio` | 按性价比 `v\_i/w\_i` 降序选取 | `O(n log n)` |

| 贪心次模优化 | `greedy\_submodular` | 背包+集合覆盖，惩罚冗余，`(1-1/e)` 近似 | `O(n²)` |



边际增益（次模）：`gain(i) = v\_i - λ · max\_{j∈S} sim(i, j)`，按 `gain(i)/w\_i` 贪心选取。



\---



\## 3. 技术栈与自动降级



| 功能 | 首选 | 自动回退（保证离线可运行） |

|------|------|----------------------------|

| 相关性 `v\_i` | `sentence-transformers` (`all-MiniLM-L6-v2`) | sklearn TF-IDF → 纯 numpy 哈希 TF-IDF |

| Token 重量 `w\_i` | `tiktoken` (`cl100k\_base`) | 字符/单词启发式估计 |

| 可视化 | `matplotlib` | 纯文本表格 |

| 数据集 | `datasets`（MS MARCO） | 内置示例语料 |



即使所有可选库都缺失，系统仍能完整跑通（用 `python main.py info` 查看当前后端）。



\---



\## 4. 目录结构



```

cs240\_project/

├── main.py              # 命令行入口（info / benchmark / demo）

├── requirements.txt

├── README.md

└── src/

&#x20;   ├── tokenizer.py     # Token 计数（重量 w\_i）

&#x20;   ├── embeddings.py    # 嵌入 + 余弦相似度（相关性 v\_i）

&#x20;   ├── algorithms.py    # 四种选择算法

&#x20;   ├── dataset.py       # 内置/MS MARCO/维基语料

&#x20;   ├── pipeline.py      # 端到端管道

&#x20;   └── evaluate.py      # 基准测试 + 图表

```



\---



\## 5. 安装



```bash

pip install -r requirements.txt

```



> 若因网络问题无法安装 `tiktoken` / `sentence-transformers`，无需担心，

> 系统会自动回退到 sklearn TF-IDF 与启发式 Token 计数。



\---



\## 6. 如何运行



```bash

\# 查看当前嵌入 / Token 后端

python main.py info



\# 在内置样本上运行四算法基准测试（默认 Token 预算 W=128）

python main.py benchmark --capacity 128



\# 用更紧的预算更明显地展示算法差异

python main.py benchmark --capacity 40



\# 使用 MS MARCO 子集（需联网+datasets，失败自动回退内置样本）

python main.py benchmark --source ms\_marco --num-samples 5 --capacity 256



\# 端到端管道演示：对单个查询执行某算法并打印生成的提示词

python main.py demo --algorithm submodular --capacity 100

```



\### 更直观的两个演示



```bash

\# A) 并排选块对比：直接「看见」四种算法各自选了哪些文本块

\#    可观察到 Top-K/DP/Greedy 会把两个近重复块都选入，

\#    而次模优化主动丢弃冗余块、转而纳入一个全新角度的块。

python main.py compare                       # 默认 HTTP 样本，最直观

python main.py compare --sample-index 0      # 强化学习样本

python main.py compare --capacity 60 --redundancy-lambda 0.9



\# B) 预算扫描曲线：相关性 vs Token 预算 W（背包问题经典分析图）

\#    展示随预算放宽各算法如何增长、DP 始终为上界。

python main.py sweep --start 20 --stop 160 --step 20



\# C) 次模优化逐步动画：展示贪心每一步如何计算边际增益、惩罚冗余、选块

\#    可看到「冗余块的惩罚随相似块被选入而升高，最终被挤出选择」。

python main.py trace                       # 默认 HTTP 样本

python main.py trace --sample-index 0      # 强化学习样本

python main.py trace --delay 1.0           # 每步停顿 1 秒，形成动画效果

```



`trace` 的核心看点（强化学习样本）：选入块 #0 后，与它高度相似的冗余块 #1

（`max\_sim=0.806`）被施加 `λ·0.806≈0.725` 的惩罚，其边际增益从 `0.734`

骤降到 `0.009`，几乎被完全抵消，从而把预算让给信息更互补的块——这就是

次模优化「动态调低与已选集合相似块分数」的机制。



`compare` 输出示例（HTTP 样本，W=60，λ=0.9）：四种算法的对比一目了然——



| 指标 | Top-K | DP | Greedy | \*\*Submodular\*\* |

|------|-------|----|--------|----------------|

| 总相关性 ↑ | 2.244 | 2.244 | 2.244 | 1.813 |

| 平均冗余 ↓ | 0.708 | 0.708 | 0.708 | \*\*0.443\*\* |

| 近似重复对 ↓ | 1 | 1 | 1 | \*\*0\*\* |



次模优化以少量相关性为代价，\*\*消除了近重复块对、显著降低冗余\*\*，

并纳入了其它算法遗漏的新信息维度——这正是多样性优化的价值。



> Windows PowerShell 若出现中文乱码，先执行 `$env:PYTHONIOENCODING="utf-8"`。



基准测试会打印指标表格，并将对比柱状图保存为 `benchmark\_results.png`。



\---



\## 7. 评估指标



\- \*\*总相关性得分\*\*：所选块 `v\_i` 之和（越高越好）

\- \*\*总 Token 数\*\*：所选块 `w\_i` 之和（需 ≤ `W`）

\- \*\*近似比\*\*：算法总相关性 / DP 最优值（越接近 1 越好）

\- \*\*冗余度\*\*：所选块两两平均余弦相似度（越低越多样）

\- \*\*执行时间\*\*：算法耗时（毫秒）



\### 典型结果解读（W=40）



| 算法 | 总相关性 | 近似比 | 冗余度 |

|------|---------|--------|--------|

| Exact DP | \*\*最高\*\* | \*\*1.000\*\* | 中 |

| Naive Top-K | 高 | 0.991 | 高 |

| Greedy ratio | 中 | 0.965 | 低 |

| Greedy Submodular | 略低 | 0.953 | \*\*最低（多样性最佳）\*\* |



DP 给出相关性最优解；次模优化以少量相关性为代价换取最低冗余，验证了

多样性优化的有效性。



\---



\## 8. 在自己的数据上使用



```python

from src.pipeline import ContextSelectionPipeline



pipeline = ContextSelectionPipeline()

out = pipeline.run(

&#x20;   query="你的查询",

&#x20;   chunks=\["候选块1", "候选块2", "..."],

&#x20;   algorithm="submodular",   # topk / dp / greedy / submodular

&#x20;   capacity=256,             # Token 预算 W

)

print(out\["result"].total\_value, out\["result"].total\_weight)

print(out\["prompt"])          # 拼接好的最终提示词

```



