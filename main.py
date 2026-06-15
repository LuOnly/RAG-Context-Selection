"""RAG 上下文选择优化系统 —— 命令行入口。

用法示例：
    # 1) 在内置样本上运行四算法基准测试（默认）
    python main.py benchmark --capacity 128

    # 2) 使用 MS MARCO 子集（需联网 + datasets 库，失败自动回退内置样本）
    python main.py benchmark --source ms_marco --num-samples 5 --capacity 256

    # 3) 演示端到端管道：对单个查询执行某算法并打印生成的提示词
    python main.py demo --algorithm submodular --capacity 100

    # 4) 检查当前环境使用的嵌入 / Token 后端
    python main.py info
"""

from __future__ import annotations

import argparse

from src.dataset import built_in_samples, load_dataset_samples
from src.evaluate import (
    capacity_sweep,
    compare_selections,
    run_benchmark,
    trace_submodular,
)
from src.pipeline import ContextSelectionPipeline


def cmd_info(_: argparse.Namespace) -> None:
    """打印当前嵌入与 Token 计数后端。"""
    pipeline = ContextSelectionPipeline()
    print("===== 运行环境后端 =====")
    print(f"嵌入模型后端 : {pipeline.embedder.backend}")
    print(f"Token 计数后端: {pipeline.counter.backend}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """运行四种算法的基准对比。"""
    samples = load_dataset_samples(source=args.source, num_samples=args.num_samples)
    run_benchmark(
        samples,
        capacity=args.capacity,
        redundancy_lambda=args.redundancy_lambda,
        plot=not args.no_plot,
        plot_path=args.plot_path,
    )


def cmd_demo(args: argparse.Namespace) -> None:
    """演示单查询端到端管道，并打印拼接出的提示词。"""
    pipeline = ContextSelectionPipeline()
    sample = built_in_samples()[args.sample_index]
    print(f"嵌入后端: {pipeline.embedder.backend} | Token 后端: {pipeline.counter.backend}")
    print(f"\n查询: {sample.query}")
    print(f"候选块数: {len(sample.chunks)}\n")

    out = pipeline.run(
        sample.query,
        sample.chunks,
        algorithm=args.algorithm,
        capacity=args.capacity,
        redundancy_lambda=args.redundancy_lambda,
    )
    res = out["result"]
    print(f"算法           : {res.algorithm}")
    print(f"选中块下标     : {res.selected_indices}")
    print(f"总相关性得分   : {res.total_value:.4f}")
    print(f"总 Token 数    : {res.total_weight} / {args.capacity}")
    print(f"执行时间       : {res.elapsed_seconds * 1000:.3f} ms")
    print("\n===== 生成的最终提示词 =====\n")
    print(out["prompt"])


def cmd_compare(args: argparse.Namespace) -> None:
    """并排展示四种算法在同一查询上各自选中的文本块。"""
    pipeline = ContextSelectionPipeline()
    sample = built_in_samples()[args.sample_index]
    compare_selections(
        sample,
        capacity=args.capacity,
        redundancy_lambda=args.redundancy_lambda,
        pipeline=pipeline,
    )


def cmd_sweep(args: argparse.Namespace) -> None:
    """扫描不同 Token 预算，绘制相关性 vs W 曲线。"""
    samples = load_dataset_samples(source=args.source, num_samples=args.num_samples)
    capacities = list(range(args.start, args.stop + 1, args.step))
    capacity_sweep(
        samples,
        capacities=capacities,
        redundancy_lambda=args.redundancy_lambda,
        out_path=args.plot_path,
    )


def cmd_trace(args: argparse.Namespace) -> None:
    """逐步动画展示次模优化的贪心选块过程。"""
    sample = built_in_samples()[args.sample_index]
    trace_submodular(
        sample,
        capacity=args.capacity,
        redundancy_lambda=args.redundancy_lambda,
        delay=args.delay,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG 上下文选择优化仿真系统（背包 / 集合覆盖 / 次模优化）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="查看当前嵌入 / Token 后端")
    p_info.set_defaults(func=cmd_info)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="四算法基准测试")
    p_bench.add_argument("--source", default="builtin",
                         choices=["builtin", "ms_marco"], help="数据来源")
    p_bench.add_argument("--num-samples", type=int, default=5,
                         help="MS MARCO 取样数量")
    p_bench.add_argument("--capacity", type=int, default=128,
                         help="Token 预算 W")
    p_bench.add_argument("--redundancy-lambda", type=float, default=0.5,
                         help="次模优化冗余惩罚强度 lambda")
    p_bench.add_argument("--no-plot", action="store_true", help="不绘制图表")
    p_bench.add_argument("--plot-path", default="benchmark_results.png",
                         help="图表保存路径")
    p_bench.set_defaults(func=cmd_benchmark)

    # demo
    p_demo = sub.add_parser("demo", help="单查询端到端管道演示")
    p_demo.add_argument("--algorithm", default="dp",
                        choices=["topk", "dp", "greedy", "submodular"],
                        help="选择算法")
    p_demo.add_argument("--sample-index", type=int, default=0,
                        help="内置样本下标")
    p_demo.add_argument("--capacity", type=int, default=100, help="Token 预算 W")
    p_demo.add_argument("--redundancy-lambda", type=float, default=0.5,
                        help="次模优化冗余惩罚强度 lambda")
    p_demo.set_defaults(func=cmd_demo)

    # compare：并排选块对比
    p_cmp = sub.add_parser("compare", help="并排展示各算法选中的文本块")
    p_cmp.add_argument("--sample-index", type=int, default=1, help="内置样本下标")
    p_cmp.add_argument("--capacity", type=int, default=60, help="Token 预算 W")
    p_cmp.add_argument("--redundancy-lambda", type=float, default=0.9,
                       help="次模优化冗余惩罚强度 lambda")
    p_cmp.set_defaults(func=cmd_compare)

    # sweep：Token 预算扫描曲线
    p_sweep = sub.add_parser("sweep", help="相关性 vs Token 预算扫描曲线")
    p_sweep.add_argument("--source", default="builtin",
                         choices=["builtin", "ms_marco"], help="数据来源")
    p_sweep.add_argument("--num-samples", type=int, default=5,
                         help="MS MARCO 取样数量")
    p_sweep.add_argument("--start", type=int, default=20, help="预算起点")
    p_sweep.add_argument("--stop", type=int, default=200, help="预算终点")
    p_sweep.add_argument("--step", type=int, default=20, help="预算步长")
    p_sweep.add_argument("--redundancy-lambda", type=float, default=0.5,
                         help="次模优化冗余惩罚强度 lambda")
    p_sweep.add_argument("--plot-path", default="capacity_sweep.png",
                         help="曲线图保存路径")
    p_sweep.set_defaults(func=cmd_sweep)

    # trace：次模优化逐步动画
    p_trace = sub.add_parser("trace", help="逐步动画展示次模优化贪心过程")
    p_trace.add_argument("--sample-index", type=int, default=1, help="内置样本下标")
    p_trace.add_argument("--capacity", type=int, default=60, help="Token 预算 W")
    p_trace.add_argument("--redundancy-lambda", type=float, default=0.9,
                         help="次模优化冗余惩罚强度 lambda")
    p_trace.add_argument("--delay", type=float, default=0.0,
                         help="每步停顿秒数（>0 形成动画效果，如 1.0）")
    p_trace.set_defaults(func=cmd_trace)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
