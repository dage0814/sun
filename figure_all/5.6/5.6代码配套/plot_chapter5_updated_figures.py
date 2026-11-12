from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from print_typography import calibrate_print_text, configure_serif_fonts


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "chapters" / "fig-chapter-5"
SOURCE_DATA = Path(__file__).resolve().parent / "source_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DATA.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": ["Times New Roman", "SimSun"],
        "font.serif": ["Times New Roman", "SimSun"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9.2,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#C9D2DC",
        "grid.linewidth": 0.65,
        "grid.alpha": 0.75,
    }
)
configure_serif_fonts()

BLUE = "#0057B8"
BLUE_LIGHT = "#00A6D6"
ORANGE = "#E66101"
GREEN = "#008A5B"
RED = "#C51B29"
PURPLE = "#7B3294"
GRAY = "#59636E"
GRAY_LIGHT = "#CBD2D9"
INK = "#111827"


source_rows: list[dict[str, object]] = []


def add_source(figure: str, configuration: str, metric: str, mean: float,
               sd: float | None = None, lower: float | None = None,
               upper: float | None = None, p_value: float | None = None) -> None:
    source_rows.append(
        {
            "figure": figure,
            "configuration": configuration,
            "metric": metric,
            "mean": mean,
            "sd": "" if sd is None else sd,
            "ci_lower": "" if lower is None else lower,
            "ci_upper": "" if upper is None else upper,
            "holm_p": "" if p_value is None else p_value,
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    # Materialize lazy tick/legend labels, then change text only.
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    if stem == "5.3.5核心机制MCC增量的Bootstrap置信区间":
        calibrate_print_text(fig)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def style_axis(ax: mpl.axes.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#66717C")
    ax.spines["bottom"].set_color("#66717C")


def draw_error_line(ax: mpl.axes.Axes, x: np.ndarray, mean: np.ndarray,
                    sd: np.ndarray, label: str, color: str, marker: str) -> None:
    ax.errorbar(
        x, mean, yerr=sd, color=color, marker=marker, markersize=5.2,
        linewidth=1.8, elinewidth=1.0, capsize=3.0, label=label, zorder=3,
    )


def figure_rq1_subperiod() -> None:
    stem = "5.2.1CSF-GP测试子区间预测稳定性"
    periods = ["前1/3", "中1/3", "后1/3"]
    x = np.arange(3)
    data = {
        "扩展算子搜索": {
            "IC": ([0.0670, 0.0740, 0.0620], [0.0091, 0.0084, 0.0100]),
            "RankIC": ([0.0760, 0.0810, 0.0640], [0.0104, 0.0092, 0.0113]),
        },
        "CSF-GP": {
            "IC": ([0.0790, 0.1050, 0.0743], [0.0070, 0.0063, 0.0077]),
            "RankIC": ([0.0680, 0.1140, 0.0856], [0.0087, 0.0071, 0.0080]),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharex=True)
    for ax, metric in zip(axes, ["IC", "RankIC"]):
        for method, color, marker in [("扩展算子搜索", GRAY, "s"), ("CSF-GP", BLUE, "o")]:
            mean, sd = map(np.asarray, data[method][metric])
            draw_error_line(ax, x, mean, sd, method, color, marker)
            for period, m, s in zip(periods, mean, sd):
                add_source(stem, f"{method}-{period}", metric, float(m), float(s))
        ax.set_title(metric)
        ax.set_xticks(x, periods)
        ax.set_xlabel("测试子区间")
        ax.set_ylabel(f"{metric}（均值 ± 标准差）")
        ax.set_ylim(0.045, 0.128)
        style_axis(ax)
    axes[0].legend(loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.90, wspace=0.24)
    save_figure(fig, stem)


def figure_rq1_complexity() -> None:
    stem = "5.2.1CSF-GP复杂度与因子质量权衡"
    labels = ["完整CSF-GP", "移除复杂度约束", "移除相似度过滤", "移除精英保留", "移除Hoist变异"]
    nodes = np.array([9.8, 18.6, 10.4, 10.1, 13.7])
    nodes_sd = np.array([0.6, 1.8, 0.7, 0.8, 1.3])
    rankic = np.array([0.0890, 0.0660, 0.0810, 0.0490, 0.0820])
    rankic_sd = np.array([0.0066, 0.0092, 0.0079, 0.0113, 0.0073])
    corr = np.array([0.38, 0.42, 0.67, 0.45, 0.47])
    norm = Normalize(vmin=0.35, vmax=0.70)
    cmap = mpl.colormaps["viridis_r"]
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    for i, label in enumerate(labels):
        color = BLUE if i == 0 else cmap(norm(corr[i]))
        ax.errorbar(nodes[i], rankic[i], xerr=nodes_sd[i], yerr=rankic_sd[i],
                    fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.9)
        ax.scatter(nodes[i], rankic[i], s=95 if i == 0 else 66,
                   color=color, edgecolor="white", linewidth=0.8, zorder=4)
        offset = (7, 5) if i != 1 else (-90, 5)
        ax.annotate(label, (nodes[i], rankic[i]), xytext=offset,
                    textcoords="offset points", fontsize=8.2, color=INK)
        add_source(stem, label, "平均公式节点数", float(nodes[i]), float(nodes_sd[i]))
        add_source(stem, label, "RankIC", float(rankic[i]), float(rankic_sd[i]))
        add_source(stem, label, "因子平均绝对相关系数", float(corr[i]))
    ax.set_xlabel("平均公式节点数（越低越简洁）")
    ax.set_ylabel("测试集RankIC（越高越好）")
    ax.set_xlim(7.8, 21.0)
    ax.set_ylim(0.030, 0.104)
    style_axis(ax, axis="both")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("因子平均绝对相关系数（越低越好）")
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.16, top=0.96)
    save_figure(fig, stem)


def figure_rq2_ranking() -> None:
    stem = "5.2.2不同文本编码方法的行业排序性能"
    methods = ["TF-IDF", "Word2Vec-Mean", "BERT-Base", "FinBERT-Sentiment", "FinFE-BERT"]
    metrics = {
        "NDCG@3": ([0.6020, 0.6270, 0.6810, 0.6940, 0.7530], [0.0100, 0.0110, 0.0120, 0.0053, 0.0080]),
        "NDCG@9": ([0.7180, 0.7390, 0.7810, 0.7700, 0.8240], [0.0082, 0.0073, 0.0093, 0.0056, 0.0060]),
        "RankIC": ([0.1020, 0.1160, 0.1640, 0.1450, 0.1570], [0.0200, 0.0190, 0.0210, 0.0170, 0.0150]),
    }
    y = np.arange(len(methods))[::-1]
    fig = plt.figure(figsize=(7.25, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    offsets = {"NDCG@3": 0.11, "NDCG@9": -0.11}
    colors = {"NDCG@3": ORANGE, "NDCG@9": BLUE}
    for metric in ["NDCG@3", "NDCG@9"]:
        mean, sd = map(np.asarray, metrics[metric])
        ax1.errorbar(mean, y + offsets[metric], xerr=sd, fmt="o", color=colors[metric],
                     markersize=5, elinewidth=1.0, capsize=2.5, label=metric)
        for method, m, s in zip(methods, mean, sd):
            add_source(stem, method, metric, float(m), float(s))
    mean, sd = map(np.asarray, metrics["RankIC"])
    point_colors = [BLUE if method == "FinFE-BERT" else GRAY for method in methods]
    for yi, m, s, c, method in zip(y, mean, sd, point_colors, methods):
        ax2.errorbar(m, yi, xerr=s, fmt="o", color=c, markersize=5,
                     elinewidth=1.0, capsize=2.5)
        add_source(stem, method, "RankIC", float(m), float(s))
    ax1.set_yticks(y, methods)
    ax1.set_xlabel("NDCG（均值 ± 标准差）")
    ax1.set_xlim(0.56, 0.85)
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.53, 0.985),
               ncol=2, frameon=False, columnspacing=1.6, handletextpad=0.6)
    ax2.tick_params(labelleft=False)
    ax2.set_xlabel("RankIC（均值 ± 标准差）")
    ax2.set_xlim(0.065, 0.205)
    for ax in [ax1, ax2]:
        style_axis(ax, axis="x")
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.16, top=0.88)
    save_figure(fig, stem)


def figure_rq2_geometry() -> None:
    stem = "5.2.2行业表示分离度与排序有效性"
    configs = ["移除行业查询注意力", "完整FinFE-BERT", "随机置换行业参数组"]
    data = {
        "行业间区分度": ([0.1040, 0.2180, 0.2160], [0.0090, 0.0140, 0.0150]),
        "轮廓系数": ([0.0610, 0.1760, 0.1720], [0.0100, 0.0120, 0.0130]),
        "NDCG@3": ([0.5140, 0.7530, 0.7210], [0.0310, 0.0080, 0.0110]),
        "RankIC": ([0.1980, 0.1570, 0.2040], [0.0200, 0.0150, 0.0190]),
    }
    y = np.arange(3)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.65), sharey=True)
    for ax, metric_pair, title in zip(
        axes,
        [("行业间区分度", "轮廓系数"), ("NDCG@3", "RankIC")],
        ["表示空间分离", "排序有效性"],
    ):
        for metric, color, offset, marker in [
            (metric_pair[0], BLUE, 0.10, "o"),
            (metric_pair[1], ORANGE, -0.10, "s"),
        ]:
            mean, sd = map(np.asarray, data[metric])
            ax.errorbar(mean, y + offset, xerr=sd, fmt=marker, color=color,
                        markersize=5, elinewidth=1.0, capsize=2.5, label=metric)
            for config, m, s in zip(configs, mean, sd):
                add_source(stem, config, metric, float(m), float(s))
        ax.set_title(title, y=1.16)
        ax.set_xlabel("指标值（均值 ± 标准差）")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
                  frameon=False, columnspacing=1.0, handletextpad=0.45)
        style_axis(ax, axis="x")
    axes[0].set_yticks(y, configs)
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlim(0.035, 0.25)
    axes[1].set_xlim(0.10, 0.80)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.16, top=0.78, wspace=0.22)
    save_figure(fig, stem)


def figure_rq3_fusion() -> None:
    stem = "5.2.3多级双路交叉注意力的递进增益"
    methods = ["线性加权", "拼接+MLP", "CA1", "CA1+CA2", "完整模型\n(CA1+CA2+CA3)"]
    data = {
        "ACC": ([0.6821, 0.6907, 0.6996, 0.7119, 0.7273], [0.0095, 0.0092, 0.0083, 0.0077, 0.0068]),
        "F1": ([0.6964, 0.7041, 0.7128, 0.7245, 0.7400], [0.0094, 0.0083, 0.0082, 0.0066, 0.0061]),
        "MCC": ([0.2116, 0.2209, 0.2337, 0.2518, 0.2714], [0.0130, 0.0130, 0.0110, 0.0110, 0.0097]),
        "AUROC": ([0.6258, 0.6354, 0.6462, 0.6598, 0.6749], [0.0110, 0.0097, 0.0093, 0.0082, 0.0076]),
    }
    # 从下至上按“简单融合→完整三级交叉注意力”排列，
    # 使均值连线呈左下到右上的增函数形态。
    y = np.arange(len(methods))
    # 采用宽而低的双行布局，减小插入论文后的纵向占幅。
    fig, axes = plt.subplots(2, 2, figsize=(8.40, 4.20), sharey=True)
    for ax, (metric, values) in zip(axes.flat, data.items()):
        mean, sd = map(np.asarray, values)
        ax.plot(mean, y, color=BLUE, linewidth=1.4, zorder=2)
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=BLUE, markersize=5,
                    elinewidth=1.0, capsize=2.5, zorder=3)
        ax.scatter([mean[-1]], [y[-1]], s=55, color=ORANGE,
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_title(metric)
        ax.margins(x=0.16, y=0.08)
        style_axis(ax, axis="x")
        for method, m, s in zip(methods, mean, sd):
            add_source(stem, method, metric, float(m), float(s))
    for ax in axes[:, 0]:
        ax.set_yticks(y, methods)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.08, top=0.94,
                        hspace=0.34, wspace=0.20)
    save_figure(fig, stem)


def draw_component_matrix(ax: mpl.axes.Axes, rows: list[list[int]], labels: list[str],
                          components: list[str], y: np.ndarray) -> None:
    ax.set_xlim(-0.55, len(components) - 0.45)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    for row_index, active in enumerate(rows):
        for col in range(len(components)):
            color = BLUE if col in active else "#E1E5EA"
            ax.scatter(col, y[row_index], s=54, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        if len(active) > 1:
            ax.plot([min(active), max(active)], [y[row_index], y[row_index]], color=BLUE, lw=1.4, zorder=2)
    ax.set_xticks(range(len(components)), components, rotation=28, ha="right")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def figure_rq4_paths() -> None:
    stem = "5.2.4量化、新闻与关系图路径的互补贡献"
    labels = ["完整三路径", "量化+新闻", "新闻+关系图", "量化+关系图", "仅新闻", "仅量化"]
    rows = [[0, 1, 2], [0, 1], [1, 2], [0, 2], [1], [0]]
    mcc = np.array([0.2714, 0.2426, 0.2348, 0.2203, 0.1952, 0.1836])
    mcc_sd = np.array([0.0097, 0.0120, 0.0120, 0.0120, 0.0130, 0.0150])
    auroc = np.array([0.6749, 0.6483, 0.6416, 0.6309, 0.6106, 0.5947])
    auroc_sd = np.array([0.0076, 0.0086, 0.0092, 0.0100, 0.0110, 0.0120])
    y = np.arange(len(labels))[::-1]
    fig = plt.figure(figsize=(7.25, 4.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.16)
    axm = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=axm)
    ax2 = fig.add_subplot(gs[0, 2], sharey=axm)
    draw_component_matrix(axm, rows, labels, ["量化", "新闻", "关系图"], y)
    for ax, mean, sd, metric, xlim in [
        (ax1, mcc, mcc_sd, "MCC", (0.155, 0.292)),
        (ax2, auroc, auroc_sd, "AUROC", (0.565, 0.695)),
    ]:
        colors = [ORANGE] + [BLUE] * (len(labels) - 1)
        for yi, m, s, color, label in zip(y, mean, sd, colors, labels):
            ax.errorbar(m, yi, xerr=s, fmt="o", color=color, markersize=5,
                        elinewidth=1.0, capsize=2.5)
            add_source(stem, label, metric, float(m), float(s))
        ax.set_xlabel(f"{metric}（均值 ± 标准差）")
        ax.set_xlim(*xlim)
        ax.tick_params(labelleft=False)
        style_axis(ax, axis="x")
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.20, top=0.97)
    save_figure(fig, stem)


def figure_external_baselines() -> None:
    stem = "5.3.1统一实验协议下的模型性能比较"
    methods = ["Adv-SHNets", "MagicNet", "MSGCA", "AST-GTN"]
    data = {
        "ACC": ([0.6937, 0.7007, 0.7106, 0.7273], [0.0086, 0.0078, 0.0070, 0.0068]),
        "F1": ([0.7079, 0.7132, 0.7241, 0.7400], [0.0075, 0.0073, 0.0067, 0.0061]),
        "MCC": ([0.2228, 0.2349, 0.2486, 0.2714], [0.0112, 0.0107, 0.0101, 0.0097]),
        "AUROC": ([0.6140, 0.6431, 0.6572, 0.6749], [0.0090, 0.0085, 0.0081, 0.0076]),
    }
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35), sharey=True)
    for ax, (metric, values) in zip(axes.flat, data.items()):
        mean, sd = map(np.asarray, values)
        for yi, m, s, method in zip(y, mean, sd, methods):
            color = ORANGE if method == "AST-GTN" else BLUE
            ax.errorbar(m, yi, xerr=s, fmt="o", color=color, markersize=5.5,
                        elinewidth=1.0, capsize=2.5)
            add_source(stem, method, metric, float(m), float(s))
        ax.set_title(metric)
        ax.set_xlabel("均值 ± 标准差")
        ax.margins(x=0.18)
        style_axis(ax, axis="x")
    for ax in axes[:, 0]:
        ax.set_yticks(y, methods)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.11, top=0.95, hspace=0.48, wspace=0.20)
    save_figure(fig, stem)


def figure_rq5_efficiency() -> None:
    stem = "5.3.2激活函数的预测性能与训练开销权衡"
    labels = ["ReLU", "LeakyReLU", "ELU", "GELU", "SGFRELU"]
    time = np.array([40.68, 55.59, 95.77, 88.10, 131.47])
    time_sd = np.array([1.20, 1.60, 2.80, 2.50, 3.26])
    auroc = np.array([0.5927, 0.6016, 0.6298, 0.6387, 0.6749])
    auroc_sd = np.array([0.0114, 0.0100, 0.0091, 0.0081, 0.0076])
    mcc = np.array([0.2088, 0.2272, 0.2549, 0.2658, 0.2714])
    sizes = 80 + 850 * (mcc - mcc.min()) / (mcc.max() - mcc.min())
    colors = [GRAY, BLUE_LIGHT, PURPLE, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.1, 3.85))
    for i, label in enumerate(labels):
        ax.errorbar(time[i], auroc[i], xerr=time_sd[i], yerr=auroc_sd[i], fmt="none",
                    ecolor=colors[i], elinewidth=1.0, capsize=2.5, alpha=0.9)
        ax.scatter(time[i], auroc[i], s=sizes[i], color=colors[i], alpha=0.88,
                   edgecolor="white", linewidth=0.8, zorder=4)
        offsets = {
            "ReLU": (7, 5),
            "LeakyReLU": (7, 5),
            "ELU": (9, 8),
            "GELU": (-55, -5),
            "SGFRELU": (-70, 7),
        }
        offset = offsets[label]
        ax.annotate(label, (time[i], auroc[i]), xytext=offset,
                    textcoords="offset points", fontsize=8.3, color=INK)
        add_source(stem, label, "训练耗时(s)", float(time[i]), float(time_sd[i]))
        add_source(stem, label, "AUROC", float(auroc[i]), float(auroc_sd[i]))
        add_source(stem, label, "MCC", float(mcc[i]))
    frontier = [0, 1, 3, 4]
    ax.plot(time[frontier], auroc[frontier], color="#7A8793", linestyle="--", linewidth=1.0,
            label="性能—耗时前沿")
    ax.set_xlabel("端到端训练耗时（s，越低越好）")
    ax.set_ylabel("AUROC（越高越好）")
    ax.set_xlim(33, 142)
    ax.set_ylim(0.57, 0.695)
    style_axis(ax, axis="both")
    size_handles = [plt.scatter([], [], s=s, color="#AAB3BC", edgecolor="white") for s in [140, 440, 800]]
    size_labels = ["MCC≈0.21", "MCC≈0.24", "MCC≈0.27"]
    leg1 = ax.legend(size_handles, size_labels, title="气泡大小", loc="lower right", frameon=False)
    ax.add_artist(leg1)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.16, top=0.96)
    save_figure(fig, stem)


def figure_rq6_temporal() -> None:
    stem = "5.3.3全局--局部时序建模的性能与状态稳健性"
    labels = ["仅Transformer", "仅TCN", "等权融合", "可学习融合"]
    mcc = np.array([0.2196, 0.2273, 0.2627, 0.2714])
    mcc_sd = np.array([0.0130, 0.0120, 0.0100, 0.0097])
    auroc = np.array([0.6018, 0.6302, 0.6583, 0.6749])
    auroc_sd = np.array([0.0110, 0.0098, 0.0084, 0.0076])
    worst = np.array([0.6016, 0.5824, 0.6127, 0.6418])
    worst_sd = np.array([0.0192, 0.0204, 0.0181, 0.0167])
    gap = np.array([0.0686, 0.1188, 0.0688, 0.0433])
    gap_sd = np.array([0.0128, 0.0156, 0.0117, 0.0096])
    colors = [GRAY, BLUE_LIGHT, PURPLE, ORANGE]
    y = np.arange(4)[::-1]
    fig = plt.figure(figsize=(7.25, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.9, 1.25], wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    for ax, mean, sd, metric, xlim in [
        (axes[0], mcc, mcc_sd, "整体MCC", (0.195, 0.292)),
        (axes[1], auroc, auroc_sd, "整体AUROC", (0.575, 0.692)),
    ]:
        for yi, m, s, color, label in zip(y, mean, sd, colors, labels):
            ax.errorbar(m, yi, xerr=s, fmt="o", color=color, markersize=5,
                        elinewidth=1.0, capsize=2.5)
            add_source(stem, label, metric, float(m), float(s))
        ax.set_xlabel(f"{metric}\n（均值 ± 标准差）")
        ax.set_xlim(*xlim)
        style_axis(ax, axis="x")
    axes[0].set_yticks(y, labels)
    axes[1].set_yticks(y, [])
    for i, label in enumerate(labels):
        axes[2].errorbar(gap[i], worst[i], xerr=gap_sd[i], yerr=worst_sd[i], fmt="o",
                         color=colors[i], markersize=5.5, elinewidth=1.0, capsize=2.5)
        axes[2].annotate(label, (gap[i], worst[i]), xytext=(5, 5),
                         textcoords="offset points", fontsize=7.8, color=INK)
        add_source(stem, label, "最差状态AUROC", float(worst[i]), float(worst_sd[i]))
        add_source(stem, label, "状态差距", float(gap[i]), float(gap_sd[i]))
    axes[2].set_xlabel("状态差距（越低越好）")
    axes[2].set_ylabel("最差状态AUROC（越高越好）")
    axes[2].set_xlim(0.025, 0.142)
    axes[2].set_ylim(0.55, 0.67)
    style_axis(axes[2], axis="both")
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.20, top=0.96)
    save_figure(fig, stem)


def figure_rq6_window() -> None:
    stem = "5.3.3历史窗口长度的参数敏感性"
    windows = np.array([4, 6, 8, 10, 12])
    data = {
        "ACC": ([0.7205, 0.7082, 0.7273, 0.7246, 0.7158], [0.0072, 0.0081, 0.0068, 0.0068, 0.0078]),
        "F1": ([0.7331, 0.7216, 0.7400, 0.7368, 0.7174], [0.0070, 0.0079, 0.0061, 0.0064, 0.0079]),
        "MCC": ([0.2646, 0.2528, 0.2714, 0.2687, 0.2582], [0.0100, 0.0110, 0.0097, 0.0099, 0.0110]),
        "AUROC": ([0.6701, 0.6608, 0.6749, 0.6761, 0.6637], [0.0081, 0.0090, 0.0076, 0.0078, 0.0088]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), sharex=True)
    for ax, (metric, values) in zip(axes.flat, data.items()):
        mean, sd = map(np.asarray, values)
        draw_error_line(ax, windows, mean, sd, metric, BLUE, "o")
        ax.axvline(8, color=ORANGE, linestyle="--", linewidth=1.1)
        ax.scatter([8], [mean[2]], s=45, color=ORANGE, edgecolor="white", zorder=5)
        ax.set_title(metric)
        ax.set_ylabel("均值 ± 标准差")
        ax.margins(y=0.18)
        style_axis(ax)
        for w, m, s in zip(windows, mean, sd):
            add_source(stem, f"W={w}", metric, float(m), float(s))
    for ax in axes[1, :]:
        ax.set_xlabel("历史窗口长度 W（交易日）")
        ax.set_xticks(windows)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.95, hspace=0.30, wspace=0.22)
    save_figure(fig, stem)


def figure_rq7_revin() -> None:
    stem = "5.3.4RevIN对整体与高波动状态的影响"
    overall_metrics = ["ACC", "F1", "MCC", "AUROC"]
    without = np.array([0.7138, 0.7267, 0.2486, 0.6508])
    without_sd = np.array([0.0081, 0.0074, 0.0115, 0.0091])
    full = np.array([0.7273, 0.7400, 0.2714, 0.6749])
    full_sd = np.array([0.0068, 0.0061, 0.0097, 0.0076])
    delta = full - without
    states = ["上涨--高波", "下跌--高波"]
    w_state = np.array([0.6392, 0.6035])
    w_state_sd = np.array([0.0160, 0.0188])
    f_state = np.array([0.6684, 0.6418])
    f_state_sd = np.array([0.0145, 0.0167])
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55), gridspec_kw={"width_ratios": [1.05, 1.25]})
    y = np.arange(len(overall_metrics))[::-1]
    axes[0].barh(y, delta, color=BLUE, height=0.58)
    for yi, value in zip(y, delta):
        axes[0].text(value + 0.0007, yi, f"+{value:.4f}", va="center", fontsize=8.2)
    axes[0].set_yticks(y, overall_metrics)
    axes[0].set_xlabel("引入RevIN后的指标增量")
    axes[0].set_xlim(0, 0.029)
    style_axis(axes[0], axis="x")
    ys = np.arange(2)[::-1]
    for yi, lo, hi in zip(ys, w_state, f_state):
        axes[1].plot([lo, hi], [yi, yi], color="#B9C1C9", linewidth=1.4, zorder=1)
    axes[1].errorbar(w_state, ys, xerr=w_state_sd, fmt="o", color=GRAY,
                     capsize=2.5, elinewidth=1.0, label="移除RevIN")
    axes[1].errorbar(f_state, ys, xerr=f_state_sd, fmt="o", color=ORANGE,
                     capsize=2.5, elinewidth=1.0, label="完整RevIN")
    axes[1].set_yticks(ys, states)
    axes[1].set_xlabel("高波动状态AUROC（均值 ± 标准差）")
    axes[1].set_xlim(0.57, 0.695)
    axes[1].legend(loc="upper left", frameon=False)
    style_axis(axes[1], axis="x")
    for metric, m0, s0, m1, s1 in zip(overall_metrics, without, without_sd, full, full_sd):
        add_source(stem, "移除RevIN", metric, float(m0), float(s0))
        add_source(stem, "完整RevIN", metric, float(m1), float(s1))
    for state, m0, s0, m1, s1 in zip(states, w_state, w_state_sd, f_state, f_state_sd):
        add_source(stem, "移除RevIN", f"{state}AUROC", float(m0), float(s0))
        add_source(stem, "完整RevIN", f"{state}AUROC", float(m1), float(s1))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.96, wspace=0.36)
    save_figure(fig, stem)


def figure_rq8_objective() -> None:
    stem = "5.3.5联合优化机制的递进贡献"
    labels = ["完整三机制", "Focal+Soft-MCC", "Focal+Mixup", "Focal Loss", "Cross-Entropy"]
    rows = [[0, 1, 2], [0, 1], [0, 2], [0], []]
    mcc = np.array([0.2714, 0.2571, 0.2498, 0.2388, 0.2215])
    mcc_sd = np.array([0.0097, 0.0100, 0.0120, 0.0130, 0.0130])
    auroc = np.array([0.6749, 0.6588, 0.6631, 0.6462, 0.6334])
    auroc_sd = np.array([0.0076, 0.0088, 0.0082, 0.0092, 0.0110])
    y = np.arange(len(labels))[::-1]
    fig = plt.figure(figsize=(7.25, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.17)
    axm = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=axm)
    ax2 = fig.add_subplot(gs[0, 2], sharey=axm)
    draw_component_matrix(axm, rows, labels, ["Focal", "Soft-MCC", "Mixup"], y)
    for ax, mean, sd, metric, xlim in [
        (ax1, mcc, mcc_sd, "MCC", (0.195, 0.29)),
        (ax2, auroc, auroc_sd, "AUROC", (0.61, 0.692)),
    ]:
        for yi, m, s, label in zip(y, mean, sd, labels):
            color = ORANGE if label == "完整三机制" else BLUE
            ax.errorbar(m, yi, xerr=s, fmt="o", color=color, markersize=5,
                        elinewidth=1.0, capsize=2.5)
            add_source(stem, label, metric, float(m), float(s))
        ax.set_xlabel(f"{metric}\n（均值 ± 标准差）")
        ax.set_xlim(*xlim)
        ax.tick_params(labelleft=False)
        style_axis(ax, axis="x")
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.22, top=0.97)
    save_figure(fig, stem)


def figure_bootstrap() -> None:
    stem = "5.3.5核心机制MCC增量的Bootstrap置信区间"
    labels = [
        "RQ3  CA3再次对齐",
        "RQ4  关系图路径",
        "RQ5  SGFRELU",
        "RQ6  可学习双路融合",
        "RQ7  RevIN",
        "RQ8  时序Mixup",
    ]
    delta = np.array([0.0196, 0.0288, 0.0056, 0.0087, 0.0228, 0.0143])
    lower = np.array([0.0068, 0.0131, -0.0019, 0.0008, 0.0094, 0.0030])
    upper = np.array([0.0325, 0.0445, 0.0134, 0.0169, 0.0366, 0.0259])
    pvals = np.array([0.012, 0.006, 0.114, 0.062, 0.010, 0.045])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.25, 3.60))
    ax.axvline(0, color="#59636E", linestyle="--", linewidth=1.0)
    for yi, d, lo, hi, p, label in zip(y, delta, lower, upper, pvals, labels):
        color = BLUE if p < 0.05 else GRAY
        ax.errorbar(d, yi, xerr=np.array([[d - lo], [hi - d]]), fmt="o",
                    color=color, markersize=6, elinewidth=1.6, capsize=3.2)
        ax.text(1.03, yi, f"Holm p={p:.3f}", va="center", ha="left",
                transform=ax.get_yaxis_transform(), fontsize=11.5, color=color)
        add_source(stem, label, "Delta MCC", float(d), lower=float(lo), upper=float(hi), p_value=float(p))
    ax.set_yticks(y, labels, fontsize=11.5)
    ax.set_xlabel("MCC配对差值及95%置信区间", fontsize=12)
    ax.set_xlim(-0.006, 0.056)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    style_axis(ax, axis="x")
    legend = [
        Line2D([0], [0], marker="o", color=BLUE, lw=1.5, label="Holm校正后 p<0.05"),
        Line2D([0], [0], marker="o", color=GRAY, lw=1.5, label="Holm校正后 p≥0.05"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.02),
               ncol=2, frameon=False, fontsize=11.5, columnspacing=1.4)
    fig.subplots_adjust(left=0.30, right=0.765, bottom=0.24, top=0.96)
    save_figure(fig, stem)


def write_source_data() -> None:
    fieldnames = ["figure", "configuration", "metric", "mean", "sd", "ci_lower", "ci_upper", "holm_p"]
    with (SOURCE_DATA / "chapter5_updated_figure_data.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)


def main() -> None:
    figure_rq1_subperiod()
    figure_rq1_complexity()
    figure_rq2_ranking()
    figure_rq2_geometry()
    figure_rq3_fusion()
    figure_rq4_paths()
    figure_external_baselines()
    figure_rq5_efficiency()
    figure_rq6_temporal()
    figure_rq6_window()
    figure_rq7_revin()
    figure_rq8_objective()
    figure_bootstrap()
    write_source_data()
    print(f"Generated 13 figure bundles in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
