"""Synchronize eight authorized thesis figures from their authoritative summaries.

Figure contract: quantitative comparisons preserve the existing panel meanings,
means and category orders; error bars show author-provided across-run SD (n=5),
not confidence intervals. No observations are synthesized, and no tests rerun.
Seven use the checked central map; the ranking figure uses its existing paper
table. The separate sensitivity figure has its own focused export entry point.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import findfont
from matplotlib.text import Text
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from print_typography import calibrate_print_text, configure_serif_fonts

ROOT = Path(__file__).resolve().parents[3]
THESIS = ROOT / "LunWen"
OUT = THESIS / "chapters/fig-chapter-5"
DATA = ROOT / "outputs/20260909-blue-style/source_data"
REPORTS = ROOT / "outputs/20260909-blue-style"
BACKUP = THESIS / "backups/pre-blue-style-20260909/plots"
MAP_FILE = ROOT / "outputs/20260909-revision/数据回填建议.json"
LEGACY_SOURCE = Path(__file__).parent / "plot_chapter5_updated_figures.py"
LEGACY_CSV = Path(__file__).parent / "source_data/chapter5_updated_figure_data.csv"
STEMS = [
    "5.2.1CSF-GP复杂度与因子质量权衡",
    "5.2.3多级双路交叉注意力的递进增益",
    "5.2.4量化、新闻与关系图路径的互补贡献",
    "5.3.1统一实验协议下的模型性能比较",
    "5.3.2激活函数的预测性能与训练开销权衡",
    "5.3.3历史窗口长度的参数敏感性",
    "5.3.5联合优化机制的递进贡献",
    "5.2.2不同文本编码方法的行业排序性能",
]
BLUE, YELLOW, GREEN = "#397CB7", "#E6C85C", "#6FAF8E"
GRAY, PALE = "#81858A", "#E5E6E8"
SOURCE_ROWS: dict[str, list[dict]] = {s: [] for s in STEMS}
RENDER_QA = []
BACKUP_MANIFEST = []
RANKING_DIFFERENCES = []

mpl.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman", "SimSun"],
    "font.size": 12.0, "axes.titlesize": 13.0, "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "grid.color": "#D4D6D9", "grid.linewidth": 0.65, "grid.alpha": 0.75,
})
configure_serif_fonts()


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_one(path: Path) -> None:
    if not path.exists():
        return
    target = BACKUP / path.relative_to(ROOT)
    if not target.resolve().is_relative_to(BACKUP.resolve()):
        raise RuntimeError("Backup path is outside the named backup directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    BACKUP_MANIFEST.append({"original": path.as_posix(), "backup": target.as_posix(),
                            "backupSha256": checksum(target)})


def setup_data() -> dict:
    proposal = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    by_key = {}
    for block in proposal["replacements"]:
        for item in block["metrics"]:
            if not item.get("tableLabel"):
                continue
            key = (item["tableLabel"], item["rowName"], item["metric"])
            if key in by_key and by_key[key] != item:
                raise RuntimeError(f"Conflicting numeric map for {key}")
            by_key[key] = item
    return by_key


NUMBERS = setup_data()


def series(stem: str, table: str, rows: list[str], metric: str):
    means, sds = [], []
    for name in rows:
        item = NUMBERS[(table, name, metric)]
        mean, sd = float(item["mean"]), float(item["sdRaw"])
        if not np.isfinite([mean, sd]).all() or sd < 0:
            raise ValueError(f"Invalid summary for {item['id']}")
        means.append(mean)
        sds.append(sd)
        SOURCE_ROWS[stem].append({
            "figure": stem, "configuration": name, "metric": metric,
            "mean": mean, "sd": sd, "source_id": item["id"],
            "excel_row": item["sourceExcelRow"], "n_author_confirmed": 5,
            "provenance": item["provenance"],
        })
    return np.array(means), np.array(sds)


def style(ax, axis="x"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")


def save(fig, stem, conclusion, panels):
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    fig.canvas.draw()
    nonblack = [t.get_text() for t in fig.findobj(match=Text)
                if t.get_text() and mpl.colors.to_hex(t.get_color()) != "#000000"]
    if nonblack:
        raise RuntimeError("Non-black text remains.")
    typography = calibrate_print_text(fig)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    source_path = DATA / f"{stem}.csv"
    records = SOURCE_ROWS[stem]
    with source_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    meta = {
        "figure": stem, "coreConclusion": conclusion, "archetype": "quantitative grid",
        "panels": panels, "backend": "python", "sourceMapping": MAP_FILE.as_posix(),
        "mappingSha256": checksum(MAP_FILE), "sourceRows": records,
        "center": "reported run-level mean", "spread": "reported across-run SD",
        "n": 5, "nBasis": "author confirmation, not per-seed verification",
        "newTestsComputed": False, "ciOrPValuesChanged": False,
        "exclusions": "No best_epoch, withdrawn subperiod, permuted-industry, or market-state evidence used.",
        "rounding": "Error bars use raw summary SD; thesis display may round the same values.",
        "sourceTemplateReuse": "structural adaptation of the existing quantitative layout",
        "palette": [BLUE, YELLOW, GREEN], "allTextBlack": True,
        "visio": "Python/SVG/PDF/PNG/TIFF synchronized; VSDX embedded images not synchronized.",
    }
    if stem == STEMS[7]:
        meta["sourceMapping"] = (THESIS / "chapters/chapter-5.tex").as_posix()
        meta["mappingSha256"] = checksum(THESIS / "chapters/chapter-5.tex")
        meta["sourceAuthority"] = "现行正文tab:rq2-ranking-result首次统计；本轮另行授权图表同步。"
        meta["oldFigureDifferences"] = RANKING_DIFFERENCES
    (DATA / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with Image.open(OUT / f"{stem}.png") as raster:
        raster_size, raster_dpi = list(raster.size), raster.info.get("dpi")
    svg = (OUT / f"{stem}.svg").read_text(encoding="utf-8")
    RENDER_QA.append({"figure": stem, "pngPixels": raster_size, "pngDpi": raster_dpi,
                      "svgEditableTextCount": svg.count("<text"), "allTextBlack": not nonblack,
                      "sourceRows": len(records), "figureSizeInches": list(fig.get_size_inches()),
                      "typography": typography,
                      "outputs": {s: checksum(OUT / f"{stem}.{s}") for s in ("png", "svg", "pdf", "tiff")}})
    plt.close(fig)
    print(f"Exported {stem}", flush=True)


def complexity():
    stem = STEMS[0]
    labels = ["完整CSF-GP", "移除复杂度约束", "移除相似度过滤", "移除精英保留", "移除Hoist变异"]
    table = "tab:gp-ablation-result"
    nodes, nodes_sd = series(stem, table, labels, "平均公式节点数")
    rankic, rankic_sd = series(stem, table, labels, "RankIC")
    corr, _ = series(stem, table, labels, "因子平均绝对相关系数")
    norm = Normalize(vmin=0.35, vmax=0.70)
    cmap = mpl.colormaps["viridis_r"]
    fig, ax = plt.subplots(figsize=(7.2, 3.92))
    for i, label in enumerate(labels):
        color = cmap(norm(corr[i]))
        ax.errorbar(nodes[i], rankic[i], xerr=nodes_sd[i], yerr=rankic_sd[i],
                    fmt="none", ecolor=color, elinewidth=1.1, capsize=2.5)
        ax.scatter(nodes[i], rankic[i], s=100 if i == 0 else 66, marker="D" if i == 0 else "o",
                   color=color, edgecolor="black", linewidth=0.7, zorder=4)
        offsets = [(8, 23), (-134, -42), (20, -40), (16, -27), (12, 14)]
        ax.annotate(f"{label}\n平均|ρ|={corr[i]:.2f}", (nodes[i], rankic[i]), xytext=offsets[i],
                    textcoords="offset points", fontsize=11.5,
                    arrowprops={"arrowstyle": "-", "color": "#606060", "linewidth": 0.65},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
    ax.set(xlabel="平均公式节点数（越低越简洁）", ylabel="测试集RankIC（越高越好）",
           xlim=(7.8, 21.0), ylim=(0.030, 0.104))
    style(ax, "both")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("因子平均绝对相关系数（越低越好）")
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.16, top=0.97)
    save(fig, stem, "复杂度与冗余约束下的完整GP保持较简洁公式和较高RankIC。",
         ["节点数–RankIC均值及两方向SD；颜色为平均绝对相关系数，完整模型为菱形。"])


def fusion():
    stem = STEMS[1]
    names = ["线性加权", "特征拼接+MLP", "CA1", "CA1+CA2", "CA1+CA2+CA3"]
    labels = ["线性加权", "拼接+MLP", "CA1", "CA1+CA2", "完整模型\n(CA1+CA2+CA3)"]
    y = np.arange(5)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.58), sharey=True)
    for ax, metric in zip(axes.flat, ["ACC", "F1", "MCC", "AUROC"]):
        mean, sd = series(stem, "tab:rq3-cross-attention-fusion", names, metric)
        ax.plot(mean, y, color=BLUE, linewidth=1.4, zorder=2)
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=BLUE, markersize=5, elinewidth=1, capsize=2.5)
        ax.scatter([mean[-1]], [y[-1]], s=55, color="#D55E00", edgecolor="black", linewidth=0.5, zorder=5)
        ax.set_title(metric, pad=9)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.margins(x=0.16, y=0.08)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_yticks(y, labels)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.285, right=0.985, bottom=0.08, top=0.94, hspace=0.38, wspace=0.28)
    save(fig, stem, "融合配置由常规融合扩展到完整交互时，四项指标均值逐级提高。", ["ACC", "F1", "MCC", "AUROC"])


def component_matrix(ax, active_rows, labels, components, y):
    ax.set(xlim=(-0.55, len(components)-0.45), ylim=(-0.55, len(labels)-0.45))
    for row_index, active in enumerate(active_rows):
        for column in range(len(components)):
            ax.scatter(column, y[row_index], s=54, color=BLUE if column in active else PALE,
                       edgecolor="white", linewidth=0.5, zorder=3)
        if len(active) > 1:
            ax.plot([min(active), max(active)], [y[row_index]]*2, color=BLUE, lw=1.4, zorder=2)
    ax.set_xticks(range(len(components)), components, rotation=35, ha="right")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def component_plot(stem, table, names, labels, active_rows, components, height, limits, conclusion):
    y = np.arange(len(labels))[::-1]
    fig = plt.figure(figsize=(7.2, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.27)
    matrix = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1], sharey=matrix), fig.add_subplot(gs[0, 2], sharey=matrix)]
    component_matrix(matrix, active_rows, labels, components, y)
    for ax, metric, bounds in zip(axes, ["MCC", "AUROC"], limits):
        means, sds = series(stem, table, names, metric)
        for i, (yi, mean, sd) in enumerate(zip(y, means, sds)):
            ax.errorbar(mean, yi, xerr=sd, fmt="D" if i == 0 else "o",
                        color="#D55E00" if i == 0 else BLUE, markersize=5, elinewidth=1, capsize=2.5)
        ax.set_xlabel(f"{metric}\n（均值 ± 标准差）")
        ax.set_xlim(*bounds)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
        ax.tick_params(labelleft=False)
        style(ax)
    fig.subplots_adjust(left=0.275, right=0.985, bottom=0.24, top=0.97)
    save(fig, stem, conclusion, ["机制/路径存在矩阵", "MCC均值±SD", "AUROC均值±SD"])


def paths():
    component_plot(STEMS[2], "tab:rq4-path-combination",
        ["完整三路径模型", "量化路径+新闻路径", "新闻路径+关系图路径", "量化路径+关系图路径", "仅新闻路径", "仅量化路径"],
        ["完整三路径", "量化+新闻", "新闻+关系图", "量化+关系图", "仅新闻", "仅量化"],
        [[0, 1, 2], [0, 1], [1, 2], [0, 2], [1], [0]], ["量化", "新闻", "关系图"], 3.50,
        [(0.155, 0.292), (0.565, 0.695)], "完整三路径相对单路径及双路径均保留正向性能增量。")


def external():
    stem = STEMS[3]
    # Unchanged external-baseline statistics are inherited literally from the
    # existing plotting source. The authorized mapping updates AST-GTN only.
    tree = ast.parse(LEGACY_SOURCE.read_text(encoding="utf-8-sig"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "figure_external_baselines")
    data = next(ast.literal_eval(n.value) for n in function.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "data" for t in n.targets))
    methods = ["Adv-SHNets", "MagicNet", "MSGCA", "AST-GTN"]
    y = np.arange(4)[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.48), sharey=True)
    for ax, (metric, values) in zip(axes.flat, data.items()):
        mean, sd = [np.array(value, dtype=float) for value in values]
        new_mean, new_sd = series(stem, "tab:unified-external-baselines", ["AST-GTN"], metric)
        if mean[-1] != new_mean[0]:
            raise ValueError("External full-model mean changed unexpectedly.")
        mean[-1], sd[-1] = new_mean[0], new_sd[0]
        for i, (yi, m, s, method) in enumerate(zip(y, mean, sd, methods)):
            ax.errorbar(m, yi, xerr=s, fmt="D" if i == 3 else "o", color="#D55E00" if i == 3 else BLUE,
                        markersize=5.5, elinewidth=1, capsize=2.5)
            if i < 3:
                SOURCE_ROWS[stem].append({"figure": stem, "configuration": method, "metric": metric,
                    "mean": float(m), "sd": float(s), "source_id": "existing_external_baseline_unchanged",
                    "excel_row": "", "n_author_confirmed": 5, "provenance": "existing_paper_and_plot_baseline_preserved_no_new_mapping"})
        ax.set_title(metric, pad=8)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.set_xlabel("均值 ± 标准差")
        ax.margins(x=0.18)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_yticks(y, methods)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.12, top=0.95, hspace=0.65, wspace=0.24)
    save(fig, stem, "统一协议下完整AST-GTN在四项指标上具有最高报告均值。", ["ACC", "F1", "MCC", "AUROC"])


def efficiency():
    stem = STEMS[4]
    labels = ["ReLU", "LeakyReLU", "ELU", "GELU", "SGFRELU"]
    table = "tab:rq5-activation-performance"
    time, time_sd = series(stem, table, labels, "训练耗时")
    auroc, auroc_sd = series(stem, table, labels, "AUROC")
    mcc, _ = series(stem, table, labels, "MCC")
    denominator = mcc.max() - mcc.min()
    if denominator <= 0:
        raise ValueError("Bubble-size normalization requires differing MCC means.")
    size = lambda value: 80 + 850 * (value - mcc.min()) / denominator
    colors = ["#606060", "#0072B2", "#E69F00", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(figsize=(7.2, 3.70))
    for i, label in enumerate(labels):
        ax.errorbar(time[i], auroc[i], xerr=time_sd[i], yerr=auroc_sd[i], fmt="none",
                    ecolor=colors[i], elinewidth=1.1, capsize=2.5, zorder=3)
        ax.scatter(time[i], auroc[i], s=size(mcc[i]), color=colors[i], alpha=0.82,
                   edgecolor="black" if i == 4 else "white", linewidth=0.6, zorder=4)
        offsets = [(8, 16), (-57, 20), (20, -13), (-70, 11), (-96, 11)]
        ax.annotate(label, (time[i], auroc[i]), xytext=offsets[i], textcoords="offset points", fontsize=11.5, zorder=6)
    frontier = [0, 1, 3, 4]
    ax.plot(time[frontier], auroc[frontier], color=GRAY, linestyle="--", linewidth=1, zorder=1)
    ax.set(xlabel="端到端训练耗时（s，越低越好）", ylabel="AUROC（越高越好）",
           xlim=(33, 142), ylim=(0.57, 0.695))
    style(ax, "both")
    ax.text(0.97, 0.07, "气泡大小表示 MCC", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11.5, color="#000000")
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.96)
    save(fig, stem, "SGFRELU的较高AUROC伴随更长的端到端训练耗时。", ["耗时–AUROC两方向SD；气泡面积编码MCC均值。"])


def window():
    stem = STEMS[5]
    windows = np.array([4, 6, 8, 10, 12])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.85), sharex=True)
    for ax, metric in zip(axes.flat, ["ACC", "F1", "MCC", "AUROC"]):
        means, sds = series(stem, "tab:rq6-window-sensitivity", [str(w) for w in windows], metric)
        ax.errorbar(windows, means, yerr=sds, color=BLUE, marker="o", markersize=5.2,
                    linewidth=1.8, elinewidth=1, capsize=3, zorder=3)
        ax.axvline(8, color=GREEN, linestyle="--", linewidth=1.1)
        ax.scatter([8], [means[2]], s=45, color="#D55E00", edgecolor="black", linewidth=0.5, zorder=5)
        ax.set(title=metric, ylabel="均值 ± 标准差")
        ax.margins(y=0.18)
        style(ax, "y")
    for ax in axes[1, :]:
        ax.set_xlabel("历史窗口长度 T（交易日）")
        ax.set_xticks(windows)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.14, top=0.95, hspace=0.38, wspace=0.48)
    save(fig, stem, "窗口长度对性能的影响非单调，8日配置在ACC/F1/MCC均值上最高。", ["ACC", "F1", "MCC", "AUROC"])


def objective():
    component_plot(STEMS[6], "tab:rq8-objective-mixup-ablation",
        ["Focal+Soft-MCC+时序Mixup", "Focal+Soft-MCC", "Focal+时序Mixup", "Focal Loss", "Cross-Entropy"],
        ["完整三机制", "Focal+Soft-MCC", "Focal+Mixup", "Focal Loss", "Cross-Entropy"],
        [[0, 1, 2], [0, 1], [0, 2], [0], []], ["Focal", "Soft-MCC", "Mixup"], 3.32,
        [(0.195, 0.292), (0.610, 0.695)], "完整联合优化配置取得本组最高MCC和AUROC均值。")


def ranking():
    stem = STEMS[7]
    path = THESIS / "chapters/chapter-5.tex"
    text = path.read_text(encoding="utf-8")
    blocks = [m for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
              if r"\label{tab:rq2-ranking-result}" in m.group()]
    if len(blocks) != 1:
        raise RuntimeError("Current authoritative ranking table is not unique.")
    block = blocks[0]
    names = ["TF-IDF", "Word2Vec-Mean", "BERT-Base", "FinBERT-Sentiment", "完整FinFE-BERT"]
    display = ["TF-IDF", "Word2Vec-Mean", "BERT-Base", "FinBERT-Sentiment", "FinFE-BERT"]
    metrics = ["NDCG@3", "NDCG@9", "RankIC"]
    data = {metric: ([], []) for metric in metrics}
    legacy_tree = ast.parse(LEGACY_SOURCE.read_text(encoding="utf-8-sig"))
    legacy_function = next(n for n in legacy_tree.body if isinstance(n, ast.FunctionDef) and n.name == "figure_rq2_ranking")
    legacy_data = next(ast.literal_eval(n.value) for n in legacy_function.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "metrics" for t in n.targets))
    for index, name in enumerate(names):
        row_matches = list(re.finditer(r"^\s*" + re.escape(name) + r"\s*&([^\n]+)", block.group(), re.M))
        if len(row_matches) != 1:
            raise RuntimeError(f"Ranking row is not unique: {name}")
        row = row_matches[0]
        values = re.findall(r"\d+\.\d+", row.group(1))
        if len(values) != 6:
            raise RuntimeError(f"Expected all three mean/SD pairs: {name}")
        source_line = text[:block.start() + row.start()].count("\n") + 1
        for j, metric in enumerate(metrics):
            mean, sd = map(float, values[2*j:2*j+2])
            if not np.isfinite([mean, sd]).all() or sd < 0:
                raise ValueError("Invalid paper ranking summary.")
            data[metric][0].append(mean)
            data[metric][1].append(sd)
            SOURCE_ROWS[stem].append({"figure": stem, "configuration": name, "metric": metric,
                "mean": mean, "sd": sd, "source_id": "tab:rq2-ranking-result", "excel_row": "",
                "n_author_confirmed": 5, "provenance": f"current_authoritative_paper_table_line_{source_line}"})
            old = (legacy_data[metric][0][index], legacy_data[metric][1][index])
            if old != (mean, sd):
                RANKING_DIFFERENCES.append({"configuration": name, "metric": metric,
                    "oldMean": old[0], "newMean": mean, "oldSD": old[1], "newSD": sd, "paperLine": source_line})
    y = np.arange(5)[::-1]
    fig = plt.figure(figsize=(7.2, 3.49))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.29)
    first = fig.add_subplot(gs[0, 0])
    second = fig.add_subplot(gs[0, 1], sharey=first)
    for metric, color, offset, marker in [("NDCG@3", "#0072B2", 0.11, "o"), ("NDCG@9", "#D55E00", -0.11, "s")]:
        mean, sd = map(np.asarray, data[metric])
        first.errorbar(mean, y + offset, xerr=sd, fmt=marker, color=color,
                       markersize=5, elinewidth=1, capsize=2.5, label=metric)
    mean, sd = map(np.asarray, data["RankIC"])
    for index, (yi, m, s) in enumerate(zip(y, mean, sd)):
        second.errorbar(m, yi, xerr=s, fmt="D" if index == 4 else "o",
                        color="#D55E00" if index == 4 else BLUE, markersize=5, elinewidth=1, capsize=2.5)
    first.set_yticks(y, display)
    first.set(xlabel="NDCG（均值 ± 标准差）", xlim=(0.56, 0.85))
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=2,
               frameon=False, columnspacing=1.6, handletextpad=0.6)
    second.tick_params(labelleft=False)
    second.set(xlabel="RankIC（均值 ± 标准差）", xlim=(0.065, 0.205))
    first.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    second.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    for ax in (first, second):
        style(ax)
    fig.subplots_adjust(left=0.275, right=0.985, bottom=0.18, top=0.85)
    save(fig, stem, "行业条件表示的优势集中于头部与整体NDCG，而非RankIC最高值。",
         ["NDCG@3与NDCG@9及SD", "RankIC及SD"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for path in [LEGACY_SOURCE, LEGACY_CSV, Path(__file__)]:
        backup_one(path)
    for stem in STEMS:
        for suffix in ("png", "svg", "pdf", "tiff", "vsdx"):
            backup_one(OUT / f"{stem}.{suffix}")
    excluded_stems = ["5.2.1CSF-GP测试子区间预测稳定性", "5.2.2行业表示分离度与排序有效性",
                      "5.3.3全局--局部时序建模的性能与状态稳健性", "5.3.4RevIN对整体与高波动状态的影响",
                      "5.3.5核心机制MCC增量的Bootstrap置信区间"]
    protected = {p: checksum(p) for s in excluded_stems for p in OUT.glob(s + ".*") if p.is_file()}
    for fn in (complexity, fusion, paths, external, efficiency, window, objective, ranking):
        fn()
    if any(checksum(p) != old for p, old in protected.items()):
        raise RuntimeError("An excluded figure changed unexpectedly.")
    validator = Path("C:/Users/User/.codex/skills/nature-figure/scripts/validate_figure.py")
    check = subprocess.run([sys.executable, str(validator), str(Path(__file__)), "--json"],
                           text=True, capture_output=True, encoding="utf-8")
    (REPORTS / "figure-validator-seven.json").write_text(check.stdout, encoding="utf-8")
    (REPORTS / "figure-validator-authorized.json").write_text(check.stdout, encoding="utf-8")
    report = {"createdAtUTC": datetime.now(timezone.utc).isoformat(), "sourceMapSha256": checksum(MAP_FILE),
        "backend": "python", "fontResolved": {"latin": findfont("Times New Roman"), "chinese": findfont("SimSun")}, "backupManifest": BACKUP_MANIFEST,
        "figures": RENDER_QA, "protectedFigureFilesUnchanged": len(protected), "validatorExitCode": check.returncode,
        "validatorStderr": check.stderr, "vsdxSynchronized": False,
        "rankingOldFigureDifferences": RANKING_DIFFERENCES,
        "notes": ["七个原授权绘图函数加另行授权的RQ2排序图；不执行旧13图总入口。", "前七图误差棒使用数据回填建议中未四舍五入的SD；RQ2排序图以现行正文15项为源；未生成逐种子数据。",
                  "外部基线三模型12项原有统计保持不变；本轮仅更新AST-GTN四项。", "历史窗口横轴从W统一为正文使用的T，不改窗口数值。",
                  "本轮仅将紫色替换为蓝色；保留绿色与黄色辅助色，全字纯黑，数值及布局不变。", "SVG/PDF保持可编辑文本；不声明VSDX已同步。",
                  "RQ2排序图已按新增授权逐项核对全部15项，与旧图8项SD差异已全部同步。",
                  "无best_epoch面板或注释出现在本八图；敏感性图由独立入口处理。"]}
    (REPORTS / "figure-update-seven-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "figure-update-authorized-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(RENDER_QA), "backups": len(BACKUP_MANIFEST),
                      "validatorExitCode": check.returncode, "report": str(REPORTS / "figure-update-seven-report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
