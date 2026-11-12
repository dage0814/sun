"""Display five author-reported summaries and an explicitly labelled IDW guide.

The surface is a smooth visual guide based on five configuration means, not
additional experimental observations or a confidence region. Only the five
configuration points carry author-reported vertical SD bars. The approved
summary JSON is the sole quantitative source; the CSV is an export, not input.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "LunWen/figures/chapter5-updated-figures"))
from print_typography import calibrate_print_text, configure_serif_fonts
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.mplot3d import proj3d


DATA_PATH = ROOT / "rq5_sensitivity_source_data.csv"
SOURCE_PATH = ROOT.parents[1] / "outputs" / "20260909-revision" / "数据回填建议.json"
TRACE_PATH = ROOT / "rq5_sensitivity_source_data.json"
GRID_PATH = ROOT / "rq5_sensitivity_simulated_grid.csv"
OUTPUT_STEM = ROOT / "rq5_sgfrelu_sensitivity_surface"
THESIS_STEM = (
    ROOT.parents[1] / "LunWen" / "chapters" / "fig-chapter-5"
    / "5.3.2敏感性分析网面图"
)

WIDTH_MM = 183
HEIGHT_MM = 104
MM_PER_INCH = 25.4


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.size": 11.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.8,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    configure_serif_fonts()


def load_source_data() -> dict:
    """Preserve first occurrence by record ID and verify repeated values."""
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    canonical = {}
    for block in source["replacements"]:
        for metric in block["metrics"]:
            if "mean" not in metric:
                if metric["id"] not in canonical or canonical[metric["id"]]["sdRaw"] != metric["sdRaw"]:
                    raise ValueError(f"Unresolved SD-only reuse: {metric['id']}")
                continue
            previous = canonical.setdefault(metric["id"], metric)
            if (previous["mean"], previous["sdRaw"]) != (metric["mean"], metric["sdRaw"]):
                raise ValueError(f"Conflicting repeated source record: {metric['id']}")
    blocks = [block for block in source["replacements"]
              if block["tableLabel"] == "tab:rq5-activation-mechanism"]
    by_name = {block["rowName"]: block for block in blocks}
    if len(blocks) != 5 or len(by_name) != 5:
        raise ValueError("Expected exactly five distinct activation configurations")
    mapping = [
        ("ReLU", "ReLU"), ("GELU", "GELU"),
        ("SGFRELU（$c_s=0.50$）", "SGFRELU_c0.50"),
        ("SGFRELU（$c_s=0.7525$）", "SGFRELU_c0.7525"),
        ("SGFRELU（$c_s=1.00$）", "SGFRELU_c1.00"),
    ]
    metric_names = {"sparsity_percent": "激活稀疏率", "gradient_cv": "梯度范数变异系数",
                    "mcc": "MCC", "auroc": "AUROC"}
    rows, records = [], []
    for row_name, configuration in mapping:
        metrics = {metric["metric"]: metric for metric in by_name[row_name]["metrics"]}
        if set(metrics) != set(metric_names.values()):
            raise ValueError(f"Unexpected or missing metric in {configuration}")
        row = {"configuration": configuration}
        for field, metric_name in metric_names.items():
            metric = canonical[metrics[metric_name]["id"]]
            mean, sd = float(metric["mean"]), float(metric["sdRaw"])
            if not np.isfinite([mean, sd]).all() or sd < 0:
                raise ValueError(f"Invalid summary value: {metric['id']}")
            if metric["provenance"] != "author_provided_central_summary_not_per_seed_verified":
                raise ValueError(f"Unapproved source: {metric['id']}")
            row[field], row[f"{field}_sd"] = mean, sd
            row[f"{field}_source_id"] = metric["id"]
            records.append({"configuration": configuration, "metric": metric_name,
                            "id": metric["id"], "sourceExcelRow": metric["sourceExcelRow"],
                            "sourceMeanCell": f"G{metric['sourceExcelRow']}",
                            "sourceSDCell": f"H{metric['sourceExcelRow']}",
                            "mean": mean, "sdRaw": sd, "sdDisplayedInPaper": metric["sdDisplayed"],
                            "provenance": metric["provenance"]})
        rows.append(row)
    with DATA_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    trace = {
        "source": str(SOURCE_PATH),
        "sourceSha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "sourceWorkbookSha256": source["sourceWorkbookSha256"],
        "sourceSheet": "数值统一记录",
        "coreConclusion": "Compare the five configurations in sparsity/gradient-CV space with their reported AUROC and MCC; the smooth surface is only a display aid.",
        "archetype": "quantitative grid, two equal paired metric panels",
        "reuseLevel": "structural adaptation; explicitly labelled original IDW style retained by author instruction",
        "counts": {"configurationsBefore": 5, "configurationsAfter": 5,
                   "metricsRetained": 4, "summaryRecords": 20,
                   "IDWGuideCoordinates": 3721, "extraExperimentalObservations": 0},
        "statistics": {"center": "author-reported mean", "spread": "author-reported SD, unrounded",
                       "nAuthorDeclared": 5, "nInWorkbook": None,
                       "replicateUnit": "author-declared independent training runs",
                       "perSeedRecordsVerified": False, "test": None, "multipleComparisonCorrection": None,
                       "verticalBars": "mean ± SD of AUROC or MCC, not CI",
                       "horizontalUncertainty": "not drawn; source SDs retained without assuming joint covariance",
                       "trainValidationTestSplit": "unchanged; not re-audited in this figure-only task"},
        "fieldMapping": {"x": "activation sparsity (%)", "y": "gradient norm coefficient of variation",
                         "zLeft": "AUROC", "zRight": "MCC"},
        "repeatRule": "first source ID occurrence is canonical; subsequent values must match",
        "exclusions": [{"metric": "best_epoch", "sourceIds": ["S136", "S139", "S142", "S147", "S150"],
                        "countBefore": 5, "countAfter": 0,
                        "reason": "entire metric excluded from experimental evidence by author instruction"}],
        "IDWGuide": {"method": "original deterministic inverse-distance weighted display",
                     "shape": [61, 61], "power": 1.4, "squaredDistanceOffset": 0.0001,
                     "coordinateNormalization": "axis ranges, x 0 to 70; y 0.20 to 0.29",
                     "newObservations": False, "confidenceRegion": False,
                     "causalOrPredictiveModel": False,
                     "caveat": "five means do not establish a continuous response; guide outside the five-point support is not experimental evidence"},
        "records": records,
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"configuration": [row["configuration"] for row in rows]}
    for field in metric_names:
        target = "sparsity" if field == "sparsity_percent" else field
        result[target] = np.asarray([row[field] for row in rows])
        result[f"{target}_sd"] = np.asarray([row[f"{field}_sd"] for row in rows])
    return result


def display_label(configuration: str) -> str:
    labels = {
        "ReLU": "ReLU",
        "GELU": "GELU",
        "SGFRELU_c0.50": r"$c_s=0.50$",
        "SGFRELU_c0.7525": r"$c_s=0.7525$",
        "SGFRELU_c1.00": r"$c_s=1.00$",
    }
    return labels[configuration]


def load_frozen_style_inputs() -> tuple[dict, tuple[np.ndarray, ...]]:
    """Read the approved five-point CSV and the existing grid without writes."""
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5:
        raise ValueError("Style-only rendering requires the existing five configurations")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    records = {(record["configuration"], record["id"]): record for record in trace["records"]}
    data = {"configuration": [row["configuration"] for row in rows]}
    for field in ("sparsity_percent", "gradient_cv", "mcc", "auroc"):
        target = "sparsity" if field == "sparsity_percent" else field
        for row in rows:
            record = records[(row["configuration"], row[f"{field}_source_id"])]
            if (float(row[field]), float(row[f"{field}_sd"])) != (record["mean"], record["sdRaw"]):
                raise ValueError("Frozen CSV and source trace disagree")
        data[target] = np.asarray([float(row[field]) for row in rows])
        data[f"{target}_sd"] = np.asarray([float(row[f"{field}_sd"]) for row in rows])
    with GRID_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        points = list(csv.DictReader(stream))
    if len(points) != 61 * 61:
        raise ValueError("Unexpected frozen grid size")
    fields = ("sparsity_percent", "gradient_cv", "idw_display_auroc_not_observation",
              "idw_display_mcc_not_observation")
    grids = tuple(np.asarray([float(point[field]) for point in points]).reshape(61, 61)
                  for field in fields)
    if not all(np.isfinite(grid).all() for grid in grids):
        raise ValueError("Frozen grid contains non-finite values")
    return data, grids


def simulate_smooth_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 1.4,
) -> np.ndarray:
    """Generate a smooth Shepard/IDW response grid without random noise.

    Coordinate ranges are normalized before distances are calculated so that
    sparsity and gradient variability contribute on comparable scales. Because
    each grid value is a convex combination of the observed responses, the
    simulated surface remains within the observed metric range.
    """

    x_lo, x_hi = float(grid_x.min()), float(grid_x.max())
    y_lo, y_hi = float(grid_y.min()), float(grid_y.max())
    observed = np.column_stack(
        ((x - x_lo) / (x_hi - x_lo), (y - y_lo) / (y_hi - y_lo))
    )
    query = np.column_stack(
        (
            (grid_x.ravel() - x_lo) / (x_hi - x_lo),
            (grid_y.ravel() - y_lo) / (y_hi - y_lo),
        )
    )
    squared_distance = np.sum(
        (query[:, None, :] - observed[None, :, :]) ** 2, axis=2
    )
    weights = 1.0 / np.power(squared_distance + 1.0e-4, power)
    simulated = (weights @ z) / np.sum(weights, axis=1)
    return simulated.reshape(grid_x.shape)


def write_simulated_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    auroc_grid: np.ndarray,
    mcc_grid: np.ndarray,
) -> None:
    with GRID_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sparsity_percent",
                "gradient_cv",
                "idw_display_auroc_not_observation",
                "idw_display_mcc_not_observation",
                "generation_method",
            ]
        )
        for sparsity, gradient, auroc, mcc in zip(
            grid_x.ravel(),
            grid_y.ravel(),
            auroc_grid.ravel(),
            mcc_grid.ravel(),
        ):
            writer.writerow(
                [
                    f"{sparsity:.6f}",
                    f"{gradient:.6f}",
                    f"{auroc:.6f}",
                    f"{mcc:.6f}",
                    "five_mean_IDW_display_only_not_experiment_or_confidence_region",
                ]
            )


def style_3d_axis(ax: mpl.axes.Axes) -> None:
    ax.computed_zorder = False
    ax.set_xlabel("激活稀疏率（%）", labelpad=6)
    ax.set_ylabel("梯度范数变异系数", labelpad=7)
    ax.view_init(elev=25, azim=-140)
    ax.set_box_aspect((1.35, 1.0, 0.72))
    ax.set_xlim(-2, 70)
    ax.set_ylim(0.292, 0.195)
    ax.set_xticks([0, 20, 40, 60])
    # Three labelled ticks keep the oblique 3D axis legible at 10.5 pt.
    ax.set_yticks([0.20, 0.24, 0.28])
    ax.tick_params(pad=1.0, width=0.6, length=2.2)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.74, 0.76, 0.78, 0.75))
        axis._axinfo["grid"].update(
            {"color": (0.74, 0.76, 0.78, 0.55), "linewidth": 0.55}
        )


def plot_surface_panel(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_sd: np.ndarray,
    configurations: list[str],
    metric_name: str,
    panel_label: str,
    z_limits: tuple[float, float],
    z_ticks: list[float],
    guide: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    if guide is None:
        grid_x_values = np.linspace(0.0, 70.0, 61)
        grid_y_values = np.linspace(0.20, 0.29, 61)
        grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
        simulated_z = simulate_smooth_surface(x, y, z, grid_x, grid_y)
    else:
        grid_x, grid_y, simulated_z = guide

    ax.plot_surface(
        grid_x,
        grid_y,
        simulated_z,
        cmap="viridis",
        rcount=61,
        ccount=61,
        linewidth=0.15,
        edgecolor="#557377",
        antialiased=False,
        shade=True,
        alpha=1.0,
        zorder=1,
    )

    observed_color = "#315B68"
    highlight_color = "#FDE725"
    highlight_index = configurations.index("SGFRELU_c0.7525")

    regular = np.arange(len(z)) != highlight_index
    # Only the five author-provided marginal metric SDs are rendered. The
    # surface receives no interval; horizontal source SDs remain in the CSV.
    ax.errorbar(x[regular], y[regular], z[regular], zerr=z_sd[regular],
                fmt="none", ecolor=observed_color, elinewidth=0.85,
                capsize=2.0, capthick=0.8, zorder=7)
    ax.errorbar([x[highlight_index]], [y[highlight_index]], [z[highlight_index]],
                zerr=[z_sd[highlight_index]], fmt="none", ecolor="#3F4D18",
                elinewidth=1.0, capsize=2.5, capthick=0.9, zorder=9)
    ax.scatter(
        x[regular],
        y[regular],
        z[regular],
        s=17,
        color=observed_color,
        edgecolor="#000000",
        linewidth=0.45,
        depthshade=False,
        zorder=8,
    )
    ax.scatter(
        [x[highlight_index]],
        [y[highlight_index]],
        [z[highlight_index]],
        marker="*",
        s=135,
        color=highlight_color,
        edgecolor="#000000",
        linewidth=0.6,
        depthshade=False,
        zorder=10,
    )

    style_3d_axis(ax)
    # A 2D metric title avoids clipping projected z-axis labels on export.
    ax.set_zlabel("")
    ax.text2D(0.015, 0.96, metric_name, transform=ax.transAxes,
              fontsize=12.5, fontweight="normal", va="top", color="#000000")
    ax.set_zlim(*z_limits)
    ax.set_zticks(z_ticks)
    label_positions = {
        "ReLU": (0.83, 0.40),
        "GELU": (0.14, 0.32),
        "SGFRELU_c0.50": (0.28, 0.60),
        "SGFRELU_c0.7525": (0.04, 0.86),
        "SGFRELU_c1.00": (0.75, 0.81),
    }
    for index, configuration in enumerate(configurations):
        px, py, _ = proj3d.proj_transform(x[index], y[index], z[index], ax.get_proj())
        ax.annotate(display_label(configuration), xy=(px, py), xycoords="data",
                    xytext=label_positions[configuration], textcoords="axes fraction",
                    fontsize=10.8, fontweight="normal",
                    color="#000000", ha="left", va="center",
                    arrowprops={"arrowstyle": "-", "color": "#315B68", "linewidth": 0.65},
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.8}, zorder=12)
    ax.text2D(
        -0.03,
        0.98,
        panel_label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="normal",
        va="top",
    )


def main(*, thesis_mode: bool = False, write_grid: bool = True, style_only: bool = False) -> None:
    configure_style()
    if style_only:
        data, frozen_grids = load_frozen_style_inputs()
        grid_x, grid_y, auroc_grid, mcc_grid = frozen_grids
    else:
        data = load_source_data()
    configurations = data["configuration"]
    sparsity = data["sparsity"]
    gradient_cv = data["gradient_cv"]

    if not style_only:
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.0, 70.0, 61), np.linspace(0.20, 0.29, 61)
        )
        auroc_grid = simulate_smooth_surface(
            sparsity, gradient_cv, data["auroc"], grid_x, grid_y
        )
        mcc_grid = simulate_smooth_surface(
            sparsity, gradient_cv, data["mcc"], grid_x, grid_y
        )
        if write_grid:
            write_simulated_grid(grid_x, grid_y, auroc_grid, mcc_grid)

    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        constrained_layout=False,
    )
    grid = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.20)
    ax_auroc = fig.add_subplot(grid[0, 0], projection="3d")
    ax_mcc = fig.add_subplot(grid[0, 1], projection="3d")

    plot_surface_panel(
        ax_auroc,
        sparsity,
        gradient_cv,
        data["auroc"],
        data["auroc_sd"],
        configurations,
        "AUROC",
        "a",
        (0.58, 0.69),
        [0.60, 0.63, 0.66, 0.69],
        guide=(grid_x, grid_y, auroc_grid),
    )
    plot_surface_panel(
        ax_mcc,
        sparsity,
        gradient_cv,
        data["mcc"],
        data["mcc_sd"],
        configurations,
        "MCC",
        "b",
        (0.19, 0.29),
        [0.20, 0.23, 0.26, 0.29],
        guide=(grid_x, grid_y, mcc_grid),
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#315B68",
            markeredgecolor="white",
            markersize=4.7,
            label="作者摘要配置点",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#FDE725",
            markeredgecolor="white",
            markersize=7.5,
            label=r"理论设定 $c_s=0.7525$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.006),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.24, top=0.985)

    if thesis_mode:
        # The retained thesis view contains neither panel letters nor a legend.
        for legend in list(fig.legends):
            legend.remove()
        for text in fig.findobj(match=Text):
            if text.get_text() in {"a", "b"}:
                text.remove()

    fig.text(0.51, 0.090, "配置点及竖直误差线：五次运行的均值±标准差（n=5）",
             ha="center", va="center", fontsize=10.8, color="#000000")
    fig.text(0.51, 0.022, "五组配置点的IDW插值曲面",
             ha="center", va="center", fontsize=10.8, color="#000000")

    # Include 3D/math/legend labels; retain their translucent backing boxes.
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    # SimSun's PDF glyph bounds need a small guard around projected labels.
    typography = calibrate_print_text(fig, pad_inches=0.20)
    output_stems = [OUTPUT_STEM, THESIS_STEM] if thesis_mode else [OUTPUT_STEM]
    for output_stem in output_stems:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        # Append extensions: the thesis filename itself contains section dots.
        fig.savefig(output_stem.parent / f"{output_stem.name}.svg", bbox_inches="tight", pad_inches=0.20)
        fig.savefig(output_stem.parent / f"{output_stem.name}.pdf", bbox_inches="tight", pad_inches=0.20)
        fig.savefig(output_stem.parent / f"{output_stem.name}.png", dpi=600, bbox_inches="tight", pad_inches=0.20)
        tiff_path = output_stem.parent / f"{output_stem.name}.tiff"
        if output_stem == OUTPUT_STEM or tiff_path.exists() or style_only:
            fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.20, pil_kwargs={"compression": "tiff_lzw"})
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    text_objects = [text for text in fig.findobj(match=Text) if text.get_text()]
    assert all(mpl.colors.to_hex(text.get_color()) == "#000000" for text in text_objects)
    trace["render"] = {"backend": "python", "widthMM": WIDTH_MM, "heightMM": HEIGHT_MM,
                       "rasterDPI": 600, "allTextBlack": True, "textObjects": len(text_objects),
                       "dataPointsPerPanel": 5, "verticalSDBarsPerPanel": 5,
                       "outputs": [], "typography": typography}
    for stem in output_stems:
        for suffix in ("png", "svg", "pdf", "tiff"):
            path = stem.parent / f"{stem.name}.{suffix}"
            if path.exists():
                trace["render"]["outputs"].append({"path": str(path), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    trace_target = TRACE_PATH
    if style_only:
        trace_target = ROOT.parents[1] / "outputs/20260909-restore-green-abstract/sensitivity_render_trace.json"
        trace_target.parent.mkdir(parents=True, exist_ok=True)
        trace["styleRevision"] = {"colormap": "viridis", "IDWRecomputed": False,
                                  "frozenGridSha256": hashlib.sha256(GRID_PATH.read_bytes()).hexdigest(),
                                  "frozenPointCSVHash": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                                  "secondLine": "五组配置点的IDW插值曲面"}
    trace_target.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"configurations": 5, "summaryRecords": len(trace["records"]),
                      "allTextBlack": True, "outputs": len(trace["render"]["outputs"])}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-mode", action="store_true",
                        help="Export the retained thesis view and synchronize its source bundle.")
    parser.add_argument("--no-write-grid", action="store_true",
                        help="Do not rewrite the derived guide grid; approved summary CSV is still refreshed.")
    parser.add_argument("--style-only", action="store_true",
                        help="Use frozen source/grid files read-only; do not recompute IDW or modify data.")
    args = parser.parse_args()
    main(thesis_mode=args.thesis_mode, write_grid=not args.no_write_grid, style_only=args.style_only)
