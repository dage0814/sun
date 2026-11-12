"""CSV columns: configuration, sparsity_percent, gradient_cv, mcc, auroc, mcc_sd, auroc_sd.
The surface interpolates configuration means with IDW; error bars show the supplied standard deviations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from mpl_toolkits.mplot3d import proj3d
if __package__:
    from .typography import configure_serif_fonts, calibrate_print_text
else:
    from typography import configure_serif_fonts, calibrate_print_text


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


def display_label(configuration: str) -> str:
    labels = {
        "ReLU": "ReLU",
        "GELU": "GELU",
        "SGFRELU_c0.50": r"$c_s=0.50$",
        "SGFRELU_c0.7525": r"$c_s=0.7525$",
        "SGFRELU_c1.00": r"$c_s=1.00$",
    }
    return labels[configuration]


def interpolate_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 1.4,
) -> np.ndarray:

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


def style_3d_axis(ax: mpl.axes.Axes) -> None:
    ax.computed_zorder = False
    ax.set_xlabel("激活稀疏率（%）", labelpad=6)
    ax.set_ylabel("梯度范数变异系数", labelpad=7)
    ax.view_init(elev=25, azim=-140)
    ax.set_box_aspect((1.35, 1.0, 0.72))
    ax.set_xlim(-2, 70)
    ax.set_ylim(0.292, 0.195)
    ax.set_xticks([0, 20, 40, 60])
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
        simulated_z = interpolate_surface(x, y, z, grid_x, grid_y)
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


def read_data(file):
    with Path(file).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"ReLU", "GELU", "SGFRELU_c0.50", "SGFRELU_c0.7525", "SGFRELU_c1.00"}
    names = [row["configuration"] for row in rows]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Provide each of the five activation configurations once.")
    fields = ("sparsity_percent", "gradient_cv", "mcc", "auroc", "mcc_sd", "auroc_sd")
    data = {"configuration": names}
    for field in fields:
        data[field] = np.asarray([float(row[field]) for row in rows])
        if not np.isfinite(data[field]).all():
            raise ValueError(f"Non-finite values: {field}")
        if field.endswith("_sd") and (data[field] < 0).any():
            raise ValueError(f"Negative standard deviation: {field}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    data = read_data(args.data)
    configure_style()
    fig = plt.figure(figsize=(183 / 25.4, 104 / 25.4))
    for index, metric in enumerate(("auroc", "mcc")):
        values, spread = data[metric], data[metric + "_sd"]
        low, high = float(np.min(values - spread)), float(np.max(values + spread))
        margin = max((high - low) * 0.15, 0.005)
        axis = fig.add_subplot(1, 2, index + 1, projection="3d")
        plot_surface_panel(axis, data["sparsity_percent"], data["gradient_cv"],
                           values, spread, data["configuration"], metric.upper(),
                           f"({chr(97 + index)})", (low - margin, high + margin),
                           np.linspace(low, high, 4).tolist())
    fig.subplots_adjust(left=0.01, right=0.97, bottom=0.08, top=0.92, wspace=0.08)
    fig.suptitle("IDW interpolation of configuration means", fontsize=11)
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    calibrate_print_text(fig)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"sensitivity.{extension}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

