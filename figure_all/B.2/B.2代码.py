"""Render the Chapter 4 SGFRELU mechanism figures with Matplotlib.

All plotted values are obtained directly from the analytical definitions used in
the thesis implementation.  No experimental observations are synthesized here.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK = SCRIPT_DIR.parents[1]
MPL_CONFIG_DIR = SCRIPT_DIR / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from print_typography import configure_serif_fonts, calibrate_print_text


TARGET_DIR = TASK / "clean-build" / "chapters" / "fig-chapter-4"
EXPORT_DIR = SCRIPT_DIR / "exports"
DATA_DIR = SCRIPT_DIR / "source_data"

SHIFT = 0.7524614220708874
TANH_COEFFICIENT = 0.044715

COLORS = {
    "hero": "#397CB7",
    "gelu": "#009E73",
    "shifted": "#E69F00",
    "relu": "#6B6B6B",
    "exact": "#3366AA",
    "approx": "#D55E00",
    "grid": "#D9DDE5",
    "zero_fill": "#E6F0F8",
    "text": "#242424",
}


mpl.rcParams.update(
    {
        "font.family": ["Times New Roman", "SimSun"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.55,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)
configure_serif_fonts()


def normal_cdf(x: np.ndarray) -> np.ndarray:
    erf_values = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in x),
        dtype=float,
        count=x.size,
    )
    return 0.5 * (1.0 + erf_values)


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    return x * normal_cdf(x)


def gelu_exact_derivative(x: np.ndarray) -> np.ndarray:
    return normal_cdf(x) + x * normal_pdf(x)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    inner = math.sqrt(2.0 / math.pi) * (
        x + TANH_COEFFICIENT * x**3
    )
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_tanh_derivative(x: np.ndarray) -> np.ndarray:
    scale = math.sqrt(2.0 / math.pi)
    inner = scale * (x + TANH_COEFFICIENT * x**3)
    tanh_inner = np.tanh(inner)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (
        1.0 - tanh_inner**2
    ) * scale * (1.0 + 3.0 * TANH_COEFFICIENT * x**2)


def bisect_root(function, left: float, right: float) -> float:
    f_left = float(function(np.array([left]))[0])
    for _ in range(100):
        middle = 0.5 * (left + right)
        f_middle = float(function(np.array([middle]))[0])
        if f_left * f_middle <= 0.0:
            right = middle
        else:
            left = middle
            f_left = f_middle
    return 0.5 * (left + right)


EXACT_ROOT = bisect_root(gelu_exact_derivative, -1.5, -0.1)
TANH_ROOT = bisect_root(gelu_tanh_derivative, -1.5, -0.1)


def sgfrelu(x: np.ndarray, shift: float = SHIFT) -> np.ndarray:
    return np.maximum(0.0, gelu_exact(x - shift))


def sgfrelu_derivative(x: np.ndarray, shift: float = SHIFT) -> np.ndarray:
    derivative = np.zeros_like(x)
    right = x > shift
    derivative[right] = gelu_exact_derivative(x[right] - shift)
    derivative[np.isclose(x, shift, atol=1e-12)] = np.nan
    return derivative


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#606060")
    ax.spines["bottom"].set_color("#606060")
    ax.tick_params(colors=COLORS["text"], direction="out")
    ax.grid(
        True,
        axis=grid_axis,
        color=COLORS["grid"],
        linewidth=0.55,
        alpha=0.72,
        zorder=0,
    )
    ax.set_axisbelow(True)


def export_figure(fig: plt.Figure, stem: str) -> None:
    # Include inset, math, and legend labels without changing their boxes/arrows.
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    calibrate_print_text(fig, pad_inches=0.04, width_mm=150, target_pt=10.5)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (TARGET_DIR, EXPORT_DIR):
        fig.savefig(
            directory / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        fig.savefig(
            directory / f"{stem}.svg",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        fig.savefig(
            directory / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    fig.savefig(
        EXPORT_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def save_source_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    u = np.linspace(-2.5, 1.5, 1601)
    with (DATA_DIR / "gelu_derivative_roots.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "u",
                "exact_gelu_derivative",
                "tanh_approx_gelu_derivative",
                "exact_root",
                "tanh_approx_root",
                "implementation_shift",
                "data_status",
            ]
        )
        for value, exact_value, approx_value in zip(
            u, gelu_exact_derivative(u), gelu_tanh_derivative(u)
        ):
            writer.writerow(
                [
                    f"{value:.8f}",
                    f"{exact_value:.10f}",
                    f"{approx_value:.10f}",
                    f"{EXACT_ROOT:.10f}",
                    f"{TANH_ROOT:.10f}",
                    f"{SHIFT:.10f}",
                    "analytical_definition",
                ]
            )

    x = np.linspace(-3.0, 3.5, 1801)
    with (DATA_DIR / "sgfrelu_output_gradient.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x",
                "relu",
                "gelu_exact",
                "shifted_gelu_exact",
                "sgfrelu",
                "relu_derivative",
                "gelu_exact_derivative",
                "sgfrelu_derivative",
                "data_status",
            ]
        )
        relu_derivative = np.where(x < 0.0, 0.0, 1.0)
        for row in zip(
            x,
            np.maximum(0.0, x),
            gelu_exact(x),
            gelu_exact(x - SHIFT),
            sgfrelu(x),
            relu_derivative,
            gelu_exact_derivative(x),
            sgfrelu_derivative(x),
        ):
            writer.writerow(
                [
                    *("" if np.isnan(value) else f"{value:.10f}" for value in row),
                    "analytical_definition",
                ]
            )


def plot_parameter_source() -> None:
    u = np.linspace(-2.5, 1.5, 2201)
    exact_derivative = gelu_exact_derivative(u)
    approx_derivative = gelu_tanh_derivative(u)

    fig, ax = plt.subplots(figsize=(160.0 / 25.4, 100.0 / 25.4))
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.19, top=0.965)

    ax.axhline(0.0, color="#505050", linewidth=0.8, zorder=1)
    ax.plot(
        u,
        exact_derivative,
        color=COLORS["exact"],
        label="精确GELU导数",
        zorder=3,
    )
    ax.plot(
        u,
        approx_derivative,
        color=COLORS["approx"],
        linestyle=(0, (4.5, 2.2)),
        label="tanh近似GELU导数",
        zorder=4,
    )
    ax.scatter(
        [EXACT_ROOT],
        [0.0],
        s=25,
        facecolor="white",
        edgecolor=COLORS["exact"],
        linewidth=1.1,
        zorder=5,
    )
    ax.scatter(
        [TANH_ROOT],
        [0.0],
        s=16,
        color=COLORS["approx"],
        zorder=6,
    )
    ax.annotate(
        r"$c_s=|\tilde{u}^{*}|=0.7524614$",
        xy=(TANH_ROOT, 0.0),
        xytext=(-2.10, 0.34),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["approx"],
            linewidth=0.9,
            shrinkA=1.5,
            shrinkB=3.5,
        ),
        color=COLORS["approx"],
        fontsize=11.5,
    )

    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-0.14, 1.14)
    ax.set_xticks(np.arange(-2.5, 1.6, 0.5))
    ax.set_yticks(np.arange(-0.1, 1.11, 0.2))
    ax.set_xlabel("输入 u", fontsize=11.5)
    ax.set_ylabel("导数值", fontsize=11.5)
    style_axis(ax)
    ax.tick_params(labelsize=10.8)
    ax.legend(loc="lower right", frameon=False, handlelength=2.3, fontsize=10.8)

    zoom = ax.inset_axes([0.13, 0.58, 0.43, 0.38])
    zoom.plot(u, exact_derivative, color=COLORS["exact"], linewidth=1.3)
    zoom.plot(
        u,
        approx_derivative,
        color=COLORS["approx"],
        linestyle=(0, (4.5, 2.2)),
        linewidth=1.3,
    )
    zoom.axhline(0.0, color="#505050", linewidth=0.65)
    zoom.axvline(EXACT_ROOT, color=COLORS["exact"], linewidth=0.75, alpha=0.75)
    zoom.axvline(TANH_ROOT, color=COLORS["approx"], linewidth=0.75, alpha=0.75)
    zoom.scatter([EXACT_ROOT], [0.0], s=17, facecolor="white", edgecolor=COLORS["exact"], linewidth=0.9, zorder=4)
    zoom.scatter([TANH_ROOT], [0.0], s=11, color=COLORS["approx"], zorder=5)
    zoom.set_xlim(-0.7532, -0.7509)
    zoom.set_ylim(-0.0007, 0.0007)
    zoom.set_xticks([EXACT_ROOT, TANH_ROOT])
    zoom.set_xticklabels(["-0.7518", "-0.7525"], rotation=0, ha="center")
    zoom.set_yticks([-0.0005, 0.0, 0.0005])
    zoom.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    zoom.tick_params(labelsize=10.5, pad=2.0)
    zoom.yaxis.get_offset_text().set_fontsize(10.5)
    for label in zoom.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.15))
    zoom.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.72)
    zoom.set_facecolor("white")
    for spine in zoom.spines.values():
        spine.set_color("#707070")
        spine.set_linewidth(0.65)
    _, connector_left, connector_right = mark_inset(
        ax, zoom, loc1=3, loc2=4, fc="none", ec="#888888", lw=0.55
    )
    # A single connector to the lower-right corner clears the parameter annotation.
    connector_left.set_visible(False)

    export_figure(fig, "4.3.1GELU导数零点与SGFRELU平移参数来源")


def plot_output_and_gradient() -> None:
    x = np.linspace(-3.0, 3.5, 2401)
    relu_output = np.maximum(0.0, x)
    gelu_output = gelu_exact(x)
    shifted_output = gelu_exact(x - SHIFT)
    sgf_output = sgfrelu(x)
    relu_gradient = np.where(x < 0.0, 0.0, 1.0)
    gelu_gradient = gelu_exact_derivative(x)
    sgf_gradient = sgfrelu_derivative(x)
    shifted_minimum_x = SHIFT + EXACT_ROOT
    shifted_minimum_y = float(gelu_exact(np.array([EXACT_ROOT]))[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(160.0 / 25.4, 104.0 / 25.4),
        gridspec_kw={"wspace": 0.36},
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.97)

    ax = axes[0]
    ax.axvspan(-3.0, SHIFT, color=COLORS["zero_fill"], alpha=0.62, zorder=0)
    ax.axhline(0.0, color="#505050", linewidth=0.75, zorder=1)
    ax.axvline(SHIFT, color=COLORS["hero"], linestyle=(0, (2.2, 2.0)), linewidth=0.9)
    ax.plot(x, relu_output, color=COLORS["relu"], linestyle=(0, (4.5, 2.2)), label="ReLU", zorder=2)
    ax.plot(x, gelu_output, color=COLORS["gelu"], linestyle=(0, (1.6, 1.5)), label="GELU", zorder=2)
    ax.plot(x, shifted_output, color=COLORS["shifted"], linewidth=1.25, label=r"$\mathrm{GELU}(x-c_s)$", zorder=3)
    ax.plot(x, sgf_output, color=COLORS["hero"], linewidth=2.2, label="SGFRELU", zorder=4)
    ax.scatter(
        [shifted_minimum_x],
        [shifted_minimum_y],
        s=22,
        facecolor="white",
        edgecolor=COLORS["shifted"],
        linewidth=1.0,
        zorder=5,
    )
    ax.annotate(
        "平移后极小点接近0",
        xy=(shifted_minimum_x, shifted_minimum_y),
        xytext=(-2.76, -0.35),
        arrowprops=dict(arrowstyle="->", color=COLORS["shifted"], linewidth=0.8),
        color="#7A4B00",
        fontsize=10.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.12),
    )
    ax.text(-1.45, 1.66, "阈值", fontsize=10.8, ha="center")
    ax.annotate(
        r"$x=c_s$", xy=(SHIFT, 1.42), xytext=(-1.45, 1.42),
        fontsize=10.8, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["hero"], linewidth=0.8),
    )
    ax.text(-1.65, 0.13, "零响应区间", color="#575B78", fontsize=10.8, ha="center")
    ax.set_xlim(-3.0, 3.5)
    ax.set_ylim(-0.48, 3.05)
    ax.set_xticks(np.arange(-3, 3.6, 1))
    ax.set_yticks(np.arange(-0.5, 3.1, 0.5))
    ax.set_xlabel("输入 x", fontsize=11.5)
    ax.set_ylabel("函数值", fontsize=11.5)
    style_axis(ax)
    ax.tick_params(labelsize=10.8)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, ncol=1, handlelength=1.7, fontsize=10.8)

    ax = axes[1]
    ax.axvspan(-3.0, SHIFT, color=COLORS["zero_fill"], alpha=0.62, zorder=0)
    ax.axhline(0.0, color="#505050", linewidth=0.75, zorder=1)
    ax.axvline(SHIFT, color=COLORS["hero"], linestyle=(0, (2.2, 2.0)), linewidth=0.9)
    ax.step(x, relu_gradient, where="post", color=COLORS["relu"], linestyle=(0, (4.5, 2.2)), label="ReLU", zorder=2)
    ax.plot(x, gelu_gradient, color=COLORS["gelu"], linestyle=(0, (1.6, 1.5)), label="GELU", zorder=2)
    ax.plot(x, sgf_gradient, color=COLORS["hero"], linewidth=2.2, label="SGFRELU", zorder=4)
    ax.scatter([SHIFT], [0.0], s=20, facecolor="white", edgecolor=COLORS["hero"], linewidth=1.0, zorder=5)
    ax.scatter([SHIFT], [0.5], s=18, color=COLORS["hero"], zorder=5)
    ax.plot([SHIFT, SHIFT], [0.0, 0.5], color=COLORS["hero"], linewidth=0.9, linestyle=(0, (1.5, 1.5)))
    ax.text(-1.80, 0.16, "零梯度区间", color="#575B78", fontsize=10.8, ha="center")
    ax.annotate(
        "阈值处\n导数跃迁",
        xy=(SHIFT, 0.27),
        xytext=(0.97, 0.15),
        arrowprops=dict(arrowstyle="->", color=COLORS["hero"], linewidth=0.8),
        color=COLORS["hero"],
        fontsize=10.8,
    )
    ax.set_xlim(-3.0, 3.5)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xticks(np.arange(-3, 3.6, 1))
    ax.set_yticks(np.arange(-0.1, 1.11, 0.2))
    ax.set_xlabel("输入 x", fontsize=11.5)
    ax.set_ylabel("导数值", fontsize=11.5)
    style_axis(ax)
    ax.tick_params(labelsize=10.8)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, handlelength=1.7, fontsize=10.8)

    export_figure(fig, "4.3.1SGFRELU与典型激活函数的输出及梯度特性")


def plot_shift_response() -> None:
    x = np.linspace(-1.0, 3.5, 1801)
    shifts = [0.50, SHIFT, 1.00]
    colors = ["#009E73", COLORS["hero"], "#E69F00"]
    line_styles = [(0, (4.5, 2.2)), "-", (0, (1.6, 1.5))]

    fig, ax = plt.subplots(figsize=(160.0 / 25.4, 70.0 / 25.4))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.94)

    ax.axhline(0.0, color="#505050", linewidth=0.75, zorder=1)
    for shift, color, line_style in zip(shifts, colors, line_styles):
        label = r"$c_s=0.7525$" if math.isclose(shift, SHIFT) else rf"$c_s={shift:.2f}$"
        ax.plot(
            x,
            sgfrelu(x, shift),
            color=color,
            linestyle=line_style,
            linewidth=2.1 if math.isclose(shift, SHIFT) else 1.55,
            label=label,
            zorder=3,
        )
        ax.axvline(shift, color=color, linewidth=0.7, alpha=0.62, linestyle=(0, (2.0, 2.0)))

    ax.annotate(
        r"$c_s$增大，零响应边界右移",
        xy=(1.0, 0.04),
        xytext=(1.48, 0.20),
        arrowprops=dict(arrowstyle="->", color="#4A4A4A", linewidth=0.85),
        color="#3A3A3A",
        fontsize=8.0,
    )
    ax.set_xlim(-1.0, 3.5)
    ax.set_ylim(-0.05, 2.85)
    ax.set_xticks(np.arange(-1.0, 3.6, 0.5))
    ax.set_yticks(np.arange(0.0, 2.6, 0.5))
    ax.set_xlabel(r"输入 $x$")
    ax.set_ylabel("SGFRELU函数值")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=2.3)

    export_figure(fig, "4.3.1不同平移参数下SGFRELU的阈值响应")


def main() -> None:
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_parameter_source()
    plot_output_and_gradient()
    print(f"exact_root={EXACT_ROOT:.10f}")
    print(f"tanh_root={TANH_ROOT:.10f}")
    print(f"implementation_shift={SHIFT:.10f}")
    print(f"shifted_exact_minimum_x={SHIFT + EXACT_ROOT:.10f}")
    print(f"outputs={TARGET_DIR}")


if __name__ == "__main__":
    main()
