"""Plotting utilities for consistent styling across notebooks and scripts.

These helpers provide small, composable wrappers around common matplotlib plots
so figures share a clean, consistent look. Functions import matplotlib lazily
to avoid importing it at module import time when not needed.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np


def hist_multi(
    arrays: Sequence[np.ndarray],
    bins: int,
    labels: Sequence[str],
    colors: Sequence[str],
    figsize: Tuple[float, float] = (10, 3),
    title: Optional[str] = None,
    stacked: bool = False,
    alpha: float = 0.6,
    edgecolor: str = "#333",
    linewidth: float = 0.4,
    scale_caption: Optional[str] = None,
) -> "tuple[object, object]":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    if stacked:
        ax.hist(arrays, bins=bins, stacked=True, label=labels, color=colors, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    else:
        for data, label, color in zip(arrays, labels, colors):
            ax.hist(data, bins=bins, alpha=alpha, color=color, label=label, edgecolor=edgecolor, linewidth=linewidth)
    if title:
        ax.set_title(title, pad=6)
    ax.legend(frameon=False)
    if scale_caption:
        ax.text(0.0, -0.18, scale_caption, transform=ax.transAxes, fontsize=10, color="#6b7280")
    plt.tight_layout()
    return fig, ax


def size_histograms(
    df: pd.DataFrame,
    columns: Sequence[str],
    bins: int,
    titles: Sequence[str],
    figsize: Tuple[float, float] = (10, 3.2),
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(columns), figsize=figsize)
    if len(columns) == 1:
        axes = [axes]
    for ax, col, ttl in zip(axes, columns, titles):
        ax.hist(df[col], bins=bins)
        if col == "aspect":
            ax.axvline(1.0, linestyle="--")
        ax.set_title(ttl); ax.set_xlabel("")
        if ax is not axes[0]: ax.set_yticklabels([])
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout(); plt.show()


__all__ = [
    "hist_multi",
    "size_histograms",
    "plot_grouped_bars_two_series",
]
 


def plot_grouped_bars_two_series(
    categories: Sequence[str],
    series_a: Sequence[float],
    series_b: Sequence[float],
    label_a: str = "before",
    label_b: str = "after",
    ylabel: str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (6.8, 3.0),
    legend_loc: str = "best",
    legend_outside: bool = False,
) -> None:
    """Plot grouped bars for two series along categorical x-axis.

    Args:
        categories: Category labels for the x-axis.
        series_a: First series values aligned to categories.
        series_b: Second series values aligned to categories.
        label_a: Legend label for first series.
        label_b: Legend label for second series.
        ylabel: Optional y-axis label.
        title: Optional plot title.
        ylim: Optional (min, max) y-limit.
        figsize: Figure size.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(categories))
    width = 0.4
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, series_a, width, label=label_a)
    ax.bar(x + width / 2, series_b, width, label=label_b)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    else:
        ax.legend(loc=legend_loc, frameon=False)
    plt.tight_layout(); plt.show()

__all__ += []


