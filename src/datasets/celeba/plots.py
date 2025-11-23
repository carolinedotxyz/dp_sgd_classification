"""Domain-specific plotting helpers for CelebA workflows.

These wrappers centralize visualizations used across notebooks and scripts,
keeping the narrative code concise and consistent.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ...visualization.general import (
    plot_grouped_bars_two_series,
    size_histograms,
    hist_multi,
)


def plot_attribute_overall(summary_all, top_n: int) -> None:
    import matplotlib.pyplot as plt

    plot_df = summary_all.sort_values("overall_pos_pct", ascending=False).head(top_n)
    fig2, ax2 = plt.subplots(figsize=(7.0, max(3.0, 0.32 * len(plot_df))))
    ax2.barh(plot_df["display_name"], plot_df["overall_pos_pct"])
    ax2.invert_yaxis(); ax2.set_xlabel("Positive rate (%)"); ax2.set_ylabel("")
    ax2.set_xlim(0, 100); ax2.set_title("Overall positive fraction per attribute", pad=8)
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    for y, v in enumerate(plot_df["overall_pos_pct"].tolist()):
        ax2.text(v + 0.5, y, f"{float(v):.1f}%", va="center", fontsize=9)
    plt.tight_layout(); plt.show()


def plot_average_and_diff(avg_orig, avg_crop, count: int, mode: str = "diff_only") -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    diff = np.abs(avg_orig - avg_crop).mean(axis=2)

    if mode == "diff_only":
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.6))
        imh = ax.imshow(diff, cmap="magma", vmin=0.0, vmax=float(diff.max()) or 1.0)
        ax.set_title("|Average original − cropped|", pad=6)
        ax.axis("off")
        plt.colorbar(imh, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.show()
        return

    # Fallback: original | cropped | abs-diff heatmap
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    axes[0].imshow(avg_orig)
    axes[0].set_title(f"Average — original (n={count})", pad=6)
    axes[0].axis("off")
    axes[1].imshow(avg_crop)
    axes[1].set_title(f"Average — cropped (n={count})", pad=6)
    axes[1].axis("off")
    imh = axes[2].imshow(diff, cmap="magma", vmin=0.0, vmax=float(diff.max()) or 1.0)
    axes[2].set_title("|Average original − cropped|", pad=6)
    axes[2].axis("off")
    cbar = fig.colorbar(imh, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout(); plt.show()


def plot_center_crop_overlays(paths: list[str], seed: int) -> None:
    from math import ceil
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from .diagnostics import sample_paths, area_retained_after_center_square

    overlay_paths = sample_paths(paths, min(3, len(paths)), seed)
    if not overlay_paths:
        return
    n = len(overlay_paths)
    cols = 3
    rows = int(ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.2, 2.6 * rows))
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
    axes = axes.reshape(rows, cols)
    for i, p in enumerate(overlay_paths):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                ax.imshow(im)
                rect = patches.Rectangle((left, top), side, side, linewidth=2, edgecolor="#E45756", facecolor="none")
                ax.add_patch(rect)
                ax.axis("off")
                kept = area_retained_after_center_square(w, h)
                ax.set_title(f"{w}×{h} — keeps ~{kept:.2f} area", fontsize=9, pad=4)
        except Exception:
            ax.axis("off")
            continue
    # Hide any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    plt.tight_layout(); plt.show()


 


def plot_channel_bars(m_b, m_a, s_b, s_a):
    """Plot means and stds before vs after; return (fig_mean, fig_std)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from ...notebooks.display import add_bar_labels, viz_style

    labels_rgb = ["R", "G", "B"]

    with viz_style():
        # Means
        fig1, ax1 = plt.subplots(figsize=(7.2, 3.0))
        x = np.arange(len(labels_rgb)); w = 0.38
        bars1 = ax1.bar(x - w/2, m_b, width=w, label="Before", color="#4C78A8")
        bars2 = ax1.bar(x + w/2, m_a, width=w, label="After", color="#54A24B")
        ax1.set_xticks(x); ax1.set_xticklabels(labels_rgb)
        ax1.set_title("Channel Means (TRAIN) — Before vs After", pad=8)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.legend(frameon=False)
        add_bar_labels(ax1, fmt=lambda v: f"{v:.3f}")
        # Delta annotations below pairs
        for i, (b, a) in enumerate(zip(m_b, m_a)):
            delta = float(a) - float(b)
            ax1.text(i, min(b, a) - 0.02 * max(1.0, ax1.get_ylim()[1]), f"Δ = {delta:+.3f}", ha="center", va="top", fontsize=10, color="#6b7280")

        # STDs
        fig2, ax2 = plt.subplots(figsize=(7.2, 3.0))
        bars1 = ax2.bar(x - w/2, s_b, width=w, label="Before", color="#E45756")
        bars2 = ax2.bar(x + w/2, s_a, width=w, label="After", color="#72B7B2")
        ax2.set_xticks(x); ax2.set_xticklabels(labels_rgb)
        ax2.set_title("Channel STDs (TRAIN) — Before vs After", pad=8)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.legend(frameon=False)
        add_bar_labels(ax2, fmt=lambda v: f"{v:.3f}")
        for i, (b, a) in enumerate(zip(s_b, s_a)):
            delta = float(a) - float(b)
            ax2.text(i, min(b, a) - 0.02 * max(1.0, ax2.get_ylim()[1]), f"Δ = {delta:+.3f}", ha="center", va="top", fontsize=10, color="#6b7280")

    return fig1, fig2


def plot_processed_pixel_hists(train_paths: list[str], bins: int, scale_01: bool, seed: int):
    import numpy as np
    from PIL import Image
    from .diagnostics import sample_paths

    hist_paths = sample_paths(train_paths, min(128, len(train_paths)), seed)
    vals = []
    for p in hist_paths:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32)
                if scale_01:
                    arr = arr / 255.0
                vals.append(arr.reshape(-1, 3))
        except (OSError, ValueError):
            continue
    if vals:
        stacked = np.concatenate(vals, axis=0)
        arrays = [stacked[:, 0], stacked[:, 1], stacked[:, 2]]
        labels = ["R", "G", "B"]
        colors = ["#E45756", "#54A24B", "#4C78A8"]
        title = "Pixel Value Distribution — Processed ([0,1])" if scale_01 else "Pixel Value Distribution — Processed ([0,255])"
        caption = "Scale: [0,1]" if scale_01 else "Scale: [0,255]"
        fig, ax = hist_multi(arrays, bins=bins, labels=labels, colors=colors, figsize=(10, 3), title=title, stacked=False, alpha=0.6, edgecolor="#333", linewidth=0.3, scale_caption=caption)
        if scale_01:
            from ...notebooks.display import percent_formatter
            ax.xaxis.set_major_formatter(percent_formatter())
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return fig


def plot_original_pixel_hists(paths: list[str], sample_n: int, scale_01: bool, seed: int):
    import numpy as np
    from PIL import Image
    from .diagnostics import sample_paths

    hist_paths = sample_paths(paths, min(sample_n, len(paths)), seed)
    vals = []
    for p in hist_paths:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32)
                if scale_01:
                    arr = arr / 255.0
                vals.append(arr.reshape(-1, 3))
        except (OSError, ValueError):
            continue
    if vals:
        stacked = np.concatenate(vals, axis=0)
        arrays = [stacked[:, 0], stacked[:, 1], stacked[:, 2]]
        labels = ["R", "G", "B"]
        colors = ["#E45756", "#54A24B", "#4C78A8"]
        title = "Pixel Value Distribution — Original ([0,1])" if scale_01 else "Pixel Value Distribution — Original ([0,255])"
        caption = "Scale: [0,1]" if scale_01 else "Scale: [0,255]"
        fig, ax = hist_multi(arrays, bins=50, labels=labels, colors=colors, figsize=(10, 3), title=title, stacked=False, alpha=0.6, edgecolor="#333", linewidth=0.3, scale_caption=caption)
        if scale_01:
            from ...notebooks.display import percent_formatter
            ax.xaxis.set_major_formatter(percent_formatter())
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return fig


def render_size_block(label: str, df_sizes) -> None:
    from IPython.display import display, HTML
    from ...notebooks.display import _h, _has_variance

    _h(f"{label} — sizes")
    has = not df_sizes.empty
    if not has:
        display(HTML("<div style='color:#6b7280'>No readable images in sample.</div>"))
        return
    w_const = not _has_variance(df_sizes["width"]) 
    h_const = not _has_variance(df_sizes["height"]) 
    a_const = not _has_variance(df_sizes["aspect"]) 
    if w_const and h_const and a_const:
        w = int(df_sizes['width'].iloc[0]); h = int(df_sizes['height'].iloc[0]); ar = float(df_sizes['aspect'].iloc[0])
        display(HTML(f"<div style='color:#374151'>All sampled images are <b>{w}×{h}</b> px (aspect <b>{ar:.2f}</b>).</div>"))
    else:
        from .diagnostics import describe_numeric
        display(describe_numeric(df_sizes, ["width","height","aspect"]))
        cols, titles = [], []
        if not w_const: cols.append("width");  titles.append("Width")
        if not h_const: cols.append("height"); titles.append("Height")
        if not a_const: cols.append("aspect"); titles.append("Aspect ratio (W/H)")
        if cols:
            size_histograms(df_sizes, columns=cols, titles=titles, bins=30, figsize=(10, 3.2))


def plot_class_balance_stacked_with_subtitle(splits: list[str], pos_counts: list[float], total_counts: list[float]):
    """Stacked bars with centered value labels and a subtitle. Returns (fig, ax)."""
    import matplotlib.pyplot as plt
    neg_counts = [float(t) - float(p) for p, t in zip(pos_counts, total_counts)]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    b1 = ax.bar(splits, neg_counts, label="Negative", color="#4C78A8")
    b2 = ax.bar(splits, pos_counts, bottom=neg_counts, label="Positive", color="#54A24B")
    ax.set_ylabel("Images"); ax.set_xlabel("")
    ax.set_title("Class Balance by Split\nTarget: 50/50 per split. Δ shown in table above.", pad=8, loc="center")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for rects in (b1, b2):
        for r in rects:
            h = r.get_height()
            y = r.get_y() + h / 2.0
            ax.text(r.get_x() + r.get_width()/2.0, y, f"{int(round(h)):,}", ha="center", va="center", fontsize=10, color="#111827")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_geometry_panel(widths: "np.ndarray", heights: "np.ndarray"):
    """Render a compact geometry panel: modal size, aspect, and small hists. Returns (fig, axes)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    try:
        size_pairs = list(zip(widths.astype(int).tolist(), heights.astype(int).tolist()))
        common = Counter(size_pairs).most_common(1)[0][0] if size_pairs else (np.nan, np.nan)
    except Exception:
        common = (np.nan, np.nan)
    w_modal, h_modal = common
    aspect = (np.nan if (not widths.size or not heights.size) else float(np.median(widths / heights)))
    axes[0].axis('off')
    axes[0].text(0.0, 0.7, "Modal size", fontsize=10, color="#6b7280", transform=axes[0].transAxes)
    axes[0].text(0.0, 0.35, f"{int(w_modal)}×{int(h_modal)}", fontsize=18, fontweight=600, transform=axes[0].transAxes)
    axes[0].text(0.0, 0.05, f"Aspect: {aspect:.2f}", fontsize=11, transform=axes[0].transAxes)
    axes[1].hist(widths, bins=15, color="#4C78A8", alpha=0.8)
    axes[1].set_title("Width", pad=6); axes[1].set_xlabel("")
    axes[2].hist(heights, bins=15, color="#54A24B", alpha=0.8)
    axes[2].set_title("Height", pad=6); axes[2].set_xlabel("")
    plt.tight_layout()
    return fig, axes


def plot_retained_area_kpi(values: "np.ndarray"):
    """Single KPI card for retained area with a slim 0→1 bar and marker. Returns (fig, ax)."""
    import numpy as np
    import matplotlib.pyplot as plt
    v = float(np.median(values)) if values.size else float("nan")
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.axis('off')
    ax.text(0.0, 0.75, "Retained Area (Center Square)", fontsize=12, color="#6b7280", transform=ax.transAxes)
    ax.text(0.0, 0.35, f"{v:.2f}", fontsize=22, fontweight=600, transform=ax.transAxes)
    ax.hlines(0.05, 0.0, 1.0, colors="#e5e7eb", linewidth=6, transform=ax.transAxes)
    ax.vlines(v, 0.05-0.06, 0.05+0.06, colors="#4C78A8", linewidth=3, transform=ax.transAxes)
    ax.text(1.0, 0.0, "~%d%% pixels kept" % int(round(v*100)), fontsize=10, ha="right", transform=ax.transAxes)
    plt.tight_layout()
    return fig, ax


def plot_channel_means_single(means: "tuple[float, float, float]"):
    """Single-series channel means bar chart (original only). Returns (fig, ax)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from ...notebooks.display import add_bar_labels

    labels = ["R", "G", "B"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(x, means, color=["#E45756", "#54A24B", "#4C78A8"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Channel Means (TRAIN, Original)", pad=8)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    add_bar_labels(ax, fmt=lambda v: f"{v:.3f}")
    plt.tight_layout()
    return fig, ax



