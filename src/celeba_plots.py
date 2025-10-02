"""Domain-specific plotting helpers for CelebA workflows.

These wrappers centralize visualizations used across notebooks and scripts,
keeping the narrative code concise and consistent.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from src.plotting import (
    plot_grouped_bars_two_series,
    size_histograms,
    hist_multi,
)


def plot_split_sizes(split_counts) -> None:
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(5.2, 2.4))
    ax1.bar(split_counts["split"], split_counts["count"])
    ax1.set_ylabel("images"); ax1.set_xlabel("")
    ax1.set_title("Split sizes", pad=8)
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for i, v in enumerate(split_counts["count"].tolist()):
        ax1.text(i, v, f"{v:,}", va="bottom", ha="center", fontsize=9)
    plt.tight_layout(); plt.show()


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


def plot_average_and_diff(avg_orig, avg_crop, count: int) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].imshow(avg_orig)
    axes[0].set_title(f"Average image — original (n={count})")
    axes[0].axis("off")
    axes[1].imshow(avg_crop)
    axes[1].set_title(f"Average image — center-cropped (n={count})")
    axes[1].axis("off")
    plt.tight_layout(); plt.show()

    diff = np.abs(avg_orig - avg_crop).mean(axis=2)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.6))
    imh = ax.imshow(diff, cmap="magma", vmin=0.0, vmax=float(diff.max()) or 1.0)
    ax.set_title("|Average original − cropped| (per-pixel mean)")
    ax.axis("off")
    plt.colorbar(imh, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()


def plot_center_crop_overlays(paths: list[str], seed: int) -> None:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from src.celeba_diagnostics import sample_paths, area_retained_after_center_square

    overlay_paths = sample_paths(paths, min(6, len(paths)), seed)
    if not overlay_paths:
        return
    rows = len(overlay_paths)
    fig = plt.figure(figsize=(6.4, 2.4 * rows))
    for i, p in enumerate(overlay_paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                ax = plt.subplot(rows, 1, i + 1)
                ax.imshow(im)
                rect = patches.Rectangle((left, top), side, side, linewidth=2, edgecolor="#E45756", facecolor="none")
                ax.add_patch(rect)
                ax.axis("off")
                kept = area_retained_after_center_square(w, h)
                ax.set_title(f"Center-square frame overlay ({w}x{h}) — keeps ~{kept:.2f} area")
        except Exception:
            continue
    plt.tight_layout(); plt.show()


 


def plot_channel_bars(m_b, m_a, s_b, s_a) -> None:
    labels_rgb = ["R", "G", "B"]
    plot_grouped_bars_two_series(
        categories=labels_rgb,
        series_a=m_b,
        series_b=m_a,
        label_a="before (original)",
        label_b="after (processed)",
        title="Channel means (TRAIN)",
        figsize=(6.4, 2.6),
        legend_loc="upper right",
        legend_outside=False,
    )
    plot_grouped_bars_two_series(
        categories=labels_rgb,
        series_a=s_b,
        series_b=s_a,
        label_a="before (original)",
        label_b="after (processed)",
        title="Channel stds (TRAIN)",
        figsize=(6.4, 2.6),
        legend_loc="upper right",
        legend_outside=False,
    )


def plot_processed_pixel_hists(train_paths: list[str], bins: int, scale_01: bool, seed: int) -> None:
    import numpy as np
    from PIL import Image
    from src.celeba_diagnostics import sample_paths

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
        hist_multi(arrays, bins=bins, labels=labels, colors=colors, figsize=(10, 3), title=f"Processed pixel value distribution ({'[0,1]' if scale_01 else '[0,255]'} scale)")


def plot_original_pixel_hists(paths: list[str], sample_n: int, scale_01: bool, seed: int) -> None:
    import numpy as np
    from PIL import Image
    from src.celeba_diagnostics import sample_paths

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
        hist_multi(arrays, bins=50, labels=labels, colors=colors, figsize=(10, 3), title=f"Original pixel value distribution ({'[0,1]' if scale_01 else '[0,255]'} scale)")


def render_size_block(label: str, df_sizes) -> None:
    from IPython.display import display, HTML
    from src.nb_display import _h, _has_variance

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
        from src.celeba_diagnostics import describe_numeric  # type: ignore
        display(describe_numeric(df_sizes, ["width","height","aspect"]))
        cols, titles = [], []
        if not w_const: cols.append("width");  titles.append("Width")
        if not h_const: cols.append("height"); titles.append("Height")
        if not a_const: cols.append("aspect"); titles.append("Aspect ratio (W/H)")
        if cols:
            size_histograms(df_sizes, columns=cols, titles=titles, bins=30, figsize=(10, 3.2))


def plot_class_balance_stacked(splits: list[str], pos_counts: list[float], total_counts: list[float], ylabel: str = "images", title: str = "Class balance by split") -> None:
    """Plot stacked bars for negative and positive counts per split.

    total_counts should be the sum of pos+neg per split; neg is computed as total-pos.
    """
    import matplotlib.pyplot as plt
    neg_counts = [float(t) - float(p) for p, t in zip(pos_counts, total_counts)]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar(splits, neg_counts, label="negative")
    ax.bar(splits, pos_counts, bottom=neg_counts, label="positive")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for i, t in enumerate(total_counts):
        ax.text(i, float(t), f"{int(t):,}", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False)
    plt.tight_layout(); plt.show()


def plot_area_retained_hist(df, bins: int = 40) -> None:
    """Plot histogram of the 'area_retained_center_square' column if present."""
    import matplotlib.pyplot as plt
    if "area_retained_center_square" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.hist(df["area_retained_center_square"], bins=bins)
    ax.set_xlabel("fraction of original area kept"); ax.set_ylabel("count")
    ax.set_title("Area retained by center square crop", pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout(); plt.show()


