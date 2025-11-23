"""
Dataset-specific implementations.

This package contains dataset-specific code organized by dataset name.
Each dataset has its own subpackage with IO, preprocessing, analysis, etc.
"""

from .celeba import (
    # IO
    validate_archive_dir,
    load_archive_paths,
    load_archive_data,
    write_archive_outputs,
    # Index
    read_processed_index_csv,
    augment_processed_index_with_sizes,
    summarize_processed_index_df,
    load_processed_index_or_raise,
    ensure_partition_and_class_columns,
    find_index_csv,
    load_subset_index,
    iter_subset_paths,
    # Analysis
    compute_attribute_summary,
    build_focus_table,
    build_balance_comparison,
    build_balance_display_table,
    compute_balance_from_df,
    # Diagnostics
    select_visual_paths,
    compute_average_original_and_cropped,
    sample_paths,
    collect_size_stats,
    describe_numeric,
    compute_channel_stats,
    center_square_crop,
    area_retained_after_center_square,
    # Plots
    plot_grouped_bars_two_series,
    size_histograms,
    hist_multi,
    plot_average_and_diff,
    plot_center_crop_overlays,
    plot_channel_bars,
    plot_processed_pixel_hists,
    plot_original_pixel_hists,
    render_size_block,
    # Workflow
    WorkflowConfig,
    build_subset as workflow_build_subset,
    preprocess_images,
    preview_center_crop_diagnostics,
    preprocess_images_only,
    review_archive,
    analyze_processed,
    analyze_original_subset,
)

__all__ = [
    # IO
    "validate_archive_dir",
    "load_archive_paths",
    "load_archive_data",
    "write_archive_outputs",
    # Index
    "read_processed_index_csv",
    "augment_processed_index_with_sizes",
    "summarize_processed_index_df",
    "load_processed_index_or_raise",
    "ensure_partition_and_class_columns",
    "find_index_csv",
    "load_subset_index",
    "iter_subset_paths",
    # Analysis
    "compute_attribute_summary",
    "build_focus_table",
    "build_balance_comparison",
    "build_balance_display_table",
    "compute_balance_from_df",
    # Diagnostics
    "select_visual_paths",
    "compute_average_original_and_cropped",
    "sample_paths",
    "collect_size_stats",
    "describe_numeric",
    "compute_channel_stats",
    "center_square_crop",
    "area_retained_after_center_square",
    # Plots
    "plot_grouped_bars_two_series",
    "size_histograms",
    "hist_multi",
    "plot_average_and_diff",
    "plot_center_crop_overlays",
    "plot_channel_bars",
    "plot_processed_pixel_hists",
    "plot_original_pixel_hists",
    "render_size_block",
    # Workflow
    "WorkflowConfig",
    "workflow_build_subset",
    "preprocess_images",
    "preview_center_crop_diagnostics",
    "preprocess_images_only",
    "review_archive",
    "analyze_processed",
    "analyze_original_subset",
]

