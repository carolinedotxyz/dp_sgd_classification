"""Subset builder for CelebA balanced binary attributes.

Provides programmatic APIs that mirror the CLI in scripts/celeba_build_subset.py,
so notebooks and other modules can build subsets without invoking a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import shutil
import pandas as pd


PARTITION_NAME_BY_ID: Dict[int, str] = {0: "train", 1: "val", 2: "test"}


@dataclass
class Selection:
    image_id: str
    label: int  # 1=positive attribute, 0=negative
    partition_name: str
    source_path: str
    dest_path: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_lookup(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, object]]:
    return df.set_index("image_id")[cols].to_dict("index")


def compute_interocular_distance(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    required = ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y"]
    for c in required:
        if c not in landmarks_df.columns:
            raise ValueError(f"Landmarks column missing: {c}")
    ld = landmarks_df.copy()
    for c in required:
        ld[c] = pd.to_numeric(ld[c], errors="coerce")
    ld["interocular"] = ((ld["lefteye_x"] - ld["righteye_x"]) ** 2 + (ld["lefteye_y"] - ld["righteye_y"]) ** 2) ** 0.5
    return ld[["image_id", "interocular"]]


def merge_and_filter_by_landmarks(merged_df: pd.DataFrame, landmarks_df: pd.DataFrame, min_iod: float | None, max_iod: float | None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    iod_df = compute_interocular_distance(landmarks_df)
    with_iod = merged_df.merge(iod_df, on="image_id", how="inner")
    if min_iod is not None:
        with_iod = with_iod[with_iod["interocular"] >= float(min_iod)]
    if max_iod is not None:
        with_iod = with_iod[with_iod["interocular"] <= float(max_iod)]
    lookup = with_iod.set_index("image_id")["interocular"].to_dict()
    return with_iod, lookup


def discretize_interocular(df: pd.DataFrame, num_bins: int) -> pd.Series:
    series = pd.to_numeric(df["interocular"], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return pd.Series(["nan"] * len(df), index=df.index)
    try:
        bins = pd.qcut(valid, q=min(num_bins, max(1, valid.nunique())), duplicates="drop")
    except Exception:
        bins = pd.cut(valid, bins=num_bins)
    out = pd.Series(["nan"] * len(df), index=df.index, dtype=object)
    out.loc[valid.index] = bins.astype(str)
    return out


def build_strata_key(df: pd.DataFrame, strat_cols: List[str]) -> pd.Series:
    return df[strat_cols].astype(str).agg("|".join, axis=1)


def allocate_per_stratum(min_counts: Dict[str, int], desired: int) -> Dict[str, int]:
    if not min_counts:
        return {}
    total_min = sum(min_counts.values())
    if total_min == 0:
        return {k: 0 for k in min_counts.keys()}
    alloc = {k: int(min_counts[k] * desired / total_min) for k in min_counts}
    remaining = desired - sum(alloc.values())
    slack = {k: min_counts[k] - alloc[k] for k in min_counts}
    for k in sorted(slack.keys(), key=lambda x: slack[x], reverse=True):
        if remaining <= 0:
            break
        if slack[k] > 0:
            alloc[k] += 1
            remaining -= 1
    return alloc


def sample_stratified_split(sub_df: pd.DataFrame, label_col: str, strat_cols: List[str], desired_per_class: int, seed: int) -> pd.DataFrame:
    work = sub_df.copy()
    work["__stratum__"] = build_strata_key(work, strat_cols)
    pos = work[work[label_col] == 1]
    neg = work[work[label_col] == 0]
    pos_counts = pos["__stratum__"].value_counts().to_dict()
    neg_counts = neg["__stratum__"].value_counts().to_dict()
    common = set(pos_counts.keys()) & set(neg_counts.keys())
    min_counts = {k: min(pos_counts[k], neg_counts[k]) for k in common}
    if not min_counts:
        limit = min(len(pos), len(neg), desired_per_class)
        return pd.concat([
            pos.sample(n=limit, random_state=seed, replace=False),
            neg.sample(n=limit, random_state=seed + 1, replace=False),
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    alloc = allocate_per_stratum(min_counts, desired_per_class)
    selected = []
    for k, n in alloc.items():
        if n <= 0:
            continue
        pos_k = pos[pos["__stratum__"] == k]
        neg_k = neg[neg["__stratum__"] == k]
        m = min(n, len(pos_k), len(neg_k))
        if m <= 0:
            continue
        selected.append(pos_k.sample(n=m, random_state=seed, replace=False))
        selected.append(neg_k.sample(n=m, random_state=seed + 1, replace=False))
    return pd.concat(selected, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def materialize_selection(selections: List[Selection], link_mode: str, overwrite: bool, dry_run: bool) -> None:
    for sel in selections:
        if dry_run:
            print(f"DRY-RUN would create: {sel.dest_path} -> {sel.source_path}")
            continue
        ensure_dir(os.path.dirname(sel.dest_path))
        if os.path.lexists(sel.dest_path):
            if overwrite:
                if os.path.islink(sel.dest_path) or os.path.isfile(sel.dest_path):
                    os.remove(sel.dest_path)
                else:
                    shutil.rmtree(sel.dest_path)
            else:
                continue
        if link_mode == "symlink":
            os.symlink(sel.source_path, sel.dest_path)
        elif link_mode == "copy":
            shutil.copy2(sel.source_path, sel.dest_path)
        else:
            raise ValueError("link_mode must be 'symlink' or 'copy'")


def build_subset(
    merged: pd.DataFrame,
    images_root: str,
    output_dir: str,
    attribute: str,
    link_mode: str,
    seed: int,
    max_per_class_by_split: Dict[str, Optional[int]],
    overwrite: bool,
    dry_run: bool,
    strict_missing: bool = False,
    fill_missing: bool = True,
    landmarks_lookup: Dict[str, float] | None = None,
    attrs_to_include: List[str] | None = None,
    attrs_lookup: Dict[str, Dict[str, object]] | None = None,
    bbox_lookup: Dict[str, Dict[str, object]] | None = None,
    landmarks_raw_lookup: Dict[str, Dict[str, object]] | None = None,
    stratify_by: List[str] | None = None,
    iod_bins: int | None = None,
) -> None:
    df = merged.copy()
    df[attribute] = pd.to_numeric(df[attribute], errors="coerce").fillna(-1).astype(int)
    df["label"] = (df[attribute] == 1).astype(int)

    if stratify_by:
        cols = list(stratify_by)
        if "interocular" in cols:
            if "interocular" not in df.columns:
                raise ValueError("Stratify by interocular requires interocular column present.")
            bins = discretize_interocular(df, iod_bins or 4)
            df["iod_bin"] = bins
            cols = [c if c != "interocular" else "iod_bin" for c in cols]
    else:
        cols = []

    selections: List[Selection] = []
    rows_meta: List[Dict[str, str | int | float]] = []
    for pname in ["train", "val", "test"]:
        sub = df[df["partition_name"] == pname]
        cap = max_per_class_by_split.get(pname)
        desired = min(int((sub["label"] == 1).sum()), int((sub["label"] == 0).sum()))
        if cap is not None:
            desired = min(desired, int(cap))
        print(f"Split {pname}: target per-class cap {desired}")
        if cols and desired > 0:
            chosen = sample_stratified_split(sub_df=sub, label_col="label", strat_cols=cols, desired_per_class=desired, seed=seed)
        else:
            pos = sub[sub["label"] == 1]
            neg = sub[sub["label"] == 0]
            limit = min(len(pos), len(neg), desired)
            chosen = pd.concat([
                pos.sample(n=limit, random_state=seed, replace=False),
                neg.sample(n=limit, random_state=seed + 1, replace=False),
            ]).sample(frac=1.0, random_state=seed)

        def try_add(pool_row: pd.Series, label_val: int) -> bool:
            image_id = str(pool_row["image_id"])
            source_local = os.path.join(images_root, image_id)
            if not dry_run and not os.path.isfile(source_local):
                if strict_missing:
                    raise FileNotFoundError(f"Image not found: {source_local}")
                print(f"Skipping missing image: {source_local}")
                return False
            class_name = "eyeglasses" if label_val == 1 else "no_eyeglasses"
            dest_local = os.path.join(output_dir, pname, class_name, image_id)
            interocular_val = None
            if landmarks_lookup is not None:
                interocular_val = landmarks_lookup.get(image_id)
            if link_mode == "copy":
                rec_source = os.path.relpath(source_local, output_dir)
                rec_dest = os.path.relpath(dest_local, output_dir)
            else:
                rec_source = source_local
                rec_dest = dest_local
            rows: Dict[str, str | int | float] = {
                "image_id": image_id,
                "label": label_val,
                "class_name": class_name,
                "partition_name": pname,
                "source_path": rec_source,
                "dest_path": rec_dest,
            }
            if interocular_val is not None:
                rows["interocular"] = interocular_val
            if attrs_to_include and attrs_lookup is not None:
                ad = attrs_lookup.get(image_id)
                if ad:
                    for k in attrs_to_include:
                        if k in ad:
                            rows[k] = ad[k]
            if bbox_lookup is not None:
                bd = bbox_lookup.get(image_id)
                if bd:
                    for k, v in bd.items():
                        rows[k] = v
            if landmarks_raw_lookup is not None:
                ld = landmarks_raw_lookup.get(image_id)
                if ld:
                    for k, v in ld.items():
                        rows[k] = v
            selections.append(
                Selection(
                    image_id=image_id,
                    label=label_val,
                    partition_name=pname,
                    source_path=source_local,
                    dest_path=dest_local,
                )
            )
            rows_meta.append(rows)
            return True

        kept_pos = kept_neg = 0
        selected_ids = set()
        for _, row in chosen.iterrows():
            label_val = int(row["label"])
            if try_add(row, label_val):
                selected_ids.add(row["image_id"])
                if label_val == 1:
                    kept_pos += 1
                else:
                    kept_neg += 1

        if fill_missing and (kept_pos < desired or kept_neg < desired):
            def backfill_for_label(target_label: int, need: int) -> int:
                if need <= 0:
                    return 0
                candidates = (
                    sub[(sub["label"] == target_label) & (~sub["image_id"].isin(selected_ids))]
                    .sample(frac=1.0, random_state=seed + (137 if target_label == 1 else 138))
                    .reset_index(drop=True)
                )
                added = 0
                for _, crow in candidates.iterrows():
                    if try_add(crow, target_label):
                        selected_ids.add(crow["image_id"])
                        added += 1
                        if added >= need:
                            break
                return added

            add_pos = backfill_for_label(1, desired - kept_pos)
            kept_pos += add_pos
            add_neg = backfill_for_label(0, desired - kept_neg)
            kept_neg += add_neg

        print(f"  Kept {kept_pos} pos, {kept_neg} neg (total {kept_pos + kept_neg})")
        if kept_pos < desired or kept_neg < desired:
            print(f"  Warning: could not reach desired cap for split '{pname}'. Available pos={kept_pos}, neg={kept_neg}.")

    ensure_dir(output_dir)
    index_csv = os.path.join(output_dir, f"subset_index_{attribute.lower()}.csv")
    index_df = pd.DataFrame(rows_meta)
    if dry_run:
        print(f"DRY-RUN would write index CSV: {index_csv} with {len(index_df)} rows")
    else:
        index_df.to_csv(index_csv, index=False)
        print(f"Wrote index CSV: {index_csv}")

    materialize_selection(selections, link_mode=link_mode, overwrite=overwrite, dry_run=dry_run)
    if not dry_run:
        print(f"Created files under: {output_dir}")


