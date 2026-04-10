"""Download/export helpers for study analysis results."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile

from fastapi import HTTPException
import numpy as np
import pandas as pd

from ..schemas import DownloadResponse
from ..utils import ensure_directory
from .analysis_service import _collect_replicate_metrics
from .channels import channel_definition_map
from .study_common import DOWNLOAD_ROOT, GRAPH_PAD_EXPORT_KEYS, _get_record, _threshold_dict


LOGGER = logging.getLogger(__name__)


def _graphpad_group_order(mouse_df: pd.DataFrame) -> List[str]:
    if mouse_df.empty or "Group" not in mouse_df:
        return []
    order: List[str] = []
    for value in mouse_df["Group"]:
        group = str(value)
        if group not in order:
            order.append(group)
    return order


def _build_graphpad_block(
    mouse_df: pd.DataFrame, metric_key: str, label: str, group_order: List[str]
) -> Tuple[Optional[pd.DataFrame], Dict[str, int]]:
    if metric_key not in mouse_df.columns or not group_order:
        return None, {}
    columns: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    max_len = 0
    for group in group_order:
        mask = mouse_df["Group"] == group
        values = mouse_df.loc[mask, metric_key].dropna().tolist()
        column_name = f"{label} - {group}"
        columns[column_name] = values
        counts[column_name] = len(values)
        max_len = max(max_len, len(values))
    if not columns:
        return None, {}
    if max_len == 0:
        max_len = 1
    padded = {name: values + [np.nan] * (max_len - len(values)) for name, values in columns.items()}
    return pd.DataFrame(padded), counts


def _build_graphpad_dataframe(
    mouse_df: pd.DataFrame,
    ratio_defs: List[Dict[str, object]],
    channel_defs: List[Dict[str, object]],
) -> pd.DataFrame:
    if mouse_df.empty:
        return pd.DataFrame()
    group_order = _graphpad_group_order(mouse_df)
    if not group_order:
        return pd.DataFrame()
    blocks: List[pd.DataFrame] = []
    counts: Dict[str, int] = {}
    channel_map = channel_definition_map(channel_defs)
    for metric_key in GRAPH_PAD_EXPORT_KEYS:
        channel_index = int(metric_key.split("_")[1])
        label = str(channel_map.get(channel_index, {}).get("label", f"Channel {channel_index}"))
        block, block_counts = _build_graphpad_block(mouse_df, metric_key, label, group_order)
        if block is None:
            continue
        blocks.append(block)
        counts.update(block_counts)
    for ratio in ratio_defs:
        column = ratio["id"]
        block, block_counts = _build_graphpad_block(mouse_df, column, ratio["label"], group_order)
        if block is None:
            continue
        blocks.append(block)
        counts.update(block_counts)
    if not blocks:
        return pd.DataFrame()
    combined = pd.concat(blocks, axis=1)
    combined.reset_index(drop=True, inplace=True)
    count_row = pd.DataFrame([{col: counts.get(col, 0) for col in combined.columns}])
    graphpad_df = pd.concat([count_row, combined], ignore_index=True)
    index_labels = ["n"] + [str(i) for i in range(1, len(graphpad_df))]
    graphpad_df.index = index_labels
    return graphpad_df


def _format_replicates_dataframe(
    individual_df: pd.DataFrame,
    ratios: List[Dict[str, object]],
    channel_defs: List[Dict[str, object]],
) -> pd.DataFrame:
    if individual_df.empty:
        return individual_df

    table = individual_df.copy()
    channel_map = channel_definition_map(channel_defs)
    ch1_label = str(channel_map.get(1, {}).get("label", "Channel 1"))
    ch2_label = str(channel_map.get(2, {}).get("label", "Channel 2"))
    ch3_label = str(channel_map.get(3, {}).get("label", "Channel 3"))
    rename_map = {
        "group": "Group",
        "mouse_id": "Mouse ID",
        "replicate_index": "Replicate #",
        "filename": "Filename",
        "channel_1_area": f"{ch1_label} Area (%)",
        "channel_2_area": f"{ch2_label} Area (%)",
        "channel_3_area": f"{ch3_label} Area (%)",
        "channel_1_3_ratio": "Channel 1 / Channel 3",
        "channel_2_3_ratio": "Channel 2 / Channel 3",
    }
    ratio_labels: List[str] = []
    for ratio in ratios:
        ratio_id = ratio.get("id")
        if not ratio_id:
            continue
        label = ratio.get("label") or ratio_id
        rename_map[ratio_id] = label
        ratio_labels.append(label)

    table.rename(columns=rename_map, inplace=True)
    ordered_columns = [
        "Group",
        "Mouse ID",
        "Replicate #",
        "Filename",
        f"{ch1_label} Area (%)",
        f"{ch2_label} Area (%)",
        f"{ch3_label} Area (%)",
    ]
    ordered_columns.extend(ratio_labels)
    existing_columns = [column for column in ordered_columns if column in table.columns]
    table = table[existing_columns]
    metric_columns = [
        column
        for column in existing_columns
        if column.startswith("Channel") or column in ratio_labels
    ]
    for column in metric_columns:
        table[column] = table[column].astype(float).round(4)
    sort_columns = [column for column in ("Group", "Mouse ID", "Replicate #") if column in table.columns]
    if sort_columns:
        table.sort_values(sort_columns, inplace=True, kind="mergesort")
    table.reset_index(drop=True, inplace=True)
    return table


def generate_downloads(study_id: str, thresholds: Dict[str, int]) -> DownloadResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(thresholds)

    mouse_averages_df = record.results.get_mouse_averages(thresholds)
    from .study_common import _ensure_ratio_columns

    _ensure_ratio_columns(mouse_averages_df, record.ratio_definitions)
    individual_images = pd.DataFrame(
        _collect_replicate_metrics(record.results.image_data, thresholds, record.ratio_definitions)
    )
    if not individual_images.empty and "ratios" in individual_images.columns:
        ratio_values = individual_images["ratios"].apply(pd.Series)
        individual_images = pd.concat([individual_images.drop(columns=["ratios"]), ratio_values], axis=1)

    download_dir = ensure_directory(DOWNLOAD_ROOT / study_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    excel_path = download_dir / f"{study_id}_thresholds_{timestamp}.xlsx"

    graphpad_df = _build_graphpad_dataframe(mouse_averages_df, record.ratio_definitions, record.channel_definitions)
    replicates_df = _format_replicates_dataframe(individual_images, record.ratio_definitions, record.channel_definitions)

    try:
        _write_excel_workbook(excel_path, graphpad_df, mouse_averages_df, replicates_df)
        final_path = excel_path
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.warning("Excel export failed (%s). Falling back to CSV bundle.", exc)
        final_path = _write_csv_bundle(
            download_dir, study_id, timestamp, graphpad_df, mouse_averages_df, replicates_df
        )

    return DownloadResponse(download_path=str(final_path), generated_at=datetime.utcnow())


def _write_excel_workbook(
    excel_path: Path, graphpad_df: pd.DataFrame, mouse_df: pd.DataFrame, replicates_df: pd.DataFrame
) -> None:
    excel_engine: Optional[str]
    try:  # pragma: no cover - optional dependency
        import openpyxl  # type: ignore  # noqa: F401

        excel_engine = "openpyxl"
    except Exception:  # pragma: no cover - optional dependency
        try:
            import xlsxwriter  # type: ignore  # noqa: F401

            excel_engine = "xlsxwriter"
        except Exception:
            excel_engine = None

    try:
        writer_factory = pd.ExcelWriter(excel_path, engine=excel_engine) if excel_engine else pd.ExcelWriter(excel_path)
    except ValueError as exc:  # pragma: no cover - optional dependency
        raise HTTPException(
            status_code=500,
            detail="Excel export requires openpyxl or xlsxwriter. Install one of these packages and restart the API.",
        ) from exc

    with writer_factory as writer:
        if not graphpad_df.empty:
            graphpad_df.to_excel(writer, sheet_name="GraphPad Data", index_label="Row")
        mouse_df.to_excel(writer, sheet_name="Mouse Averages", index=False)
        replicates_df.to_excel(writer, sheet_name="Replicates", index=False)
        if excel_engine == "openpyxl":
            replicates_sheet = writer.sheets.get("Replicates")
            if replicates_sheet is not None:
                try:
                    replicates_sheet.freeze_panes = replicates_sheet["E2"]
                except Exception:
                    pass


def _write_csv_bundle(
    base_dir: Path,
    study_id: str,
    timestamp: str,
    graphpad_df: pd.DataFrame,
    mouse_df: pd.DataFrame,
    replicates_df: pd.DataFrame,
) -> Path:
    zip_path = base_dir / f"{study_id}_thresholds_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if not graphpad_df.empty:
            archive.writestr("graphpad_data.csv", graphpad_df.to_csv(index=True))
        archive.writestr("mouse_averages.csv", mouse_df.to_csv(index=False))
        archive.writestr("replicates.csv", replicates_df.to_csv(index=False))
    return zip_path


def resolve_download_path(study_id: str, file_path: str) -> Path:
    _get_record(study_id)
    target = Path(file_path).expanduser().resolve()
    allowed_root = ensure_directory(DOWNLOAD_ROOT / study_id)
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Download path is outside of the study directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Download file not found")
    return target
