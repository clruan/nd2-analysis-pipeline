"""Study loading and analysis helpers."""

from __future__ import annotations

from collections import defaultdict, Counter
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Set

import numpy as np
import pandas as pd
from fastapi import HTTPException
import shutil
from PIL import Image

from threshold_analysis.batch_processor import load_threshold_results
from threshold_analysis.data_models import ThresholdData, ThresholdResults

from data_models import GroupConfig

from ..schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    IndividualImageRecord,
    LoadStudyRequest,
    MouseAverageRecord,
    PreviewImage,
    PreviewRequest,
    PreviewResponse,
    PreviewClearResponse,
    StatisticsRequest,
    StatisticsResponse,
    DownloadResponse,
)
from ..state import STATE, StudyRecord
from ..utils import ensure_directory, find_nd2_files, normalize_path, preview_plane_filename, slugify
from .ratios import DEFAULT_RATIO_DEFINITIONS, normalize_ratio_definitions
from image_processing import load_nd2_file


DOWNLOAD_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_downloads")
PREVIEW_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_previews")

CHANNEL_METRICS: Tuple[Dict[str, object], ...] = (
    {"id": "channel_1_area", "label": "Channel 1 Area (%)", "kind": "channel", "channel": 1},
    {"id": "channel_2_area", "label": "Channel 2 Area (%)", "kind": "channel", "channel": 2},
    {"id": "channel_3_area", "label": "Channel 3 Area (%)", "kind": "channel", "channel": 3},
)


def load_study(request: LoadStudyRequest) -> StudyRecord:
    path = normalize_path(request.file_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Threshold results not found: {path}")

    try:
        results = load_threshold_results(str(path))
    except KeyError as exc:
        missing_key = getattr(exc, "args", ["unknown field"])[0]
        # Try to help the user by suggesting nearby candidate files
        suggestions = []
        for candidate in sorted(path.parent.glob("threshold_results_*.json")):
            try:
                _ = load_threshold_results(str(candidate))
            except Exception:
                continue
            suggestions.append(str(candidate))

        message = (
            f"Threshold results file is missing required field '{missing_key}'. "
            "Try regenerating the study or ensure you selected the full JSON output from the pipeline."
        )
        if suggestions:
            message += f" Suggested valid files in the same folder: {', '.join(suggestions[:3])}"
        raise HTTPException(status_code=422, detail=message) from exc
    except Exception as exc:  # pragma: no cover - passthrough for unexpected formats
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load threshold results: {exc}",
        ) from exc

    if not results.image_data:
        raise HTTPException(
            status_code=422,
            detail="Threshold results contain no image data. Please rerun threshold generation for this study.",
        )
    study_id = slugify(results.study_name)
    run_lookup = next(
        (
            record
            for record in STATE.all_runs().values()
            if record.output_path and Path(record.output_path).resolve() == path
        ),
        None,
    )

    # Resolve ND2 root directory used to generate the study so we can build previews after restarts
    input_dir: Path
    is_3d: bool = True
    preview_plane_root: Optional[Path] = None
    ratio_definitions: Optional[List[Dict[str, object]]] = None

    meta: Dict[str, object] = {}
    meta_path = Path(str(path) + ".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    override_dir: Optional[Path] = None
    if request.input_dir_override:
        override_dir = normalize_path(request.input_dir_override).resolve()
        if not override_dir.exists():
            raise HTTPException(status_code=404, detail=f"ND2 input directory does not exist: {override_dir}")
        if not override_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"ND2 input directory is not a folder: {override_dir}")
        try:
            sample_nd2 = find_nd2_files(override_dir)
        except Exception as exc:  # pragma: no cover - filesystem edge cases
            raise HTTPException(status_code=500, detail=f"Unable to scan ND2 directory {override_dir}: {exc}") from exc
        if not sample_nd2:
            raise HTTPException(
                status_code=404,
                detail=f"No ND2 files found under {override_dir}. Select the folder that directly contains the study subdirectories.",
            )

    if run_lookup and run_lookup.ratio_definitions:
        ratio_definitions = run_lookup.ratio_definitions
    elif meta.get("ratio_definitions"):
        ratio_definitions = meta.get("ratio_definitions")

    if override_dir:
        input_dir = override_dir
    elif run_lookup and run_lookup.input_dir:
        input_dir = run_lookup.input_dir
        is_3d = run_lookup.is_3d
        preview_plane_root = run_lookup.preview_root
    elif meta.get("input_dir"):
        input_dir = Path(str(meta["input_dir"])).expanduser().resolve()
        preview_value = meta.get("preview_root")
        if preview_value:
            preview_candidate = Path(str(preview_value)).expanduser()
            if preview_candidate.exists():
                preview_plane_root = preview_candidate.resolve()
    else:
        # Last resort: search parents for any ND2 files (shallow)
        candidates = [path.parent]
        if path.parent.parent:
            candidates.append(path.parent.parent)
        found: Optional[Path] = None
        for cand in candidates:
            try:
                if find_nd2_files(cand):
                    found = cand
                    break
            except Exception:
                continue
        input_dir = found or path.parent
    if override_dir:
        preview_plane_root = None

    config_candidate: Optional[Path] = None
    if run_lookup and run_lookup.config_path:
        config_candidate = run_lookup.config_path
    elif meta.get("config_path"):
        config_candidate = Path(str(meta["config_path"])).expanduser()

    if ratio_definitions is None and config_candidate and config_candidate.exists():
        try:
            group_config = GroupConfig.from_json(str(config_candidate))
            ratio_definitions = group_config.ratios
        except Exception:
            ratio_definitions = None

    ratio_definitions = normalize_ratio_definitions(ratio_definitions)
    replicate_lookup = _build_replicate_lookup(results, input_dir)

    record = StudyRecord(
        study_id=study_id,
        results=results,
        source_path=path,
        input_dir=input_dir,
        replicate_lookup=replicate_lookup,
        is_3d=is_3d,
        preview_plane_root=preview_plane_root,
        ratio_definitions=ratio_definitions,
    )
    STATE.add_study(record)
    return record


def _build_replicate_lookup(results: ThresholdResults, input_dir: Path) -> Dict[str, Dict[str, Path]]:
    lookup: Dict[str, Dict[str, Path]] = {}
    try:
        nd2_files = find_nd2_files(input_dir)
    except Exception:
        nd2_files = []

    file_map: Dict[str, List[Path]] = defaultdict(list)
    for path in nd2_files:
        file_map[path.name].append(path)

    for entry in results.image_data:
        subject_map = lookup.setdefault(entry.mouse_id, {})
        candidates = file_map.get(entry.filename)
        if candidates:
            # Prefer files whose parent folder name contains the mouse_id; fall back to the first occurrence.
            chosen = next(
                (candidate for candidate in candidates if entry.mouse_id in candidate.parts[-2:]),
                candidates[0],
            )
            subject_map[entry.filename] = chosen
    return lookup


def _metric_definitions(record: StudyRecord) -> List[Dict[str, object]]:
    metrics = [dict(defn) for defn in CHANNEL_METRICS]
    for ratio in record.ratio_definitions:
        metrics.append(
            {
                "id": ratio["id"],
                "label": ratio["label"],
                "kind": "ratio",
                "numerator_channel": ratio["numerator_channel"],
                "denominator_channel": ratio["denominator_channel"],
            }
        )
    return metrics


def _ensure_ratio_columns(mouse_df: pd.DataFrame, ratios: List[Dict[str, object]]) -> None:
    for ratio in ratios:
        numerator_col = f"Channel_{ratio['numerator_channel']}_area"
        denominator_col = f"Channel_{ratio['denominator_channel']}_area"
        if numerator_col in mouse_df.columns and denominator_col in mouse_df.columns:
            mouse_df[ratio["id"]] = mouse_df[numerator_col] / (mouse_df[denominator_col] + 1e-3)
        else:
            mouse_df[ratio["id"]] = np.nan


def _threshold_dict(thresholds: Dict[str, int]) -> Dict[str, int]:
    defaulted = {
        "channel_1": thresholds.get("channel_1", 0),
        "channel_2": thresholds.get("channel_2", 0),
        "channel_3": thresholds.get("channel_3", 0),
    }
    return defaulted


def _get_record(study_id: str) -> StudyRecord:
    try:
        return STATE.get_study(study_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=f"Study not loaded: {study_id}") from exc


def analyze_study(study_id: str, request: AnalyzeRequest) -> AnalyzeResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)

    mouse_averages_df = record.results.get_mouse_averages(thresholds)
    _ensure_ratio_columns(mouse_averages_df, record.ratio_definitions)
    mouse_records: List[MouseAverageRecord] = []
    for row in mouse_averages_df.to_dict(orient="records"):
        ch1 = float(row.get("Channel_1_area", 0.0))
        ch2 = float(row.get("Channel_2_area", 0.0))
        ch3 = float(row.get("Channel_3_area", 0.0))
        ratio_values: Dict[str, float] = {}
        for ratio in record.ratio_definitions:
            value = float(row.get(ratio["id"], 0.0))
            ratio_values[ratio["id"]] = value
        mouse_records.append(
            MouseAverageRecord(
                Group=str(row.get("Group", "")),
                MouseID=str(row.get("MouseID", "")),
                Channel_1_area=ch1,
                Channel_2_area=ch2,
                Channel_3_area=ch3,
                ratios=ratio_values,
            )
        )

    individual_images = _collect_replicate_metrics(record.results.image_data, thresholds, record.ratio_definitions)

    return AnalyzeResponse(
        study_id=study_id,
        thresholds=thresholds,
        mouse_averages=mouse_records,
        individual_images=individual_images,
    )


def _collect_replicate_metrics(
    image_data: Iterable[ThresholdData],
    thresholds: Dict[str, int],
    ratio_definitions: List[Dict[str, object]],
) -> List[IndividualImageRecord]:
    records: List[IndividualImageRecord] = []
    replicate_counters: Counter[Tuple[str, str]] = Counter()

    for entry in image_data:
        key = (entry.group, entry.mouse_id)
        replicate_counters[key] += 1
        replicate_index = replicate_counters[key]

        ch1 = entry.get_percentage_at_threshold(1, thresholds["channel_1"])
        ch2 = entry.get_percentage_at_threshold(2, thresholds["channel_2"])
        ch3 = entry.get_percentage_at_threshold(3, thresholds["channel_3"])
        ratio_values: Dict[str, float] = {}
        for ratio in ratio_definitions:
            num_idx = int(ratio["numerator_channel"])
            den_idx = int(ratio["denominator_channel"])
            numerator = entry.get_percentage_at_threshold(num_idx, thresholds[f"channel_{num_idx}"])
            denominator = entry.get_percentage_at_threshold(den_idx, thresholds[f"channel_{den_idx}"])
            ratio_values[ratio["id"]] = float(numerator / (denominator + 1e-3))
        records.append(
            IndividualImageRecord(
                group=str(entry.group),
                mouse_id=str(entry.mouse_id),
                filename=str(entry.filename),
                channel_1_area=float(ch1),
                channel_2_area=float(ch2),
                channel_3_area=float(ch3),
                ratios=ratio_values,
                replicate_index=replicate_index,
            )
        )
    return records


def perform_statistics(study_id: str, request: StatisticsRequest) -> StatisticsResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)

    mouse_averages_df = record.results.get_mouse_averages(thresholds)
    _ensure_ratio_columns(mouse_averages_df, record.ratio_definitions)

    channels = [
        ("channel_1", "Channel_1_area"),
        ("channel_2", "Channel_2_area"),
        ("channel_3", "Channel_3_area"),
    ]

    statistics = {}
    for channel_key, column in channels:
        groups_data = {
            group: group_df[column].dropna().tolist()
            for group, group_df in mouse_averages_df.groupby("Group")
        }

        statistics[channel_key] = _analyze_groups(
            groups_data,
            request.comparison_mode,
            request.reference_group,
            request.comparison_pairs,
            request.test_type,
            request.significance_display,
        )

    for ratio in record.ratio_definitions:
        column = ratio["id"]
        groups_data = {
            group: group_df[column].dropna().tolist()
            for group, group_df in mouse_averages_df.groupby("Group")
        }
        statistics[column] = _analyze_groups(
            groups_data,
            request.comparison_mode,
            request.reference_group,
            request.comparison_pairs,
            request.test_type,
            request.significance_display,
        )

    return StatisticsResponse(
        statistics=statistics,
        thresholds=thresholds,
        test_type_used=request.test_type,
        significance_display=request.significance_display,
        ratios=record.ratio_definitions,
    )


def _analyze_groups(
    groups_data: Dict[str, List[float]],
    comparison_mode: str,
    reference_group: Optional[str],
    comparison_pairs: Optional[List[List[str]]],
    test_type: str,
    significance_display: str,
) -> Dict[str, object]:
    cleaned: Dict[str, List[float]] = {
        group: [value for value in values if np.isfinite(value)]
        for group, values in groups_data.items()
    }
    cleaned = {group: values for group, values in cleaned.items() if values}

    if len(cleaned) < 2:
        return {
            "comparison_mode": comparison_mode,
            "pairwise_comparisons": [],
            "note": "Not enough samples per group to compute statistics.",
        }

    if comparison_mode == "pairs":
        if not comparison_pairs:
            raise HTTPException(status_code=400, detail="Comparison pairs required for pairs mode")
        comparisons = []
        for pair in comparison_pairs:
            if len(pair) != 2:
                continue
            g1, g2 = pair
            if g1 not in cleaned or g2 not in cleaned:
                continue
            statistic, p_value = _perform_statistical_test(cleaned[g1], cleaned[g2], test_type)
            comparisons.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "statistic": statistic,
                    "p_value": p_value,
                    "significance": _format_significance(p_value, significance_display),
                }
            )
        return {
            "comparison_mode": comparison_mode,
            "pairwise_comparisons": comparisons,
            "note": "Comparisons skipped for groups without samples." if not comparisons else None,
        }

    if comparison_mode == "all_pairs":
        group_names = sorted(cleaned.keys())
        comparisons = []
        for idx, group_a in enumerate(group_names):
            for group_b in group_names[idx + 1 :]:
                statistic, p_value = _perform_statistical_test(cleaned[group_a], cleaned[group_b], test_type)
                comparisons.append(
                    {
                        "group1": group_a,
                        "group2": group_b,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significance": _format_significance(p_value, significance_display),
                    }
                )

        overall_stat, overall_p = _perform_anova(cleaned, test_type)

        overall_block = None
        if test_type != "t_test":
            overall_block = {
                "statistic": overall_stat,
                "p_value": overall_p,
                "significance": _format_significance(overall_p, significance_display),
            }

        return {
            "comparison_mode": comparison_mode,
            "overall_test": overall_block,
            "pairwise_comparisons": comparisons,
            "note": "Comparisons skipped for groups without samples." if not comparisons else None,
        }

    reference = reference_group or next(iter(cleaned))
    if reference not in cleaned:
        reference = next(iter(cleaned))

    comparisons = []
    for group_name, data in cleaned.items():
        if group_name == reference:
            continue
        statistic, p_value = _perform_statistical_test(cleaned[reference], data, test_type)
        comparisons.append(
            {
                "group1": reference,
                "group2": group_name,
                "statistic": statistic,
                "p_value": p_value,
                "significance": _format_significance(p_value, significance_display),
            }
        )

    overall_stat, overall_p = _perform_anova(cleaned, test_type)

    overall_block = None
    if test_type != "t_test":
        overall_block = {
            "statistic": overall_stat,
            "p_value": overall_p,
            "significance": _format_significance(overall_p, significance_display),
        }

    return {
        "comparison_mode": comparison_mode,
        "reference_group": reference,
        "overall_test": overall_block,
        "pairwise_comparisons": comparisons,
    }


def _perform_statistical_test(group1: List[float], group2: List[float], test_type: str) -> Tuple[float, float]:
    from scipy import stats

    group1 = [value for value in group1 if np.isfinite(value)]
    group2 = [value for value in group2 if np.isfinite(value)]

    if len(group1) < 2 or len(group2) < 2:
        return 0.0, 1.0

    if not group1 or not group2:
        return 0.0, 1.0

    if test_type in {"anova_parametric", "t_test"}:
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    elif test_type == "anova_non_parametric":
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    else:
        use_parametric = _is_normal(group1) and _is_normal(group2)
        if use_parametric:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        else:
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    return float(statistic), float(p_value)


def _perform_anova(groups_data: Dict[str, List[float]], test_type: str) -> Tuple[float, float]:
    from scipy import stats

    clean_groups = [
        [value for value in values if np.isfinite(value)]
        for values in groups_data.values()
        if values
    ]
    clean_groups = [group for group in clean_groups if len(group) >= 2]
    if len(clean_groups) < 2:
        return 0.0, 1.0

    if test_type == "t_test":
        return 0.0, 1.0
    if test_type == "anova_parametric":
        statistic, p_value = stats.f_oneway(*clean_groups)
    elif test_type == "anova_non_parametric":
        statistic, p_value = stats.kruskal(*clean_groups)
    else:
        all_normal = all(_is_normal(group) for group in clean_groups)
        if all_normal:
            statistic, p_value = stats.f_oneway(*clean_groups)
        else:
            statistic, p_value = stats.kruskal(*clean_groups)

    return float(statistic), float(p_value)


def _is_normal(data: Iterable[float]) -> bool:
    from scipy import stats

    data = [value for value in data if np.isfinite(value)]
    if len(data) < 3:
        return True
    _, p_value = stats.shapiro(data)
    return p_value > 0.05


def _format_significance(p_value: float, mode: str) -> str:
    if mode == "p_values":
        return f"p={p_value:.4f}"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


ALL_PREVIEW_METRICS: Tuple[str, ...] = (
    "channel_1_area",
    "channel_2_area",
    "channel_3_area",
    "channel_1_3_ratio",
    "channel_2_3_ratio",
)

DEFAULT_PREVIEW_METRIC = ALL_PREVIEW_METRICS[0]

GRAPH_PAD_EXPORT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("Channel_1_area", "Channel 1"),
    ("Channel_2_area", "Channel 2"),
    ("Channel_3_area", "Channel 3"),
    ("Channel_1_3_ratio", "Channel 1 / Channel 3"),
    ("Channel_2_3_ratio", "Channel 2 / Channel 3"),
)


class PreviewVariant(NamedTuple):
    variant: str
    channels: Tuple[int, ...]
    cacheable: bool


def _normalize_metric(metric: Optional[str]) -> str:
    if metric in ALL_PREVIEW_METRICS:
        return metric
    return DEFAULT_PREVIEW_METRIC


def _normalize_metrics(
    metrics: Optional[Iterable[str]],
    fallback: Optional[str] = None,
    available: Optional[Iterable[str]] = None,
) -> List[str]:
    allowed = list(available) if available else list(ALL_PREVIEW_METRICS)
    if metrics:
        ordered: List[str] = []
        for metric in metrics:
            if metric in allowed and metric not in ordered:
                ordered.append(metric)
        if ordered:
            return ordered
    if fallback:
        normalized = _normalize_metric(fallback)
        return [normalized]
    return allowed


def _preview_variants_for_metric(metric: Dict[str, object]) -> List[PreviewVariant]:
    if metric.get("kind") == "channel":
        channel = int(metric.get("channel", 1))
        return [
            PreviewVariant("raw", (channel,), True),
            PreviewVariant("mask", (channel,), False),
            PreviewVariant("overlay", (channel,), False),
        ]

    numerator = int(metric.get("numerator_channel", 1))
    denominator = int(metric.get("denominator_channel", 3))
    return [
        PreviewVariant("raw", (numerator,), True),
        PreviewVariant("raw", (denominator,), True),
        PreviewVariant("mask", (numerator,), False),
        PreviewVariant("mask", (denominator,), False),
        PreviewVariant("overlay", (numerator, denominator), False),
    ]


def _build_variant_cache_key(
    cache_base: str, variant: PreviewVariant, threshold_key: str, metric: str
) -> str:
    channel_tag = "-".join(str(ch) for ch in variant.channels) or "all"
    if variant.cacheable:
        return f"{cache_base}|{variant.variant}|{channel_tag}"
    return f"{cache_base}|{threshold_key}|{metric}|{variant.variant}|{channel_tag}"


def _build_variant_filename(
    safe_base: str, variant: PreviewVariant, metric_slug: str, cacheable: bool
) -> str:
    channel_tag = "-".join(f"ch{ch}" for ch in variant.channels) or "all"
    parts = [safe_base, variant.variant, channel_tag]
    if not cacheable:
        parts.append(metric_slug)
    return "_".join(parts) + ".png"


def _is_valid_preview_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as candidate:
            return max(candidate.size) <= 320
    except Exception:
        return False


def _metadata_path_for(image_path: Path) -> Path:
    return Path(str(image_path) + ".meta.json")


def _write_preview_metadata(image_path: Path, metric_id: str, variant: PreviewVariant) -> None:
    metadata = {
        "variant": variant.variant,
        "channels": list(variant.channels),
        "cache_scope": "global" if variant.cacheable else metric_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = _metadata_path_for(image_path)
    try:
        meta_path.write_text(json.dumps(metadata))
    except Exception:
        # Metadata is advisory; ignore failures but ensure stale files are not reused later.
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)


def _metadata_matches(image_path: Path, metric_id: str, variant: PreviewVariant) -> bool:
    meta_path = _metadata_path_for(image_path)
    if not meta_path.exists():
        return False
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return False
    expected_scope = "global" if variant.cacheable else metric_id
    if data.get("variant") != variant.variant:
        return False
    if data.get("channels") != list(variant.channels):
        return False
    return data.get("cache_scope") == expected_scope


def _cached_variant_valid(path: Path, metric_id: str, variant: PreviewVariant) -> bool:
    path = Path(path)
    if not _is_valid_preview_file(path):
        return False
    return _metadata_matches(path, metric_id, variant)


def _prune_preview_thresholds(preview_dir: Path, retain_keys: Set[str], max_sets: int = 5, max_age_days: int = 7) -> None:
    try:
        candidates = [
            child
            for child in preview_dir.iterdir()
            if child.is_dir() and child.name not in {"raw", "planes"} and child.name not in retain_keys
        ]
    except FileNotFoundError:
        return

    now = datetime.utcnow()
    max_age = timedelta(days=max_age_days)
    scored: List[Tuple[datetime, Path]] = []
    for child in candidates:
        try:
            mtime = datetime.fromtimestamp(child.stat().st_mtime)
        except OSError:
            continue
        scored.append((mtime, child))

    scored.sort(key=lambda item: item[0], reverse=True)
    slots = max(max_sets - len(retain_keys), 0)
    for index, (mtime, path) in enumerate(scored):
        if index >= slots or (now - mtime) > max_age:
            shutil.rmtree(path, ignore_errors=True)


def clear_preview_cache(study_id: str, scope: str = "thresholds", threshold_key: Optional[str] = None) -> PreviewClearResponse:
    record = _get_record(study_id)
    preview_dir = ensure_directory(PREVIEW_ROOT / study_id)
    allowed_scopes = {"thresholds", "all"}
    if scope not in allowed_scopes:
        raise HTTPException(status_code=400, detail=f"Invalid scope '{scope}'. Expected one of {sorted(allowed_scopes)}.")

    def removable_children(filter_raw: bool) -> List[Path]:
        targets: List[Path] = []
        try:
            for child in preview_dir.iterdir():
                if not child.is_dir():
                    continue
                if not filter_raw or child.name not in {"raw", "planes"}:
                    targets.append(child)
        except FileNotFoundError:
            return []
        return targets

    removed: List[str] = []

    if scope == "thresholds":
        targets: List[Path]
        if threshold_key:
            candidate = preview_dir / threshold_key
            targets = [candidate] if candidate.exists() and candidate.is_dir() else []
        else:
            targets = removable_children(filter_raw=True)
        for path in targets:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    else:  # scope == "all"
        targets = removable_children(filter_raw=False)
        for path in targets:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        record.preview_cache.clear()
        record.raw_cache.clear()
        record.preview_plane_root = None

    return PreviewClearResponse(removed_directories=removed)


def generate_previews(study_id: str, request: PreviewRequest) -> PreviewResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)
    metric_defs = _metric_definitions(record)
    metric_map = {metric["id"]: metric for metric in metric_defs}

    preview_dir = ensure_directory(PREVIEW_ROOT / study_id)
    raw_dir = ensure_directory(preview_dir / "raw")
    if record.preview_plane_root is None:
        record.preview_plane_root = preview_dir
    else:
        try:
            ensure_directory(record.preview_plane_root / "planes")
        except Exception:
            record.preview_plane_root = preview_dir

    threshold_key = f"{thresholds['channel_1']}-{thresholds['channel_2']}-{thresholds['channel_3']}"
    threshold_dir = ensure_directory(preview_dir / threshold_key)
    _prune_preview_thresholds(preview_dir, retain_keys={threshold_key})
    available_metric_ids = [metric["id"] for metric in metric_defs]
    metric_candidates = _normalize_metrics(request.metrics, request.metric, available_metric_ids)
    metric_ids = [metric for metric in metric_candidates if metric in metric_map] or available_metric_ids
    metric_slugs = {metric_id: slugify(metric_id) for metric_id in metric_ids}
    metric_dirs = {metric_id: ensure_directory(threshold_dir / metric_slugs[metric_id]) for metric_id in metric_ids}
    variant_plans = {metric_id: _preview_variants_for_metric(metric_map[metric_id]) for metric_id in metric_ids}

    groups_requested = set(request.groups) if request.groups else None
    group_counts: Dict[str, int] = defaultdict(int)
    per_group_limits: Dict[str, int] = {
        group: max(1, min(6, int(value)))
        for group, value in (request.group_sample_limits or {}).items()
        if isinstance(value, int)
    }
    max_samples = max(1, min(6, request.max_samples_per_group))
    nd2_available = record.input_dir.exists()
    preview_images: List[PreviewImage] = []

    for entry in record.results.image_data:
        if groups_requested and entry.group not in groups_requested:
            continue
        limit = per_group_limits.get(entry.group, max_samples)
        if group_counts[entry.group] >= limit:
            continue

        cache_base = f"{entry.group}|{entry.mouse_id}|{entry.filename}"
        safe_base = f"{slugify(entry.group)}_{slugify(entry.mouse_id)}_{slugify(entry.filename)}"
        channel_arrays: Optional[Dict[int, np.ndarray]] = None

        for metric_id in metric_ids:
            variant_plan = variant_plans[metric_id]
            metric_dir = metric_dirs[metric_id]
            metric_slug = metric_slugs[metric_id]
            existing_files: Dict[str, Path] = {}

            for variant in variant_plan:
                cache_key = _build_variant_cache_key(cache_base, variant, threshold_key, metric_id)
                output_dir = raw_dir if variant.cacheable else metric_dir
                safe_name = _build_variant_filename(safe_base, variant, metric_slug, variant.cacheable)
                image_path = output_dir / safe_name

                cached_path = record.preview_cache.get(cache_key) if variant.cacheable else None
                if cached_path and _cached_variant_valid(cached_path, metric_id, variant):
                    existing_files[cache_key] = cached_path
                    continue

                if _cached_variant_valid(image_path, metric_id, variant):
                    existing_files[cache_key] = image_path
                    if variant.cacheable:
                        record.preview_cache[cache_key] = image_path
                    continue

                if channel_arrays is None:
                    channel_arrays = _load_channels(record, entry.group, entry.mouse_id, entry.filename)
                if not channel_arrays:
                    continue

                image_array = _render_preview_variant(channel_arrays, thresholds, variant)
                if image_array is None:
                    continue

                _write_image(image_array, image_path)
                _write_preview_metadata(image_path, metric_id, variant)
                if variant.cacheable:
                    record.preview_cache[cache_key] = image_path
                existing_files[cache_key] = image_path

            if not existing_files:
                continue

            for variant in variant_plan:
                cache_key = _build_variant_cache_key(cache_base, variant, threshold_key, metric_id)
                image_path = existing_files.get(cache_key)
                if not image_path:
                    continue
                channel = variant.channels[0] if len(variant.channels) == 1 else None
                preview_images.append(
                    PreviewImage(
                        metric=metric_id,
                        variant=variant.variant,
                        channel=channel,
                        group=entry.group,
                        subject_id=entry.mouse_id,
                        filename=entry.filename,
                        image_path=str(image_path),
                    )
                )

        group_counts[entry.group] += 1

    metric_rank = {metric["id"]: index for index, metric in enumerate(metric_defs)}
    preview_images.sort(
        key=lambda item: (
            metric_rank.get(item.metric, len(metric_rank)),
            item.group,
            item.subject_id,
            item.variant,
            item.channel or 0,
        )
    )

    group_sample_counts = dict(group_counts)

    return PreviewResponse(
        study_id=study_id,
        generated_at=datetime.utcnow(),
        images=preview_images,
        nd2_available=nd2_available,
        nd2_source=str(record.input_dir),
        max_samples_per_group=max_samples,
        ratio_definitions=record.ratio_definitions,
        group_sample_counts=group_sample_counts,
    )


def list_ratio_definitions(study_id: str) -> List[Dict[str, object]]:
    record = _get_record(study_id)
    return record.ratio_definitions


def update_ratio_definitions(study_id: str, ratios: List[Dict[str, object]]) -> List[Dict[str, object]]:
    record = _get_record(study_id)
    normalized = normalize_ratio_definitions(ratios)
    record.ratio_definitions = normalized
    record.preview_cache.clear()
    _persist_ratio_metadata(record)
    return normalized


def _persist_ratio_metadata(record: StudyRecord) -> None:
    meta_path = Path(str(record.source_path) + ".meta.json")
    try:
        if meta_path.exists():
            payload = json.loads(meta_path.read_text())
        else:
            payload = {}
        payload["ratio_definitions"] = record.ratio_definitions
        if record.input_dir:
            payload.setdefault("input_dir", str(record.input_dir))
        if record.preview_plane_root:
            payload.setdefault("preview_root", str(record.preview_plane_root))
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


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


def _build_graphpad_dataframe(mouse_df: pd.DataFrame, ratio_defs: List[Dict[str, object]]) -> pd.DataFrame:
    if mouse_df.empty:
        return pd.DataFrame()
    group_order = _graphpad_group_order(mouse_df)
    if not group_order:
        return pd.DataFrame()
    blocks: List[pd.DataFrame] = []
    counts: Dict[str, int] = {}
    for metric_key, label in GRAPH_PAD_EXPORT_ORDER:
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


def _format_replicates_dataframe(individual_df: pd.DataFrame, ratios: List[Dict[str, object]]) -> pd.DataFrame:
    if individual_df.empty:
        return individual_df

    table = individual_df.copy()
    rename_map = {
        "group": "Group",
        "mouse_id": "Mouse ID",
        "replicate_index": "Replicate #",
        "filename": "Filename",
        "channel_1_area": "Channel 1 Area (%)",
        "channel_2_area": "Channel 2 Area (%)",
        "channel_3_area": "Channel 3 Area (%)",
        "channel_1_3_ratio": "Channel 1 / Channel 3",
        "channel_2_3_ratio": "Channel 2 / Channel 3",
    }
    table.rename(columns=rename_map, inplace=True)
    ordered_columns = [
        "Group",
        "Mouse ID",
        "Replicate #",
        "Filename",
        "Channel 1 Area (%)",
        "Channel 2 Area (%)",
        "Channel 3 Area (%)",
    ]
    ratio_labels: List[str] = []
    for ratio in ratios:
        ratio_id = ratio["id"]
        label = ratio["label"]
        rename_map[ratio_id] = label
        ratio_labels.append(label)
    ordered_columns.extend(ratio_labels)
    existing_columns = [column for column in ordered_columns if column in table.columns]
    table = table[existing_columns]
    metric_columns = [
        column
        for column in existing_columns
        if column.startswith("Channel")
    ]
    for column in metric_columns:
        table[column] = table[column].astype(float).round(4)
    table.sort_values(["Group", "Mouse ID", "Replicate #"], inplace=True, kind="mergesort")
    table.reset_index(drop=True, inplace=True)
    return table


def generate_downloads(study_id: str, thresholds: Dict[str, int]) -> DownloadResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(thresholds)

    mouse_averages_df = record.results.get_mouse_averages(thresholds)
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

    graphpad_df = _build_graphpad_dataframe(mouse_averages_df, record.ratio_definitions)
    replicates_df = _format_replicates_dataframe(individual_images, record.ratio_definitions)

    # Prefer openpyxl for better formatting, but fall back gracefully if it is unavailable.
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

    writer_factory = pd.ExcelWriter(excel_path, engine=excel_engine) if excel_engine else pd.ExcelWriter(excel_path)

    with writer_factory as writer:
        if not graphpad_df.empty:
            graphpad_df.to_excel(writer, sheet_name="GraphPad Data", index_label="Row")
        mouse_averages_df.to_excel(writer, sheet_name="Mouse Averages", index=False)
        replicates_df.to_excel(writer, sheet_name="Replicates", index=False)
        if excel_engine == "openpyxl":
            replicates_sheet = writer.sheets.get("Replicates")
            if replicates_sheet is not None:
                try:
                    replicates_sheet.freeze_panes = replicates_sheet["E2"]
                except Exception:
                    pass

    return DownloadResponse(download_path=str(excel_path), generated_at=datetime.utcnow())


def resolve_preview_path(study_id: str, file_path: str) -> Path:
    _get_record(study_id)
    target = Path(file_path).expanduser().resolve()
    allowed_root = ensure_directory(PREVIEW_ROOT / study_id)
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Preview path is outside of the study directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Preview image not found")
    return target


def resolve_download_path(study_id: str, file_path: str) -> Path:
    _get_record(study_id)
    target = Path(file_path).expanduser().resolve()
    allowed_root = ensure_directory(DOWNLOAD_ROOT / study_id)
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Download path is outside of the study directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Download file not found")
    return target


def _load_channels(record: StudyRecord, group: str, subject_id: str, filename: str) -> Optional[Dict[int, np.ndarray]]:
    cache_key = f"{group}|{subject_id}|{filename}"
    if cache_key in record.raw_cache:
        return record.raw_cache[cache_key]

    channel_arrays: Dict[int, np.ndarray] = {}

    if record.preview_plane_root:
        plane_dir = ensure_directory(record.preview_plane_root / "planes")
        for channel_index in (1, 2, 3):
            plane_path = plane_dir / preview_plane_filename(group, subject_id, filename, channel_index)
            if not plane_path.exists():
                channel_arrays.clear()
                break
            try:
                array = np.load(plane_path, allow_pickle=False)
                channel_arrays[channel_index] = array.astype(np.uint16, copy=False)
            except Exception:
                channel_arrays.clear()
                break

    if not channel_arrays:
        subject_path: Optional[Path] = None
        for subject_map in record.replicate_lookup.values():
            if filename in subject_map:
                subject_path = subject_map[filename]
                break
        if subject_path and subject_path.exists():
            try:
                ch1, ch2, ch3 = load_nd2_file(str(subject_path), is_3d=record.is_3d)
                channel_arrays = {1: ch1, 2: ch2, 3: ch3}
                _persist_preview_planes(record, group, subject_id, filename, channel_arrays)
            except Exception:
                channel_arrays = {}

    if not channel_arrays:
        channel_arrays = _load_channels_from_cached_png(record, group, subject_id, filename)
        if channel_arrays:
            _persist_preview_planes(record, group, subject_id, filename, channel_arrays)

    if not channel_arrays:
        return None

    record.raw_cache[cache_key] = channel_arrays
    return channel_arrays


def _persist_preview_planes(
    record: StudyRecord, group: str, subject_id: str, filename: str, channel_arrays: Dict[int, np.ndarray]
) -> None:
    if not record.preview_plane_root:
        return
    try:
        plane_dir = ensure_directory(record.preview_plane_root / "planes")
    except Exception:
        return

    for channel_index, array in channel_arrays.items():
        plane_path = plane_dir / preview_plane_filename(group, subject_id, filename, channel_index)
        if plane_path.exists():
            continue
        try:
            np.save(plane_path, array.astype(np.uint16, copy=False), allow_pickle=False)
        except Exception:
            continue


def _load_channels_from_cached_png(
    record: StudyRecord, group: str, subject_id: str, filename: str
) -> Optional[Dict[int, np.ndarray]]:
    preview_dir = PREVIEW_ROOT / record.study_id
    raw_dir = preview_dir / "raw"
    if not raw_dir.exists():
        return None

    safe_base = f"{slugify(group)}_{slugify(subject_id)}_{slugify(filename)}"
    channel_arrays: Dict[int, np.ndarray] = {}

    for channel_index in (1, 2, 3):
        image_name = f"{safe_base}_raw_ch{channel_index}.png"
        image_path = raw_dir / image_name
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                grayscale = image.convert("L")
                array = np.asarray(grayscale, dtype=np.float32)
                # Raw previews are scaled to 0-255; re-expand to 16-bit space so thresholds still apply.
                rescaled = (array / 255.0) * 4095.0
                channel_arrays[channel_index] = rescaled.astype(np.uint16)
        except Exception:
            continue

    return channel_arrays or None


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    channel = np.clip(channel, 0, None)
    channel -= channel.min()
    max_val = channel.max()
    if max_val <= 0:
        return np.zeros_like(channel, dtype=np.uint8)
    channel /= max_val
    channel *= 255.0
    return channel.astype(np.uint8)


def _generate_raw_image(channels: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    red_source = channels.get(2) or channels.get(1)
    green_source = channels.get(1) or channels.get(2)
    blue_source = channels.get(3) or channels.get(1)
    if red_source is None or green_source is None or blue_source is None:
        return None
    red = _normalize_channel(red_source)
    green = _normalize_channel(green_source)
    blue = _normalize_channel(blue_source)
    return np.stack([red, green, blue], axis=-1)


def _build_binary_mask(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], channel_id: int
) -> Optional[np.ndarray]:
    channel = channels.get(channel_id)
    threshold = thresholds.get(f"channel_{channel_id}")
    if channel is None or threshold is None:
        return None
    return (channel > threshold).astype(np.uint8)


def _generate_channel_raw_image(
    channels: Dict[int, np.ndarray], channel_id: int
) -> Optional[np.ndarray]:
    channel = channels.get(channel_id)
    if channel is None:
        return None
    normalized = _normalize_channel(channel)
    zeros = np.zeros_like(normalized)
    if channel_id == 1:
        return np.stack([zeros, normalized, zeros], axis=-1)
    if channel_id == 2:
        return np.stack([normalized, zeros, zeros], axis=-1)
    if channel_id == 3:
        return np.stack([zeros, zeros, normalized], axis=-1)
    return np.stack([normalized, normalized, normalized], axis=-1)


def _generate_channel_mask_image(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], channel_id: int
) -> Optional[np.ndarray]:
    binary = _build_binary_mask(channels, thresholds, channel_id)
    if binary is None:
        return None
    mask = (binary * 255).astype(np.uint8)
    return np.stack([mask, mask, mask], axis=-1)


def _generate_mask_image(channels: Dict[int, np.ndarray], thresholds: Dict[str, int]) -> Optional[np.ndarray]:
    masks = [
        _build_binary_mask(channels, thresholds, channel_id)
        for channel_id in (1, 2, 3)
    ]
    masks = [mask for mask in masks if mask is not None]
    if not masks:
        sample = next(iter(channels.values()), None)
        if sample is None:
            return None
        shape = sample.shape
        return np.zeros((*shape, 3), dtype=np.uint8)
    combined = np.maximum.reduce(masks)
    mask = (combined * 255).astype(np.uint8)
    return np.stack([mask, mask, mask], axis=-1)


def _apply_highlight(raw: np.ndarray, mask: np.ndarray, strength: float = 0.45) -> np.ndarray:
    base = raw.astype(np.float32)
    alpha = (mask[..., :1].astype(np.float32) / 255.0) * strength
    highlighted = np.clip(base * (1.0 - alpha) + 255.0 * alpha, 0, 255)
    result = np.where(mask[..., :1] > 0, highlighted, base)
    return result.astype(np.uint8)


def _generate_channel_overlay_image(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], channel_id: int
) -> Optional[np.ndarray]:
    raw = _generate_channel_raw_image(channels, channel_id)
    if raw is None:
        return None
    mask = _generate_channel_mask_image(channels, thresholds, channel_id)
    if mask is None:
        return raw
    return _apply_highlight(raw, mask)


def _generate_ratio_overlay_image(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], channel_pair: Tuple[int, ...]
) -> Optional[np.ndarray]:
    if len(channel_pair) != 2:
        return None
    raw_images: List[np.ndarray] = []
    for channel_id in channel_pair:
        raw_image = _generate_channel_raw_image(channels, channel_id)
        if raw_image is None:
            return None
        raw_images.append(raw_image.astype(np.float32))
    stack = np.stack(raw_images, axis=0)
    base = np.max(stack, axis=0).astype(np.uint8)
    overlay = base
    for channel_id in channel_pair:
        mask = _generate_channel_mask_image(channels, thresholds, channel_id)
        if mask is not None:
            overlay = _apply_highlight(overlay, mask)
    return overlay


def _generate_overlay_image(channels: Dict[int, np.ndarray], thresholds: Dict[str, int]) -> Optional[np.ndarray]:
    raw = _generate_raw_image(channels)
    if raw is None:
        return None
    mask = _generate_mask_image(channels, thresholds)
    if mask is None:
        return raw
    return _apply_highlight(raw, mask, strength=0.35)


def _render_preview_variant(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], variant: PreviewVariant
) -> Optional[np.ndarray]:
    if variant.variant == "raw":
        if len(variant.channels) == 1:
            return _generate_channel_raw_image(channels, variant.channels[0])
        return _generate_raw_image(channels)
    if variant.variant == "mask":
        if len(variant.channels) == 1:
            return _generate_channel_mask_image(channels, thresholds, variant.channels[0])
        return _generate_mask_image(channels, thresholds)
    if variant.variant == "overlay":
        if len(variant.channels) == 1:
            return _generate_channel_overlay_image(channels, thresholds, variant.channels[0])
        if len(variant.channels) == 2:
            return _generate_ratio_overlay_image(channels, thresholds, variant.channels)
        return _generate_overlay_image(channels, thresholds)
    return None


def _write_image(image_array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(image_array)
    max_dim = max(image.size)
    if max_dim > 320:
        scale = 320 / max_dim
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    image.save(path)
