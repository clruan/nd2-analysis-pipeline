"""Statistics helpers for loaded studies."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

from ..schemas import StatisticsRequest, StatisticsResponse
from .analysis_service import _analysis_tables_for_thresholds
from .study_common import _analysis_cache_key, _get_record, _record_channel_ids, _threshold_dict


def _statistics_cache_key(thresholds: Dict[str, int], request: StatisticsRequest) -> str:
    pairs = request.comparison_pairs or []
    pairs_token = "|".join(
        "::".join(sorted((str(pair[0]), str(pair[1]))))
        for pair in sorted((pair for pair in pairs if len(pair) == 2), key=lambda pair: tuple(sorted(pair)))
    ) or "none"
    reference_group = request.reference_group or "none"
    return (
        f"{_analysis_cache_key(thresholds)}|{request.comparison_mode}|{reference_group}|"
        f"{pairs_token}|{request.test_type}|{request.significance_display}"
    )


def perform_statistics(study_id: str, request: StatisticsRequest) -> StatisticsResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds, channel_ids=_record_channel_ids(record))
    cache_key = _statistics_cache_key(thresholds, request)
    cached = record.statistics_cache.get(cache_key)
    if isinstance(cached, StatisticsResponse):
        return cached
    uses_ratio_metrics = True
    mouse_averages_df, _ = _analysis_tables_for_thresholds(record, thresholds)

    statistics = {}
    for channel in _record_channel_ids(record):
        channel_key = f"channel_{channel}_area"
        column = f"Channel_{channel}_area"
        groups_data = {
            group: group_df[column].dropna().tolist()
            for group, group_df in mouse_averages_df.groupby("Group")
            if column in group_df
        }

        statistics[channel_key] = _analyze_groups(
            groups_data,
            request.comparison_mode,
            request.reference_group,
            request.comparison_pairs,
            request.test_type,
            request.significance_display,
        )

    if uses_ratio_metrics:
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

    response = StatisticsResponse(
        statistics=statistics,
        thresholds=thresholds,
        test_type_used=request.test_type,
        significance_display=request.significance_display,
        ratios=record.ratio_definitions,
    )
    record.statistics_cache[cache_key] = response
    return response


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
