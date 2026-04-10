"""Preview rendering primitives and cache-token helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

from .channels import (
    DEFAULT_CHANNEL_DEFINITIONS,
    channel_color_rgb,
    channel_definition_map,
    normalize_channel_definitions,
)
from .study_common import ALL_PREVIEW_METRICS, DEFAULT_PREVIEW_METRIC


MASK_RENDER_VERSION = "binary-bw-v1"


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


def _normalize_channel_ranges(payload: Optional[Dict[str, object]]) -> Dict[int, Tuple[float, float]]:
    """Convert client-provided channel range payload into numeric overrides."""
    if not payload:
        return {}
    normalized: Dict[int, Tuple[float, float]] = {}

    def _extract_bounds(candidate: object) -> Optional[Tuple[float, float]]:
        raw_min = None
        raw_max = None
        if isinstance(candidate, dict):
            raw_min = candidate.get("vmin")
            raw_max = candidate.get("vmax")
        else:
            raw_min = getattr(candidate, "vmin", None)
            raw_max = getattr(candidate, "vmax", None)
            if raw_min is None and hasattr(candidate, "model_dump"):
                try:
                    data = candidate.model_dump()
                except Exception:
                    data = {}
                raw_min = data.get("vmin", raw_min)
                raw_max = data.get("vmax", raw_max)

        if raw_min is None and raw_max is None:
            return None

        vmin = float(raw_min if raw_min is not None else 0)
        vmax = float(raw_max if raw_max is not None else vmin + 1)
        if vmax <= vmin:
            vmax = vmin + 1
        return max(0.0, vmin), max(vmax, vmin + 1e-3)

    for key, values in payload.items():
        channel_match = re.match(r"channel_(\d+)", key)
        if not channel_match:
            continue
        bounds = _extract_bounds(values)
        if bounds is None:
            continue
        channel_index = int(channel_match.group(1))
        normalized[channel_index] = bounds
    return normalized


def _normalize_channel_ids(channel_ids: Optional[Iterable[int]] = None) -> Tuple[int, ...]:
    if channel_ids is None:
        return (1, 2, 3)
    normalized = sorted({int(channel_id) for channel_id in channel_ids if int(channel_id) in {1, 2, 3}})
    return tuple(normalized) or (1, 2, 3)


def _channel_range_token(
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    active_channels = _normalize_channel_ids(channel_ids)
    parts = []
    for channel_index in active_channels:
        bounds = channel_ranges.get(channel_index)
        if bounds is None:
            continue
        vmin, vmax = bounds
        parts.append(f"{channel_index}:{int(vmin)}-{int(vmax)}")
    if not parts:
        return "default"
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"rng-{digest}"


def _channel_color_token(
    channel_definitions: List[Dict[str, object]],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    active_channels = _normalize_channel_ids(channel_ids)
    normalized = normalize_channel_definitions(channel_definitions)
    defaults = normalize_channel_definitions(DEFAULT_CHANNEL_DEFINITIONS)
    normalized_map = {int(entry["channel"]): entry for entry in normalized}
    default_map = {int(entry["channel"]): entry for entry in defaults}
    parts = []
    for channel_index in active_channels:
        current = normalized_map.get(channel_index)
        default = default_map.get(channel_index)
        if current is None or default is None:
            continue
        if current == default:
            continue
        parts.append(f"{channel_index}:{current['color']}")
    if not parts:
        return "default"
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"clr-{digest}"


def _preview_style_token(
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    range_token = _channel_range_token(channel_ranges, channel_ids)
    color_token = _channel_color_token(channel_definitions, channel_ids)
    if range_token == "default" and color_token == "default":
        return "default"
    digest = hashlib.sha1(f"{range_token}|{color_token}".encode("utf-8")).hexdigest()[:10]
    return f"style-{digest}"


def _threshold_dependency_token(
    thresholds: Dict[str, int],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    active_channels = _normalize_channel_ids(channel_ids)
    if not active_channels:
        return "default"
    parts = [f"{channel_index}:{thresholds.get(f'channel_{channel_index}', 0)}" for channel_index in active_channels]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"thr-{digest}"


def _preview_dependency_token(
    thresholds: Dict[str, int],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
    variant: PreviewVariant,
) -> str:
    active_channels = variant.channels or _normalize_channel_ids()
    tokens: List[str] = []
    if variant.variant in {"mask", "overlay"}:
        tokens.append(MASK_RENDER_VERSION)
        tokens.append(_threshold_dependency_token(thresholds, active_channels))
    if variant.variant in {"raw", "overlay"}:
        tokens.append(_channel_range_token(channel_ranges, active_channels))
    tokens.append(_channel_color_token(channel_definitions, active_channels))
    non_default = [token for token in tokens if token != "default"]
    if not non_default:
        return "default"
    digest = hashlib.sha1("|".join(non_default).encode("utf-8")).hexdigest()[:10]
    return f"dep-{digest}"


def _visualizer_range_payload(channel_ranges: Dict[int, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for channel_index, (vmin, vmax) in channel_ranges.items():
        payload[f"channel_{channel_index}"] = {"vmin": vmin, "vmax": vmax}
    return payload


def _build_variant_cache_key(
    cache_base: str,
    variant: PreviewVariant,
    metric: str,
    dependency_token: str,
) -> str:
    channel_tag = "-".join(str(ch) for ch in variant.channels) or "all"
    token = dependency_token or "default"
    if variant.cacheable:
        return f"{cache_base}|{variant.variant}|{channel_tag}|{token}"
    return f"{cache_base}|{metric}|{variant.variant}|{channel_tag}|{token}"


def _build_variant_filename(
    safe_base: str,
    variant: PreviewVariant,
    metric_slug: str,
    cacheable: bool,
    dependency_token: str,
) -> str:
    channel_tag = "-".join(f"ch{ch}" for ch in variant.channels) or "all"
    parts = [safe_base, variant.variant, channel_tag]
    if not cacheable:
        parts.append(metric_slug)
    if dependency_token and dependency_token != "default":
        parts.append(dependency_token)
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


def _write_preview_metadata(image_path: Path, metric_id: str, variant: PreviewVariant, dependency_token: str) -> None:
    metadata = {
        "variant": variant.variant,
        "channels": list(variant.channels),
        "cache_scope": "global" if variant.cacheable else metric_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "dependency_token": dependency_token or "default",
    }
    meta_path = _metadata_path_for(image_path)
    try:
        meta_path.write_text(json.dumps(metadata))
    except Exception:
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)


def _metadata_matches(image_path: Path, metric_id: str, variant: PreviewVariant, dependency_token: str) -> bool:
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
    if data.get("cache_scope") != expected_scope:
        return False
    stored_token = data.get("dependency_token", data.get("range_token", "default"))
    return stored_token == (dependency_token or "default")


def _cached_variant_valid(path: Path, metric_id: str, variant: PreviewVariant, dependency_token: str) -> bool:
    path = Path(path)
    if not _is_valid_preview_file(path):
        return False
    return _metadata_matches(path, metric_id, variant, dependency_token)


def _preview_cache_token(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except OSError:
        return "0"


def _normalize_channel(channel: np.ndarray, range_override: Optional[Tuple[float, float]] = None) -> np.ndarray:
    channel = channel.astype(np.float32)
    channel = np.clip(channel, 0, None)
    if range_override:
        vmin, vmax = range_override
        channel = np.clip(channel, vmin, vmax)
        channel -= vmin
        denom = max(vmax - vmin, 1e-6)
    else:
        channel -= channel.min()
        denom = channel.max()
    if denom <= 0:
        return np.zeros_like(channel, dtype=np.uint8)
    channel /= denom
    channel *= 255.0
    return channel.astype(np.uint8)


def _generate_raw_image(
    channels: Dict[int, np.ndarray],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    sample = next((array for array in channels.values() if array is not None), None)
    if sample is None:
        return None
    rgb = np.zeros((*sample.shape, 3), dtype=np.float32)
    color_map = channel_definition_map(channel_definitions)
    for channel_id in (1, 2, 3):
        source = channels.get(channel_id)
        if source is None:
            continue
        normalized = _normalize_channel(source, channel_ranges.get(channel_id)).astype(np.float32) / 255.0
        channel_color = str(color_map.get(channel_id, {}).get("color", "#ffffff"))
        red, green, blue = channel_color_rgb(channel_color)
        rgb[..., 0] += normalized * red
        rgb[..., 1] += normalized * green
        rgb[..., 2] += normalized * blue
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_binary_mask(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], channel_id: int
) -> Optional[np.ndarray]:
    channel = channels.get(channel_id)
    threshold = thresholds.get(f"channel_{channel_id}")
    if channel is None or threshold is None:
        return None
    return (channel > threshold).astype(np.uint8)


def _generate_channel_raw_image(
    channels: Dict[int, np.ndarray],
    channel_id: int,
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    channel = channels.get(channel_id)
    if channel is None:
        return None
    normalized = _normalize_channel(channel, channel_ranges.get(channel_id)).astype(np.float32) / 255.0
    color_map = channel_definition_map(channel_definitions)
    channel_color = str(color_map.get(channel_id, {}).get("color", "#ffffff"))
    red, green, blue = channel_color_rgb(channel_color)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized * red
    rgb[..., 1] = normalized * green
    rgb[..., 2] = normalized * blue
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _generate_channel_mask_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_id: int,
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    binary = _build_binary_mask(channels, thresholds, channel_id)
    if binary is None:
        return None
    mask = (binary.astype(np.uint8) * 255)[..., None]
    return np.repeat(mask, 3, axis=2)


def _generate_mask_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    masks = {channel_id: _build_binary_mask(channels, thresholds, channel_id) for channel_id in (1, 2, 3)}
    valid_masks = [mask for mask in masks.values() if mask is not None]
    if not valid_masks:
        sample = next(iter(channels.values()), None)
        if sample is None:
            return None
        shape = sample.shape
        return np.zeros((*shape, 3), dtype=np.uint8)
    combined = np.maximum.reduce(valid_masks).astype(np.uint8) * 255
    return np.repeat(combined[..., None], 3, axis=2)


def _apply_highlight(raw: np.ndarray, mask: np.ndarray, strength: float = 0.45) -> np.ndarray:
    base = raw.astype(np.float32)
    alpha = (mask[..., :1].astype(np.float32) / 255.0) * strength
    highlighted = np.clip(base * (1.0 - alpha) + 255.0 * alpha, 0, 255)
    result = np.where(mask[..., :1] > 0, highlighted, base)
    return result.astype(np.uint8)


def _apply_white_mask(raw: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 image for raw overlay")
    if binary_mask.ndim != 2:
        raise ValueError("Expected 2D mask")
    overlay = raw.copy()
    overlay[binary_mask.astype(bool)] = 255
    return overlay


def _generate_channel_overlay_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_id: int,
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    raw = _generate_channel_raw_image(channels, channel_id, channel_ranges, channel_definitions)
    if raw is None:
        return None
    binary = _build_binary_mask(channels, thresholds, channel_id)
    if binary is None:
        return raw
    return _apply_white_mask(raw, binary)


def _generate_ratio_overlay_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_pair: Tuple[int, ...],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    if len(channel_pair) != 2:
        return None
    raw_images: List[np.ndarray] = []
    for channel_id in channel_pair:
        raw_image = _generate_channel_raw_image(channels, channel_id, channel_ranges, channel_definitions)
        if raw_image is None:
            return None
        raw_images.append(raw_image.astype(np.float32))
    stack = np.stack(raw_images, axis=0)
    base = np.max(stack, axis=0).astype(np.uint8)
    masks = [_build_binary_mask(channels, thresholds, channel_id) for channel_id in channel_pair]
    masks = [mask for mask in masks if mask is not None]
    if not masks:
        return base
    combined = np.maximum.reduce(masks)
    return _apply_white_mask(base, combined)


def _generate_overlay_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    raw = _generate_raw_image(channels, channel_ranges, channel_definitions)
    if raw is None:
        return None
    masks = [_build_binary_mask(channels, thresholds, channel_id) for channel_id in (1, 2, 3)]
    masks = [mask for mask in masks if mask is not None]
    if not masks:
        return raw
    combined = np.maximum.reduce(masks)
    return _apply_white_mask(raw, combined)


def _render_preview_variant(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    variant: PreviewVariant,
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    if variant.variant == "raw":
        if len(variant.channels) == 1:
            return _generate_channel_raw_image(channels, variant.channels[0], channel_ranges, channel_definitions)
        return _generate_raw_image(channels, channel_ranges, channel_definitions)
    if variant.variant == "mask":
        if len(variant.channels) == 1:
            return _generate_channel_mask_image(channels, thresholds, variant.channels[0], channel_definitions)
        return _generate_mask_image(channels, thresholds, channel_definitions)
    if variant.variant == "overlay":
        if len(variant.channels) == 1:
            return _generate_channel_overlay_image(
                channels,
                thresholds,
                variant.channels[0],
                channel_ranges,
                channel_definitions,
            )
        if len(variant.channels) == 2:
            return _generate_ratio_overlay_image(
                channels,
                thresholds,
                variant.channels,
                channel_ranges,
                channel_definitions,
            )
        return _generate_overlay_image(channels, thresholds, channel_ranges, channel_definitions)
    return None


def _write_image(image_array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(image_array)
    max_dim = max(image.size)
    if max_dim > 320:
        scale = 320 / max_dim
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    image.save(path)
