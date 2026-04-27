"""Core image processing functions for ND2 files."""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union, Any, Sequence
from pathlib import Path
import logging

import pyclesperanto as cle
from nd2reader import ND2Reader, Nd2

from data_models import ChannelData, ImageMetrics
from config import DEFAULT_THRESHOLDS, DEFAULT_MARKER, DEFAULT_MARKER_2D

# Set up logging
logger = logging.getLogger(__name__)
SUPPORTED_MICROSCOPY_EXTENSIONS = (".nd2", ".czi", ".oib", ".oif")
OPTIONAL_AICSIMAGEIO_EXTENSIONS = (".czi", ".oib", ".oif")
BIOFORMATS_REQUIRED_EXTENSIONS = (".oib", ".oif")
SUBJECT_TOKEN_PATTERN = re.compile(r"([A-Za-z]+)(\d{1,4})")
CHANNEL_XML_NAME_PATTERN = re.compile(r"<Channel\b[^>]*\bName=\"([^\"]+)\"")
DEFAULT_CHANNEL_COLORS = (
    "#00ff00",
    "#ff0000",
    "#0000ff",
    "#ffff00",
    "#00ffff",
    "#ff00ff",
    "#ffffff",
    "#ff8800",
)


def _project_channel_stack(channel_data: np.ndarray, projection: str) -> np.ndarray:
    if channel_data.ndim <= 2:
        return channel_data
    if projection == "max":
        return np.max(channel_data, axis=0)
    if projection == "sum":
        return np.sum(channel_data, axis=0)
    if projection == "none":
        return channel_data
    raise ValueError(f"Unsupported projection mode: {projection}")


def _default_channel_color(position: int) -> str:
    if 1 <= position <= len(DEFAULT_CHANNEL_COLORS):
        return DEFAULT_CHANNEL_COLORS[position - 1]
    return "#ffffff"


def _clean_channel_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _extract_channel_label_value(candidate: Any) -> str:
    if candidate is None:
        return ""
    if isinstance(candidate, bytes):
        return _clean_channel_label(candidate.decode("utf-8", errors="ignore"))
    if isinstance(candidate, str):
        return _clean_channel_label(candidate)
    if isinstance(candidate, dict):
        for key in ("label", "name", "channel_name", "fluor"):
            if key in candidate:
                label = _extract_channel_label_value(candidate.get(key))
                if label:
                    return label
        return ""

    for attr in ("label", "name", "channel_name", "fluor"):
        if hasattr(candidate, attr):
            label = _extract_channel_label_value(getattr(candidate, attr, None))
            if label:
                return label
    return ""


def _dedupe_labels(labels: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for label in labels:
        cleaned = _clean_channel_label(label)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _channel_labels_from_text(text: str) -> List[str]:
    if not text:
        return []
    matches = [_clean_channel_label(match) for match in CHANNEL_XML_NAME_PATTERN.findall(text)]
    return _dedupe_labels(matches)


def _coerce_channel_labels(candidate: Any) -> List[str]:
    if candidate is None:
        return []
    if isinstance(candidate, bytes):
        return _coerce_channel_labels(candidate.decode("utf-8", errors="ignore"))
    if isinstance(candidate, str):
        parsed = _channel_labels_from_text(candidate)
        if parsed:
            return parsed
        cleaned = _clean_channel_label(candidate)
        if cleaned and len(cleaned) <= 120:
            return [cleaned]
        return []
    if isinstance(candidate, dict):
        for key in ("channel_names", "channels", "metadata", "ome_metadata"):
            labels = _coerce_channel_labels(candidate.get(key))
            if labels:
                return labels
        return []
    if isinstance(candidate, Sequence):
        labels = [_extract_channel_label_value(item) for item in candidate]
        return _dedupe_labels([label for label in labels if label])

    for attr in ("channel_names", "channels"):
        labels = _coerce_channel_labels(getattr(candidate, attr, None))
        if labels:
            return labels

    metadata = getattr(candidate, "metadata", None)
    if metadata is not None and metadata is not candidate:
        labels = _coerce_channel_labels(metadata)
        if labels:
            return labels

    ome_metadata = getattr(candidate, "ome_metadata", None)
    if ome_metadata is not None and ome_metadata is not candidate:
        if hasattr(ome_metadata, "to_xml"):
            try:
                labels = _channel_labels_from_text(ome_metadata.to_xml())
            except Exception:
                labels = []
            if labels:
                return labels
        labels = _coerce_channel_labels(ome_metadata)
        if labels:
            return labels

    if hasattr(candidate, "to_xml"):
        try:
            labels = _channel_labels_from_text(candidate.to_xml())
        except Exception:
            labels = []
        if labels:
            return labels

    return _channel_labels_from_text(_clean_channel_label(candidate))


def _extract_channel_labels_from_nd2_metadata(metadata: Any) -> List[str]:
    if metadata is None:
        return []
    labels = _coerce_channel_labels(getattr(metadata, "channels", None))
    if labels:
        return labels
    labels = _coerce_channel_labels(getattr(metadata, "channel_names", None))
    if labels:
        return labels
    return _coerce_channel_labels(metadata)


def _extract_channel_labels_from_aics_image(image: Any) -> List[str]:
    labels = _coerce_channel_labels(getattr(image, "channel_names", None))
    if labels:
        return labels
    ome_metadata = getattr(image, "ome_metadata", None)
    if ome_metadata is not None:
        if hasattr(ome_metadata, "to_xml"):
            try:
                labels = _channel_labels_from_text(ome_metadata.to_xml())
            except Exception:
                labels = []
            if labels:
                return labels
        labels = _coerce_channel_labels(ome_metadata)
        if labels:
            return labels
    return _coerce_channel_labels(getattr(image, "metadata", None))


def _select_channel_indices(channel_count: int, limit: Optional[int] = None) -> List[int]:
    if channel_count <= 0:
        return []
    if limit is None or limit <= 0:
        return list(range(channel_count))
    return list(range(min(channel_count, limit)))


def detect_channel_definitions(filepath: str, limit: Optional[int] = None) -> Optional[List[Dict[str, object]]]:
    """
    Detect display channel labels from image metadata.
    """
    file_suffix = Path(filepath).suffix.lower()
    labels: List[str] = []

    if file_suffix == ".nd2":
        try:
            with ND2Reader(filepath) as nd2:
                labels = _extract_channel_labels_from_nd2_metadata(getattr(nd2, "metadata", None))
        except Exception as exc:
            logger.debug(f"Could not read ND2 channel metadata from {filepath}: {exc}")
    else:
        try:
            from aicsimageio import AICSImage  # type: ignore

            image = AICSImage(filepath)
            labels = _extract_channel_labels_from_aics_image(image)
        except Exception as exc:
            logger.debug(f"Could not read channel metadata via AICSImage from {filepath}: {exc}")

    if not labels:
        return None

    selected_indices = _select_channel_indices(len(labels), limit=limit)
    if not selected_indices:
        return None

    definitions: List[Dict[str, object]] = []
    for position, source_index in enumerate(selected_indices, start=1):
        label = labels[source_index] if source_index < len(labels) else ""
        if not label:
            continue
        definitions.append({"channel": position, "label": label, "color": _default_channel_color(position)})
    return definitions or None

def detect_pixel_size(filepath: str) -> Optional[float]:
    """
    Attempt to read the pixel-to-micrometer ratio from an ND2 file.

    Args:
        filepath: Path to an ND2 image.

    Returns:
        Detected pixel size in micrometers, or None if unavailable.
    """
    file_suffix = Path(filepath).suffix.lower()
    if file_suffix == ".nd2":
        try:
            with ND2Reader(filepath) as nd2:
                pixel_size = getattr(nd2.metadata, "pixel_microns", None)
                value = _extract_pixel_value(pixel_size)
                if value is not None and value > 0:
                    return float(value)
        except Exception as exc:
            logger.debug(f"Could not read pixel size from {filepath}: {exc}")
        try:
            nd2 = Nd2(filepath)
            pixel_size = getattr(nd2.metadata, "pixel_microns", None)
            nd2.close()
            value = _extract_pixel_value(pixel_size)
            if value is not None and value > 0:
                return float(value)
        except Exception as exc:
            logger.debug(f"Fallback pixel size read failed for {filepath}: {exc}")
        return None

    try:
        from aicsimageio import AICSImage  # type: ignore

        image = AICSImage(filepath)
        pixel_sizes = getattr(image, "physical_pixel_sizes", None)
        if pixel_sizes is None:
            return None
        x_size = getattr(pixel_sizes, "X", None)
        value = _extract_pixel_value(x_size)
        if value is not None and value > 0:
            return float(value)
    except Exception as exc:
        logger.debug(f"Could not read pixel size via AICSImage from {filepath}: {exc}")
    return None


def _extract_pixel_value(pixel_size: Optional[Union[Sequence[float], float]]) -> Optional[float]:
    """Extract the first numeric value from pixel metadata."""
    if pixel_size is None:
        return None
    if isinstance(pixel_size, (int, float)):
        return float(pixel_size)
    if isinstance(pixel_size, Sequence) and pixel_size:
        try:
            return float(pixel_size[0])
        except (TypeError, ValueError):
            return None
    return None


def ensure_microscopy_reader_dependencies(filepaths: Sequence[Union[str, Path]]) -> None:
    """
    Validate that optional readers needed by the discovered file extensions
    exist in the active Python environment before processing begins.
    """
    required_extensions = sorted(
        {
            Path(filepath).suffix.lower()
            for filepath in filepaths
            if Path(filepath).suffix.lower() in OPTIONAL_AICSIMAGEIO_EXTENSIONS
        }
    )
    if not required_extensions:
        return

    try:
        import aicsimageio  # type: ignore  # noqa: F401
    except Exception as exc:
        extensions = ", ".join(required_extensions)
        raise ImportError(
            f"Found microscopy files with extensions {extensions}, but 'aicsimageio' is not "
            "installed in the active Python environment. Install it to process .czi/.oib/.oif files."
        ) from exc

    bioformats_extensions = sorted(
        {
            Path(filepath).suffix.lower()
            for filepath in filepaths
            if Path(filepath).suffix.lower() in BIOFORMATS_REQUIRED_EXTENSIONS
        }
    )
    if not bioformats_extensions:
        return

    try:
        import bioformats_jar  # type: ignore  # noqa: F401
    except Exception as exc:
        extensions = ", ".join(bioformats_extensions)
        raise ImportError(
            f"Found Olympus microscopy files with extensions {extensions}, but 'bioformats_jar' is not "
            "installed in the active Python environment. Install it to process .oib/.oif files."
        ) from exc


def _configure_bioformats_runtime() -> None:
    """
    Route Bio-Formats / Java caches into a writable runtime directory when one
    is provided by the caller environment.
    """
    cache_root_value = os.environ.get("MICROSCOPY_RUNTIME_CACHE_DIR")
    if not cache_root_value:
        return

    cache_root = Path(cache_root_value)
    cache_root.mkdir(parents=True, exist_ok=True)

    xdg_cache = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    cjdk_cache = Path(os.environ.setdefault("CJDK_CACHE_DIR", str(cache_root / "cjdk")))
    scyjava_cache = Path(os.environ.get("SCYJAVA_CACHE_DIR", str(cache_root / "scyjava")))
    scyjava_m2_repo = Path(os.environ.get("SCYJAVA_M2_REPO", str(cache_root / "m2")))

    for path in (xdg_cache, cjdk_cache, scyjava_cache, scyjava_m2_repo):
        path.mkdir(parents=True, exist_ok=True)

    try:
        import scyjava.config as scyjava_config  # type: ignore
    except Exception:
        return

    scyjava_config.set_java_constraints(
        fetch="auto",
        vendor=os.environ.get("SCYJAVA_JAVA_VENDOR", "zulu"),
        version=os.environ.get("SCYJAVA_JAVA_VERSION", "11"),
    )
    scyjava_config.set_cache_dir(scyjava_cache)
    scyjava_config.set_m2_repo(scyjava_m2_repo)

def get_nd2_files(directory: str) -> List[str]:
    """
    Recursively find supported microscopy files in a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of paths to supported files
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    directory_path = Path(directory)
    microscopy_files = [
        str(p.absolute())
        for p in directory_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_MICROSCOPY_EXTENSIONS
    ]

    logger.info(f"Found {len(microscopy_files)} microscopy files in {directory}")
    return sorted(microscopy_files)


def _canonical_mouse_token(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value)).upper()


def _loose_mouse_token(value: str) -> str:
    match = re.match(r"^([A-Z]+)(\d+)$", value)
    if not match:
        return value
    prefix, digits = match.groups()
    return f"{prefix}{digits.lstrip('0') or '0'}"


def _prepared_mouse_candidates(mouse_ids: List[str]) -> List[Tuple[str, str, str, int]]:
    prepared: List[Tuple[str, str, str, int]] = []
    for candidate in mouse_ids:
        canonical = _canonical_mouse_token(candidate)
        if not canonical:
            continue
        prepared.append((candidate, canonical, _loose_mouse_token(canonical), len(canonical)))
    return prepared

def parse_mouse_id(filename: str, marker: str = DEFAULT_MARKER, mouse_ids: list = None) -> str:
    """
    Extract mouse ID from filename using direct token matching.
    
    Args:
        filename: Path to the file
        marker: String marker that precedes mouse ID (for backward compatibility)
        mouse_ids: List of known mouse IDs to search for
        
    Returns:
        Extracted mouse ID
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split()
    
    # If mouse_ids list is provided, use direct token matching
    if mouse_ids:
        prepared_candidates = _prepared_mouse_candidates(list(mouse_ids))
        token_matches: List[Tuple[int, int, int, int, str]] = []
        for match in SUBJECT_TOKEN_PATTERN.finditer(base_name):
            token = _canonical_mouse_token(match.group(0))
            loose_token = _loose_mouse_token(token)
            digit_count = len(match.group(2))
            position = match.start()
            for candidate, canonical, loose, candidate_length in prepared_candidates:
                if token == canonical:
                    token_matches.append((2, digit_count, candidate_length, -position, candidate))
                elif loose_token == loose:
                    token_matches.append((1, digit_count, candidate_length, -position, candidate))
        if token_matches:
            token_matches.sort(reverse=True)
            return token_matches[0][4]

        sanitized_base = _canonical_mouse_token(base_name)
        substring_matches: List[Tuple[int, int, int, str]] = []
        for candidate, canonical, loose, candidate_length in prepared_candidates:
            if canonical and canonical in sanitized_base:
                substring_matches.append((2, candidate_length, len(loose), candidate))
            elif loose and loose in sanitized_base:
                substring_matches.append((1, candidate_length, len(loose), candidate))
        if substring_matches:
            substring_matches.sort(reverse=True)
            return substring_matches[0][3]
        for part in parts:
            if part in mouse_ids:
                return part
        raise ValueError(f"No known mouse ID found in filename: {filename}")
    
    # Fallback to original marker-based approach for backward compatibility
    try:
        marker_index = parts.index(marker)
        if marker_index == 0:
            raise ValueError(f"Marker '{marker}' found at beginning of filename")
        return parts[marker_index - 1]
    except (ValueError, IndexError):
        raise ValueError(f"Could not find marker '{marker}' in filename: {filename}")

def load_microscopy_channels(
    filepath: str,
    is_3d: bool = True,
    projection: str = "max",
    channel_indices: Optional[Sequence[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Load a microscopy file and return projected channel data keyed by 1-based channel index.

    Args:
        filepath: Path to the microscopy file
        is_3d: Whether the file contains 3D data
        projection: Z-axis projection for 3D stacks ("max", "sum", "none")
        channel_indices: Optional 1-based channel subset to load. When omitted, all channels are returned.

    Returns:
        Mapping of channel index -> numpy array
    """
    requested_indices = None
    if channel_indices is not None:
        requested_indices = {int(index) for index in channel_indices if int(index) > 0}
    try:
        file_suffix = Path(filepath).suffix.lower()
        channel_arrays: Dict[int, np.ndarray] = {}
        if file_suffix == ".nd2":
            if is_3d:
                nd2 = ND2Reader(filepath)
                nd2.bundle_axes = ("c", "y", "x")
                nd2.iter_axes = "z"

                image = np.asarray(nd2)
                logger.debug(f"Loaded 3D image: {filepath}, shape: {image.shape}")

                channel_count = image.shape[1] if image.ndim >= 2 else 0
                selected_indices = _select_channel_indices(channel_count)
                for idx in selected_indices:
                    channel_id = idx + 1
                    if requested_indices is not None and channel_id not in requested_indices:
                        continue
                    channel_stack = image[:, idx, :, :]
                    channel_arrays[channel_id] = _project_channel_stack(channel_stack, projection)

                nd2.close()
            else:
                nd2 = Nd2(filepath)
                image = np.asarray(nd2)
                nd2.close()

                logger.debug(f"Loaded 2D image: {filepath}, shape: {image.shape}")

                if image.ndim >= 3:
                    channel_axis = 0
                    channel_count = image.shape[channel_axis]
                    selected_indices = _select_channel_indices(channel_count)
                    for idx in selected_indices:
                        channel_id = idx + 1
                        if requested_indices is not None and channel_id not in requested_indices:
                            continue
                        channel_arrays[channel_id] = image[idx]
                else:
                    raise ValueError(f"Unexpected ND2 image shape for 2D data: {image.shape}")
        else:
            if file_suffix in BIOFORMATS_REQUIRED_EXTENSIONS:
                _configure_bioformats_runtime()
                try:
                    from aicsimageio.readers.bioformats_reader import BioformatsReader  # type: ignore
                except Exception as exc:
                    raise ImportError(
                        "Loading .oib/.oif files requires both 'aicsimageio' and 'bioformats_jar'."
                    ) from exc
                image = BioformatsReader(filepath)
                data = image.get_image_data("CZYX", T=0)
            else:
                try:
                    from aicsimageio import AICSImage  # type: ignore
                except Exception as exc:
                    raise ImportError(
                        "Loading .czi files requires the 'aicsimageio' dependency."
                    ) from exc

                image = AICSImage(filepath)
                data = image.get_image_data("CZYX", T=0)
            logger.debug(f"Loaded microscopy image via AICSImage: {filepath}, shape: {data.shape}")

            channel_count = data.shape[0] if data.ndim >= 1 else 0
            selected_indices = _select_channel_indices(channel_count)
            for idx in selected_indices:
                channel_id = idx + 1
                if requested_indices is not None and channel_id not in requested_indices:
                    continue
                channel_stack = np.asarray(data[idx])
                if is_3d:
                    channel_arrays[channel_id] = _project_channel_stack(channel_stack, projection)
                else:
                    if channel_stack.ndim == 3:
                        channel_arrays[channel_id] = channel_stack[0]
                    else:
                        channel_arrays[channel_id] = channel_stack

        if not channel_arrays:
            raise ValueError(f"No channels found in microscopy file: {filepath}")
        return channel_arrays
        
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        raise


def load_nd2_file(filepath: str, is_3d: bool = True, projection: str = "max") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a microscopy file and extract the first three channels for legacy callers.

    Args:
        filepath: Path to the microscopy file
        is_3d: Whether the file contains 3D data
        projection: Z-axis projection for 3D stacks ("max", "sum", "none")

    Returns:
        Tuple of (channel_1, channel_2, channel_3) data
    """
    channel_map = load_microscopy_channels(filepath, is_3d=is_3d, projection=projection)
    ordered = [channel_map[channel_id] for channel_id in sorted(channel_map)]
    actual_channels = len(ordered)
    reference = ordered[0]
    while len(ordered) < 3:
        ordered.append(np.zeros_like(reference))
    if actual_channels < 3:
        logger.warning(
            f"Padded microscopy file {os.path.basename(filepath)} from {actual_channels} to 3 channels with zeros"
        )
    return ordered[0], ordered[1], ordered[2]

def analyze_channel(
    image: np.ndarray, 
    threshold_value: float
) -> ChannelData:
    """
    Analyze a single channel with thresholding and statistics.
    
    Args:
        image: 2D image data for the channel
        threshold_value: Value for thresholding
        
    Returns:
        ChannelData object with analysis results
    """
    try:
        # Ensure image is not empty and has valid data
        if image is None or image.size == 0:
            logger.warning("Empty or null image data encountered")
            # Create empty stats DataFrame with required columns
            stats = pd.DataFrame({
                'area': [0],
                'mean_intensity': [0],
                'sum_intensity': [0],
                'min_intensity': [0],
                'max_intensity': [0]
            })
            # Create a zero threshold mask with same shape as image if possible
            threshold = np.zeros_like(image) if image is not None else np.array([[0]])
            return ChannelData(
                raw=image, 
                threshold=threshold, 
                stats=stats,
                threshold_value=threshold_value
            )
        
        # Create threshold mask
        threshold = cle.greater_constant(image, None, threshold_value)
        
        # Compute statistics - handle case where no regions are found
        stats_dict = cle.statistics_of_labelled_pixels(image, threshold)
        stats = pd.DataFrame(stats_dict)
        
        # If no statistics were found (no regions above threshold), create default row
        if len(stats) == 0:
            stats = pd.DataFrame({
                'area': [0],
                'mean_intensity': [0],
                'sum_intensity': [0],
                'min_intensity': [0],
                'max_intensity': [0]
            })
        else:
            # Keep only relevant columns
            required_cols = ['area', 'mean_intensity', 'sum_intensity', 'min_intensity', 'max_intensity']
            available_cols = [col for col in required_cols if col in stats.columns]
            stats = stats[available_cols]
        
        return ChannelData(
            raw=image, 
            threshold=threshold, 
            stats=stats,
            threshold_value=threshold_value
        )
        
    except Exception as e:
        logger.warning(f"Error in channel analysis (returning zeros): {e}")
        # Return default values when analysis fails
        stats = pd.DataFrame({
            'area': [0],
            'mean_intensity': [0],
            'sum_intensity': [0],
            'min_intensity': [0],
            'max_intensity': [0]
        })
        threshold = np.zeros_like(image) if image is not None else np.array([[0]])
        return ChannelData(
            raw=image, 
            threshold=threshold, 
            stats=stats,
            threshold_value=threshold_value
        )

def safe_get_row(stats: pd.DataFrame) -> pd.Series:
    """
    Safely extract statistics row, handling empty results.
    
    Args:
        stats: DataFrame with statistics
        
    Returns:
        Series with statistics or NaNs if empty
    """
    if len(stats) > 1:
        return stats.iloc[1]  # Use second row (first is background)
    elif len(stats) == 1:
        return stats.iloc[0]  # Use first row if only one exists
    else:
        # Return NaNs if no rows
        return pd.Series({
            col: np.nan for col in ['area', 'mean_intensity', 'sum_intensity', 
                                     'min_intensity', 'max_intensity']
        })

def process_single_file(
    filepath: str,
    mouse_lookup: Dict,
    thresholds: Dict[str, float],
    is_3d: bool = True,
    marker: str = None
) -> Optional[ImageMetrics]:
    """
    Process a single ND2 file and extract metrics.
    
    Args:
        filepath: Path to the ND2 file
        mouse_lookup: Dictionary mapping mouse IDs to groups
        thresholds: Dictionary of threshold values for each channel
        is_3d: Whether the file contains 3D data
        marker: Filename marker for mouse ID extraction
        
    Returns:
        ImageMetrics object or None if error
    """
    try:
        # Extract mouse ID and lookup group
        if marker is None:
            marker = DEFAULT_MARKER if is_3d else DEFAULT_MARKER_2D
        
        # Extract known mouse IDs from mouse_lookup for direct matching
        known_mouse_ids = list(mouse_lookup.keys())
        mouse_id = parse_mouse_id(filepath, marker, known_mouse_ids)
        
        if mouse_id not in mouse_lookup:
            logger.warning(f"Mouse ID {mouse_id} not found in groups")
            return None
            
        group_name = mouse_lookup[mouse_id]["group"]
        
        # Load channel data
        channel_1, channel_2, channel_3 = load_nd2_file(filepath, is_3d)
        total_area = channel_1.shape[0] * channel_1.shape[1]
        
        # Analyze channels
        channel_1_data = analyze_channel(channel_1, thresholds['channel_1'])
        channel_2_data = analyze_channel(channel_2, thresholds['channel_2'])
        channel_3_data = analyze_channel(channel_3, thresholds['channel_3'])
        
        # Extract statistics
        row1 = safe_get_row(channel_1_data.stats)
        row2 = safe_get_row(channel_2_data.stats)
        row3 = safe_get_row(channel_3_data.stats)
        
        # Calculate metrics
        # Areas as percentages
        channel_1_area = row1['area'] / total_area * 100 if not np.isnan(row1['area']) else 0.0
        channel_2_area = row2['area'] / total_area * 100 if not np.isnan(row2['area']) else 0.0
        channel_3_area = row3['area'] / total_area * 100 if not np.isnan(row3['area']) else 0.0
        
        # Ratio calculations
        channel_2_per_channel_3_area = (
            row2['area'] / row3['area'] * 100 
            if not np.isnan(row3['area']) and row3['area'] != 0 
            else np.nan
        )
        channel_1_per_channel_3_area = (
            row1['area'] / row3['area'] * 100 
            if not np.isnan(row3['area']) and row3['area'] != 0 
            else np.nan
        )
        
        # Intensity ratios
        channel_2_per_channel_3_mean = (
            row2['mean_intensity'] / row3['mean_intensity'] 
            if not np.isnan(row3['mean_intensity']) and row3['mean_intensity'] != 0 
            else np.nan
        )
        channel_1_per_channel_3_mean = (
            row1['mean_intensity'] / row3['mean_intensity'] 
            if not np.isnan(row3['mean_intensity']) and row3['mean_intensity'] != 0 
            else np.nan
        )
        channel_2_per_channel_3_sum = (
            row2['sum_intensity'] / row3['sum_intensity'] 
            if not np.isnan(row3['sum_intensity']) and row3['sum_intensity'] != 0 
            else np.nan
        )
        channel_1_per_channel_3_sum = (
            row1['sum_intensity'] / row3['sum_intensity'] 
            if not np.isnan(row3['sum_intensity']) and row3['sum_intensity'] != 0 
            else np.nan
        )
        
        # Create ImageMetrics object
        metrics = ImageMetrics(
            group=group_name,
            mouse_id=mouse_id,
            filename=os.path.basename(filepath),
            channel_1_area=channel_1_area,
            channel_2_area=channel_2_area,
            channel_3_area=channel_3_area,
            channel_2_per_channel_3_area=channel_2_per_channel_3_area,
            channel_1_per_channel_3_area=channel_1_per_channel_3_area,
            channel_1_mean_intensity=row1['mean_intensity'],
            channel_2_mean_intensity=row2['mean_intensity'],
            channel_3_mean_intensity=row3['mean_intensity'],
            channel_1_sum_intensity=row1['sum_intensity'],
            channel_2_sum_intensity=row2['sum_intensity'],
            channel_3_sum_intensity=row3['sum_intensity'],
            channel_2_per_channel_3_mean_intensity=channel_2_per_channel_3_mean,
            channel_1_per_channel_3_mean_intensity=channel_1_per_channel_3_mean,
            channel_2_per_channel_3_sum_intensity=channel_2_per_channel_3_sum,
            channel_1_per_channel_3_sum_intensity=channel_1_per_channel_3_sum,
            channel_1_min_intensity=row1['min_intensity'],
            channel_2_min_intensity=row2['min_intensity'],
            channel_3_min_intensity=row3['min_intensity'],
            channel_1_max_intensity=row1['max_intensity'],
            channel_2_max_intensity=row2['max_intensity'],
            channel_3_max_intensity=row3['max_intensity']
        )
        
        logger.debug(f"Successfully processed {os.path.basename(filepath)}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None

def calculate_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mouse-level averages for group summary.
    Returns one row per mouse with averaged values across all images from that mouse.
    
    Args:
        df: DataFrame with individual image results
        
    Returns:
        DataFrame with mouse averages (one row per mouse)
    """
    # Get numeric columns for aggregation
    numeric_cols = [col for col in df.columns 
                   if col not in ["Group", "MouseID", "Filename"] 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        logger.warning("No numeric columns found for mouse averages")
        return pd.DataFrame()
    
    # Calculate mouse averages within each group (one row per mouse)
    mouse_averages = df.groupby(["Group", "MouseID"])[numeric_cols].mean().reset_index()
    
    logger.info(f"Calculated mouse averages for {len(mouse_averages)} mice across {len(df['Group'].unique())} groups")
    return mouse_averages

def identify_representative_images(
    df: pd.DataFrame, 
    metric: str = 'Channel_2_area',
    top_n: int = 3
) -> Dict[str, List[str]]:
    """
    Identify the most representative images for each group.
    
    Args:
        df: DataFrame with individual image results
        metric: Metric to use for selection
        top_n: Number of representative images per group
        
    Returns:
        Dictionary mapping group names to lists of representative filenames
    """
    representative_images = {}
    
    for group in df['Group'].unique():
        group_data = df[df['Group'] == group].copy()
        
        if len(group_data) == 0:
            representative_images[group] = []
            continue
            
        if metric not in group_data.columns:
            logger.warning(f"Metric {metric} not found, using first available numeric column")
            numeric_cols = [col for col in group_data.columns 
                           if pd.api.types.is_numeric_dtype(group_data[col])]
            if numeric_cols:
                metric = numeric_cols[0]
            else:
                representative_images[group] = []
                continue
        
        # Calculate distance from group mean
        group_mean = group_data[metric].mean()
        group_data['distance_from_mean'] = abs(group_data[metric] - group_mean)
        
        # Select top N closest to mean
        closest_to_mean = group_data.nsmallest(top_n, 'distance_from_mean')
        representative_images[group] = closest_to_mean['Filename'].tolist()
        
        logger.debug(f"Selected {len(representative_images[group])} representative images for group {group}")
    
    return representative_images
