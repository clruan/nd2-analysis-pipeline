"""Generate threshold analysis data from microscopy files."""

import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path

# Import from existing pipeline (no changes to existing code)
from config import DEFAULT_MARKER, DEFAULT_MARKER_2D
from image_processing import load_microscopy_channels, parse_mouse_id
from data_models import GroupConfig
from .data_models import ThresholdData

logger = logging.getLogger(__name__)


def _compute_threshold_percentages(channel_data: np.ndarray, max_threshold: int) -> np.ndarray:
    """Compute positive-pixel percentages for every threshold in one pass."""
    flattened = np.asarray(channel_data, dtype=np.float32).reshape(-1)
    if flattened.size == 0:
        return np.zeros(max_threshold + 1, dtype=np.float32)

    # A value contributes to threshold t when value > t. Ceil keeps that
    # relation intact for non-integer microscope intensities.
    quantized = np.ceil(flattened).astype(np.int32, copy=False)
    clipped = np.clip(quantized, 0, max_threshold)
    histogram = np.bincount(clipped, minlength=max_threshold + 1)
    above_max = int(np.count_nonzero(quantized > max_threshold))

    greater_than = np.zeros(max_threshold + 1, dtype=np.float64)
    if max_threshold > 0:
        cumulative = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
        greater_than[:-1] = cumulative[1:]
    greater_than += above_max

    return ((greater_than / flattened.size) * 100.0).astype(np.float32, copy=False)

def analyze_single_image_all_thresholds(
    filepath: str,
    mouse_lookup: Dict,
    is_3d: bool = True,
    marker: str = None,
    max_threshold: int = 0
) -> Optional[ThresholdData]:
    """
    Analyze a single image at all threshold values.
    
    MINIMAL IMPLEMENTATION: Focus on core functionality only.
    
    Args:
        filepath: Path to source image file
        mouse_lookup: Dictionary mapping mouse IDs to groups
        is_3d: Whether file contains 3D data
        marker: Filename marker for mouse ID extraction
        max_threshold: Maximum threshold value to test. Use 0 to auto-size from the image data.
        
    Returns:
        ThresholdData object or None if error
    """
    try:
        # Reuse existing functions - NO CHANGES to existing pipeline
        # Use existing pipeline's default marker logic and direct ID matching
        if marker is None:
            marker = DEFAULT_MARKER if is_3d else DEFAULT_MARKER_2D
        known_mouse_ids = list(mouse_lookup.keys())
        mouse_id = parse_mouse_id(filepath, marker, known_mouse_ids)
        
        if mouse_id not in mouse_lookup:
            logger.warning(f"Mouse ID {mouse_id} not found in groups")
            return None
            
        group_name = mouse_lookup[mouse_id]["group"]
        
        channel_arrays = load_microscopy_channels(filepath, is_3d)
        if not channel_arrays:
            raise ValueError(f"No channels were loaded from {filepath}")

        resolved_max_threshold = int(max_threshold)
        if resolved_max_threshold <= 0:
            channel_maxima = [
                int(np.ceil(np.max(np.asarray(channel)))) if np.asarray(channel).size else 0
                for channel in channel_arrays.values()
            ]
            resolved_max_threshold = max(channel_maxima, default=0)

        channel_percentages = {
            channel: _compute_threshold_percentages(channel_data, resolved_max_threshold)
            for channel, channel_data in channel_arrays.items()
        }

        return ThresholdData(
            mouse_id=mouse_id,
            group=group_name,
            filename=Path(filepath).name,
            channel_percentages=channel_percentages,
        )
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None

# MINIMAL TEST FUNCTION
def test_single_file(filepath: str, config_path: str) -> bool:
    """Test threshold generation on a single file."""
    try:
        config = GroupConfig.from_json(config_path)
        mouse_lookup = config.build_mouse_info()
        
        result = analyze_single_image_all_thresholds(filepath, mouse_lookup)
        
        if result is None:
            print(f"Failed to process {filepath}")
            return False
            
        print(f"Success! Processed {result.filename}")
        print(f"Mouse: {result.mouse_id}, Group: {result.group}")
        for channel in result.channel_ids:
            print(f"Channel {channel} at threshold 1000: {result.get_percentage_at_threshold(channel, 1000):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
