"""Generate threshold analysis data from ND2 files."""

import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path

# Import from existing pipeline (no changes to existing code)
from image_processing import load_nd2_file, parse_mouse_id
from data_models import GroupConfig
from .data_models import ThresholdData

logger = logging.getLogger(__name__)

def analyze_single_image_all_thresholds(
    filepath: str,
    mouse_lookup: Dict,
    is_3d: bool = True,
    marker: str = None,
    max_threshold: int = 4095
) -> Optional[ThresholdData]:
    """
    Analyze a single image at all threshold values.
    
    MINIMAL IMPLEMENTATION: Focus on core functionality only.
    
    Args:
        filepath: Path to ND2 file
        mouse_lookup: Dictionary mapping mouse IDs to groups
        is_3d: Whether file contains 3D data
        marker: Filename marker for mouse ID extraction
        max_threshold: Maximum threshold value to test (default 4095)
        
    Returns:
        ThresholdData object or None if error
    """
    try:
        # Reuse existing functions - NO CHANGES to existing pipeline
        # Use existing pipeline's default marker logic
        mouse_id = parse_mouse_id(filepath, marker)
        
        if mouse_id not in mouse_lookup:
            logger.warning(f"Mouse ID {mouse_id} not found in groups")
            return None
            
        group_name = mouse_lookup[mouse_id]["group"]
        
        # Load image data using existing function
        channel_1, channel_2, channel_3 = load_nd2_file(filepath, is_3d)
        total_pixels = channel_1.shape[0] * channel_1.shape[1]
        
        # Pre-allocate arrays for all thresholds
        ch1_percentages = np.zeros(max_threshold + 1)
        ch2_percentages = np.zeros(max_threshold + 1)  
        ch3_percentages = np.zeros(max_threshold + 1)
        
        # Calculate positive pixel percentages for each threshold
        # MINIMAL: Simple numpy operations, no GPU acceleration yet
        for threshold in range(max_threshold + 1):
            ch1_positive = np.sum(channel_1 > threshold)
            ch2_positive = np.sum(channel_2 > threshold)
            ch3_positive = np.sum(channel_3 > threshold)
            
            ch1_percentages[threshold] = (ch1_positive / total_pixels) * 100
            ch2_percentages[threshold] = (ch2_positive / total_pixels) * 100
            ch3_percentages[threshold] = (ch3_positive / total_pixels) * 100
        
        return ThresholdData(
            mouse_id=mouse_id,
            group=group_name,
            filename=Path(filepath).name,
            channel_1_percentages=ch1_percentages,
            channel_2_percentages=ch2_percentages,
            channel_3_percentages=ch3_percentages
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
        print(f"Channel 1 at threshold 1000: {result.get_percentage_at_threshold(1, 1000):.2f}%")
        print(f"Channel 2 at threshold 1000: {result.get_percentage_at_threshold(2, 1000):.2f}%")
        print(f"Channel 3 at threshold 1000: {result.get_percentage_at_threshold(3, 1000):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
