"""Core image processing functions for ND2 files."""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
import logging

import pyclesperanto_prototype as cle
from nd2reader import ND2Reader, Nd2

from data_models import ChannelData, ImageMetrics
from config import DEFAULT_THRESHOLDS, DEFAULT_MARKER, DEFAULT_MARKER_2D

# Set up logging
logger = logging.getLogger(__name__)

def get_nd2_files(directory: str) -> List[str]:
    """
    Recursively find all ND2 files in a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of paths to ND2 files
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    directory_path = Path(directory)
    nd2_files = [str(p.absolute()) for p in directory_path.glob("**/*.nd2")]
    
    logger.info(f"Found {len(nd2_files)} ND2 files in {directory}")
    return nd2_files

def parse_mouse_id(filename: str, marker: str = DEFAULT_MARKER) -> str:
    """
    Extract mouse ID from filename based on marker position.
    
    Args:
        filename: Path to the file
        marker: String marker that precedes mouse ID
        
    Returns:
        Extracted mouse ID
    """
    parts = os.path.splitext(os.path.basename(filename))[0].split()
    try:
        marker_index = parts.index(marker)
        if marker_index == 0:
            raise ValueError(f"Marker '{marker}' found at beginning of filename")
        return parts[marker_index - 1]
    except (ValueError, IndexError):
        raise ValueError(f"Could not find marker '{marker}' in filename: {filename}")

def load_nd2_file(filepath: str, is_3d: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an ND2 file and extract channel data.
    
    Args:
        filepath: Path to the ND2 file
        is_3d: Whether the file contains 3D data
        
    Returns:
        Tuple of (channel_1, channel_2, channel_3) data
    """
    try:
        if is_3d:
            # Handle 3D data using ND2Reader
            nd2 = ND2Reader(filepath)
            nd2.bundle_axes = ('c', 'y', 'x')
            nd2.iter_axes = 'z'
            
            image = np.asarray(nd2)
            logger.debug(f"Loaded 3D image: {filepath}, shape: {image.shape}")
            
            # Max projections over z-axis
            channel_1 = np.max(image[:, 0, :, :], axis=0)  # Channel 0 -> Channel 1
            channel_2 = np.max(image[:, 1, :, :], axis=0)  # Channel 1 -> Channel 2
            channel_3 = np.max(image[:, 2, :, :], axis=0)  # Channel 2 -> Channel 3
            
            nd2.close()
            
        else:
            # Handle 2D data using Nd2
            nd2 = Nd2(filepath)
            image = np.asarray(nd2)
            nd2.close()
            
            logger.debug(f"Loaded 2D image: {filepath}, shape: {image.shape}")
            
            channel_1 = image[0]
            channel_2 = image[1]
            channel_3 = image[2]
            
        return channel_1, channel_2, channel_3
        
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        raise

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
    # Create threshold mask
    threshold = cle.greater_constant(image, None, threshold_value)
    
    # Compute statistics
    stats_dict = cle.statistics_of_labelled_pixels(image, threshold)
    stats = pd.DataFrame(stats_dict)
    
    if len(stats) > 0:
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
            
        mouse_id = parse_mouse_id(filepath, marker)
        
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
    mouse_averages = df.groupby(["Group", "MouseID"])[numeric_cols].mean().round(3).reset_index()
    
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
