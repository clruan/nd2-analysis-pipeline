"""Data structures and models for ND2 image analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path

@dataclass
class GroupConfig:
    """Configuration for treatment groups."""
    groups: Dict[str, List[str]]
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    pixel_size_um: Optional[float] = None  # Pixel size in micrometers
    
    def build_mouse_info(self) -> Dict[str, Dict]:
        """Build a unified dictionary of mouse information."""
        lookup = {}
        for group_name, mouse_list in self.groups.items():
            for mouse_id in mouse_list:
                if mouse_id in lookup:
                    raise ValueError(f"Mouse ID {mouse_id} is duplicated!")
                lookup[mouse_id] = {"group": group_name}
        return lookup
    
    @classmethod
    def from_json(cls, json_path: str) -> 'GroupConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(
            groups=data.get('groups', {}),
            thresholds=data.get('thresholds', None),
            pixel_size_um=data.get('pixel_size_um', None)
        )
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        data = {
            'groups': self.groups,
            'thresholds': self.thresholds,
            'pixel_size_um': self.pixel_size_um
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

@dataclass
class ChannelData:
    """Stores data for a single channel."""
    raw: np.ndarray
    threshold: Optional[np.ndarray] = None
    stats: Optional[pd.DataFrame] = None
    threshold_value: Optional[float] = None

@dataclass
class ImageMetrics:
    """Container for all metrics from a single image."""
    group: str
    mouse_id: str
    filename: str
    
    # Channel areas (as percentages)
    channel_1_area: float
    channel_2_area: float
    channel_3_area: float
    
    # Channel ratios (as percentages)
    channel_2_per_channel_3_area: float
    channel_1_per_channel_3_area: float
    
    # Intensity metrics
    channel_1_mean_intensity: float
    channel_2_mean_intensity: float
    channel_3_mean_intensity: float
    
    channel_1_sum_intensity: float
    channel_2_sum_intensity: float
    channel_3_sum_intensity: float
    
    # Intensity ratios
    channel_2_per_channel_3_mean_intensity: float
    channel_1_per_channel_3_mean_intensity: float
    channel_2_per_channel_3_sum_intensity: float
    channel_1_per_channel_3_sum_intensity: float
    
    # Min/Max intensities
    channel_1_min_intensity: float
    channel_2_min_intensity: float
    channel_3_min_intensity: float
    
    channel_1_max_intensity: float
    channel_2_max_intensity: float
    channel_3_max_intensity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'Group': self.group,
            'MouseID': self.mouse_id,
            'Filename': self.filename,
            'Channel_1_area': self.channel_1_area,
            'Channel_2_area': self.channel_2_area,
            'Channel_3_area': self.channel_3_area,
            'Channel_2_per_Channel_3_area': self.channel_2_per_channel_3_area,
            'Channel_1_per_Channel_3_area': self.channel_1_per_channel_3_area,
            'Channel_1_mean_intensity': self.channel_1_mean_intensity,
            'Channel_2_mean_intensity': self.channel_2_mean_intensity,
            'Channel_3_mean_intensity': self.channel_3_mean_intensity,
            'Channel_1_sum_intensity': self.channel_1_sum_intensity,
            'Channel_2_sum_intensity': self.channel_2_sum_intensity,
            'Channel_3_sum_intensity': self.channel_3_sum_intensity,
            'Channel_2_per_Channel_3_mean_intensity': self.channel_2_per_channel_3_mean_intensity,
            'Channel_1_per_Channel_3_mean_intensity': self.channel_1_per_channel_3_mean_intensity,
            'Channel_2_per_Channel_3_sum_intensity': self.channel_2_per_channel_3_sum_intensity,
            'Channel_1_per_Channel_3_sum_intensity': self.channel_1_per_channel_3_sum_intensity,
            'Channel_1_min_intensity': self.channel_1_min_intensity,
            'Channel_2_min_intensity': self.channel_2_min_intensity,
            'Channel_3_min_intensity': self.channel_3_min_intensity,
            'Channel_1_max_intensity': self.channel_1_max_intensity,
            'Channel_2_max_intensity': self.channel_2_max_intensity,
            'Channel_3_max_intensity': self.channel_3_max_intensity
        }

@dataclass
class ProcessingResults:
    """Container for all processing results."""
    raw_data: pd.DataFrame
    group_summary: pd.DataFrame
    representative_images: Dict[str, List[str]]
    processing_stats: Dict[str, Any]
    
    def save_to_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        data = {
            'raw_data': self.raw_data.to_dict('records'),
            'group_summary': self.group_summary.to_dict('records'),
            'representative_images': self.representative_images,
            'processing_stats': self.processing_stats
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ProcessingResults':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            raw_data=pd.DataFrame(data['raw_data']),
            group_summary=pd.DataFrame(data['group_summary']),
            representative_images=data['representative_images'],
            processing_stats=data['processing_stats']
        )

@dataclass
class VisualizationConfig:
    """Configuration for image visualization."""
    scale_bar_um: float = 50
    scale_bar_color: str = 'white'
    scale_bar_thickness: int = 3
    figure_size: tuple = (16, 4)
    dpi: int = 300
    colormap_1: str = 'Greens'
    colormap_2: str = 'Reds'
    colormap_3: str = 'Blues'
    
    def get_channel_colormap(self, channel: int) -> str:
        """Get colormap for specific channel."""
        colormaps = {
            1: self.colormap_1,
            2: self.colormap_2,
            3: self.colormap_3
        }
        return colormaps.get(channel, 'gray')
