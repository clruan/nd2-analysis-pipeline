"""Data models for interactive threshold analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class ThresholdData:
    """Stores threshold analysis data for a single image."""
    mouse_id: str
    group: str
    filename: str
    
    # Arrays of positive pixel percentages for each threshold (0-4095)
    channel_1_percentages: np.ndarray  # Shape: (4096,)
    channel_2_percentages: np.ndarray  # Shape: (4096,)
    channel_3_percentages: np.ndarray  # Shape: (4096,)
    
    def get_percentage_at_threshold(self, channel: int, threshold: int) -> float:
        """Get positive pixel percentage for specific channel and threshold."""
        if not 0 <= threshold <= 4095:
            raise ValueError(f"Threshold must be 0-4095, got {threshold}")
        
        if channel == 1:
            return float(self.channel_1_percentages[threshold])
        elif channel == 2:
            return float(self.channel_2_percentages[threshold])
        elif channel == 3:
            return float(self.channel_3_percentages[threshold])
        else:
            raise ValueError(f"Channel must be 1, 2, or 3, got {channel}")

@dataclass
class ThresholdResults:
    """Container for all threshold analysis results from a study."""
    study_name: str
    image_data: List[ThresholdData]
    group_info: Dict[str, List[str]]  # group_name -> list of mouse_ids
    
    def get_mouse_averages(self, thresholds: Dict[str, int]) -> pd.DataFrame:
        """Calculate mouse averages for given thresholds."""
        results = []
        
        for data in self.image_data:
            mouse_avg = {
                'Group': data.group,
                'MouseID': data.mouse_id,
                'Channel_1_area': data.get_percentage_at_threshold(1, thresholds['channel_1']),
                'Channel_2_area': data.get_percentage_at_threshold(2, thresholds['channel_2']),
                'Channel_3_area': data.get_percentage_at_threshold(3, thresholds['channel_3'])
            }
            results.append(mouse_avg)
        
        df = pd.DataFrame(results)
        # Average by mouse (in case multiple images per mouse)
        return df.groupby(['Group', 'MouseID']).mean().reset_index()

