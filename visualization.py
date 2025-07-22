"""Image visualization tools for ND2 analysis."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import logging
from matplotlib.colors import LinearSegmentedColormap

from image_processing import load_nd2_file, parse_mouse_id
from data_models import VisualizationConfig
from config import VISUALIZATION_RANGES, CHANNEL_COLORS, DEFAULT_MARKER, DEFAULT_MARKER_2D

logger = logging.getLogger(__name__)

# Custom colormaps
colors_r = ["black", "red"]
colors_g = ["black", "green"] 
colors_b = ["black", "blue"]

# Create custom colormaps
red_cmap = LinearSegmentedColormap.from_list("custom_red", colors_r)
green_cmap = LinearSegmentedColormap.from_list("custom_green", colors_g)
blue_cmap = LinearSegmentedColormap.from_list("custom_blue", colors_b)

# Channel colormap mapping
CHANNEL_COLORMAPS = {
    'channel_1': green_cmap,  # Green channel
    'channel_2': red_cmap,    # Red channel
    'channel_3': blue_cmap    # Blue channel
}

class ND2Visualizer:
    """Advanced visualization tools for ND2 images."""
    
    def __init__(self, config: VisualizationConfig = None, viz_ranges: Dict = None, pixel_size_um: float = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration object
            viz_ranges: Custom visualization ranges from JSON config
            pixel_size_um: Pixel size in micrometers from JSON config
        """
        self.config = config or VisualizationConfig()
        
        # Use custom ranges if provided, otherwise fall back to default
        if viz_ranges:
            self.viz_ranges = viz_ranges
        else:
            from config import VISUALIZATION_RANGES
            self.viz_ranges = VISUALIZATION_RANGES
            
        # Use custom pixel size if provided, otherwise use default
        self.pixel_size_um = pixel_size_um or 0.222
        
        # Set up matplotlib for better rendering
        plt.style.use('default')
        
    def add_scale_bar(self, ax, image_shape: Tuple[int, int], 
                     scale_bar_um: float = None, 
                     pixel_size_um: float = None) -> None:
        """
        Add a scale bar to an image.
        
        Args:
            ax: Matplotlib axis object
            image_shape: Shape of the image (height, width)
            scale_bar_um: Scale bar size in micrometers
            pixel_size_um: Pixel size in micrometers
        """
        if scale_bar_um is None:
            scale_bar_um = self.config.scale_bar_um
        
        if pixel_size_um is None:
            pixel_size_um = self.pixel_size_um
            
        # Calculate scale bar length in pixels
        scale_bar_pixels = scale_bar_um / pixel_size_um
        
        # Position scale bar in bottom-right corner (slightly more centered)
        height, width = image_shape
        x_start = width - scale_bar_pixels - 30  # Moved slightly more towards center
        y_start = height - 60  # Moved slightly up
        
        # Add scale bar rectangle
        scale_bar = patches.Rectangle(
            (x_start, y_start), scale_bar_pixels, self.config.scale_bar_thickness,
            linewidth=1, edgecolor=self.config.scale_bar_color, 
            facecolor=self.config.scale_bar_color, alpha=0.8
        )
        ax.add_patch(scale_bar)
        
        # Add scale bar text (smaller font)
        ax.text(x_start + scale_bar_pixels/2, y_start+15, f'{scale_bar_um} μm',
               fontsize=3, color=self.config.scale_bar_color, ha='center', va='top',
               weight='bold')

    def visualize_channels(
        self,
        channel_1: np.ndarray,
        channel_2: np.ndarray, 
        channel_3: np.ndarray,
        title: str = None,
        save_path: Optional[str] = None,
        add_scale_bar: bool = True
    ) -> plt.Figure:
        """
        Visualize the three channels of an ND2 image.
        
        Args:
            channel_1: Channel 1 data (green)
            channel_2: Channel 2 data (red)
            channel_3: Channel 3 data (blue)
            title: Title for the figure
            save_path: Path to save the figure
            add_scale_bar: Whether to add scale bar
            
        Returns:
            matplotlib Figure object
        """
        fig, axs = plt.subplots(1, 4, figsize=self.config.figure_size)
        
        # Individual channels (no titles on representative images)
        im1 = axs[0].imshow(channel_1, cmap=green_cmap, 
                           **self.viz_ranges['channel_1'])
        axs[0].axis('off')
        if add_scale_bar:
            self.add_scale_bar(axs[0], channel_1.shape)
        
        im2 = axs[1].imshow(channel_2, cmap=red_cmap,
                           **self.viz_ranges['channel_2'])
        axs[1].axis('off')
        if add_scale_bar:
            self.add_scale_bar(axs[1], channel_2.shape)
        
        im3 = axs[2].imshow(channel_3, cmap=blue_cmap, 
                           **self.viz_ranges['channel_3'])
        axs[2].axis('off')
        if add_scale_bar:
            self.add_scale_bar(axs[2], channel_3.shape)
        
        # RGB composite
        rgb = self._create_rgb_composite(channel_1, channel_2, channel_3)
        axs[3].imshow(rgb)
        axs[3].axis('off')
        if add_scale_bar:
            self.add_scale_bar(axs[3], rgb.shape[:2])
        
        # Only add title if explicitly provided (not for representative images)
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved visualization: {save_path}")
        
        return fig

    def _create_rgb_composite(self, channel_1: np.ndarray, 
                             channel_2: np.ndarray, 
                             channel_3: np.ndarray) -> np.ndarray:
        """
        Create RGB composite image from three channels.
        
        Args:
            channel_1: Channel 1 data (mapped to green)
            channel_2: Channel 2 data (mapped to red)
            channel_3: Channel 3 data (mapped to blue)
            
        Returns:
            RGB composite image
        """
        rgb = np.zeros((channel_1.shape[0], channel_1.shape[1], 3), dtype=np.float32)
        
        # Normalize each channel to [0, 1]
        rgb[:,:,0] = self._normalize_channel(channel_2, self.viz_ranges['channel_2'])  # Red
        rgb[:,:,1] = self._normalize_channel(channel_1, self.viz_ranges['channel_1'])  # Green
        rgb[:,:,2] = self._normalize_channel(channel_3, self.viz_ranges['channel_3'])  # Blue
        
        return np.clip(rgb, 0, 1)

    def _normalize_channel(self, channel: np.ndarray, range_dict: Dict) -> np.ndarray:
        """Normalize channel data to [0, 1] range."""
        vmin, vmax = range_dict['vmin'], range_dict['vmax']
        normalized = (channel - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1)

    def visualize_single_file(
        self,
        filepath: str,
        output_path: str,
        is_3d: bool = True,
        title: str = None
    ) -> bool:
        """
        Visualize a single ND2 file.
        
        Args:
            filepath: Path to ND2 file
            output_path: Path to save visualization
            is_3d: Whether file contains 3D data
            title: Custom title for the image (None for no title)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image data
            channel_1, channel_2, channel_3 = load_nd2_file(filepath, is_3d)
            
            # Create visualization
            fig = self.visualize_channels(
                channel_1, channel_2, channel_3,
                title=title, save_path=output_path
            )
            
            plt.close(fig)  # Free memory
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing file {filepath}: {e}")
            return False

    def create_representative_gallery(
        self,
        input_dir: str,
        output_dir: str,
        representative_images: Dict[str, List[str]],
        is_3d: bool = True
    ) -> Dict[str, List[str]]:
        """
        Create visualization gallery of representative images.
        
        Args:
            input_dir: Directory containing ND2 files
            output_dir: Directory to save visualizations
            representative_images: Dict mapping groups to representative filenames
            is_3d: Whether files contain 3D data
            
        Returns:
            Dictionary mapping groups to created visualization paths
        """
        os.makedirs(output_dir, exist_ok=True)
        gallery_paths = {}
        
        for group, filenames in representative_images.items():
            group_dir = os.path.join(output_dir, f"Group_{group}")
            os.makedirs(group_dir, exist_ok=True)
            
            group_paths = []
            
            for i, filename in enumerate(filenames):
                # Find the full path to the file
                nd2_path = self._find_file_path(input_dir, filename)
                if nd2_path is None:
                    logger.warning(f"Could not find file: {filename}")
                    continue
                
                # Create output path (keep filename in file path, not displayed title)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(group_dir, f"{base_name}_representative_{i+1}.png")
                
                # Create visualization (no titles for representative images)
                success = self.visualize_single_file(nd2_path, output_path, is_3d, title=None)
                
                if success:
                    group_paths.append(output_path)
            
            gallery_paths[group] = group_paths
            logger.info(f"Created {len(group_paths)} representative images for group {group}")
        
        return gallery_paths

    def create_complete_gallery(
        self,
        input_dir: str,
        output_dir: str,
        mouse_lookup: Dict,
        is_3d: bool = True,
        marker: str = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Create complete visualization gallery organized by group and mouse.
        
        Args:
            input_dir: Directory containing ND2 files
            output_dir: Directory to save visualizations
            mouse_lookup: Dictionary mapping mouse IDs to groups
            is_3d: Whether files contain 3D data
            marker: Filename marker for mouse ID extraction
            
        Returns:
            Nested dictionary: {group: {mouse_id: [visualization_paths]}}
        """
        from image_processing import get_nd2_files
        
        os.makedirs(output_dir, exist_ok=True)
        gallery_structure = {}
        
        if marker is None:
            marker = DEFAULT_MARKER if is_3d else DEFAULT_MARKER_2D
        
        # Get all ND2 files
        nd2_files = get_nd2_files(input_dir)
        
        # Organize files by group and mouse
        for filepath in nd2_files:
            try:
                mouse_id = parse_mouse_id(filepath, marker)
                if mouse_id not in mouse_lookup:
                    continue
                    
                group = mouse_lookup[mouse_id]["group"]
                filename = os.path.basename(filepath)
                
                # Create directory structure
                group_dir = os.path.join(output_dir, f"Group_{group}")
                mouse_dir = os.path.join(group_dir, f"Mouse_{mouse_id}")
                os.makedirs(mouse_dir, exist_ok=True)
                
                # Create visualization
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(mouse_dir, f"{base_name}.png")
                
                # No titles for gallery images - just filename preserved in path
                success = self.visualize_single_file(filepath, output_path, is_3d, title=None)
                
                if success:
                    # Update gallery structure
                    if group not in gallery_structure:
                        gallery_structure[group] = {}
                    if mouse_id not in gallery_structure[group]:
                        gallery_structure[group][mouse_id] = []
                    
                    gallery_structure[group][mouse_id].append(output_path)
                    
            except Exception as e:
                logger.warning(f"Could not process file {filepath}: {e}")
                continue
        
        # Log summary
        total_images = sum(
            len(mouse_files) 
            for group_data in gallery_structure.values() 
            for mouse_files in group_data.values()
        )
        logger.info(f"Created complete gallery with {total_images} visualizations")
        
        return gallery_structure

    def _find_file_path(self, input_dir: str, filename: str) -> Optional[str]:
        """Find the full path to a file given its filename."""
        from image_processing import get_nd2_files
        
        nd2_files = get_nd2_files(input_dir)
        for filepath in nd2_files:
            if os.path.basename(filepath) == filename:
                return filepath
        return None

    def plot_group_comparisons(
        self,
        df: pd.DataFrame,
        metrics: List[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create plots comparing groups for different metrics based on mouse averages.
        
        Args:
            df: DataFrame with processed results
            metrics: List of metrics to plot (if None, use common metrics)
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping metric names to figure objects
        """
        if metrics is None:
            # Default metrics to plot
            metrics = [
                'Channel_1_area', 'Channel_2_area', 'Channel_3_area',
                'Channel_2_per_Channel_3_area', 'Channel_1_per_Channel_3_area',
                'Channel_1_mean_intensity', 'Channel_2_mean_intensity', 'Channel_3_mean_intensity'
            ]
        
        # Calculate mouse averages first
        numeric_cols = [col for col in df.columns 
                       if col not in ["Group", "MouseID", "Filename"] 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        mouse_averages = df.groupby(["Group", "MouseID"])[numeric_cols].mean().reset_index()
        
        figures = {}
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for metric in metrics:
            if metric not in mouse_averages.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue
                
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create boxplot using mouse averages
            sns.boxplot(
                data=mouse_averages,
                x='Group',
                y=metric,
                palette='Set1',
                ax=ax
            )
            
            # Add individual mouse data points
            sns.stripplot(
                data=mouse_averages,
                x='Group',
                y=metric,
                color='black',
                alpha=0.7,
                size=6,
                ax=ax
            )
            
            # Format plot
            ax.set_title(f"{metric} by Treatment Group (Mouse Averages)", fontsize=14, pad=20)
            ax.set_xlabel('Treatment Group', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save if output directory provided
            if output_dir:
                safe_metric_name = metric.replace('/', '_per_').replace(' ', '_')
                save_path = os.path.join(output_dir, f"{safe_metric_name}_comparison.png")
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                logger.info(f"Saved comparison plot: {save_path}")
                
            figures[metric] = fig
            
        return figures
