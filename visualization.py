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
        Add a scale bar to an image with adaptive placement.
        
        Args:
            ax: Matplotlib axis object
            image_shape: Shape of the image (height, width)
            scale_bar_um: Scale bar size in micrometers
            pixel_size_um: Pixel size in micrometers
        """
        requested_um = scale_bar_um if scale_bar_um is not None else self.config.scale_bar_um
        
        if pixel_size_um is None or pixel_size_um <= 0:
            pixel_size_um = self.pixel_size_um
        
        height, width = image_shape
        if height == 0 or width == 0:
            return
        
        scale_bar_pixels = max(1, int(round(requested_um / pixel_size_um)))
        
        margin_x = max(12, int(width * 0.05))
        margin_y = max(18, int(height * 0.08))
        max_bar_pixels = max(5, width - margin_x - 5)
        if scale_bar_pixels > max_bar_pixels:
            scale_bar_pixels = max_bar_pixels
            effective_scale_um = round(scale_bar_pixels * pixel_size_um, 2)
        else:
            effective_scale_um = requested_um
        
        # Set bar geometry
        bar_height = max(self.config.scale_bar_thickness, int(max(height, width) * 0.01))
        x_start = width - margin_x - scale_bar_pixels
        y_start = height - margin_y - bar_height
        x_start = max(5, x_start)
        y_start = max(5, y_start)
        
        scale_bar = patches.Rectangle(
            (x_start, y_start), scale_bar_pixels, bar_height,
            linewidth=0, edgecolor=self.config.scale_bar_color, 
            facecolor=self.config.scale_bar_color, alpha=0.9
        )
        ax.add_patch(scale_bar)
        
        # Label below the bar, stay inside frame
        label_offset = max(8, int(max(height, width) * 0.024))
        label_y = min(height - 8, y_start + bar_height + label_offset)
        font_size_override = getattr(self.config, 'scale_bar_font_size', None)
        if font_size_override is not None:
            font_size = font_size_override
        else:
            font_size = max(7, min(16, int(min(height, width) * 0.02)))
        ax.text(
            x_start + scale_bar_pixels / 2,
            label_y,
            f'{effective_scale_um:g} μm',
            fontsize=font_size,
            color=self.config.scale_bar_color,
            ha='center',
            va='top',
            weight='bold'
        )

    def _resolve_panel_order(self, panel_order: Optional[List[str]]) -> List[str]:
        """Validate and normalize requested panel ordering."""
        default_order = ['channel_1', 'channel_2', 'channel_3', 'composite']
        if not panel_order:
            return default_order
        normalized = []
        for panel in panel_order:
            if panel in {'channel_1', 'channel_2', 'channel_3', 'composite'} and panel not in normalized:
                normalized.append(panel)
        return normalized or default_order

    def _resolve_channel_ranges(self, overrides: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Merge override ranges with defaults."""
        resolved = {
            'channel_1': dict(self.viz_ranges.get('channel_1', {'vmin': 0, 'vmax': 4095})),
            'channel_2': dict(self.viz_ranges.get('channel_2', {'vmin': 0, 'vmax': 4095})),
            'channel_3': dict(self.viz_ranges.get('channel_3', {'vmin': 0, 'vmax': 4095}))
        }
        if overrides:
            for channel, values in overrides.items():
                if channel not in resolved or not isinstance(values, dict):
                    continue
                vmin = values.get('vmin', resolved[channel]['vmin'])
                vmax = values.get('vmax', resolved[channel]['vmax'])
                if vmin >= vmax:
                    vmax = vmin + 1e-3
                resolved[channel]['vmin'] = float(vmin)
                resolved[channel]['vmax'] = float(vmax)
        return resolved

    def _render_panel(
        self,
        ax,
        panel_id: str,
        channels: Dict[str, np.ndarray],
        resolved_ranges: Dict[str, Dict[str, float]],
        add_scale_bar: bool,
        composite_cache: Dict[str, np.ndarray]
    ) -> None:
        """Render a single panel into the provided axis."""
        ax.axis('off')
        if panel_id == 'composite':
            if 'composite' not in composite_cache:
                composite_cache['composite'] = self._create_rgb_composite(
                    channels['channel_1'], channels['channel_2'], channels['channel_3']
                )
            ax.imshow(composite_cache['composite'])
            if add_scale_bar:
                self.add_scale_bar(ax, composite_cache['composite'].shape[:2])
            return
        channel_data = channels.get(panel_id)
        cmap = CHANNEL_COLORMAPS.get(panel_id, 'gray')
        if channel_data is None:
            ax.imshow(np.zeros((1, 1)))
            return
        ax.imshow(channel_data, cmap=cmap, **resolved_ranges.get(panel_id, {}))
        if add_scale_bar:
            self.add_scale_bar(ax, channel_data.shape)

    def visualize_channels(
        self,
        channel_1: np.ndarray,
        channel_2: np.ndarray, 
        channel_3: np.ndarray,
        title: str = None,
        save_path: Optional[str] = None,
        add_scale_bar: bool = True,
        panel_order: Optional[List[str]] = None,
        channel_ranges: Optional[Dict[str, Dict[str, float]]] = None
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
        order = self._resolve_panel_order(panel_order)
        resolved_ranges = self._resolve_channel_ranges(channel_ranges)
        base_width, base_height = self.config.figure_size
        width_scale = max(len(order) / 4, 0.5)
        fig_width = max(4, base_width * width_scale)
        fig, axs = plt.subplots(1, len(order), figsize=(fig_width, base_height))
        if len(order) == 1:
            axs = [axs]
        
        channel_map = {
            'channel_1': channel_1,
            'channel_2': channel_2,
            'channel_3': channel_3
        }
        composite_cache: Dict[str, np.ndarray] = {}
        for axis, panel_id in zip(axs, order):
            self._render_panel(axis, panel_id, channel_map, resolved_ranges, add_scale_bar, composite_cache)
        
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
        title: str = None,
        panel_order: Optional[List[str]] = None,
        channel_ranges: Optional[Dict[str, Dict[str, float]]] = None,
        extra_outputs: Optional[Dict[str, str]] = None
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
            channel_map = {
                'channel_1': channel_1,
                'channel_2': channel_2,
                'channel_3': channel_3
            }
            resolved_ranges = self._resolve_channel_ranges(channel_ranges)
            
            # Create visualization
            fig = self.visualize_channels(
                channel_1, channel_2, channel_3,
                title=title, save_path=output_path,
                panel_order=panel_order,
                channel_ranges=channel_ranges
            )
            if extra_outputs:
                composite_cache: Dict[str, np.ndarray] = {}
                for variant, path in extra_outputs.items():
                    if variant not in {'channel_1', 'channel_2', 'channel_3', 'composite'}:
                        continue
                    self._save_single_panel(
                        channel_map,
                        variant,
                        path,
                        resolved_ranges,
                        composite_cache,
                        add_scale_bar=True
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

    def _save_single_panel(
        self,
        channels: Dict[str, np.ndarray],
        variant: str,
        save_path: str,
        resolved_ranges: Dict[str, Dict[str, float]],
        composite_cache: Dict[str, np.ndarray],
        add_scale_bar: bool = True
    ) -> None:
        """Save a single panel variant to disk."""
        fig, ax = plt.subplots(1, 1, figsize=(max(4, self.config.figure_size[0] / 4), self.config.figure_size[1]))
        self._render_panel(ax, variant, channels, resolved_ranges, add_scale_bar, composite_cache)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

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
                hue='Group',
                palette='Set1',
                legend=False,
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
