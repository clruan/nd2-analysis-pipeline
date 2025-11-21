"""Main processing pipeline for ND2 image analysis."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import logging
import re
from joblib import Parallel, delayed

from config import (DEFAULT_THRESHOLDS, DEFAULT_PARALLEL_JOBS, DEFAULT_GROUPS, 
                   OUTPUT_FILES, REPRESENTATIVE_SELECTION)
from data_models import GroupConfig, ProcessingResults, VisualizationConfig
from image_processing import (get_nd2_files, process_single_file, 
                             calculate_group_statistics, identify_representative_images,
                             detect_pixel_size)
from excel_output import ExcelReporter
from visualization import ND2Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ND2Pipeline:
    """Main pipeline for ND2 image analysis."""
    
    def __init__(self, config: GroupConfig = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Group configuration object
        """
        self.config = config or GroupConfig(groups=DEFAULT_GROUPS)
        self.mouse_lookup = self.config.build_mouse_info()
        self.excel_reporter = ExcelReporter()
        
    def process_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        is_3d: bool = True,
        n_jobs: int = DEFAULT_PARALLEL_JOBS,
        marker: str = None,
        scale_bar_um: float = 50,
        create_visualizations: bool = True,
        viz_ranges: dict = None,
        pixel_size_um: float = None
    ) -> ProcessingResults:
        """
        Process all ND2 files in a directory.
        
        Args:
            input_dir: Directory containing ND2 files
            output_dir: Directory for output files
            is_3d: Whether files contain 3D data
            n_jobs: Number of parallel jobs
            marker: Filename marker for mouse ID extraction
            scale_bar_um: Scale bar size for visualizations
            create_visualizations: Whether to create visualization gallery
            viz_ranges: Custom visualization ranges from JSON config
            pixel_size_um: Pixel size in micrometers from JSON config
            
        Returns:
            ProcessingResults object with all analysis results
        """
        start_time = time.time()
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(input_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        log_path = os.path.join(output_dir, OUTPUT_FILES['log_file'])
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Starting ND2 analysis pipeline")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Processing mode: {'3D' if is_3d else '2D'}")
        logger.info(f"Parallel jobs: {n_jobs}")
        
        # Find ND2 files
        nd2_files = get_nd2_files(input_dir)
        if not nd2_files:
            raise ValueError(f"No ND2 files found in {input_dir}")
        
        logger.info(f"Found {len(nd2_files)} ND2 files")

        effective_pixel_size = pixel_size_um or self.config.pixel_size_um
        if effective_pixel_size is None:
            detected_pixel_size = detect_pixel_size(nd2_files[0])
            if detected_pixel_size:
                effective_pixel_size = detected_pixel_size
                logger.info(f"Detected pixel size from {os.path.basename(nd2_files[0])}: {detected_pixel_size} µm/pixel")
            else:
                logger.warning("Pixel size not provided and could not be detected; scale bars may be inaccurate.")
        
        # Get thresholds
        thresholds = self._get_thresholds(is_3d)
        logger.info(f"Using thresholds: {thresholds}")
        
        # Process files in parallel
        logger.info(f"Starting parallel processing with {n_jobs} workers")
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_file)(
                filepath, self.mouse_lookup, thresholds, is_3d, marker
            ) for filepath in nd2_files
        )
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            raise ValueError("No files were processed successfully")
        
        logger.info(f"Successfully processed {len(valid_results)}/{len(nd2_files)} files")
        
        # Create DataFrames
        raw_data = pd.DataFrame([result.to_dict() for result in valid_results])
        column_metadata: Dict[str, Any] = {}
        flexible_config = getattr(self, 'flexible_config', None)
        if flexible_config is not None:
            column_metadata = self._apply_flexible_annotations(raw_data, flexible_config)
        group_summary = calculate_group_statistics(raw_data)
        
        # Identify representative images
        representative_images = identify_representative_images(
            raw_data, 
            metric=REPRESENTATIVE_SELECTION['metric'],
            top_n=REPRESENTATIVE_SELECTION['top_n']
        )
        
        # Processing statistics
        processing_stats = {
            'total_files_found': len(nd2_files),
            'files_processed_successfully': len(valid_results),
            'processing_time_seconds': round(time.time() - start_time, 2),
            'groups_analyzed': len(raw_data['Group'].unique()),
            'mice_analyzed': len(raw_data['MouseID'].unique()),
            'dimension': '3D' if is_3d else '2D',
            'thresholds_used': thresholds,
            'parallel_jobs': n_jobs,
            'pixel_size_um': effective_pixel_size
        }
        
        # Create results object
        results = ProcessingResults(
            raw_data=raw_data,
            group_summary=group_summary,
            representative_images=representative_images,
            processing_stats=processing_stats,
            column_metadata=column_metadata
        )
        
        # Save results
        self._save_results(results, output_dir)
        
        # Create visualizations
        if create_visualizations:
            self._create_visualizations(
                results, input_dir, output_dir, is_3d, scale_bar_um, viz_ranges, effective_pixel_size
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        
        return results
    
    def _get_thresholds(self, is_3d: bool) -> Dict[str, float]:
        """Get threshold values for processing."""
        dimension = '3d' if is_3d else '2d'
        
        if self.config.thresholds and dimension in self.config.thresholds:
            return self.config.thresholds[dimension]
        else:
            return DEFAULT_THRESHOLDS[dimension]

    @staticmethod
    def _legacy_channel_prefix(internal_id: str) -> str:
        """Convert flexible channel ID to legacy DataFrame prefix."""
        if internal_id.startswith('channel_'):
            return f"Channel_{internal_id.split('_')[1]}"
        return internal_id

    @staticmethod
    def _format_ratio_suffix(metric_suffix: str) -> str:
        """Convert metric suffix to a readable label."""
        suffix_map = {
            'area': 'Area (%)',
            'mean_intensity': 'Mean Intensity Ratio',
            'sum_intensity': 'Total Intensity Ratio',
            'min_intensity': 'Minimum Intensity Ratio',
            'max_intensity': 'Maximum Intensity Ratio'
        }
        return suffix_map.get(metric_suffix, metric_suffix.replace('_', ' ').title())

    def _apply_flexible_annotations(self, raw_data: pd.DataFrame, flexible_config) -> Dict[str, Any]:
        """Augment raw data with flexible ratios and prepare display metadata."""
        if raw_data.empty:
            return {}

        rename_map: Dict[str, str] = {}
        ratio_records: List[Dict[str, str]] = []
        channel_lookup: Dict[str, str] = {}

        ordered_channels = flexible_config.get_ordered_channels()

        # Base channel labels
        for channel in ordered_channels:
            prefix = self._legacy_channel_prefix(channel.internal_id)
            channel_lookup[prefix] = channel.name

            labeled_columns = [
                (f"{prefix}_area", f"{channel.name} Area (%)"),
                (f"{prefix}_mean_intensity", f"{channel.name} Mean Intensity"),
                (f"{prefix}_sum_intensity", f"{channel.name} Total Intensity"),
                (f"{prefix}_min_intensity", f"{channel.name} Minimum Intensity"),
                (f"{prefix}_max_intensity", f"{channel.name} Maximum Intensity")
            ]
            for column_name, display_label in labeled_columns:
                if column_name in raw_data.columns:
                    rename_map[column_name] = display_label

        # Ratio definitions from configuration
        ratio_name_lookup: Dict[tuple, str] = {}
        for ratio in flexible_config.ratios:
            numerator_prefix = self._legacy_channel_prefix(ratio.numerator_channel)
            denominator_prefix = self._legacy_channel_prefix(ratio.denominator_channel)
            ratio_name_lookup[(numerator_prefix, denominator_prefix)] = ratio.name

            numerator_area = f"{numerator_prefix}_area"
            denominator_area = f"{denominator_prefix}_area"
            area_ratio_column = f"{numerator_prefix}_per_{denominator_prefix}_area"

            if (
                area_ratio_column not in raw_data.columns
                and numerator_area in raw_data.columns
                and denominator_area in raw_data.columns
            ):
                denominator_values = raw_data[denominator_area]
                raw_data[area_ratio_column] = np.where(
                    denominator_values > 0,
                    raw_data[numerator_area] / denominator_values * 100.0,
                    np.nan
                )

            if area_ratio_column in raw_data.columns:
                rename_map[area_ratio_column] = f"{ratio.name} (Area %)"
                ratio_records.append({
                    'column': area_ratio_column,
                    'label': rename_map[area_ratio_column]
                })

            # Additional intensity ratios
            for suffix, descriptor in [
                ('_mean_intensity', ' Mean Intensity Ratio'),
                ('_sum_intensity', ' Total Intensity Ratio')
            ]:
                numerator_metric = f"{numerator_prefix}{suffix}"
                denominator_metric = f"{denominator_prefix}{suffix}"
                ratio_metric_column = f"{numerator_prefix}_per_{denominator_prefix}{suffix}"

                if (
                    ratio_metric_column not in raw_data.columns
                    and numerator_metric in raw_data.columns
                    and denominator_metric in raw_data.columns
                ):
                    denominator_values = raw_data[denominator_metric]
                    raw_data[ratio_metric_column] = np.where(
                        denominator_values != 0,
                        raw_data[numerator_metric] / denominator_values,
                        np.nan
                    )

                if ratio_metric_column in raw_data.columns:
                    rename_map[ratio_metric_column] = f"{ratio.name}{descriptor}"
                    ratio_records.append({
                        'column': ratio_metric_column,
                        'label': rename_map[ratio_metric_column]
                    })

        # Provide readable names for any remaining ratio columns
        ratio_pattern = re.compile(r'^(Channel_\d+)_per_(Channel_\d+)_([A-Za-z_]+)$')
        for column_name in [c for c in raw_data.columns if '_per_' in c]:
            if column_name in rename_map:
                continue

            match = ratio_pattern.match(column_name)
            if not match:
                continue

            numerator_prefix, denominator_prefix, metric_suffix = match.groups()
            numerator_name = channel_lookup.get(numerator_prefix, numerator_prefix)
            denominator_name = channel_lookup.get(denominator_prefix, denominator_prefix)

            base_label = ratio_name_lookup.get(
                (numerator_prefix, denominator_prefix),
                f"{numerator_name} per {denominator_name}"
            )
            metric_label = self._format_ratio_suffix(metric_suffix)
            rename_map[column_name] = f"{base_label} {metric_label}"
            ratio_records.append({
                'column': column_name,
                'label': rename_map[column_name]
            })

        return {
            'display_names': rename_map,
            'channels': [
                {
                    'internal_id': channel.internal_id,
                    'name': channel.name,
                    'color': channel.color_name,
                    'display_order': channel.display_order
                }
                for channel in ordered_channels
            ],
            'ratio_columns': ratio_records
        }
    
    def _save_results(self, results: ProcessingResults, output_dir: str) -> None:
        """Save analysis results to files."""
        # Save to Excel
        excel_path = os.path.join(output_dir, OUTPUT_FILES['excel_report'])
        try:
            # If flexible configuration is available on the pipeline config, prefer enriched report
            success = False
            flexible_config = getattr(self, 'flexible_config', None)
            if flexible_config is not None and hasattr(self.excel_reporter, 'create_flexible_report'):
                success = self.excel_reporter.create_flexible_report(results, flexible_config, excel_path)
            else:
                success = self.excel_reporter.create_simple_report(results, excel_path)
            if success:
                logger.info(f"Excel report saved: {excel_path}")
            else:
                logger.warning("Excel report creation failed, falling back to CSV")
                self._save_csv_fallback(results, output_dir)
        except Exception as e:
            logger.warning(f"Excel export failed: {e}. Saving as CSV.")
            self._save_csv_fallback(results, output_dir)
        
        # Save JSON data
        json_path = os.path.join(output_dir, OUTPUT_FILES['processed_data'])
        results.save_to_json(json_path)
        logger.info(f"JSON data saved: {json_path}")
    
    def _save_csv_fallback(self, results: ProcessingResults, output_dir: str) -> None:
        """Save results as CSV files if Excel fails."""
        csv_files = self.excel_reporter.export_to_csv(results, output_dir)
        for name, path in csv_files.items():
            logger.info(f"CSV file saved: {path}")
    
    def _create_visualizations(
        self,
        results: ProcessingResults,
        input_dir: str,
        output_dir: str,
        is_3d: bool,
        scale_bar_um: float,
        viz_ranges: dict = None,
        pixel_size_um: float = None
    ) -> None:
        """Create visualization galleries."""
        vis_config = VisualizationConfig(scale_bar_um=scale_bar_um)
        visualizer = ND2Visualizer(vis_config, viz_ranges, pixel_size_um)
        
        # Create representative images gallery
        repr_dir = os.path.join(output_dir, "representative_images")
        gallery_paths = visualizer.create_representative_gallery(
            input_dir, repr_dir, results.representative_images, is_3d
        )
        
        logger.info(f"Representative images gallery created: {repr_dir}")
        
        # Create comparison plots
        plots_dir = os.path.join(output_dir, "comparison_plots")
        comparison_figures = visualizer.plot_group_comparisons(
            results.raw_data, output_dir=plots_dir
        )
        
        logger.info(f"Comparison plots created: {plots_dir}")
        
        # Close figures to free memory
        import matplotlib.pyplot as plt
        for fig in comparison_figures.values():
            plt.close(fig)

def create_sample_config(output_path: str) -> None:
    """Create a sample configuration file."""
    sample_config = GroupConfig(
        groups=DEFAULT_GROUPS,
        thresholds=DEFAULT_THRESHOLDS
    )
    sample_config.to_json(output_path)
    logger.info(f"Sample configuration created: {output_path}")

# Standalone visualization functions
def visualize_single_file(
    filepath: str,
    output_path: str,
    is_3d: bool = True,
    scale_bar_um: float = 50,
    viz_ranges: dict = None,
    pixel_size_um: float = None
) -> bool:
    """
    Visualize a single ND2 file.
    
    Args:
        filepath: Path to ND2 file
        output_path: Path to save visualization
        is_3d: Whether file contains 3D data
        scale_bar_um: Scale bar size in micrometers
        viz_ranges: Custom visualization ranges
        pixel_size_um: Pixel size in micrometers
        
    Returns:
        True if successful, False otherwise
    """
    vis_config = VisualizationConfig(scale_bar_um=scale_bar_um)
    visualizer = ND2Visualizer(vis_config, viz_ranges, pixel_size_um)
    
    return visualizer.visualize_single_file(filepath, output_path, is_3d)

def create_complete_gallery(
    input_dir: str,
    output_dir: str,
    config: GroupConfig,
    is_3d: bool = True,
    scale_bar_um: float = 50,
    viz_ranges: dict = None,
    pixel_size_um: float = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create complete visualization gallery organized by group and mouse.
    
    Args:
        input_dir: Directory containing ND2 files
        output_dir: Directory to save visualizations
        config: Group configuration
        is_3d: Whether files contain 3D data
        scale_bar_um: Scale bar size in micrometers
        viz_ranges: Custom visualization ranges
        pixel_size_um: Pixel size in micrometers
        
    Returns:
        Gallery structure dictionary
    """
    vis_config = VisualizationConfig(scale_bar_um=scale_bar_um)
    visualizer = ND2Visualizer(vis_config, viz_ranges, pixel_size_um)
    
    mouse_lookup = config.build_mouse_info()
    
    return visualizer.create_complete_gallery(
        input_dir, output_dir, mouse_lookup, is_3d
    )
