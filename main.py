"""Main entry point for ND2 image analysis."""

import argparse
import logging
import sys
import os
import json
from typing import Dict, List
from pathlib import Path

from data_models import GroupConfig
from processing_pipeline import ND2Pipeline, create_sample_config
from config import (DEFAULT_PARALLEL_JOBS, DEFAULT_SCALE_BAR_UM, 
                   DEFAULT_MARKER, DEFAULT_MARKER_2D, DEFAULT_GROUPS)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ND2 Image Analysis Pipeline - Professional analysis of multi-channel microscopy images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python main.py --input "data/" --output "results/"
  
  # Advanced analysis with custom settings
  python main.py --input "data/" --output "results/" --dimension 3d --jobs 8 --config config.json --scale-bar 100
  
  # Generate sample configuration
  python main.py --create-config sample_config.json
        """
    )
    
    parser.add_argument('--input', '-i', 
                       help='Directory containing ND2 files')
    
    parser.add_argument('--output', '-o', 
                       help='Output directory for results (default: [input]/results)')
    
    parser.add_argument('--config', '-c', 
                       help='Path to JSON configuration file with group definitions')
    
    parser.add_argument('--dimension', '-d', choices=['2d', '3d'], default='3d',
                       help='Dimension of the data (default: 3d)')
                       
    parser.add_argument('--jobs', '-j', type=int, default=DEFAULT_PARALLEL_JOBS,
                       help=f'Number of parallel jobs (default: {DEFAULT_PARALLEL_JOBS})')
                       
    parser.add_argument('--scale-bar', '-s', type=float, default=DEFAULT_SCALE_BAR_UM,
                       help=f'Scale bar size in micrometers (default: {DEFAULT_SCALE_BAR_UM})')

    parser.add_argument('--pixel-size-um', type=float, dest='pixel_size_um',
                       help='Pixel size in micrometers per pixel. Overrides config detection.')
                       
    parser.add_argument('--marker', '-m', 
                       help=f'Filename marker for mouse ID extraction (default: C1 for 3D, 20X for 2D)')
    
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization creation to speed up processing')
                       
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--create-config', 
                       help='Create a sample configuration file and exit')
                       
    return parser.parse_args()

def setup_logging(verbose: bool = False, output_dir: str = None):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress excessive logging from other libraries
    logging.getLogger('nd2reader').setLevel(logging.WARNING)
    logging.getLogger('joblib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def load_config(config_file: str = None) -> Dict:
    """
    Load complete configuration from file or use defaults.
    
    Args:
        config_file: Path to JSON config file
        
    Returns:
        Dictionary with GroupConfig and additional config data
    """
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            group_config = GroupConfig(
                groups=config_data.get('groups', DEFAULT_GROUPS),
                thresholds=config_data.get('thresholds', None),
                pixel_size_um=config_data.get('pixel_size_um', None)
            )
            
            return {
                'group_config': group_config,
                'visualization_ranges': config_data.get('VISUALIZATION_RANGES', None),
                'pixel_size_um': config_data.get('pixel_size_um', None),
                'raw_config': config_data
            }
        except Exception as e:
            logging.warning(f"Error loading config file {config_file}: {e}")
            logging.info("Using default configuration")
    
    # Use default configuration
    return {
        'group_config': GroupConfig(groups=DEFAULT_GROUPS),
        'visualization_ranges': None,
        'pixel_size_um': None,
        'raw_config': {}
    }

def validate_arguments(args) -> None:
    """Validate command line arguments."""
    if args.create_config:
        return  # Skip validation for config creation
    
    if not args.input:
        raise ValueError("Input directory is required. Use --input or -i")
    
    if not os.path.exists(args.input):
        raise ValueError(f"Input directory does not exist: {args.input}")
    
    if not os.path.isdir(args.input):
        raise ValueError(f"Input path is not a directory: {args.input}")
    
    if args.jobs < 1:
        raise ValueError("Number of jobs must be at least 1")
    
    if args.scale_bar <= 0:
        raise ValueError("Scale bar size must be positive")
    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        raise ValueError("Pixel size must be positive when provided")

def print_summary(results, args):
    """Print a summary of the analysis results."""
    stats = results.processing_stats
    
    print("\n" + "="*60)
    print("ND2 ANALYSIS PIPELINE - SUMMARY")
    print("="*60)
    print(f"Total files found:       {stats['total_files_found']}")
    print(f"Files processed:         {stats['files_processed_successfully']}")
    print(f"Groups analyzed:         {stats['groups_analyzed']}")
    print(f"Mice analyzed:          {stats['mice_analyzed']}")
    print(f"Processing time:         {stats['processing_time_seconds']} seconds")
    print(f"Dimension:              {stats['dimension']}")
    print(f"Parallel jobs:          {stats['parallel_jobs']}")
    pixel_size = stats.get('pixel_size_um')
    if pixel_size:
        print(f"Pixel size:             {pixel_size} µm/pixel")
    
    print(f"\nThresholds used:")
    for channel, value in stats['thresholds_used'].items():
        print(f"  {channel}: {value}")
    
    print(f"\nRepresentative images per group:")
    for group, files in results.representative_images.items():
        print(f"  {group}: {len(files)} images")
    
    print(f"\nOutput files:")
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.input, "results")
    
    print(f"  Excel report:     {os.path.join(output_dir, 'analysis_results.xlsx')}")
    print(f"  JSON data:        {os.path.join(output_dir, 'processed_data.json')}")
    print(f"  Visualizations:   {os.path.join(output_dir, 'representative_images/')}")
    print("="*60)

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Handle config creation
        if args.create_config:
            create_sample_config(args.create_config)
            print(f"Sample configuration file created: {args.create_config}")
            return 0
        
        # Validate arguments
        validate_arguments(args)
        
        # Set up logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        # Load configuration
        config_data = load_config(args.config)
        config = config_data['group_config']
        logger.info(f"Loaded configuration with {len(config.groups)} groups")
        
        # Extract visualization config
        viz_ranges = config_data.get('visualization_ranges')
        pixel_size_override = args.pixel_size_um if hasattr(args, 'pixel_size_um') else None
        pixel_size_um = pixel_size_override if pixel_size_override is not None else config_data.get('pixel_size_um')
        
        if viz_ranges:
            logger.info("Using custom visualization ranges from config file")
        if pixel_size_um:
            logger.info(f"Using pixel size: {pixel_size_um} µm/pixel from config file")
        
        # Set marker based on dimension if not specified
        marker = args.marker
        if marker is None:
            marker = DEFAULT_MARKER if args.dimension == '3d' else DEFAULT_MARKER_2D
        
        # Create pipeline
        pipeline = ND2Pipeline(config)
        
        # Run analysis
        results = pipeline.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            is_3d=(args.dimension == '3d'),
            n_jobs=args.jobs,
            marker=marker,
            scale_bar_um=args.scale_bar,
            create_visualizations=not args.no_visualization,
            viz_ranges=viz_ranges,
            pixel_size_um=pixel_size_um
        )
        
        # Print summary
        print_summary(results, args)
        
        logger.info("Analysis completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
        
    except Exception as e:
        if args.verbose if 'args' in locals() else False:
            logging.exception("Error in main process")
        else:
            print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
