"""Standalone visualization tool for ND2 files."""

import argparse
import logging
import sys
import os
from pathlib import Path

from data_models import GroupConfig, VisualizationConfig
from processing_pipeline import visualize_single_file, create_complete_gallery
from visualization import ND2Visualizer
from config import DEFAULT_SCALE_BAR_UM, DEFAULT_GROUPS

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ND2 Visualization Tool - Create publication-quality visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single file
  python visualize.py --file "sample.nd2" --output "sample_viz.png"
  
  # Create gallery for all files in directory
  python visualize.py --input "data/" --output "gallery/" --mode gallery
  
  # Create representative images only
  python visualize.py --input "data/" --output "representatives/" --mode representatives --config config.json
        """
    )
    
    parser.add_argument('--file', '-f',
                       help='Single ND2 file to visualize')
    
    parser.add_argument('--input', '-i',
                       help='Directory containing ND2 files (for gallery mode)')
    
    parser.add_argument('--output', '-o', required=True,
                       help='Output path (file for single mode, directory for gallery)')
    
    parser.add_argument('--mode', choices=['single', 'gallery', 'representatives'], 
                       default='single',
                       help='Visualization mode (default: single)')
    
    parser.add_argument('--config', '-c',
                       help='Path to JSON configuration file')
    
    parser.add_argument('--dimension', '-d', choices=['2d', '3d'], default='3d',
                       help='Data dimension (default: 3d)')
    
    parser.add_argument('--scale-bar', '-s', type=float, default=DEFAULT_SCALE_BAR_UM,
                       help=f'Scale bar size in micrometers (default: {DEFAULT_SCALE_BAR_UM})')
    
    parser.add_argument('--title', '-t',
                       help='Custom title for single file visualization')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def validate_arguments(args):
    """Validate command line arguments."""
    if args.mode == 'single':
        if not args.file:
            raise ValueError("Single mode requires --file argument")
        if not os.path.exists(args.file):
            raise ValueError(f"File does not exist: {args.file}")
    
    elif args.mode in ['gallery', 'representatives']:
        if not args.input:
            raise ValueError(f"{args.mode} mode requires --input argument")
        if not os.path.exists(args.input):
            raise ValueError(f"Input directory does not exist: {args.input}")
        if not os.path.isdir(args.input):
            raise ValueError(f"Input path is not a directory: {args.input}")

def visualize_single_mode(args):
    """Handle single file visualization."""
    logger = logging.getLogger(__name__)
    
    success = visualize_single_file(
        filepath=args.file,
        output_path=args.output,
        is_3d=(args.dimension == '3d'),
        scale_bar_um=args.scale_bar
    )
    
    if success:
        logger.info(f"Visualization saved: {args.output}")
        return True
    else:
        logger.error("Visualization failed")
        return False

def visualize_gallery_mode(args):
    """Handle complete gallery creation."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = GroupConfig.from_json(args.config)
    else:
        config = GroupConfig(groups=DEFAULT_GROUPS)
        if args.config:
            logger.warning(f"Config file not found: {args.config}. Using defaults.")
    
    logger.info(f"Creating gallery with {len(config.groups)} groups")
    
    gallery_structure = create_complete_gallery(
        input_dir=args.input,
        output_dir=args.output,
        config=config,
        is_3d=(args.dimension == '3d'),
        scale_bar_um=args.scale_bar
    )
    
    # Print summary
    total_visualizations = sum(
        len(mouse_files) 
        for group_data in gallery_structure.values() 
        for mouse_files in group_data.values()
    )
    
    logger.info(f"Gallery created with {total_visualizations} visualizations")
    logger.info(f"Output directory: {args.output}")
    
    return True

def visualize_representatives_mode(args):
    """Handle representative images visualization."""
    logger = logging.getLogger(__name__)
    
    # This mode requires running analysis first to identify representatives
    logger.error("Representatives mode requires running the main analysis pipeline first.")
    logger.info("Run: python main.py --input <dir> --output <results_dir>")
    logger.info("Then use the representative images from the results.")
    
    return False

def main():
    """Main entry point for visualization tool."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate arguments
        validate_arguments(args)
        
        # Set up logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting ND2 visualization in {args.mode} mode")
        
        # Route to appropriate handler
        if args.mode == 'single':
            success = visualize_single_mode(args)
        elif args.mode == 'gallery':
            success = visualize_gallery_mode(args)
        elif args.mode == 'representatives':
            success = visualize_representatives_mode(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        if success:
            logger.info("Visualization completed successfully!")
            return 0
        else:
            logger.error("Visualization failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
        return 1
        
    except Exception as e:
        if args.verbose if 'args' in locals() else False:
            logging.exception("Error in visualization")
        else:
            print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
