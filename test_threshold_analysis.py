"""Simple test script for threshold analysis - run from main directory."""

import sys
import os
from pathlib import Path

def test_threshold_analysis(nd2_file: str, config_file: str, marker: str = None):
    """Test the threshold analysis on a single file."""
    print("🔬 Testing Threshold Analysis")
    print("=" * 50)
    print(f"ND2 File: {nd2_file}")
    print(f"Config: {config_file}")
    print(f"Marker: {marker if marker else 'Auto-detect'}")
    print()
    
    try:
        # Import the test function
        from threshold_analysis.generator import analyze_single_image_all_thresholds
        from data_models import GroupConfig
        
        # Load config
        config = GroupConfig.from_json(config_file)
        mouse_lookup = config.build_mouse_info()
        
        # Run the test
        print("📊 Processing file with all thresholds (0-4095)...")
        result = analyze_single_image_all_thresholds(nd2_file, mouse_lookup, marker=marker)
        
        if result:
            print(f"\n✅ SUCCESS: Threshold analysis completed!")
            print(f"Mouse: {result.mouse_id}, Group: {result.group}")
            print(f"Channel 1 at threshold 1000: {result.get_percentage_at_threshold(1, 1000):.2f}%")
            print(f"Channel 2 at threshold 1000: {result.get_percentage_at_threshold(2, 1000):.2f}%")
            print(f"Channel 3 at threshold 1000: {result.get_percentage_at_threshold(3, 1000):.2f}%")
            return True
        else:
            print("\n❌ FAILED: Threshold analysis failed.")
            print("Check the error messages above.")
            return False
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running from the main nd2-analysis-pipeline directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Single file: python test_threshold_analysis.py <nd2_file> <config_file> [marker]")
        print("  Batch mode:  python test_threshold_analysis.py --batch <input_dir> <config_file> [marker]")
        print()
        print("Arguments:")
        print("  nd2_file    : Path to single ND2 file to analyze")
        print("  input_dir   : Directory containing ND2 files (batch mode)")
        print("  config_file : Path to the JSON configuration file")
        print("  marker      : (Optional) Marker string to find mouse ID")
        print()
        print("Examples:")
        print('# Single file with custom marker')
        print('python test_threshold_analysis.py "file.nd2" "config.json" "VWF(R)"')
        print()
        print('# Batch processing all files in directory')
        print('python test_threshold_analysis.py --batch "input_dir/" "config.json" "VWF(R)"')
        sys.exit(1)
    
    # Check for batch mode
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Batch mode requires: --batch <input_dir> <config_file> [marker]")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        config_file = sys.argv[3]
        marker = sys.argv[4] if len(sys.argv) > 4 else None
        
        success = test_batch_processing(input_dir, config_file, marker)
        sys.exit(0 if success else 1)
    
    # Single file mode
    if len(sys.argv) > 4:
        print("Too many arguments for single file mode")
        sys.exit(1)
    
    nd2_file = sys.argv[1]
    config_file = sys.argv[2]
    marker = sys.argv[3] if len(sys.argv) == 4 else None
    
    # Check if files exist
    if not Path(nd2_file).exists():
        print(f"❌ ND2 file not found: {nd2_file}")
        sys.exit(1)
        
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    success = test_threshold_analysis(nd2_file, config_file, marker)
    sys.exit(0 if success else 1)

def test_batch_processing(input_dir: str, config_file: str, marker: str = None):
    """Test batch processing of multiple ND2 files."""
    print("🔬 Testing Batch Threshold Analysis")
    print("=" * 50)
    print(f"Input Directory: {input_dir}")
    print(f"Config: {config_file}")
    print(f"Marker: {marker if marker else 'Auto-detect'}")
    print()
    
    try:
        # Import the batch processor
        from threshold_analysis.batch_processor import process_directory_all_thresholds
        
        # Check if directories exist
        if not Path(input_dir).exists():
            print(f"❌ Input directory not found: {input_dir}")
            return False
            
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        # Run batch processing with automatic saving
        output_file = f"threshold_results_{Path(input_dir).name.replace(' ', '_')}.json"
        print("📊 Processing all ND2 files in directory...")
        print(f"💾 Results will be saved to: {output_file}")
        
        results = process_directory_all_thresholds(
            input_dir=input_dir,
            config_path=config_file,
            output_file=output_file,
            marker=marker,
            n_jobs=1,  # Start with 1 for stability
            save_intermediate=True  # Save every 5 files
        )
        
        if results and len(results.image_data) > 0:
            print(f"\n✅ SUCCESS: Processed {len(results.image_data)} images!")
            
            # Show sample mouse averages at different thresholds
            print("\n📈 Sample Results:")
            sample_thresholds = {'channel_1': 1000, 'channel_2': 1000, 'channel_3': 300}
            mouse_averages = results.get_mouse_averages(sample_thresholds)
            
            print(f"Mouse averages at thresholds {sample_thresholds}:")
            print(mouse_averages.head())
            
            return True
        else:
            print("\n❌ FAILED: No images were processed successfully.")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    main()
