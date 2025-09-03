"""Batch processing for threshold analysis - handles multiple ND2 files."""

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

# Import from existing pipeline (reuse functions)
from image_processing import get_nd2_files
from data_models import GroupConfig
from .data_models import ThresholdData, ThresholdResults
from .generator import analyze_single_image_all_thresholds

logger = logging.getLogger(__name__)

def process_directory_all_thresholds(
    input_dir: str,
    config_path: str,
    output_file: str = None,
    is_3d: bool = True,
    marker: str = None,
    n_jobs: int = 1,  # Start with 1 for stability
    max_threshold: int = 4095,
    save_intermediate: bool = True,
    progress_interval: int = 1  # Show progress every N files
) -> ThresholdResults:
    """
    Process all ND2 files in a directory with threshold analysis.
    
    Mimics the original pipeline's batch processing approach but generates
    threshold data for all values 0-4095.
    
    Args:
        input_dir: Directory containing ND2 files
        config_path: Path to JSON configuration file
        output_file: Optional path to save results
        is_3d: Whether files contain 3D data
        marker: Filename marker for mouse ID extraction
        n_jobs: Number of parallel jobs (default 1 for stability)
        max_threshold: Maximum threshold value to compute
        
    Returns:
        ThresholdResults object with all processed data
    """
    start_time = time.time()
    
    logger.info("🔬 Starting batch threshold analysis")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Processing mode: {'3D' if is_3d else '2D'}")
    logger.info(f"Marker: {marker if marker else 'Auto-detect'}")
    logger.info(f"Parallel jobs: {n_jobs}")
    
    # Load configuration
    config = GroupConfig.from_json(config_path)
    mouse_lookup = config.build_mouse_info()
    
    # Find all ND2 files (reuse existing function)
    print("🔍 Searching for ND2 files...")
    nd2_files = get_nd2_files(input_dir)
    if not nd2_files:
        raise ValueError(f"No ND2 files found in {input_dir}")
    
    print(f"📊 Found {len(nd2_files)} ND2 files")
    logger.info(f"Found {len(nd2_files)} ND2 files")
    
    # Test accessibility of first few files
    print("🔍 Testing file accessibility...")
    accessible_count = 0
    for i, filepath in enumerate(nd2_files[:5]):  # Test first 5 files
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                accessible_count += 1
                print(f"   ✅ {Path(filepath).name} - {os.path.getsize(filepath)/1024/1024:.1f} MB")
            else:
                print(f"   ❌ {Path(filepath).name} - Not accessible or empty")
        except Exception as e:
            print(f"   ❌ {Path(filepath).name} - Error: {str(e)}")
    
    if accessible_count == 0:
        raise ValueError("No ND2 files are accessible. Check network connection and file permissions.")
    
    print(f"✅ {accessible_count}/5 test files accessible")
    
    # Process files in parallel (like original pipeline)
    logger.info(f"Starting parallel processing with {n_jobs} workers")
    
    if n_jobs == 1:
        # Sequential processing with detailed progress tracking
        results = []
        failed_files = []
        
        for i, filepath in enumerate(nd2_files):
            file_start_time = time.time()
            filename = Path(filepath).name
            
            print(f"\n📁 Processing file {i+1}/{len(nd2_files)}")
            print(f"   File: {filename}")
            print(f"   Progress: {((i+1)/len(nd2_files)*100):.1f}%")
            
            logger.info(f"Processing file {i+1}/{len(nd2_files)}: {filename}")
            
            try:
                result = analyze_single_image_all_thresholds(
                    filepath, mouse_lookup, is_3d, marker, max_threshold
                )
                
                file_elapsed = time.time() - file_start_time
                
                if result:
                    results.append(result)
                    print(f"   ✅ Success: {result.mouse_id} ({result.group}) - {file_elapsed:.1f}s")
                    logger.info(f"✅ Success: {result.mouse_id} ({result.group}) - {file_elapsed:.1f}s")
                    
                    # Save intermediate results periodically
                    if save_intermediate and (i + 1) % 5 == 0:
                        intermediate_file = f"intermediate_results_{i+1}_files.json"
                        save_intermediate_results(results, intermediate_file, input_dir, config.groups)
                        print(f"   💾 Saved intermediate results: {intermediate_file}")
                        
                else:
                    failed_files.append(filename)
                    print(f"   ❌ Failed: {filename} - {file_elapsed:.1f}s")
                    logger.warning(f"❌ Failed: {filename} - {file_elapsed:.1f}s")
                    
            except Exception as e:
                file_elapsed = time.time() - file_start_time
                failed_files.append(filename)
                print(f"   ❌ Error: {filename} - {str(e)} - {file_elapsed:.1f}s")
                logger.error(f"❌ Error processing {filename}: {str(e)} - {file_elapsed:.1f}s")
            
            # Show estimated time remaining
            if i > 0:
                avg_time_per_file = (time.time() - start_time) / (i + 1)
                remaining_files = len(nd2_files) - (i + 1)
                estimated_remaining = avg_time_per_file * remaining_files
                print(f"   ⏱️  Estimated time remaining: {estimated_remaining/60:.1f} minutes")
                
        if failed_files:
            print(f"\n⚠️  Failed to process {len(failed_files)} files:")
            for failed in failed_files[:5]:  # Show first 5 failed files
                print(f"   - {failed}")
            if len(failed_files) > 5:
                print(f"   ... and {len(failed_files) - 5} more")
                
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(analyze_single_image_all_thresholds)(
                filepath, mouse_lookup, is_3d, marker, max_threshold
            ) for filepath in nd2_files
        )
        # Filter out None results
        results = [r for r in results if r is not None]
    
    if not results:
        raise ValueError("No files were processed successfully")
    
    logger.info(f"Successfully processed {len(results)}/{len(nd2_files)} files")
    
    # Create study name from input directory
    study_name = Path(input_dir).name
    
    # Create ThresholdResults object
    threshold_results = ThresholdResults(
        study_name=study_name,
        image_data=results,
        group_info=config.groups
    )
    
    # Save results if output file specified
    if output_file:
        save_threshold_results(threshold_results, output_file)
        logger.info(f"Results saved to: {output_file}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
    
    # Print summary (like original pipeline)
    print_batch_summary(threshold_results)
    
    return threshold_results

def save_intermediate_results(results: List[ThresholdData], filepath: str, study_name: str, group_info: Dict) -> None:
    """Save intermediate results during processing."""
    try:
        data = {
            'study_name': study_name,
            'group_info': group_info,
            'processed_count': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_data': []
        }
        
        for img_data in results:
            img_dict = {
                'mouse_id': img_data.mouse_id,
                'group': img_data.group,
                'filename': img_data.filename,
                'channel_1_percentages': img_data.channel_1_percentages.tolist(),
                'channel_2_percentages': img_data.channel_2_percentages.tolist(),
                'channel_3_percentages': img_data.channel_3_percentages.tolist()
            }
            data['image_data'].append(img_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved intermediate results: {len(results)} files processed")
        
    except Exception as e:
        logger.error(f"Error saving intermediate results: {e}")

def save_threshold_results(results: ThresholdResults, filepath: str) -> None:
    """Save ThresholdResults to JSON file."""
    try:
        data = {
            'study_name': results.study_name,
            'group_info': results.group_info,
            'image_data': []
        }
        
        for img_data in results.image_data:
            img_dict = {
                'mouse_id': img_data.mouse_id,
                'group': img_data.group,
                'filename': img_data.filename,
                'channel_1_percentages': img_data.channel_1_percentages.tolist(),
                'channel_2_percentages': img_data.channel_2_percentages.tolist(),
                'channel_3_percentages': img_data.channel_3_percentages.tolist()
            }
            data['image_data'].append(img_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved threshold results to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def load_threshold_results(filepath: str) -> ThresholdResults:
    """Load ThresholdResults from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct ThresholdData objects
        image_data = []
        for img_dict in data['image_data']:
            threshold_data = ThresholdData(
                mouse_id=img_dict['mouse_id'],
                group=img_dict['group'],
                filename=img_dict['filename'],
                channel_1_percentages=np.array(img_dict['channel_1_percentages']),
                channel_2_percentages=np.array(img_dict['channel_2_percentages']),
                channel_3_percentages=np.array(img_dict['channel_3_percentages'])
            )
            image_data.append(threshold_data)
        
        return ThresholdResults(
            study_name=data['study_name'],
            image_data=image_data,
            group_info=data['group_info']
        )
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise

def print_batch_summary(results: ThresholdResults) -> None:
    """Print summary of batch processing results (like original pipeline)."""
    print("\n" + "="*60)
    print("📊 BATCH THRESHOLD ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Study: {results.study_name}")
    print(f"Total images processed: {len(results.image_data)}")
    print(f"Groups analyzed: {len(results.group_info)}")
    
    # Count mice per group
    mice_per_group = {}
    images_per_group = {}
    
    for img_data in results.image_data:
        group = img_data.group
        mouse_id = img_data.mouse_id
        
        if group not in mice_per_group:
            mice_per_group[group] = set()
            images_per_group[group] = 0
        
        mice_per_group[group].add(mouse_id)
        images_per_group[group] += 1
    
    print(f"Total mice analyzed: {sum(len(mice) for mice in mice_per_group.values())}")
    print()
    
    print("Group breakdown:")
    for group_name in results.group_info.keys():
        if group_name in mice_per_group:
            mice_count = len(mice_per_group[group_name])
            image_count = images_per_group[group_name]
            print(f"  {group_name}: {mice_count} mice, {image_count} images")
        else:
            print(f"  {group_name}: 0 mice, 0 images (no data found)")
    
    print()
    print("✅ Ready for interactive threshold analysis!")
    print("   - All threshold values (0-4095) pre-computed")
    print("   - Mouse averages can be calculated for any threshold combination")
    print("   - Data ready for web interface")
    print("="*60)
