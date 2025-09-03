# Robust Batch Processing Guide

This guide explains how to run batch processing with proper monitoring, progress tracking, and error recovery.

## 🚀 Quick Start (Improved Version)

### 1. Start Batch Processing
```bash
# Run with improved progress tracking
python test_threshold_analysis.py --batch "\\nas.uic.umn.edu\user\User Data Goes Here\Julia Nguyen\Study 2 Octapharrma Lung 20X vWF Pselectin CdD31 05072025" "examples/configs/C1inh_lung_study2.json" "VWF(R)"
```

### 2. Monitor Progress (In Another Terminal)
```bash
# Open a second terminal window
python monitor_progress.py
```

## 📊 What You'll See Now

### Improved Output with Timing:
```
🔍 Searching for ND2 files...
📊 Found 45 ND2 files

🔍 Testing file accessibility...
   ✅ Study 2 Set #1... - 125.3 MB
   ✅ Study 2 Set #2... - 98.7 MB
   ✅ Study 2 Set #3... - 112.4 MB
   ✅ Study 2 Set #4... - 103.8 MB
   ✅ Study 2 Set #5... - 119.2 MB
✅ 5/5 test files accessible

📊 Processing all ND2 files in directory...
💾 Results will be saved to: threshold_results_Study_2_Octapharrma_Lung_20X_vWF_Pselectin_CdD31_05072025.json

📁 Processing file 1/45
   File: Study 2 Set #1  Octapharma 20X HbSS Untreated Y3   VWF(R) Psel (G) CD31 (B).nd2
   Progress: 2.2%
   ✅ Success: Y3 (HbSS Untreated) - 45.3s
   ⏱️  Estimated time remaining: 33.2 minutes

📁 Processing file 2/45
   File: Study 2 Set #2...
   Progress: 4.4%
   ✅ Success: Y80 (HbSS Untreated) - 42.1s
   ⏱️  Estimated time remaining: 30.8 minutes

...

📁 Processing file 5/45
   File: Study 2 Set #5...
   Progress: 11.1%
   ✅ Success: Y65 (HbSS ATIII) - 38.7s
   💾 Saved intermediate results: intermediate_results_5_files.json
   ⏱️  Estimated time remaining: 25.4 minutes
```

### Progress Monitor Output:
```
📊 Batch Processing Monitor
========================================
Latest file: intermediate_results_15_files.json
Files processed: 15
Last update: 2025-01-07 14:23:45
Processing rate: 1.3 files/minute

Recent files processed:
  - Study 2 Set #13  Octapharma 20X HbAA Untreated Y21... (Y21, HbAA Untreated)
  - Study 2 Set #14  Octapharma 20X HbSS 100Ug-kg Y9... (Y9, HbSS 100Ug-kg)
  - Study 2 Set #15  Octapharma 20X HbSS 300Ug-kg Y34... (Y34, HbSS 300Ug-kg)

⏱️  Monitoring... (Press Ctrl+C to stop)
```

## 🛠️ Key Improvements

### 1. **Real-Time Progress Tracking**
- **File-by-file progress** with percentage completion
- **Time per file** and estimated time remaining
- **Success/failure status** for each file
- **Processing rate** calculation

### 2. **Intermediate Results Saving**
- **Automatic saves every 5 files** (`intermediate_results_5_files.json`, etc.)
- **Timestamp and progress info** in each save
- **Recovery capability** if process is interrupted
- **Final results** saved to named file

### 3. **File Accessibility Testing**
- **Pre-flight check** of first 5 files
- **File size verification** to ensure files aren't corrupted
- **Network connectivity test** before starting long process
- **Early failure detection**

### 4. **Error Handling & Reporting**
- **Individual file error tracking** without stopping the batch
- **Detailed error messages** with timing information
- **Failed files list** at the end
- **Partial success handling**

## 📁 Output Files

### Intermediate Files (Every 5 Files):
- `intermediate_results_5_files.json`
- `intermediate_results_10_files.json`
- `intermediate_results_15_files.json`
- etc.

### Final Results File:
- `threshold_results_Study_2_Octapharrma_Lung_20X_vWF_Pselectin_CdD31_05072025.json`

### Log File:
- Detailed logging information for troubleshooting

## 🔄 Recovery Options

### If Process Gets Interrupted:
```bash
# Check what was completed
python -c "
import json
with open('intermediate_results_15_files.json') as f:
    data = json.load(f)
print(f'Last processed: {data[\"processed_count\"]} files')
print(f'Timestamp: {data[\"timestamp\"]}')
"

# Resume processing (manual approach for now)
# You can see which files were completed and restart from there
```

### If Network Issues:
```bash
# Test connectivity first
python -c "
import os
test_file = '\\\\nas.uic.umn.edu\\user\\User Data Goes Here\\Julia Nguyen\\Study 2 Octapharrma Lung 20X vWF Pselectin CdD31 05072025\\SS Untreated\\Study 2 Set #1  Octapharma 20X HbSS Untreated Y3   VWF(R) Psel (G) CD31 (B).nd2'
print('File accessible:', os.path.exists(test_file))
print('File size:', os.path.getsize(test_file) if os.path.exists(test_file) else 'N/A')
"
```

## ⚡ Performance Optimization

### Current Settings (Conservative):
- **n_jobs = 1**: Sequential processing for stability
- **save_intermediate = True**: Save every 5 files
- **Network files**: Direct access (may be slow)

### For Faster Processing:
```bash
# Copy files locally first (if you have space)
robocopy "\\nas.uic.umn.edu\user\User Data Goes Here\Julia Nguyen\Study 2 Octapharrma Lung 20X vWF Pselectin CdD31 05072025" "C:\temp\local_nd2_files" *.nd2 /S

# Then process locally
python test_threshold_analysis.py --batch "C:\temp\local_nd2_files" "examples/configs/C1inh_lung_study2.json" "VWF(R)"
```

### Parallel Processing (Advanced):
```python
# Modify the call in test_threshold_analysis.py
results = process_directory_all_thresholds(
    input_dir=input_dir,
    config_path=config_file,
    output_file=output_file,
    marker=marker,
    n_jobs=2,  # Try 2 parallel jobs
    save_intermediate=True
)
```

## 🔍 Troubleshooting

### Problem: "Hours with no output"
**Solution**: Now you get output every file, plus monitoring script

### Problem: Network timeouts
**Solution**: File accessibility test catches this early

### Problem: Process interrupted
**Solution**: Intermediate files let you see progress and resume

### Problem: Memory issues
**Solution**: Processing one file at a time, with garbage collection

### Problem: Unknown errors
**Solution**: Detailed error logging with file-specific information

## 📈 Expected Performance

### Typical Processing Times:
- **Small files (50-100 MB)**: 30-60 seconds per file
- **Large files (100-200 MB)**: 60-120 seconds per file
- **Network files**: Add 20-50% overhead

### For 45 Files:
- **Estimated total time**: 30-60 minutes
- **Progress updates**: Every 1-2 minutes
- **Intermediate saves**: Every 5 files (every 5-10 minutes)

## 🎯 Next Steps

1. **Run the improved batch processing** with the new progress tracking
2. **Use the monitor script** in a second terminal to watch progress
3. **Check intermediate files** if you need to stop and resume
4. **Load results into web interface** once processing is complete

This robust approach ensures you never lose hours of processing work and always know what's happening!


