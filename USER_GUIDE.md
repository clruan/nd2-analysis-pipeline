# 🔬 ND2 Interactive Threshold Analysis - User Guide

> **Transform your static ND2 image analysis into an interactive web-based system with real-time threshold adjustment and statistical analysis**

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📊 System Overview](#-system-overview)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🔧 Data Processing](#-data-processing)
- [🌐 Web Interface](#-web-interface)
- [📈 Statistical Analysis](#-statistical-analysis)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📁 Project Structure](#-project-structure)
- [📚 Additional Resources](#-additional-resources)

---

## 🚀 Quick Start

### For Impatient Users (5-Minute Setup)

```bash
# 1. Setup environment
cd nd2-analysis-pipeline
python -m venv venv_threshold
venv_threshold\Scripts\activate
pip install -r requirements.txt
pip install -r threshold_analysis/requirements.txt

# 2. Process your data (replace with your paths)
python test_threshold_analysis.py --batch "your_study_directory" "examples/configs/your_config.json" "VWF(R)"

# 3. Start the system
# Terminal 1: API Server
python -m threshold_analysis.web_api.main

# Terminal 2: Web Interface  
cd threshold_analysis/web_interface
npm install
npm start

# 4. Open browser: http://localhost:3000
```

**Result**: Interactive web interface with draggable threshold sliders and real-time boxplot updates showing all your treatment groups across all channels.

---

## 📊 System Overview

### What This System Does

**Before (Static Analysis)**:
- Fixed thresholds → Single analysis → Excel output
- Manual threshold testing → Reprocess entire dataset
- Static visualizations → No real-time exploration

**After (Interactive Analysis)**:
- Pre-compute ALL thresholds (0-4095) → Store comprehensive data
- Web interface → Drag sliders → Instant visualization updates
- Real-time statistical analysis → Interactive group comparisons

### Key Features

✅ **Real-time threshold adjustment** for all 3 channels (0-4095 range)  
✅ **Interactive boxplots** showing all treatment groups simultaneously  
✅ **Statistical analysis** with parametric/non-parametric tests  
✅ **Individual mouse visualization** with replicate data on hover  
✅ **5 analysis channels**: RGB channels + Green/Blue + Red/Blue ratios  
✅ **Color palette customization** for presentations  
✅ **Professional visualization** matching your existing pipeline  

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 14+** with npm
- **Virtual environment** (recommended)
- **Your existing ND2 files** and configuration files

### Step 1: Python Environment Setup

```bash
# Navigate to your project directory
cd C:\Users\Duke\Downloads\nd2-analysis-pipeline

# Create virtual environment
python -m venv venv_threshold

# Activate virtual environment
venv_threshold\Scripts\activate  # Windows
# source venv_threshold/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt
pip install -r threshold_analysis/requirements.txt
```

### Step 2: Frontend Dependencies

```bash
# Navigate to frontend directory
cd threshold_analysis/web_interface

# Install Node.js dependencies
npm install
```

### Step 3: Verify Installation

```bash
# Test Python environment
python -c "import numpy, scipy, fastapi; print('✅ Python dependencies OK')"

# Test Node environment
cd threshold_analysis/web_interface
npm list react plotly.js axios
```

---

## 🔧 Data Processing

### Understanding File Structure

Your ND2 files need proper mouse ID extraction. The system uses a **marker** approach:

**Example filename**: `Study 2 Set #1 Octapharma 20X HbSS Untreated Y3 VWF(R) Psel (G) CD31 (B).nd2`

**Marker selection**:
- Marker `"VWF(R)"` → Extracts mouse ID: `"Y3"` ✅
- Marker `"20X"` → Extracts mouse ID: `"Octapharma"` ❌
- Marker `"Psel"` → Extracts mouse ID: `"VWF(R)"` ❌

**Rule**: The system takes the word **immediately before** your chosen marker as the mouse ID.

### Single File Testing

Test with one file first to verify everything works:

```bash
python test_threshold_analysis.py "path/to/your/file.nd2" "examples/configs/your_config.json" "VWF(R)"
```

**Expected output**:
```
🔬 Testing Threshold Analysis
==================================================
ND2 File: [your file path]
Config: examples/configs/your_config.json
Marker: VWF(R)

📊 Processing file with all thresholds (0-4095)...

✅ SUCCESS: Threshold analysis completed!
Mouse: Y3, Group: HbSS Untreated
Channel 1 at threshold 1000: 15.23%
Channel 2 at threshold 1000: 8.45%
Channel 3 at threshold 1000: 25.67%
```

### Batch Processing (Recommended)

Process all files in your study directory:

```bash
python test_threshold_analysis.py --batch "C:\path\to\your\study\directory" "examples/configs/your_config.json" "VWF(R)"
```

**What happens**:
1. **Finds all ND2 files** in directory and subdirectories
2. **Processes each file** for all 4096 thresholds per channel
3. **Calculates mouse averages** across multiple images per mouse
4. **Groups data by treatment** according to your config file
5. **Saves comprehensive dataset** for web interface
6. **Shows progress tracking** with time estimates

**Expected output**:
```
🔬 Batch Processing Started
==================================================
📁 Scanning directory: C:\path\to\your\study\directory
📊 Found 80 ND2 files to process

⏳ Processing files:
[1/80] ✅ file1.nd2 (Mouse: Y3, Group: HbSS Untreated) - 45.2s
[2/80] ✅ file2.nd2 (Mouse: Y44, Group: HbSS ATIII) - 43.8s
...
[80/80] ✅ file80.nd2 (Mouse: X47, Group: HbAA Untreated) - 44.1s

📊 BATCH SUMMARY:
✅ Successfully processed: 80 files
📊 Total mice found: 20
📊 Groups identified: 6
📊 Average processing time: 44.3s per file
💾 Results saved to: threshold_results_[timestamp].json
```

---

## 🌐 Web Interface

### Starting the System

**Terminal 1 - API Server**:
```bash
cd nd2-analysis-pipeline
venv_threshold\Scripts\activate
python -m threshold_analysis.web_api.main
```
*Server runs at: http://localhost:8000*

**Terminal 2 - Web Interface**:
```bash
cd nd2-analysis-pipeline/threshold_analysis/web_interface  
npm start
```
*Interface runs at: http://localhost:3000*

### Loading Your Data

1. **Open browser**: Navigate to http://localhost:3000
2. **Load study data**: 
   - Enter filename: `threshold_results_[your_study_name].json`
   - Click "Load Study" 
   - Status should show: "✅ Real data loaded"

### Using the Interface

#### **Threshold Controls**
- **3 sliders**: Channel 1 (Green), Channel 2 (Red), Channel 3 (Blue)
- **Range**: 0-4095 for each channel
- **Real-time updates**: Drag any slider → All plots update instantly (500ms debounce)
- **Current values**: Displayed next to each slider

#### **Visualization**
- **5 boxplots**: 3 RGB channels + 2 ratio channels (Green/Blue, Red/Blue)
- **All treatment groups**: Every group appears in every boxplot
- **Mouse-level data**: Each dot represents one mouse average
- **Individual replicates**: Hover over mouse dots to see individual image data
- **Color coding**: Groups are consistently colored across all plots

#### **Interactive Features**
- **Hover effects**: Mouse dots show individual image replicates as dimmer connected dots
- **Color palettes**: Choose from Default, Pastel, Bright, Scientific, Colorblind Friendly
- **Responsive design**: Plots adapt to browser window size
- **Professional quality**: Suitable for presentations and publications

---

## 📈 Statistical Analysis

### Enabling Statistics

1. **Check "Enable Statistical Analysis"**
2. **Choose comparison mode**:
   - **All vs One Group**: Compare all other groups to a reference group
   - **Specified Pairs**: Define custom group comparisons

### Statistical Options

#### **Test Types**
- **Auto-detect**: System chooses parametric vs non-parametric based on normality tests
- **Parametric**: t-test (2 groups) or ANOVA (multiple groups)
- **Non-parametric**: Mann-Whitney U (2 groups) or Kruskal-Wallis (multiple groups)

#### **Significance Display**
- **Stars**: `***` (p<0.001), `**` (p<0.01), `*` (p<0.05), `ns` (not significant)
- **P-values**: Exact p-values displayed

#### **Comparison Modes**

**All vs One Group**:
- Select reference group (e.g., "Neg" control)
- All other groups compared to reference
- Overall ANOVA result shown

**Specified Pairs**:
- **Add Pair**: Manually select two groups to compare
- **All vs Control**: Automatically compare all treatments to control
- **Clear All**: Remove all comparisons
- **Visual feedback**: Comparison bars appear between selected groups

### Statistical Visualization

- **Comparison bars**: Horizontal lines between compared groups
- **Significance markers**: Stars or p-values above comparison bars
- **Color coding**: Red for significant (p<0.05), Gray for non-significant
- **Real-time updates**: Statistics recalculate as you adjust thresholds

---

## 🛠️ Troubleshooting

### Common Errors and Solutions

#### **Error: "Could not find marker 'X' in filename"**

**Problem**: Marker not found in filename.

**Solutions**:
1. **Check filename structure**: Look at your actual filenames
2. **Find mouse ID in config**: Verify mouse ID exists in your config file  
3. **Choose correct marker**: Pick a word that comes after your mouse ID
4. **Test different markers**: Try various unique words from your filename

**Example debugging**:
```
Filename: Study 2 Set #1 Octapharma 20X HbSS Untreated Y3 VWF(R) Psel (G) CD31 (B).nd2
Config mouse IDs: ["Y80", "Y44", "X47", "Y3"]
Marker "VWF(R)" → Extracts "Y3" ✅ (found in config)
Marker "20X" → Extracts "Octapharma" ❌ (not in config)
```

#### **Error: "Mouse ID X not found in groups"**

**Problem**: Extracted mouse ID doesn't match config file.

**Solutions**:
1. **Verify mouse ID extraction**: Check what ID was extracted
2. **Update config file**: Add missing mouse ID to appropriate group
3. **Fix marker choice**: Use marker that extracts correct mouse ID

#### **Web Interface Issues**

**Problem**: "Using mock data" instead of real data.

**Solutions**:
1. **Check file path**: Ensure JSON file exists and path is correct
2. **Verify API server**: Confirm http://localhost:8000 is running
3. **Check browser console**: Look for error messages (F12 → Console)
4. **Restart servers**: Stop and restart both API and frontend servers

**Problem**: Sliders not updating plots.

**Solutions**:
1. **Wait for debounce**: Allow 500ms after slider movement
2. **Check network**: Verify API calls in browser Network tab (F12 → Network)
3. **Reload page**: Refresh browser and reload data
4. **Check console errors**: Look for JavaScript errors

#### **Performance Issues**

**Problem**: Batch processing very slow or stuck.

**Solutions**:
1. **Check progress output**: System shows file-by-file progress
2. **Verify file access**: Ensure all ND2 files are accessible
3. **Monitor memory usage**: Large files may require more RAM
4. **Process smaller batches**: Split large directories into smaller groups

**Problem**: Web interface slow or unresponsive.

**Solutions**:
1. **Reduce debounce delay**: Modify 500ms timeout in code if needed
2. **Check browser resources**: Close other tabs, check memory usage
3. **Simplify visualization**: Temporarily disable hover effects
4. **Use faster hardware**: SSD storage and more RAM help significantly

### Environment Issues

#### **Python Import Errors**

```bash
# Verify virtual environment is active
which python  # Should show venv_threshold path

# Reinstall dependencies
pip install -r requirements.txt
pip install -r threshold_analysis/requirements.txt

# Test specific imports
python -c "import numpy, scipy, fastapi"
```

#### **Node.js Issues**

```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Verify installation
npm list react plotly.js axios
```

---

## 📁 Project Structure

### File Organization

```
nd2-analysis-pipeline/
├── USER_GUIDE.md                    # This comprehensive guide
├── requirements.txt                 # Main project dependencies
├── test_threshold_analysis.py       # CLI tool for data processing
│
├── threshold_analysis/              # Interactive analysis module
│   ├── __init__.py
│   ├── requirements.txt             # Additional dependencies
│   ├── data_models.py               # Data structures (ThresholdData, ThresholdResults)
│   ├── generator.py                 # Core processing (analyze_single_image_all_thresholds)
│   ├── batch_processor.py           # Batch processing with progress tracking
│   │
│   ├── web_api/                     # Backend API server
│   │   ├── __init__.py
│   │   └── main.py                  # FastAPI application with endpoints
│   │
│   └── web_interface/               # Frontend React application
│       ├── package.json             # Node.js dependencies
│       ├── public/
│       │   └── index.html
│       └── src/
│           ├── index.js             # React entry point
│           ├── index.css            # Styling
│           └── App.js               # Main React component
│
├── examples/                        # Example configurations (kept local)
│   └── configs/
│       ├── high_resolution_study.json
│       └── neuroblastoma_study.json
│
└── output/                          # Generated results (kept local)
    └── threshold_results_*.json     # Processed threshold data
```

### Key Components

#### **Data Models** (`threshold_analysis/data_models.py`)
- `ThresholdData`: Stores 4096 threshold values per channel per image
- `ThresholdResults`: Aggregates all data for a study
- Methods for threshold lookup and mouse average calculation

#### **Processing Engine** (`threshold_analysis/generator.py`)
- `analyze_single_image_all_thresholds()`: Core processing function
- Calculates positive pixel percentages for all thresholds (0-4095)
- Integrates with existing ND2 pipeline functions

#### **Web API** (`threshold_analysis/web_api/main.py`)
- `/api/studies/load`: Load processed JSON data
- `/api/studies/{study_id}/analyze`: Get analysis for specific thresholds
- `/api/studies/{study_id}/statistics`: Perform statistical analysis
- FastAPI with automatic OpenAPI documentation

#### **Frontend** (`threshold_analysis/web_interface/src/App.js`)
- React application with real-time threshold sliders
- Plotly.js boxplots showing all treatment groups
- Statistical analysis interface with comparison tools
- Color palette customization and hover effects

---

## 🎯 Workflow Summary

### Development Workflow
1. **Setup environment** → Install Python and Node.js dependencies
2. **Test single file** → Verify marker extraction and processing
3. **Process batch data** → Generate comprehensive threshold dataset  
4. **Start servers** → Launch API backend and React frontend
5. **Load data** → Import your processed JSON file into web interface
6. **Interactive analysis** → Drag sliders, view real-time updates, perform statistics

### Data Flow
```
ND2 Files → Marker Extraction → Mouse ID → Group Assignment → 
Threshold Processing (0-4095) → JSON Storage → Web API → 
Real-time Interface → Statistical Analysis → Visualization
```

### Typical Session
1. **Load study**: Import your threshold_results_*.json file
2. **Explore thresholds**: Drag sliders to find optimal values
3. **Enable statistics**: Compare treatment groups with appropriate tests
4. **Analyze results**: Use boxplots to identify significant differences
5. **Export findings**: Screenshot plots for presentations/publications

## 📚 Additional Resources

### Learning and Development
- **[INTERACTIVE_TUTORIAL.ipynb](INTERACTIVE_TUTORIAL.ipynb)** - Step-by-step web development tutorial
- **[Installation Guide](INSTALL.md)** - Detailed setup instructions
- **[Configuration Examples](examples/README.md)** - Sample study configurations

### Project Documentation
- **[Changelog](CHANGELOG.md)** - Version history and updates
- **[Upgrade Guide](UPGRADE_GUIDE.md)** - Migration between versions
- **[example_usage.ipynb](example_usage.ipynb)** - Traditional analysis examples

### Web Development Resources
- **React Documentation**: https://react.dev/
- **Plotly.js Documentation**: https://plotly.com/javascript/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/

---

**🚀 Ready to transform your ND2 analysis from static to interactive!**

For technical implementation details, see the existing documentation files in your project directory. This system provides the foundation for real-time, interactive analysis of your microscopy data with professional-quality visualizations and robust statistical testing.

