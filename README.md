# 🔬 ND2 Image Analysis Pipeline

> **A comprehensive Python package for analyzing ND2 microscopy files with interactive web-based threshold analysis, advanced visualization, and statistical analysis capabilities.**

## ✨ Interactive Threshold Analysis (New!)

🎯 **Transform your static analysis into real-time, interactive exploration!**

- **🎛️ Real-time threshold sliders** for all 3 channels (0-4095 range)
- **📊 Interactive boxplots** showing all treatment groups simultaneously  
- **📈 Statistical analysis** with parametric/non-parametric tests
- **🐭 Individual mouse visualization** with replicate data on hover
- **🌈 5 analysis channels**: RGB + Green/Blue + Red/Blue ratios
- **🎨 Customizable color palettes** for professional presentations

## 🚀 Quick Start

### Interactive Analysis (Recommended)

```bash
# 1. Setup environment
git clone <repository-url>
cd nd2-analysis-pipeline
python -m venv venv_threshold
venv_threshold\Scripts\activate
pip install -r requirements.txt
pip install -r threshold_analysis/requirements.txt

# 2. Process your data
python test_threshold_analysis.py --batch "your_study_directory" "examples/configs/example_study.json" "MARKER"

# 3. Start interactive system
# Terminal 1: API Server
python -m threshold_analysis.web_api.main

# Terminal 2: Web Interface
cd threshold_analysis/web_interface && npm install && npm start

# 4. Open browser: http://localhost:3000
```

**Result**: Drag threshold sliders → See instant updates across all treatment groups in real-time boxplots!

### Traditional Analysis (Original)

```bash
# Standard pipeline with fixed thresholds
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Process files
python main.py --input "path/to/nd2/files/" --output "results/"
```

## 📚 Complete Documentation

### **📖 [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive Usage Guide**
> **Everything you need: installation, data processing, web interface, statistical analysis, and troubleshooting**

**Quick Links**:
- [Installation & Setup](USER_GUIDE.md#️-installation--setup)
- [Data Processing Guide](USER_GUIDE.md#-data-processing)  
- [Web Interface Usage](USER_GUIDE.md#-web-interface)
- [Statistical Analysis](USER_GUIDE.md#-statistical-analysis)
- [Troubleshooting](USER_GUIDE.md#️-troubleshooting)

### Additional Resources
- [Installation Guide](INSTALL.md) - Detailed setup instructions
- [Configuration Examples](examples/README.md) - Sample study configurations
- [Changelog](CHANGELOG.md) - Version history and updates
- [Upgrade Guide](UPGRADE_GUIDE.md) - Migration between versions

## 🎯 Two Analysis Modes

### 🌐 Interactive Mode (New!)
- **Web-based interface** with real-time threshold adjustment
- **Pre-compute all thresholds** (0-4095) for instant response
- **Statistical comparison tools** with visual significance markers
- **Professional visualization** suitable for presentations
- **Multi-channel analysis** with ratio calculations

### 📊 Traditional Mode (Original)
- **Command-line processing** with fixed thresholds
- **Batch analysis** for large datasets
- **Excel output** with detailed statistics
- **Established workflow** for routine analysis

## 🔧 Core Features

### Image Processing
- **ND2 file support** with automatic metadata extraction
- **Multi-channel analysis** (Green, Red, Blue channels)
- **Flexible thresholding** with customizable parameters
- **Batch processing** with progress tracking
- **GPU acceleration** using pyclesperanto

### Statistical Analysis
- **Parametric/non-parametric tests** (t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis)
- **Multiple comparison corrections** with significance markers
- **Group-based comparisons** with visual annotations
- **Real-time statistics** that update with threshold changes

### Visualization
- **Interactive boxplots** with mouse-level data points
- **Individual replicate visualization** on hover
- **Color palette customization** (5 professional schemes)
- **Statistical comparison bars** with significance indicators
- **High-resolution output** for publications

### Data Export
- **JSON format** for web applications and sharing
- **Excel output** with multiple sheets (traditional mode)
- **Real-time analysis** without file generation needed

## 🔍 System Requirements

### Interactive Analysis
- **Python 3.8+** with pip
- **Node.js 14+** with npm  
- **8GB RAM minimum** (16GB recommended for large studies)
- **Modern web browser** (Chrome, Firefox, Edge, Safari)

### Traditional Analysis
- **Python 3.8+** with pip
- **4GB RAM minimum** for basic processing
- **GPU with OpenCL support** (recommended for performance)

## 🏗️ Project Structure

```
nd2-analysis-pipeline/
├── USER_GUIDE.md                   # Comprehensive usage guide
├── README.md                       # This file
├── requirements.txt                # Main project dependencies
├── test_threshold_analysis.py      # CLI tool for data processing
│
├── threshold_analysis/             # Interactive analysis module
│   ├── requirements.txt            # Additional dependencies
│   ├── data_models.py              # Data structures
│   ├── generator.py                # Core processing functions
│   ├── batch_processor.py          # Batch processing with progress
│   ├── web_api/                    # Backend API server
│   │   └── main.py                 # FastAPI application
│   └── web_interface/              # Frontend React application
│       ├── package.json            # Node.js dependencies
│       └── src/                    # React components
│
├── main.py                         # Traditional CLI interface
├── processing_pipeline.py          # Original batch processing
├── image_processing.py             # Core image analysis functions
├── visualization.py                # Traditional visualization tools
├── excel_output.py                 # Excel report generation
│
└── examples/                       # Example configurations (kept local)
    └── configs/                    # Sample study configurations
```

## 🤝 Getting Help

### Quick Solutions
1. **📖 Read [USER_GUIDE.md](USER_GUIDE.md)** - Comprehensive troubleshooting section
2. **🐛 Check GitHub Issues** - Search existing problems and solutions
3. **💬 Start a Discussion** - Ask questions and share experiences

### Common Issues
- **Marker extraction problems** → See filename pattern guide in USER_GUIDE.md
- **Web interface not loading** → Check server startup and browser console
- **Performance issues** → Review hardware requirements and optimization tips

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Core Technologies
- **Scientific Computing**: numpy, scipy, pandas
- **Web Framework**: FastAPI (backend), React (frontend)
- **Visualization**: Plotly.js for interactive plots
- **Image Processing**: pyclesperanto for GPU acceleration
- **File Handling**: nd2reader for ND2 file support

### Special Features
- **Real-time Analysis**: Pre-computation strategy for instant threshold updates
- **Statistical Integration**: Seamless parametric/non-parametric test selection
- **Professional UI**: Publication-ready visualizations with customizable styling

---

**🚀 Ready to make your ND2 analysis interactive? Start with [USER_GUIDE.md](USER_GUIDE.md)!**