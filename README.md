# ND2 Image Analysis Pipeline

A professional-grade, cross-platform package for analyzing multi-channel ND2 microscopy images with advanced visualization and Excel reporting capabilities.

**Cross-Platform Support**: Works on Windows and macOS  
**GPU-Accelerated**: Uses pyclesperanto for fast image processing  
**Robust Error Handling**: Handles GPU memory issues and null data gracefully  
**Flexible Configuration**: Study-specific pixel sizes and visualization settings  
**Professional Output**: Excel reports with statistical analysis and publication-quality visualizations

## Features

- **Cross-Platform**: Runs on Windows and macOS
- **GPU-Accelerated Processing**: Fast image analysis using pyclesperanto with OpenCL
- **Multi-channel Analysis**: Process 3-channel ND2 files (Channel 1, 2, 3)
- **Robust Error Handling**: Graceful handling of GPU memory issues and "Host data is null" errors
- **Study-Specific Configuration**: Configure pixel sizes and settings per study/batch
- **Custom Visualization Ranges**: Fine-tune channel visibility and contrast
- **Parallel Processing**: High-performance batch processing with configurable CPU cores
- **Excel Reports**: Professional Excel output with formatting and representative image highlighting
- **Advanced Visualization**: Interactive image viewers with customizable scale bars
- **Representative Image Selection**: Automatically identifies most representative images for each group
- **Flexible Grouping**: Single study design with customizable treatment groups
- **Comprehensive Metrics**: Area, intensity, and ratio calculations for all channels
- **Memory Management**: Automatic GPU memory clearing and optimization

## Recent Improvements

- **Enhanced Error Handling**: Robust handling of GPU memory issues and null data
- **Performance Optimization**: GPU memory management and batch processing
- **Better Logging**: Detailed logging of processing steps and errors
- **Visualization Controls**: Option to skip visualizations for faster processing (`--no-visualization`)
- **Configurable Parallel Jobs**: Optimize performance based on your hardware

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows or macOS
- Git (for installation from GitHub)
- GPU with OpenCL support (recommended for best performance)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/clruan/nd2-analysis-pipeline.git
cd nd2-analysis-pipeline
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux  
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Analysis
```bash
python main.py --input "path/to/nd2/files/" --output "results/"
```

### High-Performance Analysis
```bash
# Use more CPU cores and skip visualizations for speed
python main.py --input "data/" --output "results/" --jobs 8 --no-visualization
```

### Custom Configuration
```bash
python main.py --input "data/" --output "results/" --config my_config.json --jobs 4
```

## Performance Optimization

### For Best Performance:
- **Use `--no-visualization`** for 50-70% speed improvement
- **Increase `--jobs`** based on your CPU cores (check with `nproc` on Linux/macOS)
- **Ensure GPU drivers are up to date** for optimal pyclesperanto performance
- **Monitor GPU memory** if processing large datasets

### Troubleshooting GPU Issues:
If you encounter "Host data is null" errors:
1. Reduce parallel jobs: `--jobs 2`
2. Restart Python to clear GPU memory
3. Update GPU drivers
4. The pipeline will automatically handle these errors and continue processing

### 2. Jupyter Notebook Usage
Open `example_usage.ipynb` for interactive analysis and visualization.

### 3. Configuration File
Create a JSON configuration file for your specific study. See `examples/configs/` for templates:

```json
{
  "study_name": "neuroblastoma_batch1",
  "pixel_size_um": 0.65,
  "description": "Neuroblastoma study with custom settings",
  "groups": {
    "Control": ["C01", "C02", "C03", "C04"],
    "Treatment_Low": ["T01", "T02", "T03", "T04"],
    "Treatment_High": ["T05", "T06", "T07", "T08"]
  },
  "thresholds": {
    "2d": {
      "channel_1": 800.0,
      "channel_2": 600.0,
      "channel_3": 200.0
    },
    "3d": {
      "channel_1": 2000.0,
      "channel_2": 1800.0,
      "channel_3": 250.0
    }
  },
  "VISUALIZATION_RANGES": {
    "channel_1": {"vmin": 50, "vmax": 3000},
    "channel_2": {"vmin": 80, "vmax": 2500},
    "channel_3": {"vmin": 20, "vmax": 800}
  }
}
```

## Output Files

The pipeline generates several types of output:

### Excel Reports
- **`analysis_results.xlsx`**: Main results file with multiple sheets
  - `Raw_Data`: All individual image measurements
  - `Group_Summary`: Aggregated statistics by treatment group
  - `Representative_Images`: Highlighted most representative images
  - `Statistical_Analysis`: Group comparisons and statistics

### Visualization
- **`representative_images/`**: Most representative image from each group
- **`all_visualizations/`**: Complete visualization gallery organized by group and mouse
- **`individual_files/`**: Single file visualizations (on demand)

### Data Files
- **`processed_data.json`**: Machine-readable results for further analysis

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Directory containing ND2 files | Required |
| `--output` | `-o` | Output directory for results | `{input}/results` |
| `--config` | `-c` | Path to JSON configuration file | Built-in config |
| `--dimension` | `-d` | Data dimension (2d/3d) | `3d` |
| `--jobs` | `-j` | Number of parallel jobs | `4` |
| `--scale-bar` | `-s` | Scale bar size in micrometers | `50` |
| `--marker` | `-m` | Filename marker for mouse ID extraction | `C1` |
| `--verbose` | `-v` | Enable verbose logging | `False` |

## Visualization Options

### 1. Representative Images (Automatic)
```powershell
# Automatically generated - most representative image per group
python main.py --input "data/" --output "results/"
```

### 2. Specific File Visualization
```powershell
# Visualize a specific file
python visualize.py --file "path\to\specific\file.nd2" --output "results\specific\"
```

### 3. Group Gallery
```powershell
# Create complete visualization gallery
python visualize.py --input "data/" --output "results\gallery\" --mode gallery --scale-bar 100
```

## Project Structure

```
ND2_Analysis_Pipeline/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── config.py                # Default configuration parameters
├── data_models.py           # Data structures and models
├── image_processing.py      # Core image analysis functions
├── excel_output.py          # Excel report generation
├── visualization.py         # Image visualization tools
├── processing_pipeline.py   # Main processing workflows
├── main.py                  # Command line interface
├── visualize.py            # Standalone visualization tool
├── example_usage.ipynb     # Example Jupyter notebook
├── examples/               # Example configurations and documentation
│   ├── README.md          # Configuration guide
│   └── configs/           # JSON configuration files
│       ├── neuroblastoma_study.json
│       ├── thrombosis_study.json
│       └── high_resolution_study.json
└── tests/                  # Unit tests
    ├── __init__.py
    ├── test_image_processing.py
    └── test_pipeline.py
```

## Advanced Usage

### Custom Thresholds
Modify thresholds in your config file or programmatically:

```python
from processing_pipeline import ND2Pipeline

pipeline = ND2Pipeline()
pipeline.config.thresholds['3d']['channel_1'] = 3000.0
results = pipeline.process_directory('data/')
```

### Visualization Customization
```python
from visualization import ND2Visualizer

visualizer = ND2Visualizer(scale_bar_um=100)
visualizer.create_representative_gallery('results/', 'gallery/')
visualizer.visualize_single_file('file.nd2', 'output.png')
```

### Excel Report Customization
```python
from excel_output import ExcelReporter

reporter = ExcelReporter()
reporter.create_report(results_df, 'custom_report.xlsx', 
                      highlight_top_n=3, include_charts=True)
```

## Troubleshooting

### Common Issues

1. **ND2 files not found**
   - Check file paths (use forward slashes or escaped backslashes)
   - Ensure files have `.nd2` extension

2. **Memory issues with large files**
   - Reduce number of parallel jobs: `--jobs 2`
   - Process files in smaller batches

3. **Mouse ID extraction errors**
   - Check filename format and marker position
   - Use custom marker: `--marker "20X"`

### Performance Optimization

- **SSD storage**: Process files from SSD for better performance
- **RAM**: 16GB+ recommended for large datasets
- **CPU cores**: Use `--jobs` equal to your CPU core count

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```
ND2 Image Analysis Pipeline (2025)
GitHub repository: https://github.com/clruan/nd2-analysis-pipeline
```

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review example usage in the Jupyter notebook

---

*Last updated: June 16, 2025*
