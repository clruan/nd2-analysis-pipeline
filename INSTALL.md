# Installation Guide

## Quick Installation (Recommended)

### From GitHub
```bash
pip install git+https://github.com/clruan/nd2-analysis-pipeline.git
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/clruan/nd2-analysis-pipeline.git
cd nd2-analysis-pipeline

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Platform-Specific Instructions

### Windows
```powershell
# Using PowerShell
python -m pip install git+https://github.com/clruan/nd2-analysis-pipeline.git

# Verify installation
nd2-analysis --help
```

### macOS
```bash
# Using Terminal
python3 -m pip install git+https://github.com/clruan/nd2-analysis-pipeline.git

# Verify installation
nd2-analysis --help
```

## Virtual Environment Setup

### Windows
```powershell
python -m venv nd2-env
nd2-env\Scripts\activate
pip install git+https://github.com/clruan/nd2-analysis-pipeline.git
```

### macOS/Linux
```bash
python3 -m venv nd2-env
source nd2-env/bin/activate
pip install git+https://github.com/clruan/nd2-analysis-pipeline.git
```

## Dependencies

The package will automatically install required dependencies:
- numpy
- scipy  
- pandas
- matplotlib
- scikit-image
- nd2reader
- openpyxl
- joblib

## Verification

Test your installation:
```bash
# Check if commands are available
nd2-analysis --help
nd2-visualize --help

# Run with example config
nd2-analysis --input "your_data/" --config "examples/configs/neuroblastoma_study.json"
```

## Troubleshooting

### Common Issues

1. **Permission errors on macOS**: Use `python3 -m pip install --user`
2. **Path issues**: Ensure Python Scripts folder is in PATH
3. **Missing dependencies**: Run `pip install -r requirements.txt` manually

### Getting Help
- Check the main [README.md](README.md)
- Review [example configurations](examples/README.md)
- Open an issue on GitHub for bugs
