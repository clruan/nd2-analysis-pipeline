# Configuration Examples

This folder contains example configuration files for different types of studies. Each JSON file includes:

- **Study identification** (name, description)
- **Pixel size configuration** (pixel_size_um)
- **Treatment groups** (mouse IDs per group)
- **Processing thresholds** (2D and 3D)
- **Visualization ranges** (vmin/vmax for each channel)

## Using Configuration Files

### Command Line Usage

```powershell
# Use a specific configuration
python main.py -i "data/" -o "results/" -c "examples/configs/neuroblastoma_study.json"
```

### Configuration Structure

```json
{
  "study_name": "your_study_name",
  "pixel_size_um": 0.65,
  "description": "Study description",
  "groups": {
    "Control": ["C01", "C02", "C03"],
    "Treatment": ["T01", "T02", "T03"]
  },
  "thresholds": {
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

## Available Examples

- **`example_study.json`** - Standard 3-group study template (0.222 µm/pixel)
- **`kidney_study.json`** - Multi-group kidney study with 5 treatment groups (0.222 µm/pixel)
- **`lung_study.json`** - Lung study with negative control and 5 experimental groups (0.444 µm/pixel)

## Customizing for Your Study

1. Copy an example file closest to your study type
2. Modify the `groups` section with your mouse IDs
3. Adjust `pixel_size_um` based on your imaging setup
4. Tune `VISUALIZATION_RANGES` to optimize channel visibility
5. Update `thresholds` if needed for your analysis

## Pixel Size Guidelines

Common pixel sizes by objective:
- **4X objective**: ~1.6 µm/pixel
- **10X objective**: ~0.65 µm/pixel  
- **20X objective**: ~0.32 µm/pixel
- **40X objective**: ~0.16 µm/pixel
- **63X objective**: ~0.10 µm/pixel

Check your microscope specifications for exact values.
