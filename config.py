"""Configuration parameters for ND2 image analysis."""

# Default threshold values for different channels and dimensions
DEFAULT_THRESHOLDS = {
    '2d': {
        'channel_1': 800.0,   # Previously psel (green)
        'channel_2': 600.0,   # Previously vwf (red)
        'channel_3': 200.0    # Previously cd31 (blue)
    },
    '3d': {
        'channel_1': 2500.0,  # Previously psel (green)
        'channel_2': 2500.0,  # Previously vwf (red)
        'channel_3': 300.0    # Previously cd31 (blue)
    }
}

# Visualization ranges for each channel (not for thresholding)
VISUALIZATION_RANGES = {
    'channel_1': {'vmin': 100, 'vmax': 2200},   # Green channel
    'channel_2': {'vmin': 150, 'vmax': 2200},   # Red channel
    'channel_3': {'vmin': 50,  'vmax': 2200}     # Blue channel
}

# Channel color mappings for visualization
CHANNEL_COLORS = {
    'channel_1': 'green',
    'channel_2': 'red',
    'channel_3': 'blue'
}

# Channel names for output
CHANNEL_NAMES = {
    'channel_1': 'Channel_1',
    'channel_2': 'Channel_2', 
    'channel_3': 'Channel_3'
}

# Default processing parameters
DEFAULT_PARALLEL_JOBS = 4
DEFAULT_SCALE_BAR_UM = 50  # micrometers
DEFAULT_MARKER = "C1"  # for 3D files
DEFAULT_MARKER_2D = "20X"  # for 2D files

# Default group configuration (can be overridden by config file)
DEFAULT_GROUPS = {
    "Control": ["N00"],
    "Treatment_A": ["X12", "W97", "W88", "X85"],
    "Treatment_B": ["X84", "X3", "W95", "W87"],
    "Treatment_C": ["X83", "W108", "W102", "W82"],
    "Treatment_D": ["X82", "W105", "W81", "W80"],
    "Treatment_E": ["Y26", "W85", "W103", "X39"]
}

# Excel formatting settings
EXCEL_SETTINGS = {
    'highlight_color': 'FFFF00',  # Yellow highlight for representative images
    'header_color': '4F81BD',     # Blue header
    'freeze_panes': (1, 3),       # Freeze first row and first 3 columns
    'column_width': 12,           # Default column width
    'number_format': '#,##0.00'   # Number format for metrics
}

# Output file names
OUTPUT_FILES = {
    'excel_report': 'analysis_results.xlsx',
    'processed_data': 'processed_data.json',
    'log_file': 'processing.log'
}

# Representative image selection
REPRESENTATIVE_SELECTION = {
    'top_n': 3,  # Number of representative images to highlight
    'metric': 'Channel_2_area',  # Primary metric for selection
    'method': 'closest_to_mean'  # Selection method
}
