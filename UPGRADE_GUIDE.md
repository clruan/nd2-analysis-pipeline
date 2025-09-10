# 🚀 ND2 Analysis Pipeline - Upgrade Roadmap

> **Future Features and Enhancements for the Interactive Threshold Analysis System**

This document outlines planned upgrades and new features for the ND2 analysis pipeline. These features are **proposed** and not yet implemented.

## 📋 Table of Contents

- [🔬 Multi-Format File Support](#-multi-format-file-support)
- [🎨 Flexible Channel Configuration](#-flexible-channel-configuration)  
- [📊 Variable Channel Count Support](#-variable-channel-count-support)
- [⚡ Real-Time Visualization Enhancements](#-real-time-visualization-enhancements)
- [🛠️ Implementation Timeline](#️-implementation-timeline)
- [💡 Technical Considerations](#-technical-considerations)

---

## 🔬 Multi-Format File Support

### Current State
- **Supported**: ND2 files only (Nikon format)
- **Limitation**: Single vendor lock-in

### Proposed Enhancement
**Support for multiple microscopy file formats:**

#### Target Formats
- **Zeiss**: `.czi`, `.lsm`, `.zvi` files
- **Olympus**: `.oib`, `.oif`, `.vsi` files  
- **Leica**: `.lif`, `.lei` files
- **Generic**: `.tiff`, `.tif` multi-channel stacks
- **Bio-Formats**: Universal support via `python-bioformats`

#### Implementation Plan
```python
# New file reader abstraction
class UniversalImageReader:
    def __init__(self, file_path):
        self.format = self._detect_format(file_path)
        self.reader = self._get_reader(self.format)
    
    def _detect_format(self, path):
        """Auto-detect file format from extension and metadata"""
        
    def _get_reader(self, format):
        """Return appropriate reader (nd2reader, czifile, etc.)"""
        
    def get_channels(self):
        """Standardized channel access across formats"""
        
    def get_metadata(self):
        """Unified metadata extraction"""
```

#### User Benefits
- **Vendor flexibility**: Use any microscope brand
- **Lab compatibility**: Support mixed equipment environments
- **Data migration**: Easy transition between microscope systems
- **Collaboration**: Share analysis workflows across institutions

---

## 🎨 Flexible Channel Configuration

### Current State
- **Fixed mapping**: Channel 1 = Green, Channel 2 = Red, Channel 3 = Blue
- **Limitation**: Cannot adapt to different staining protocols

### Proposed Enhancement
**User-configurable channel assignments and color schemes:**

#### Configuration Interface
```json
{
  "channel_configuration": {
    "channel_1": {
      "name": "DAPI",
      "color": "blue",
      "display_order": 3,
      "wavelength": "405nm"
    },
    "channel_2": {
      "name": "FITC", 
      "color": "green",
      "display_order": 1,
      "wavelength": "488nm"
    },
    "channel_3": {
      "name": "Texas Red",
      "color": "red", 
      "display_order": 2,
      "wavelength": "594nm"
    }
  },
  "visualization_order": ["channel_2", "channel_3", "channel_1"],
  "color_palette": "custom"
}
```

#### Web Interface Updates
```javascript
// New channel configuration panel
const ChannelConfigPanel = () => {
  return (
    <div className="channel-config">
      <h3>Channel Configuration</h3>
      {channels.map(channel => (
        <ChannelEditor 
          key={channel.id}
          channel={channel}
          onNameChange={updateChannelName}
          onColorChange={updateChannelColor}
          onOrderChange={updateDisplayOrder}
        />
      ))}
    </div>
  );
};
```

#### User Benefits
- **Flexible protocols**: Adapt to any staining combination
- **Intuitive naming**: Use meaningful marker names instead of numbers
- **Visual clarity**: Match colors to actual fluorophores
- **Professional output**: Publication-ready channel labels

---

## 📊 Variable Channel Count Support

### Current State
- **Fixed**: 3 channels only (RGB)
- **Limitation**: Cannot handle 2, 4, 5+ channel images

### Proposed Enhancement
**Dynamic support for 1-8 channels with automatic UI adaptation:**

#### Dynamic Data Models
```python
@dataclass
class FlexibleThresholdData:
    """Supports variable number of channels"""
    mouse_id: str
    group: str
    filename: str
    channel_count: int
    channel_percentages: Dict[int, np.ndarray]  # channel_id -> percentages
    
    def get_percentage_at_threshold(self, channel: int, threshold: int) -> float:
        """Works with any number of channels"""
        if channel not in self.channel_percentages:
            raise ValueError(f"Channel {channel} not found")
        return float(self.channel_percentages[channel][threshold])
```

#### Adaptive Web Interface
```javascript
// Dynamic slider generation
const ThresholdControls = ({ channelCount, thresholds, onChange }) => {
  const channels = Array.from({length: channelCount}, (_, i) => i + 1);
  
  return (
    <div className="threshold-controls">
      {channels.map(channel => (
        <ChannelSlider
          key={channel}
          channelId={channel}
          value={thresholds[`channel_${channel}`]}
          onChange={onChange}
          color={getChannelColor(channel)}
        />
      ))}
    </div>
  );
};
```

#### Automatic Ratio Calculations
```python
def calculate_all_ratios(channels):
    """Generate all possible channel ratios dynamically"""
    ratios = []
    for i in range(len(channels)):
        for j in range(len(channels)):
            if i != j:
                ratios.append(f"channel_{i+1}_{j+1}_ratio")
    return ratios
```

#### User Benefits
- **Experimental flexibility**: Support any channel combination
- **Cost efficiency**: Use 2-channel protocols when appropriate
- **Advanced imaging**: Support high-content screening (6+ channels)
- **Future-proofing**: Ready for new microscopy technologies

---

## ⚡ Real-Time Visualization Enhancements

### Current State
- **Interactive sliders**: Working with 500ms debounce
- **Basic plots**: Boxplots with statistical annotations
- **Limitation**: Limited visual feedback and customization

### Proposed Enhancement
**Enhanced real-time visualization with advanced features:**

#### Live Preview System
```javascript
// Real-time image preview with threshold overlay
const LivePreview = ({ imageData, thresholds }) => {
  const [previewImage, setPreviewImage] = useState(null);
  
  useEffect(() => {
    // Generate threshold overlay in real-time
    const overlay = generateThresholdOverlay(imageData, thresholds);
    setPreviewImage(overlay);
  }, [thresholds]);
  
  return (
    <div className="live-preview">
      <img src={previewImage} alt="Threshold Preview" />
      <div className="threshold-info">
        Positive pixels: {calculatePositivePixels()}%
      </div>
    </div>
  );
};
```

#### Advanced Plot Types
```javascript
// Multiple visualization options
const VisualizationSelector = () => {
  const plotTypes = [
    { id: 'boxplot', name: 'Box Plot', component: BoxPlot },
    { id: 'violin', name: 'Violin Plot', component: ViolinPlot },
    { id: 'scatter', name: 'Scatter Plot', component: ScatterPlot },
    { id: 'heatmap', name: 'Correlation Heatmap', component: Heatmap },
    { id: 'timeseries', name: 'Threshold Response', component: TimeSeries }
  ];
  
  return (
    <select onChange={handlePlotTypeChange}>
      {plotTypes.map(type => (
        <option key={type.id} value={type.id}>{type.name}</option>
      ))}
    </select>
  );
};
```

#### Performance Optimizations
```javascript
// Web Workers for heavy computations
const useWebWorkerCalculations = () => {
  const worker = useRef(null);
  
  useEffect(() => {
    worker.current = new Worker('/threshold-calculator.worker.js');
    return () => worker.current.terminate();
  }, []);
  
  const calculateThresholds = useCallback((data, thresholds) => {
    return new Promise((resolve) => {
      worker.current.postMessage({ data, thresholds });
      worker.current.onmessage = (e) => resolve(e.data);
    });
  }, []);
  
  return { calculateThresholds };
};
```

#### User Benefits
- **Instant feedback**: See threshold effects immediately
- **Visual validation**: Preview actual thresholded images
- **Flexible analysis**: Multiple plot types for different insights
- **Smooth performance**: No lag even with large datasets

---

## 🛠️ Implementation Timeline

### Phase 1: Multi-Format Support (2-3 months)
- [ ] Research and integrate Bio-Formats library
- [ ] Implement universal file reader abstraction
- [ ] Add format detection and validation
- [ ] Test with Zeiss, Olympus, and TIFF files
- [ ] Update documentation and examples

### Phase 2: Channel Configuration (1-2 months)
- [ ] Design channel configuration schema
- [ ] Implement backend channel mapping
- [ ] Create web interface for channel editing
- [ ] Add color palette selection
- [ ] Test with various staining protocols

### Phase 3: Variable Channel Count (2-3 months)
- [ ] Refactor data models for flexibility
- [ ] Update API endpoints for dynamic channels
- [ ] Implement adaptive UI components
- [ ] Add automatic ratio calculations
- [ ] Comprehensive testing with 2-8 channel images

### Phase 4: Real-Time Enhancements (1-2 months)
- [ ] Implement live image preview
- [ ] Add Web Workers for performance
- [ ] Create additional plot types
- [ ] Optimize rendering pipeline
- [ ] User experience testing and refinement

---

## 💡 Technical Considerations

### Dependencies
```python
# Additional packages needed
python-bioformats>=4.0.0  # Universal microscopy format support
czifile>=2022.9.26        # Zeiss format support
tifffile>=2023.4.12       # Enhanced TIFF support
aicsimageio>=4.9.4        # Multi-format image I/O
```

### Performance Impact
- **File loading**: Slight increase due to format detection
- **Memory usage**: Variable based on channel count
- **Processing time**: Minimal impact with optimized algorithms
- **Storage**: JSON files scale with channel count

### Backward Compatibility
- **Existing data**: Full compatibility maintained
- **API endpoints**: Versioned to support both old and new formats
- **Configuration files**: Auto-migration from 3-channel format

### Testing Strategy
- **Unit tests**: Each file format and channel configuration
- **Integration tests**: End-to-end workflows with real data
- **Performance tests**: Large datasets and high channel counts
- **User acceptance tests**: Real laboratory workflows

---

## 🎯 Migration Guide (When Implemented)

### For Existing Users
1. **Backup current setup**: Save configurations and results
2. **Update dependencies**: Install new packages
3. **Test compatibility**: Verify existing workflows work
4. **Explore new features**: Try multi-format and flexible channels
5. **Update documentation**: Revise lab protocols as needed

### For New Users
1. **Choose file format**: Select based on your microscope
2. **Configure channels**: Set up meaningful names and colors
3. **Define protocols**: Create templates for common experiments
4. **Train team members**: Ensure everyone understands new features

---

**🚨 Important Note**: These are **proposed features** and are not currently available. This document serves as a roadmap for future development. Implementation will depend on user feedback, development resources, and priority assessment.

**📞 Feedback Welcome**: If you're interested in any of these features or have additional suggestions, please open an issue on GitHub or contact the development team.
