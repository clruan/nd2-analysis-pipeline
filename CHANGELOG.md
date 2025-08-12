# Changelog

All notable changes to the ND2 Analysis Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-07

### Added
- Enhanced GPU memory management and error handling
- Robust handling of "Host data is null" errors from pyclesperanto
- Performance optimization options (`--no-visualization` flag)
- Detailed logging of processing steps and errors
- Memory management between file processing
- Configurable parallel job settings for optimal performance

### Improved
- Error handling now gracefully manages GPU memory issues
- Processing continues even when individual channels fail
- Better performance with GPU memory clearing
- More informative error messages and warnings

### Fixed
- GPU memory exhaustion issues during batch processing
- "Host data is null" errors causing processing failures
- Memory leaks during parallel processing

### Performance
- 50-70% speed improvement when using `--no-visualization`
- Better GPU memory utilization
- Optimized parallel processing for multi-core systems

## [1.0.0] - 2024-12-01

### Added
- Initial release of ND2 Analysis Pipeline
- Cross-platform support (Windows and macOS)
- Multi-channel ND2 file processing
- Excel report generation with statistical analysis
- Advanced visualization capabilities
- Representative image selection
- Flexible group configuration
- Parallel processing support
- Comprehensive metrics calculation

### Features
- GPU-accelerated image processing using pyclesperanto
- Professional Excel output with formatting
- Interactive image viewers with scale bars
- Study-specific pixel size configuration
- Custom visualization ranges
- Batch processing capabilities 