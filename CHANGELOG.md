# Changelog

All notable changes to SONATA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-06-20

### Added
- PyPI package deployment improvements
- Enhanced package metadata

### Changed
- Updated version information across the codebase
- Minor code optimizations

## [0.1.0] - 2024-06-18

### Added
- First stable release
- Improved documentation in multiple languages (English, Korean, Japanese, Chinese)
- Enhanced error handling and stability improvements

### Changed
- Optimized audio processing for better performance
- Improved speaker diarization accuracy
- Updated dependency requirements for better compatibility

## [0.0.9] - 2024-06-13

### Changed
- Minor version update and code maintenance
- Updated package dependencies and compatibility

## [0.0.8] - 2024-05-30

### Added
- Custom audio event thresholds: Ability to fine-tune detection sensitivity for specific audio events
- Improved validation for word data in ASR processor
- Better handling of boundary cases in audio event detection

### Changed
- Updated default audio event thresholds for improved detection accuracy
- Optimized audio event detection for various use cases
- Enhanced documentation with custom threshold examples and usage

## [0.0.7] - 2024-04-17

### Added
- Offline diarization support: Use speaker diarization without HuggingFace token
- Added `--offline-diarize` and `--setup-offline` CLI options for diarization
- Created detailed offline diarization documentation

### Changed
- Reorganized multilingual README files to i18n directory
- Enhanced code structure for better maintainability
- Improved error handling and user feedback
- Suppressed warnings during model loading for cleaner output

## [0.0.6] - 2024-04-17

### Changed
- Improved package stability
- Enhanced code readability
- Updated PyPI deployment

## [0.0.5] - 2024-04-16

### Changed
- Updated package build system to use Poetry
- Improved metadata in pyproject.toml
- Updated EmotiveDetector to only use AudioSet AST model
- Removed Korean comments and debug prints
- Converted comments to English for better code readability

## [0.0.4] - 2024-04-XX

### Added
- Multiple transcript format options: concise, default, and extended
- Text output file option
- Command-line arguments for format selection

## [0.0.3] - 2024-04-XX 

### Added
- Initial public release on PyPI
- Basic audio transcription with WhisperX
- Emotive event detection
- Integrated transcription with timestamps

[0.0.9]: https://github.com/hwk06023/SONATA/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/hwk06023/SONATA/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/hwk06023/SONATA/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/hwk06023/SONATA/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/hwk06023/SONATA/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/hwk06023/SONATA/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/hwk06023/SONATA/releases/tag/v0.0.3 
[0.1.0]: https://github.com/hwk06023/SONATA/compare/v0.0.9...v0.1.0 
[0.1.1]: https://github.com/hwk06023/SONATA/compare/v0.1.0...v0.1.1 