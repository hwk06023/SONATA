# Changelog

All notable changes to SONATA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.7]: https://github.com/hwk06023/SONATA/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/hwk06023/SONATA/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/hwk06023/SONATA/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/hwk06023/SONATA/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/hwk06023/SONATA/releases/tag/v0.0.3 