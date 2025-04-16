# SONATA

**SOund and Narrative Advanced Transcription Assistant**

SONATA is an advanced Automatic Speech Recognition (ASR) system that captures the symphony of human expression by recognizing and transcribing both verbal content and emotive sounds.

## Features

- High-accuracy speech-to-text transcription
- Recognition of emotive sounds and non-verbal cues
- Support for tags like `<laugh>`, `<sigh>`, `<yawn>`, `<surprise>`, `<inhale>`, `<groan>`, `<cough>`, `<sneeze>`, `<sniffle>`
- Open-source and extensible architecture

## Installation

Install the package from PyPI:

```bash
pip install sonata-asr
```

Or install from source:

```bash
git clone https://github.com/hwk06023/SONATA.git
cd SONATA
pip install -e .
```

## Usage Examples

### Basic Transcription

```python
from sonata import Transcriber

# Initialize the transcriber
transcriber = Transcriber()

# Transcribe an audio file
result = transcriber.transcribe("path/to/audio.wav")
print(result)
```

### Detecting Emotive Sounds

```python
from sonata.core import EmotiveDetector

# Initialize the emotive detector
detector = EmotiveDetector(threshold=0.6)

# Detect emotive events in an audio file
events = detector.detect_events("path/to/audio.wav")

# Print the detected events
for event in events:
    print(f"{event.type}: {event.start_time:.2f}s - {event.end_time:.2f}s (confidence: {event.confidence:.2f})")
```

### Full Pipeline

```python
from sonata import Sonata

# Initialize SONATA with default settings
sonata = Sonata()

# Process audio file - transcribes speech and detects emotive sounds
result = sonata.process("path/to/audio.wav")

# Print the text with emotive tags
print(result.text_with_tags)

# Save the result
sonata.save_output(result, "output.json")
```

## Command Line Interface

SONATA also provides a CLI for quick transcription:

```bash
# Basic usage
sonata-asr path/to/audio.wav

# Save output to specific file
sonata-asr path/to/audio.wav --output result.json

# Set threshold for emotive detection
sonata-asr path/to/audio.wav --threshold 0.7
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details. This license ensures that derivative works must also be open source and use the same license.

## Acknowledgements

This project leverages the following key open source components:

- [WhisperX](https://github.com/m-bain/whisperX) - Fast speech recognition with word-level timestamps
- [Laughter-Detection](https://github.com/jrgillick/laughter-detection) - Automatic detection of laughter in audio files (MIT License)

We are grateful to the developers and contributors of these libraries for their valuable work.