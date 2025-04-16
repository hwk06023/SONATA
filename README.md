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
from sonata.core.transcriber import IntegratedTranscriber

# Initialize the transcriber
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# Transcribe an audio file
result = transcriber.process_audio("path/to/audio.wav", language="en")

# Save the result to a file
transcriber.save_result(result, "output.json")

# Get the plain text transcript
plain_text = result["integrated_transcript"]["plain_text"]
print(plain_text)
```

### Extracting Timestamps

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize the transcriber
transcriber = IntegratedTranscriber()

# Process audio with timestamps
result = transcriber.process_audio("path/to/audio.wav")

# Extract words with their timestamps
for item in result["integrated_transcript"]["rich_text"]:
    if item["type"] == "word":
        word = item["content"]
        start_time = item["start"]
        end_time = item["end"]
        print(f"{word}: {start_time:.2f}s - {end_time:.2f}s")
```

### Processing with GPU Acceleration

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize with CUDA device
transcriber = IntegratedTranscriber(
    asr_model="large-v3",
    device="cuda",
    compute_type="float16"  # Use float16 for faster GPU processing
)

# Process audio
result = transcriber.process_audio("path/to/audio.wav")
```

## Command Line Interface

SONATA provides a command-line interface for quick transcription:

```bash
# Basic usage
sonata-asr path/to/audio.wav

# Save output to specific file
sonata-asr path/to/audio.wav --output result.json

# Use GPU acceleration
sonata-asr path/to/audio.wav --device cuda

# Process audio with preprocessing (format conversion and silence trimming)
sonata-asr path/to/audio.wav --preprocess

# Split and process long audio files
sonata-asr path/to/audio.wav --split --split-length 30 --split-overlap 5
```

## Inference Tools

The `test` directory contains additional inference tools for more advanced usage:

### Basic Inference Script

```bash
# Process a single file
python test/infer.py path/to/audio.wav

# Specify output file and use GPU
python test/infer.py path/to/audio.wav -o output.json -d cuda
```

### Advanced Processing

The advanced inference script supports batch processing and additional features:

```bash
# Process a directory of audio files in parallel
python test/advanced_infer.py path/to/audio_directory/ --batch --max-workers 4

# Preprocess audio before transcription
python test/advanced_infer.py path/to/audio.wav --preprocess
```

The preprocessing option performs two important operations:
1. Converts audio to WAV format for maximum compatibility
2. Trims silence from the beginning and end, improving accuracy and reducing processing time

See the [inference tools documentation](test/README.md) for more details.

## Future Roadmap

SONATA is under active development. Here are some planned features and improvements:

### Enhanced Multilingual Support
- Expand language coverage beyond current supported languages
- Improve transcription quality for non-English languages
- Add language auto-detection capabilities

### ASR Model Diversity
- Support for additional ASR models beyond WhisperX
- Integration with local models for offline/private use
- Finetuned models for specific domains (medical, legal, etc.)

### Advanced Emotive Detection
- Expand the range of detectable emotive events
- Improve accuracy of emotive event detection
- Add custom emotive event training capabilities

### Performance Improvements
- Optimize processing for large audio files
- Enhance parallel processing capabilities
- Reduce memory footprint for resource-constrained environments

### User Interface
- Add web-based UI for transcription monitoring
- Develop visualization tools for speech analytics
- Create interactive transcript editor

We welcome contributions in any of these areas!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details. This license ensures that derivative works must also be open source and use the same license.

## Acknowledgements

This project leverages the following key open source components:

- [WhisperX](https://github.com/m-bain/whisperX) - Fast speech recognition with word-level timestamps
- [Laughter-Detection](https://github.com/jrgillick/laughter-detection) - Automatic detection of laughter in audio files (MIT License)

We are grateful to the developers and contributors of these libraries for their valuable work.