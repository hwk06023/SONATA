# SONATA 🎵🔊

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/sonata-asr.svg)](https://badge.fury.io/py/sonata-asr)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/hwk06023/SONATA?style=social)](https://github.com/hwk06023/SONATA/stargazers)

<div align="right">
<a href="README.md">English</a> |
<a href="i18n/README.ko.md">한국어</a> |
<a href="i18n/README.zh.md">中文</a> |
<a href="i18n/README.ja.md">日本語</a>
</div>

**SOund and Narrative Advanced Transcription Assistant**

SONATA is an advanced Automatic Speech Recognition (ASR) system that captures the symphony of human expression by recognizing and transcribing both verbal content and emotive sounds.

## ✨ Features

- 🎙️ High-accuracy speech-to-text transcription using WhisperX
- 😀 Recognition of 523+ emotive sounds and non-verbal cues
- 🌍 Multi-language support with 10 languages
- 👥 Speaker diarization for multi-speaker transcription (online and offline modes)
- ⏱️ Rich timestamp information at the word level
- 🔄 Audio preprocessing capabilities

[📚 See detailed features documentation](docs/FEATURES.md)

## 🚀 Installation

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

## 📖 Quick Start

### Basic Transcription

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize the transcriber
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# Transcribe an audio file
result = transcriber.process_audio("path/to/audio.wav", language="en")
print(result["integrated_transcript"]["plain_text"])
```

### CLI Usage

```bash
# Basic usage
sonata-asr path/to/audio.wav

# With speaker diarization
sonata-asr path/to/audio.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN

# With offline speaker diarization (no token needed after setup)
sonata-asr path/to/audio.wav --diarize --offline-diarize --offline-config ~/.sonata/models/offline_config.yaml
```

[📚 See full usage documentation](docs/USAGE.md)  
[⌨️ See complete CLI documentation](docs/CLI.md)  
[🎤 See offline diarization guide](docs/OFFLINE_DIARIZATION.md)

## 🗣️ Supported Languages

SONATA supports 10 languages including English, Korean, Chinese, Japanese, French, German, Spanish, Italian, Portuguese, and Russian.

[🌐 See languages documentation](docs/LANGUAGES.md)

## 🔊 Audio Event Detection

SONATA can detect over 500 different audio events, from laughter and applause to ambient sounds and music.

[🎵 See audio events documentation](docs/AUDIO_EVENTS.md)

## 🛣️ Roadmap

- 🌐 Enhanced multilingual support
- 🧠 Advanced ASR model diversity
- 😢 Improved emotive detection
- 🔊 Better speaker diarization
- ⚡ Performance optimization

[📋 See full roadmap](docs/ROADMAP.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[📝 See contribution guidelines](docs/CONTRIBUTING.md)

## 📄 License

This project is licensed under the GNU General Public License v3.0.

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - Fast speech recognition
- [AudioSet AST](https://github.com/YuanGongND/ast) - Audio event detection
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP tools