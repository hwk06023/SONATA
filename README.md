# SONATA 🎵🔊

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub stars](https://img.shields.io/github/stars/hwk06023/SONATA?style=social)](https://github.com/hwk06023/SONATA/stargazers)

<div align="right">
<a href="README.md">English</a> |
<a href="i18n/README.ko.md">한국어</a> |
<a href="i18n/README.zh.md">中文</a> |
<a href="i18n/README.ja.md">日本語</a>
</div>

**SOund and Narrative Advanced Transcription Assistant**

SONATA(SOund and Narrative Advanced Transcription Assistant) is advanced ASR system that captures human expressions including emotive sounds and non-verbal cues.

## ✨ Features

- 🎙️ High-accuracy speech-to-text transcription using WhisperX
- 😀 Recognition of 523+ emotive sounds and non-verbal cues
- 🌍 Multi-language support with 10 languages
- 👥 SOTA speaker diarization using Silero VAD and WavLM embeddings
- ⏱️ Rich timestamp information at the word level
- 🔄 Audio preprocessing capabilities

[📚 See detailed features documentation](https://github.com/hwk06023/SONATA/blob/main/docs/FEATURES.md)

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
sonata-asr path/to/audio.wav --diarize

# Set number of speakers if known
sonata-asr path/to/audio.wav --diarize --num-speakers 3
```

#### Common CLI Options:

```
General:
  -o, --output FILE           Save transcript to specified JSON file
  -l, --language LANG         Language code (en, ko, zh, ja, fr, de, es, it, pt, ru)
  -m, --model NAME            WhisperX model size (tiny, small, medium, large-v3, etc.)
  -d, --device DEVICE         Device to run models on (cpu, cuda)
  --text-output               Save transcript to text file (defaults to input_name.txt)
  --preprocess                Preprocess audio (convert format and trim silence)

Diarization:
  --diarize                   Enable SOTA speaker diarization using Silero VAD and WavLM
  --num-speakers NUM          Set exact number of speakers (optional)

Audio Events:
  --threshold VALUE           Threshold for audio event detection (0.0-1.0)
  --custom-thresholds FILE    Path to JSON file with custom audio event thresholds
  --deep-detect               Enable multi-scale audio event detection for better accuracy
  --deep-detect-scales NUM    Number of scales for deep detection (1-3, default: 3)
  --deep-detect-window-sizes  Custom window sizes for deep detection (comma-separated)
  --deep-detect-hop-sizes     Custom hop sizes for deep detection (comma-separated)
```

[📚 See full usage documentation](https://github.com/hwk06023/SONATA/blob/main/docs/USAGE.md)  
[⌨️ See complete CLI documentation](https://github.com/hwk06023/SONATA/blob/main/docs/CLI.md)

## 🗣️ Supported Languages

SONATA supports 10 languages including English, Korean, Chinese, Japanese, French, German, Spanish, Italian, Portuguese, and Russian.

[🌐 See languages documentation](https://github.com/hwk06023/SONATA/blob/main/docs/LANGUAGES.md)

## 🔊 Audio Event Detection

SONATA can detect over 500 different audio events, from laughter and applause to ambient sounds and music. The customizable event detection thresholds allow you to fine-tune sensitivity for specific audio events to match your unique use cases, such as podcast analysis, meeting transcription, or nature recording analysis.

[🎵 See audio events documentation](https://github.com/hwk06023/SONATA/blob/main/docs/AUDIO_EVENTS.md)

## 👥 Speaker Diarization

SONATA provides state-of-the-art speaker diarization to identify and separate different speakers in recordings. The system uses Silero VAD for speech detection and WavLM embeddings for speaker identification, making it ideal for transcribing multi-speaker content like meetings, interviews, and podcasts.

[🎙️ See speaker diarization documentation](https://github.com/hwk06023/SONATA/blob/main/docs/SPEAKER_DIARIZATION.md)

## 🚀 Next Steps

- 🧠 Advanced ASR model diversity
- 😢 Improved emotive detection
- 🔊 Better speaker diarization
- ⚡ Performance optimization
- 🛠️ Fix parallel processing issues in deep detection mode for improved reliability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[📝 See contribution guidelines](https://github.com/hwk06023/SONATA/blob/main/docs/CONTRIBUTING.md)

## 📄 License

This project is licensed under the GNU General Public License v3.0.

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - Fast speech recognition
- [AudioSet AST](https://github.com/YuanGongND/ast) - Audio event detection
  - [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) - Pretrained model for audio event classification
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection for speaker diarization
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) - Microsoft's advanced audio understanding model
  - [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) - Speaker verification model for speaker embeddings
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - Speaker diarization and embedding extraction
- [PyAnnote](https://github.com/pyannote) - Advanced speaker diarization toolkit
  - [pyannote/segmentation](https://github.com/pyannote/pyannote-audio) - Speaker change detection
  - [pyannote/clustering](https://github.com/pyannote/pyannote-audio) - Speaker clustering
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP tools and transformer models