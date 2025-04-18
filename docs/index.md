---
layout: home
title: SONATA
nav_order: 1
permalink: /
---

# SONATA: SOund and Narrative Advanced Transcription Assistant
{: .fs-9 }

SONATA is an advanced audio transcription system that captures human expressions including emotive sounds and non-verbal cues, providing rich and contextual transcription results.
{: .fs-6 .fw-300 }

[View on GitHub](https://github.com/hwk06023/SONATA){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Get Started](#quick-start){: .btn .fs-5 .mb-4 .mb-md-0 }

<div class="language-selector">
Language: 
<a href="#">English</a> |
<a href="#" class="disabled">한국어 (Coming Soon)</a> |
<a href="#" class="disabled">中文 (Coming Soon)</a> |
<a href="#" class="disabled">日本語 (Coming Soon)</a>
</div>

---

## Features

- 🎙️ High-accuracy speech-to-text transcription using WhisperX
- 😀 Recognition of 523+ emotive sounds and non-verbal cues
- 🌍 Multi-language support with 10 languages
- 👥 Speaker diarization for multi-speaker transcription (online and offline modes)
- ⏱️ Rich timestamp information at the word level
- 🔊 Customizable audio event detection thresholds

## Quick Start
{: #quick-start }

### Installation

```bash
pip install sonata-asr
```

### Basic Usage

**Python API:**

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize the transcriber
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# Transcribe an audio file
result = transcriber.process_audio("path/to/audio.wav", language="en")
print(result["integrated_transcript"]["plain_text"])
```

**CLI Command:**

```bash
# Basic usage
sonata-asr path/to/audio.wav

# With speaker diarization
sonata-asr path/to/audio.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN
```

## Documentation

- [Features](./FEATURES.html)
- [Usage Guide](./USAGE.html)
- [CLI Reference](./CLI.html)
- [Audio Event Detection](./AUDIO_EVENTS.html)
- [Offline Diarization](./OFFLINE_DIARIZATION.html) 