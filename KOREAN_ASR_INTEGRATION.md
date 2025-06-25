# Korean ASR Integration

This document describes the integration of the Korean Conformer Transducer model from HuggingFace as a replacement for MFA (Montreal Forced Alignment) in SONATA.

## Overview

The Korean ASR integration uses the `eesungkim/stt_kr_conformer_transducer_large` model from HuggingFace, which is based on NVIDIA NeMo's Conformer-Transducer architecture. This model provides high-quality Korean speech recognition without requiring the complex MFA pipeline setup.

## Key Changes

### 1. New Korean ASR Model (`sonata/models/korean_asr.py`)
- Implements `KoreanASRModel` class using NeMo toolkit
- Provides Korean speech-to-text transcription
- Generates approximate word-level timestamps
- Falls back gracefully if NeMo is not available

### 2. Updated ASR Model (`sonata/models/asr.py`)
- Integrates Korean ASR model alongside existing WhisperX
- Automatically detects Korean language (`ko`, `kor`, `korean`, `kr`)
- Uses Korean model for Korean audio, WhisperX for other languages

### 3. Updated Core ASR Processor (`sonata/core/asr.py`)
- Replaced MFA-based processing with Korean Conformer Transducer
- Removed TextGrid and subprocess dependencies for Korean processing
- Simplified Korean audio processing pipeline

### 4. Dependencies
- Added `nemo-toolkit[all]>=1.23.0` to requirements.txt and pyproject.toml
- Removed reliance on external MFA tools for Korean processing

## Usage

### Basic Korean Transcription
```python
from sonata.models.korean_asr import KoreanASRModel

# Initialize Korean ASR model
korean_asr = KoreanASRModel()

# Transcribe Korean audio
result = korean_asr.transcribe("korean_audio.wav", language="ko")
print(result["text"])
```

### Using ASR Processor
```python
from sonata.core.asr import ASRProcessor

# Initialize ASR processor
processor = ASRProcessor()

# Process Korean audio (automatically uses Korean model)
segments = processor.process_audio("korean_audio.wav", language="ko")
for segment in segments:
    print(f"{segment['start']:.2f}-{segment['end']:.2f}: {segment['content']}")
```

## Model Details

**Model**: `eesungkim/stt_kr_conformer_transducer_large`
- **Architecture**: Conformer-Transducer
- **Training Data**: KsponSpeech dataset (965 hours)
- **Performance**: 
  - eval_clean CER: 6.94%
  - eval_other CER: 7.38%
  - eval_clean WER: 19.49%
  - eval_other WER: 22.73%
- **Input**: 16kHz mono audio
- **Output**: Korean text transcription

## Testing

Run the test script to verify the integration:

```bash
python test/korean_asr_test.py
```

With audio file:
```bash
python test/korean_asr_test.py --audio path/to/korean_audio.wav
```

## Installation

Install the required dependencies:

```bash
pip install nemo-toolkit[all]>=1.23.0
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Benefits

1. **No External Dependencies**: Eliminates need for MFA tool installation
2. **Better Korean Support**: Specialized Korean ASR model trained on Korean data
3. **Simplified Pipeline**: Direct transcription without intermediate TextGrid processing
4. **HuggingFace Integration**: Leverages pre-trained models from HuggingFace Hub
5. **Fallback Support**: Gracefully falls back to WhisperX if Korean model unavailable

## Limitations

1. **Word Timestamps**: Currently provides approximate word timestamps, not as precise as MFA forced alignment
2. **Korean Only**: Specialized model only works for Korean language
3. **Model Size**: Large model may require significant memory/compute resources
4. **NeMo Dependency**: Requires NeMo toolkit installation

## Future Improvements

1. **Enhanced Timestamps**: Implement more precise word-level timestamp extraction
2. **Model Variants**: Support for different sized Korean models (small, medium, large)
3. **Real-time Processing**: Optimize for streaming/real-time Korean transcription
4. **Fine-tuning**: Support for domain-specific Korean model fine-tuning 