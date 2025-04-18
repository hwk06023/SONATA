---
layout: default
title: Features
nav_order: 2
permalink: /FEATURES
---

# SONATA Features 🎵🔊

SONATA offers a comprehensive suite of audio transcription and analysis features. This document provides details on each major feature.

## 🎙️ High-Accuracy Speech Recognition

SONATA uses WhisperX, an enhanced version of Whisper that provides:

- State-of-the-art transcription accuracy across multiple languages
- Word-level timestamps for precise text alignment
- Support for various Whisper models (tiny, base, small, medium, large, large-v2, large-v3)
- Automatic language detection capabilities
- Model optimization for various hardware (CPU, CUDA, MPS)

## 😀 Audio Event Detection

SONATA can detect 523+ distinct audio events using the AudioSet AST (Audio Spectrogram Transformer) model, including:

### Human Sounds
- Laughter, crying, sighing, whispering, screaming, shouting
- Conversation, narration, male/female/child speech
- Breathing, coughing, sneezing, sniffing, throat clearing

### Physical Sounds
- Clapping, finger snapping, footsteps, heartbeat
- Hands, applause, cheering

### Animal Sounds
- Dog barking, cat meowing, bird calls
- Various other animal vocalizations

### Musical and Environmental Sounds
- Musical instruments (piano, guitar, drum, violin)
- Natural sounds (wind, rain, thunder, water, fire)
- Mechanical sounds (engines, vehicles, alarms)

## 🌍 Multi-Language Support

SONATA supports 10 languages:
- English (en)
- Korean (ko)
- Chinese (zh)
- Japanese (ja)
- French (fr)
- German (de)
- Spanish (es)
- Italian (it)
- Portuguese (pt)
- Russian (ru)

## 👥 Speaker Diarization

- Identify and label different speakers in multi-speaker audio
- Set minimum and maximum speaker constraints
- Integrated with PyAnnote's diarization models
- Speaker-attributed transcripts with formatting options

## ⏱️ Rich Timestamp Information

- Word-level timestamps for all transcribed content
- Precise timing for audio events
- Multiple output formats with varying levels of timestamp detail
- Support for extracting specific time ranges

## 🔄 Audio Preprocessing

- Audio format conversion for maximum compatibility
- Silence detection and trimming to improve transcription quality
- Audio segmentation for long files
- Custom segment length and overlap controls

## 📊 Multiple Output Formats

- **Concise**: Simple text with integrated audio event tags
- **Default**: Text with timestamps
- **Extended**: Includes confidence scores
- JSON output with comprehensive metadata

## 📱 Convenient Interfaces

- Python API for integration into other applications
- Command-line interface for quick usage
- Batch processing capabilities
- Progress indicators for long-running operations 