---
layout: default
title: CLI Reference
nav_order: 4
permalink: /CLI
---

# SONATA Command Line Interface ⌨️

SONATA provides a powerful command-line interface for quick and efficient audio transcription.

## Basic Usage

```bash
sonata-asr <audio_file>
```

This will transcribe the audio file using default settings and save the results to `<filename>.json`.

## Command Line Options

### Input and Output

| Option | Description |
|--------|-------------|
| `<audio_file>` | Path to the input audio file |
| `-o, --output FILE` | Path to save JSON output (default: `<filename>.json`) |
| `--text-output` | Save transcript to text file (default: `<filename>.txt`) |
| `--format [TYPE]` | Format for text output: concise, default, extended (default: default) |

### Model Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Whisper model size to use: tiny, base, small, medium, large, large-v2, large-v3 (default: large-v3) |
| `-d, --device DEVICE` | Device to run on: cpu, cuda, mps (default: cpu) |
| `-e, --audio-model PATH` | Path to custom audio event detection model |
| `-t, --threshold FLOAT` | Threshold for audio event detection (default: 0.5) |
| `--custom-thresholds PATH` | Path to JSON file with custom audio event thresholds |

### Language Options

| Option | Description |
|--------|-------------|
| `-l, --language LANG` | Language code for transcription: en (English), ko (Korean), zh (Chinese), ja (Japanese), fr (French), de (German), es (Spanish), it (Italian), pt (Portuguese), ru (Russian), and other languages supported by WhisperX (default: en) |

### Preprocessing Options

| Option | Description |
|--------|-------------|
| `--preprocess` | Preprocess audio (convert format and trim silence) |
| `--split` | Split long audio into segments |
| `--split-length SECONDS` | Length of split segments in seconds (default: 30) |
| `--split-overlap SECONDS` | Overlap between split segments in seconds (default: 5) |

### Speaker Diarization Options

| Option | Description |
|--------|-------------|
| `--diarize` | Enable SOTA speaker diarization using Silero VAD and WavLM embeddings |
| `--num-speakers N` | Number of speakers if known (estimated automatically if not provided) |

### Audio Event Detection Options

| Option | Description |
|--------|-------------|
| `--deep-detect` | Enable multi-scale audio event detection with parallel window sizes for better paralinguistic detection |
| `--deep-detect-scales {1,2,3}` | Number of scales to use for deep detection (default: 3, fewer scales = faster processing) |
| `--deep-detect-window-sizes SIZES` | Comma-separated list of window sizes in seconds for deep detection (default: 0.2,1.0,2.5) |
| `--deep-detect-hop-sizes SIZES` | Comma-separated list of hop sizes in seconds for deep detection (default: 0.1,0.5,1.0) |
| `--deep-detect-parallel` | Use parallel processing for multi-scale detection (automatically enables --deep-detect) |
| `--deep-detect-progress` | Show detailed progress bars for deep detection processing |

### Miscellaneous

| Option | Description |
|--------|-------------|
| `--version` | Show SONATA version and exit |
| `--help` | Show help message and exit |

## Examples

### Basic Transcription

```bash
sonata-asr recording.wav
```

### Using GPU with a Specific Model

```bash
sonata-asr recording.wav --device cuda --model medium
```

### Transcribing Non-English Audio

```bash
sonata-asr korean_speech.mp3 --language ko
```

### With Speaker Diarization

```bash
sonata-asr interview.wav --diarize --num-speakers 3
```

### Enhanced Audio Event Detection

```bash
sonata-asr podcast.wav --deep-detect --deep-detect-parallel
```

### Preprocessing Long Audio

```bash
sonata-asr long_podcast.mp3 --preprocess --split --split-length 60 --split-overlap 10
```

### Customizing Output

```bash
sonata-asr meeting.wav --output meeting_data.json --text-output --format extended
```

### Adjusting Detection Sensitivity

```bash
sonata-asr comedy_show.wav --threshold 0.2
```

Lower threshold values will detect more audio events but may increase false positives.

## Exit Codes

- **0**: Success
- **1**: Invalid arguments or file not found
- **2**: Processing error

## Advanced Examples

```bash
# Using custom audio event thresholds
sonata-asr audio.wav --custom-thresholds thresholds.json

# Deep detection with custom settings
sonata-asr interview.wav --deep-detect --deep-detect-scales 2 --deep-detect-window-sizes 0.5,2.0 --deep-detect-hop-sizes 0.2,1.0

# Combining multiple options
sonata-asr audio.mp3 --language ko --device cuda --diarize --num-speakers 2 --deep-detect --custom-thresholds custom_thresholds.json
```

> **Note**: Speaker diarization works with all supported languages. The language option affects only the transcription part, not the speaker identification. 