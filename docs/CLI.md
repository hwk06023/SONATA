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

This will transcribe the audio file using default settings and save the results to `<filename>_transcript.json` and `<filename>_transcript.txt`.

## Command Line Options

### Input and Output

| Option | Description |
|--------|-------------|
| `<audio_file>` | Path to the input audio file |
| `-o, --output FILE` | Path to save JSON output (default: `<filename>_transcript.json`) |
| `--text-output FILE` | Path to save formatted text transcript (default: `<filename>_transcript.txt`) |

### Model Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | WhisperX model size to use: tiny, base, small, medium, large, large-v2, large-v3 (default: large-v3) |
| `-d, --device DEVICE` | Device to run on: cpu, cuda, mps (default: cpu) |
| `-e, --audio-model PATH` | Path to custom audio event detection model |
| `-t, --threshold FLOAT` | Threshold for audio event detection (default: 0.3) |

### Language Options

| Option | Description |
|--------|-------------|
| `-l, --language LANG` | Language code for transcription: en, ko, zh, ja, fr, de, es, it, pt, ru (default: en) |

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
| `--diarize` | Enable speaker diarization |
| `--hf-token TOKEN` | HuggingFace token for diarization models (required for diarization) |
| `--min-speakers N` | Minimum number of speakers for diarization |
| `--max-speakers N` | Maximum number of speakers for diarization |

### Output Format Options

| Option | Description |
|--------|-------------|
| `--format FORMAT` | Format for text output: concise, default, extended (default: default) |

### Miscellaneous

| Option | Description |
|--------|-------------|
| `--version` | Show SONATA version and exit |
| `--help` | Show help message and exit |

## Format Types

- **concise**: Simple text with integrated audio event tags and speaker labels
- **default**: Text with timestamps
- **extended**: Text with timestamps and confidence scores

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
sonata-asr interview.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN
```

### Preprocessing Long Audio

```bash
sonata-asr long_podcast.mp3 --preprocess --split --split-length 60 --split-overlap 10
```

### Customizing Output

```bash
sonata-asr meeting.wav --output meeting_data.json --text-output meeting_transcript.txt --format concise
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

## Additional Information

- For speaker diarization, sign up on [HuggingFace](https://huggingface.co) and accept the user agreement for the pyannote/speaker-diarization models
- Large audio files are automatically processed in segments regardless of the `--split` option, but with this flag you can customize the segmentation parameters
- The `--preprocess` option is recommended for noisy or poorly formatted audio files 

# Transcription Options

| Option | Default | Description |
|--------|---------|-------------|
| `-l`, `--language` | en | Language code (en, ko, zh, ja, fr, de, etc.) |
| `-m`, `--model` | large-v3 | WhisperX model size |
| `-d`, `--device` | cpu | Device to run models on (cpu/cuda) |
| `-e`, `--audio-model` | (None) | Path to custom audio event detection model |
| `-t`, `--threshold` | 0.5 | Threshold for audio event detection |
| `--custom-thresholds` | (None) | Path to JSON file with custom audio event thresholds |
| `--format` | default | Format for text output (concise/default/extended) |
| `--text-output` | (None) | Path to save formatted transcript text file |
| `--preprocess` | (False) | Preprocess audio (convert format and trim silence) |

# Advanced Examples

```bash
# Using custom audio event thresholds
sonata-asr audio.wav --custom-thresholds thresholds.json

# Combining multiple options
sonata-asr audio.mp3 --language ko --device cuda --diarize --offline-diarize --custom-thresholds custom_thresholds.json
``` 