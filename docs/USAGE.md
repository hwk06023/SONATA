---
layout: default
title: Usage Guide
nav_order: 3
permalink: /USAGE
---

# SONATA Usage Guide 📝

This document provides detailed examples for using SONATA in different scenarios.

## Basic Usage

### Initializing the Transcriber

```python
from sonata.core.transcriber import IntegratedTranscriber

# Basic initialization (CPU)
transcriber = IntegratedTranscriber()

# With specific model and device
transcriber = IntegratedTranscriber(
    asr_model="large-v3",  # Options: tiny, base, small, medium, large, large-v2, large-v3
    device="cuda",         # Options: cpu, cuda, mps (Mac)
    compute_type="float16" # Options: float32, float16, int8
)
```

### Processing Audio Files

```python
# Basic transcription with default options
result = transcriber.process_audio("path/to/audio.wav")

# Transcription with specific language
result = transcriber.process_audio(
    audio_path="path/to/audio.wav",
    language="ko"  # Specify language code (e.g., en, ko, zh)
)

# Adjust audio event detection threshold (lower = more sensitive)
result = transcriber.process_audio(
    audio_path="path/to/audio.wav",
    audio_threshold=0.2  # Default is 0.3
)
```

### Saving and Retrieving Results

```python
# Save output to JSON file
transcriber.save_result(result, "output.json")

# Get plain text transcript
plain_text = result["integrated_transcript"]["plain_text"]

# Get formatted transcript with different options
concise_format = transcriber.get_formatted_transcript(result, format_type="concise")
default_format = transcriber.get_formatted_transcript(result, format_type="default")
extended_format = transcriber.get_formatted_transcript(result, format_type="extended")

# Save formatted transcript to a text file
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(concise_format)
```

## Advanced Usage

### Speaker Diarization

```python
# Process audio with speaker diarization
result = transcriber.process_audio(
    audio_path="path/to/audio.wav",
    diarize=True,
    hf_token="YOUR_HUGGINGFACE_TOKEN",  # Required for diarization
    min_speakers=2,  # Optional: minimum number of speakers
    max_speakers=5   # Optional: maximum number of speakers
)

# Get transcript with speaker labels
transcript = transcriber.get_formatted_transcript(result, format_type="concise")
```

### Working with Timestamps

```python
# Extract words with timestamps
for item in result["integrated_transcript"]["rich_text"]:
    if item["type"] == "word":
        word = item["content"]
        start_time = item["start"]
        end_time = item["end"]
        print(f"{word}: {start_time:.2f}s - {end_time:.2f}s")
        
    # You can also access audio events
    elif item["type"] == "audio_event":
        event_type = item["event_type"]
        start_time = item["start"]
        confidence = item["confidence"]
        print(f"{event_type}: {start_time:.2f}s (confidence: {confidence:.2f})")
```

### Processing Long Audio

For very long audio files, use the split option:

```python
from sonata.utils.audio import split_audio

# Split audio into manageable segments
split_dir = "splits_directory"
segments = split_audio(
    "path/to/long_audio.wav",
    split_dir,
    segment_length=30,  # Length of each segment in seconds
    overlap=5           # Overlap between segments in seconds
)

# Process each segment and combine results
all_results = []
for segment in segments:
    result = transcriber.process_audio(segment["path"])
    all_results.append(result)
    
# Combine results (simplified example)
combined_plain_text = " ".join([r["integrated_transcript"]["plain_text"] for r in all_results])
```

### Audio Preprocessing

```python
from sonata.utils.audio import convert_audio_file, trim_silence

# Convert file format
converted_file = convert_audio_file("input.mp3", "output.wav")

# Trim silence
trimmed_file = trim_silence("input.wav", silence_threshold=-50, min_silence_len=500)

# Process preprocessed audio
result = transcriber.process_audio(trimmed_file)
```

## Working with Results

The result dictionary contains three main sections:

```python
{
    "raw_asr": {
        # Original WhisperX output
        "segments": [...],  # List of transcript segments
        "language": "en"    # Detected language
    },
    "audio_events": [
        # List of detected audio events
        {
            "type": "laughter",
            "start": 12.5,
            "end": 14.2,
            "confidence": 0.89
        },
        ...
    ],
    "integrated_transcript": {
        # Integrated transcript with words and audio events
        "plain_text": "This is a transcript with [laughter] and more text.",
        "rich_text": [
            # Detailed list of words and events with timing information
            {"type": "word", "content": "This", "start": 0.5, "end": 0.7, ...},
            ...
        ]
    }
}
```

## Performance Tips

- Use GPU acceleration (`device="cuda"`) for faster processing
- Set `compute_type="float16"` for reduced memory usage with minimal accuracy loss
- For long audio files, use the split audio functionality
- Preprocess audio to remove silence for faster and more accurate results 