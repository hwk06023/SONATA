# SONATA Inference Tools

This directory contains tools for running inference using the SONATA ASR package.

## Basic Inference

The `infer.py` script provides a simple way to transcribe a single audio file:

```bash
# Basic usage
python infer.py /path/to/audio.wav

# Specify output file
python infer.py /path/to/audio.wav -o /path/to/output.json

# Change language and model
python infer.py /path/to/audio.wav -l en -m large-v2

# Use CUDA if available
python infer.py /path/to/audio.wav -d cuda

# Also generate formatted text output
python infer.py /path/to/audio.wav --format
```

## Advanced Inference

The `advanced_infer.py` script provides additional features such as batch processing and audio preprocessing:

```bash
# Process a single file with preprocessing
python advanced_infer.py /path/to/audio.wav --preprocess

# Process all audio files in a directory
python advanced_infer.py /path/to/audio_directory/ -o /path/to/output_dir

# Batch process with parallel workers (recommended for multiple files)
python advanced_infer.py /path/to/audio_directory/ --batch --max-workers 4

# Complete example with all options
python advanced_infer.py /path/to/audio_directory/ \
  -o /path/to/output_dir \
  -l en \
  -m large-v3 \
  -d cuda \
  --format \
  --preprocess \
  --batch \
  --max-workers 4
```

The `--preprocess` option performs two key operations:
1. Converts audio to WAV format for optimal compatibility
2. Trims silence from the beginning and end of the file, improving transcription accuracy and reducing processing time

## Python Usage Examples

For those who prefer to use SONATA directly in Python code, we provide examples of integrating the transcription functionality into your applications:

### Simple Transcription

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize transcriber
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# Process audio file
result = transcriber.process_audio("path/to/audio.wav", language="en")

# Access the plain text transcript
transcript = result["integrated_transcript"]["plain_text"]
print(transcript)
```

### Working with Timestamps

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize and process
transcriber = IntegratedTranscriber()
result = transcriber.process_audio("path/to/audio.wav")

# Extract words with timestamps
for item in result["integrated_transcript"]["rich_text"]:
    if item["type"] == "word":
        print(f"{item['content']}: {item['start']:.2f}s - {item['end']:.2f}s")
```

See [`usage_example.py`](usage_example.py) for a complete example demonstrating:
- Basic transcription
- Word-level timestamp extraction
- Working with emotive events
- Additional formatting options

## Output Format

The output JSON file contains:

1. `raw_asr`: The raw output from WhisperX
2. `emotive_events`: Detected emotive events (laughter, sighs, etc.)
3. `integrated_transcript`: The final transcript with:
   - `plain_text`: Text-only transcript
   - `rich_text`: Array of words and events with timestamps

When using the `--format` option, a text file with timestamped content is also generated.

## Example

Transcribe the `podcast_bobbylee.wav` file using the CUDA device:

```bash
python infer.py ../sonata/data/podcast_bobbylee.wav -d cuda
```

This will produce `podcast_bobbylee_transcript.json` in the current directory.

## Performance Tips

1. Use CUDA for faster processing if you have a compatible GPU
2. For long audio files, consider using the advanced script with preprocessing
3. Batch processing can significantly improve throughput when processing multiple files 