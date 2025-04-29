# 🎙️ Speaker Diarization in SONATA

SONATA provides state-of-the-art speaker diarization capabilities that can identify different speakers in audio recordings. This feature is especially useful for transcribing conversations, interviews, meetings, podcasts, and other multi-speaker audio content.

## Overview

Speaker diarization is the process of partitioning an audio stream into segments according to the speaker identity. SONATA implements an advanced diarization pipeline that includes:

1. **Voice Activity Detection (VAD)** - Identifies speech segments in the audio using Silero VAD
2. **Speaker Embedding Extraction** - Creates speaker-specific embeddings using WavLM
3. **Speaker Change Detection** - Identifies points where speakers change
4. **Speaker Clustering** - Groups speech segments by speaker identity
5. **Speaker Assignment** - Assigns speaker labels to each speech segment and integrates with transcription

## Technical Implementation

SONATA's speaker diarization uses the following components:

### Voice Activity Detection

[Silero VAD](https://github.com/snakers4/silero-vad) provides accurate voice activity detection, identifying speech segments while filtering out silence and non-speech audio. It creates a foundation for further speaker analysis by focusing only on relevant speech segments.

### Speaker Embeddings

SONATA uses [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) (specifically the `microsoft/wavlm-base-plus-sv` model) to extract speaker embeddings - high-dimensional vectors that capture the unique vocal characteristics of each speaker. These embeddings are used to distinguish between different speakers in the recording.

WavLM is Microsoft's state-of-the-art model for audio and speech understanding, and the speaker verification variant is specifically optimized for tasks like speaker diarization.

### Speaker Clustering

The extracted embeddings are clustered using hierarchical clustering algorithms (specifically Agglomerative Clustering) to group speech segments by speaker identity. If the number of speakers is known in advance, this can be specified; otherwise, SONATA will attempt to estimate the optimal number of speakers using silhouette analysis.

## Usage

### Command Line Interface

```bash
# Basic diarization
sonata-asr path/to/audio.wav --diarize

# Specify the number of speakers
sonata-asr path/to/audio.wav --diarize --num-speakers 3

# Full transcription with speaker diarization
sonata-asr path/to/audio.wav --diarize --output transcript.json
```

### Python API

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize the transcriber
transcriber = IntegratedTranscriber(device="cpu")

# Transcribe with speaker diarization
result = transcriber.process_audio(
    audio_path="path/to/audio.wav",
    language="en",
    diarize=True,
    num_speakers=3  # Optional: specify if known
)

# Access speaker-labeled transcript
for segment in result["integrated_transcript"]["rich_text"]:
    if "speaker" in segment:
        print(f"{segment['speaker']}: {segment['text']}")
```

### Standalone Diarization

For applications that only need speaker diarization without transcription:

```python
from sonata.core.speaker_diarization import SpeakerDiarizer

# Initialize the diarizer
diarizer = SpeakerDiarizer(device="cpu")

# Perform diarization
speaker_segments = diarizer.diarize(
    audio_path="path/to/audio.wav",
    num_speakers=3,  # Optional: specify if known
    show_progress=True
)

# Print results
for segment in speaker_segments:
    print(f"{segment.start:.2f} - {segment.end:.2f}: {segment.speaker}")
```

## Output Format

The speaker diarization output is integrated into SONATA's transcription results:

```json
{
  "integrated_transcript": {
    "rich_text": [
      {
        "text": "Hello, welcome to the podcast.",
        "start": 0.5,
        "end": 2.3,
        "speaker": "SPEAKER_1"
      },
      {
        "text": "Thanks for having me today.",
        "start": 2.8,
        "end": 4.1,
        "speaker": "SPEAKER_2"
      }
    ]
  }
}
```

## Performance Considerations

- Speaker diarization is a computationally intensive process, especially for longer recordings.
- The accuracy of speaker identification improves with longer speech segments per speaker.
- Performance is generally better when the number of speakers is provided in advance.
- For optimal results, use high-quality audio with minimal background noise.
- GPU acceleration is recommended for processing longer recordings.

## Limitations

- Speaker identities are assigned arbitrary labels (SPEAKER_1, SPEAKER_2, etc.) rather than actual names.
- Very short utterances (less than 1-2 seconds) may be difficult to assign to the correct speaker.
- Overlapping speech presents challenges and may be assigned to the dominant speaker.
- Performance may vary based on audio quality, background noise, and the number of speakers.

## Advanced Configuration

For advanced users, SONATA provides ways to customize the diarization process:

1. **Custom VAD parameters** - Adjust voice activity detection sensitivity
2. **Clustering algorithms** - Use different speaker clustering methods
3. **Embedding models** - Configure or substitute alternative speaker embedding extraction

Please refer to the API documentation for details on advanced configuration options. 