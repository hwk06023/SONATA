---
layout: default
title: Offline Diarization
nav_order: 6
permalink: /OFFLINE_DIARIZATION
---

# Offline Diarization Guide 🎙️

SONATA provides an offline mode for speaker diarization that allows you to use this feature without requiring a HuggingFace token for each run.

## Why Use Offline Diarization?

Speaker diarization in SONATA uses PyAnnote models hosted on HuggingFace. These models are gated, meaning they require user authentication through a token. The offline mode offers several advantages:

- 🔒 Works in environments without internet access
- 🚀 Faster execution without authentication checks
- 🤖 Suitable for automated pipelines and batch processing
- 🔄 No need to manage tokens in production environments

## Setup Process

### Step 1: Initial Setup (One-time Only)

To use offline mode, you must first download the necessary model files. This initial setup requires a HuggingFace token **once**.

```bash
# Set up offline models
sonata-asr --setup-offline --hf-token YOUR_HUGGINGFACE_TOKEN
```

This command will:
1. Download `config.yaml` from pyannote/speaker-diarization-3.1
2. Download `pytorch_model.bin` from pyannote/segmentation-3.0
3. Automatically modify the configuration file to point to the local model files
4. Save everything in `~/.sonata/models/` directory

### Step 2: Using Offline Mode

Once setup is complete, you can use diarization without a HuggingFace token:

```bash
sonata-asr path/to/audio.wav --diarize --offline-diarize --offline-config ~/.sonata/models/offline_config.yaml
```

## Integration in Python Code

You can also use offline diarization directly in your Python code:

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize with offline diarization
transcriber = IntegratedTranscriber(
    asr_model="large-v3", 
    device="cuda",
    offline_diarization=True,
    offline_config_path="~/.sonata/models/offline_config.yaml"
)

# Process audio with diarization
result = transcriber.process_audio(
    "path/to/audio.wav",
    diarize=True,  # Enable diarization
    # No hf_token needed!
)

# Get the transcript with speaker labels
formatted_transcript = transcriber.get_formatted_transcript(result, format_type="concise")
print(formatted_transcript)
```

## Important Notes

1. **License Compliance**: Even when using offline mode, you are still bound by the PyAnnote models' license terms.
2. **Storage Location**: Model files are stored in the `~/.sonata/models/` directory by default.
3. **Internet Requirement**: Internet connection is only needed for the initial setup, not for subsequent usage.
4. **File Size**: The downloaded model files require approximately 400MB of disk space.
5. **Updates**: If PyAnnote releases new model versions, you may need to run the setup again to update your local files.

## Troubleshooting

### Common Issues

1. **Setup Fails**:
   - Ensure your HuggingFace token has the appropriate permissions
   - Verify you've accepted the user agreements for both models on HuggingFace

2. **Model Not Found**:
   - Check that the paths in the `--offline-config` are correct
   - Verify the model files exist in the specified locations

3. **Permission Errors**:
   - Ensure your user has read/write access to `~/.sonata/models/`

### Manual Setup Alternative

If the automatic setup doesn't work, you can perform the process manually:

1. Download `config.yaml` from [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1/blob/main/config.yaml)
2. Download `pytorch_model.bin` from [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0/blob/main/pytorch_model.bin)
3. Create a directory: `mkdir -p ~/.sonata/models/`
4. Place both files in this directory
5. Edit `config.yaml` to update the segmentation field:
   ```yaml
   segmentation: '~/.sonata/models/pytorch_model.bin'
   ```
6. Use this modified config file with the `--offline-config` flag 