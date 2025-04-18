# 🔊 Audio Event Detection

SONATA includes advanced audio event detection capabilities that can identify over 500 different types of sounds in your audio files.

## Supported Audio Events

SONATA can detect a wide range of audio events, including:

- Human sounds (laughter, crying, coughing, etc.)
- Animal sounds (barks, meows, bird calls, etc.)
- Musical sounds (instruments, singing, etc.)
- Environmental sounds (wind, rain, thunder, etc.)
- Mechanical sounds (engines, vehicles, alarms, etc.)
- Domestic sounds (doors, typing, microwave, etc.)

For a complete list of detectable audio events, see the [`AudioEventType` enum in constants.py](../sonata/constants.py).

## Detection Sensitivity

SONATA uses a sophisticated detection system that employs different sensitivity thresholds for different types of audio events. The default threshold for most events is 0.5, but certain subtle events like laughter, sighs, or breathing have lower thresholds to ensure they're detected properly.

### Custom Audio Event Thresholds

You can customize the detection sensitivity for specific audio events to better suit your particular use case. This is useful when you need to:

- Increase sensitivity for certain events you're particularly interested in
- Decrease sensitivity for events that might be triggering false positives
- Fine-tune the detection based on your specific audio environment

#### Using Custom Thresholds in Python

```python
from sonata.core.transcriber import IntegratedTranscriber

# Define custom thresholds (values between 0.0 and 1.0)
custom_thresholds = {
    "laughter": 0.05,      # More sensitive (lower threshold)
    "cough": 0.2,          # Less sensitive (higher threshold)
    "music": 0.7,          # Much less sensitive
    "dog": 0.3             # Custom threshold for dog sounds
}

# Initialize transcriber with custom thresholds
transcriber = IntegratedTranscriber(
    asr_model="large-v3",
    device="cpu",
    custom_audio_thresholds=custom_thresholds
)

# Process audio with custom thresholds applied
result = transcriber.process_audio("path/to/audio.wav", language="en")
```

#### Using Custom Thresholds with CLI

Create a JSON file with your custom thresholds:

```json
{
  "laughter": 0.05,
  "giggle": 0.05,
  "cough": 0.2,
  "sneeze": 0.2,
  "music": 0.7
}
```

Then use the `--custom-thresholds` parameter:

```bash
sonata-asr path/to/audio.wav --custom-thresholds path/to/thresholds.json
```

## Example

Here's a sample of how audio events appear in the transcript:

```
[00:05.234] The presenter walked onto the stage [applause]
[00:10.567] Thank you everyone for coming today [laughter]
[00:15.123] Let me share some exciting news about our latest product [music]
```

## Advanced Use Cases

Custom thresholds can be particularly useful in scenarios like:

- **Podcast Analysis**: Increase sensitivity for laughter to capture audience reactions
- **Meeting Transcription**: Reduce sensitivity for keyboard/typing sounds that might be prevalent
- **Nature Recordings**: Customize thresholds for specific bird or animal sounds
- **Music Analysis**: Fine-tune detection of specific instruments or musical elements

For a complete example of using custom thresholds, see the [custom_thresholds_example.py](../test/custom_thresholds_example.py) script. 