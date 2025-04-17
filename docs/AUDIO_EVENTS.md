# SONATA Audio Event Detection 🔊

SONATA incorporates the AudioSet AST (Audio Spectrogram Transformer) model to detect a wide range of audio events in your recordings. This document provides details on the audio event detection capabilities.

## Overview

The audio event detection system can identify over 500 different sound categories from the AudioSet ontology. These events are seamlessly integrated into transcripts using tags like `[laughter]` or `[applause]`.

## Audio Event Categories

### 😀 Human Sounds

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Speech | Various speech types | `[male_speech]`, `[female_speech]`, `[child_speech]` |
| Conversation | Multiple people speaking | `[conversation]` |
| Laughter | Different types of laughter | `[laughter]`, `[giggle]`, `[chuckle]` |
| Crying | Crying and sobbing sounds | `[crying]`, `[baby_cry]`, `[whimper]` |
| Vocalizations | Various vocal sounds | `[whistling]`, `[sigh]`, `[groan]` |
| Breathing | Breathing-related sounds | `[breathing]`, `[snoring]`, `[wheeze]` |
| Coughing | Cough and related sounds | `[cough]`, `[throat_clearing]`, `[sneeze]` |

### 👏 Physical Sounds

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Hand Sounds | Sounds made with hands | `[clapping]`, `[finger_snapping]` |
| Body Sounds | Sounds from the body | `[footsteps]`, `[heartbeat]` |
| Crowd Sounds | Sounds from groups | `[cheering]`, `[applause]` |

### 🐕 Animal Sounds

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Dogs | Dog-related sounds | `[dog]`, `[bark]` |
| Cats | Cat-related sounds | `[cat]`, `[meow]` |
| Birds | Bird-related sounds | `[bird_vocalization]` |
| Farm Animals | Farm animal sounds | `[horse]`, `[cow]`, `[moo]`, `[sheep]` |

### 🎵 Music and Instruments

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Music | General music detection | `[music]` |
| String Instruments | String instrument sounds | `[guitar]`, `[violin]` |
| Percussion | Percussion sounds | `[drum]`, `[drum_kit]` |
| Wind Instruments | Wind instrument sounds | `[wind_instrument]`, `[flute]` |
| Bells | Bell sounds | `[bell]` |

### 🌧️ Environmental Sounds

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Weather | Weather-related sounds | `[wind]`, `[rain]`, `[thunder]` |
| Water | Water-related sounds | `[water]`, `[stream]`, `[ocean]`, `[waves]` |
| Fire | Fire-related sounds | `[fire]` |

### 🚗 Mechanical and Transport

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Vehicles | Various vehicle sounds | `[vehicle]`, `[car]`, `[train]`, `[airplane]` |
| Engines | Engine sounds | `[engine]`, `[motor_vehicle]` |
| Emergency | Emergency vehicle sounds | `[police_car]`, `[ambulance]`, `[fire_truck]` |

### 🏠 Domestic Sounds

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Doors | Door-related sounds | `[door]`, `[doorbell]`, `[knock]`, `[slam]` |
| Appliances | Appliance sounds | `[microwave]`, `[vacuum]` |
| Alarms | Various alarm sounds | `[alarm]`, `[telephone]`, `[fire_alarm]` |

### 💥 Miscellaneous

| Category | Description | Tag Example |
|----------|-------------|-------------|
| Impacts | Impact sounds | `[explosion]`, `[gunshot]`, `[bang]`, `[smash]`, `[breaking]` |
| Silence | Detected silence | `[silence]` |
| Noise | General noise | `[noise]` |

## Configuration

### Threshold Settings

You can adjust the sensitivity of audio event detection using the `audio_threshold` parameter:

```python
# More sensitive detection (more events but potentially more false positives)
result = transcriber.process_audio("audio.wav", audio_threshold=0.2)

# Less sensitive detection (fewer events but higher confidence)
result = transcriber.process_audio("audio.wav", audio_threshold=0.4)
```

The default threshold is 0.3. Lower values will detect more audio events but may include more false positives.

### Event-Specific Thresholds

SONATA uses different thresholds for different event types. Subtle sounds like breathing or sighing have lower thresholds than more distinctive sounds like explosions or alarms.

## Technical Implementation

The audio event detection uses the Audio Spectrogram Transformer (AST) model trained on the AudioSet dataset, which includes 527 audio event classes. SONATA converts these to user-friendly tags and integrates them into the transcript according to their timestamps.

## Accessing Audio Events

You can access detailed information about detected audio events:

```python
# Get all detected audio events
audio_events = result["audio_events"]

# Example event data
# {
#   "type": "laughter",
#   "start": 12.5,
#   "end": 14.2,
#   "confidence": 0.89
# }

# Filter for specific event types
laughter_events = [e for e in audio_events if e["type"] == "laughter"]

# Get events above a certain confidence
high_confidence_events = [e for e in audio_events if e["confidence"] > 0.8]
```

## Output Example

Here's an example of audio events integrated into transcript output:

```
[00:05] [SPEAKER_1]: I think we should discuss the project timeline.
[00:08] [laughter]
[00:10] [SPEAKER_2]: That's funny because we just had that meeting yesterday.
[00:15] [door]
[00:17] [SPEAKER_3]: Sorry I'm late everyone.
``` 