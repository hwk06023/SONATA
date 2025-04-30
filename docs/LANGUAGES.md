---
layout: default
title: Supported Languages
nav_order: 5
---

# Supported Languages

SONATA leverages WhisperX (based on Whisper large-v3) to provide high-quality speech recognition across multiple languages with varying levels of transcription accuracy.

## Language Performance

Whisper large-v3 shows improved performance (10-20% error reduction) over previous versions across a wide variety of languages. Below is a summary of supported languages and their performance characteristics.

| Language | Code | Transcription Accuracy | Audio Event Detection | Speaker Diarization | Notes |
|----------|------|------------------------|------------------------|---------------------|-------|
| English (US/UK) | `en` | Excellent (<5% WER) | Full Support | Full Support | Best overall performance |
| Spanish | `es` | Very High (~7% WER) | Full Support | Full Support | Strong performance for most dialects |
| French | `fr` | Very High (~8% WER) | Full Support | Full Support | Good handling of accents |
| German | `de` | Very High (~7% WER) | Full Support | Full Support | Reliable for most contexts |
| Portuguese | `pt` | Very High (~9% WER) | Full Support | Full Support | Works well with both European and Brazilian variants |
| Italian | `it` | Very High (~8% WER) | Full Support | Full Support | Consistent performance |
| Dutch | `nl` | High (~10% WER) | Full Support | Full Support | Good overall reliability |
| Polish | `pl` | High (~12% WER) | Full Support | Full Support | Strong for clear speech |
| Ukrainian | `uk` | High (~12% WER) | Full Support | Full Support | Improved in large-v3 |
| Russian | `ru` | High (~13% WER) | Full Support | Full Support | Better with formal speech |
| Japanese | `ja` | High* (~11% CER) | Full Support | Full Support | *Uses Character Error Rate |
| Chinese (Mandarin) | `zh` | High* (~12% CER) | Full Support | Full Support | *Uses Character Error Rate, supports both traditional and simplified characters |
| Cantonese | `yue` | High* (~15% CER) | Full Support | Full Support | *New in large-v3, uses Character Error Rate |
| Korean | `ko` | High* (~12% CER) | Full Support | Full Support | *Uses Character Error Rate |
| Hindi | `hi` | Moderate (~17% WER) | Full Support | Full Support | Better with standard accents |
| Turkish | `tr` | Moderate (~16% WER) | Full Support | Full Support | Improved in large-v3 |
| Arabic | `ar` | Moderate (~18% WER) | Full Support | Full Support | Handles Modern Standard Arabic best |
| Vietnamese | `vi` | Moderate (~15% WER) | Full Support | Full Support | Better with northern dialects |
| Tamil | `ta` | Moderate (~19% WER) | Full Support | Full Support | Improving with model updates |
| Urdu | `ur` | Moderate (~19% WER) | Full Support | Full Support | Better with clear speech |

Additionally, Whisper large-v3 supports approximately 80 more languages with varying levels of accuracy. Languages with less training data typically have higher error rates (20-60% WER/CER).

## Using Different Languages

To specify a language, use the language code parameter in both the API and CLI:

### CLI Usage

```bash
# Specify the language with the -l or --language flag
sonata-asr audio.wav -l ko

# Example with multiple options
sonata-asr audio.wav -l zh --diarize --deep-detect
```

### Python API Usage

```python
from sonata.core.transcriber import IntegratedTranscriber

# Initialize the transcriber
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# Specify the language in process_audio
result = transcriber.process_audio("audio.wav", language="ja")
```

## Language Auto-detection

If no language is specified, SONATA will attempt to auto-detect the language in the audio. This works best with longer audio samples (>30 seconds).

```bash
# Auto-detect language
sonata-asr audio.wav
```

```python
# Auto-detect language in Python
result = transcriber.process_audio("audio.wav")  # language not specified
detected_language = result["detected_language"]
```

## Performance Factors

Several factors influence transcription accuracy:

1. **Audio Quality**: Clear audio with minimal background noise yields better results
2. **Speaker Accents**: Non-native accents may reduce accuracy
3. **Technical Terminology**: Specialized vocabulary can be challenging
4. **Speech Clarity**: Mumbling or very fast speech may increase error rates
5. **Audio Length**: Longer samples provide more context for accurate transcription

## Tips for Better Transcription

1. **Choose the Right Model**: Use the largest model your hardware can support for best accuracy
2. **Specify Language**: Always specify the language when known rather than relying on auto-detection
3. **Clean Audio**: When possible, use audio with minimal background noise
4. **Pre-processing**: Use audio normalization for better results with quiet recordings
5. **Post-processing**: For specialized domains, consider implementing domain-specific post-processing

## Future Language Support

The SONATA team continuously works to improve language support through:

1. Integration of the latest WhisperX/Whisper models as they become available
2. Fine-tuning for specific languages and domains
3. Expanding audio event detection capabilities across languages
4. Improved handling of code-switching (language mixing within a single audio file)

If you're interested in contributing to language support or have testing data in specific languages, please see our [contribution guidelines](CONTRIBUTING.html). 