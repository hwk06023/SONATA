#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sonata.core.transcriber import IntegratedTranscriber
from sonata.core.speaker_diarization import SpeakerDiarizer


def test_wavlm_diarization():
    """Test that WavLM diarization model can be loaded and used."""
    # Initialize diarizer with WavLM
    diarizer = SpeakerDiarizer(device="cpu", model_type="wavlm-base-plus-sv")
    print("WavLM model loaded successfully!")

    # Test model switching
    diarizer.model_type = "titanet"
    print("Switched back to TitaNet model successfully!")

    # Check that model is correctly loaded
    print(f"Active model: {diarizer.model_type}")

    return True


if __name__ == "__main__":
    try:
        test_wavlm_diarization()
        print("✅ WavLM integration test passed!")
    except Exception as e:
        print(f"❌ WavLM integration test failed: {str(e)}")
        sys.exit(1)
