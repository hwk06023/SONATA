import os
import numpy as np
import torch
import whisperx
from typing import Dict, List, Union, Tuple, Optional


class ASRProcessor:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.align_model = None
        self.align_metadata = None

    def load_models(self):
        self.model = whisperx.load_model(
            self.model_name, self.device, compute_type=self.compute_type
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

    def process_audio(
        self, audio_path: str, batch_size: int = 16, language: str = "en"
    ) -> Dict:
        """Process audio file with WhisperX to get transcription with timestamps."""
        if self.model is None:
            self.load_models()

        # Transcribe with whisperx
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=batch_size, language=language)

        # Align timestamps
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
        )

        return result

    def get_word_timestamps(self, result: Dict) -> List[Dict]:
        """Extract word-level timestamps from whisperx result."""
        words_with_timestamps = []

        for segment in result["segments"]:
            for word in segment["words"]:
                words_with_timestamps.append(
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("confidence", 1.0),
                    }
                )

        return words_with_timestamps
