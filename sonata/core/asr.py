import os
import numpy as np
import torch
import whisperx
import ssl
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
        self.current_language = None

    def load_models(self, language_code: str = "en"):
        # SSL 인증서 문제 해결을 위한 컨텍스트 설정
        ssl._create_default_https_context = ssl._create_unverified_context

        self.model = whisperx.load_model(
            self.model_name, self.device, compute_type=self.compute_type
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device
        )
        self.current_language = language_code

    def process_audio(
        self, audio_path: str, batch_size: int = 16, language: str = "en"
    ) -> Dict:
        """Process audio file with WhisperX to get transcription with timestamps."""
        if self.model is None or self.current_language != language:
            try:
                self.load_models(language_code=language)
            except Exception as e:
                print(
                    f"Warning: Could not load alignment model for {language}. Falling back to transcription without alignment."
                )
                if self.model is None:
                    self.model = whisperx.load_model(
                        self.model_name, self.device, compute_type=self.compute_type
                    )

        # Transcribe with whisperx
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=batch_size, language=language)

        # Align timestamps if alignment model is available
        if self.align_model is not None:
            try:
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                )
            except Exception as e:
                print(
                    f"Warning: Alignment failed. Using original timestamps. Error: {e}"
                )

        return result

    def get_word_timestamps(self, result: Dict) -> List[Dict]:
        """Extract word-level timestamps from whisperx result."""
        words_with_timestamps = []

        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    words_with_timestamps.append(
                        {
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "confidence": word.get("confidence", 1.0),
                        }
                    )
            else:
                # If word-level alignment is not available, use segment-level timing
                words_with_timestamps.append(
                    {
                        "word": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "confidence": 1.0,
                    }
                )

        return words_with_timestamps
