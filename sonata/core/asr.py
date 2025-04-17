import os
import numpy as np
import torch
import whisperx
import ssl
from typing import Dict, List, Union, Tuple, Optional
from sonata.constants import LanguageCode


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

    def load_models(self, language_code: str = LanguageCode.ENGLISH.value):
        """Load WhisperX and alignment models for the specified language.

        Args:
            language_code: ISO language code (e.g., "en", "ko", "zh")
        """
        ssl._create_default_https_context = ssl._create_unverified_context

        # Load model with explicit language parameter
        self.model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=language_code,  # Pass language parameter directly
        )

        # Ensure preset_language is set
        if hasattr(self.model, "preset_language"):
            self.model.preset_language = language_code

        try:
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
            self.current_language = language_code
        except Exception as e:
            print(
                f"Warning: Could not load alignment model for {language_code}. Falling back to transcription without alignment."
            )
            self.align_model = None
            self.align_metadata = None
            self.current_language = language_code

    def process_audio(
        self,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        batch_size: int = 16,
    ) -> Dict:
        """Process audio file with WhisperX to get transcription with timestamps.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            batch_size: Batch size for processing

        Returns:
            Dictionary containing transcription results
        """
        # Ensure batch_size is an integer
        if not isinstance(batch_size, int):
            print(
                f"Warning: batch_size must be an integer. Got {type(batch_size)}. Using default value 16."
            )
            batch_size = 16

        # Always check if models need to be loaded or reloaded
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

        # Print parameters for debugging
        print(
            f"Transcribing with parameters - language: {language}, batch_size: {batch_size}"
        )

        # Transcribe with whisperx
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,  # Explicitly pass language parameter
        )

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

        # First, check if the result has the expected structure
        if "segments" not in result:
            print(
                f"Warning: WhisperX result does not contain 'segments'. Keys: {list(result.keys())}"
            )
            # Create a minimal output with the whole text if available
            if "text" in result:
                return [
                    {
                        "word": result["text"],
                        "start": 0.0,
                        "end": 1.0,
                        "confidence": 1.0,
                    }
                ]
            return []

        for segment in result["segments"]:
            try:
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        # Check if word has the required keys
                        if "word" in word and "start" in word and "end" in word:
                            words_with_timestamps.append(
                                {
                                    "word": word["word"],
                                    "start": word["start"],
                                    "end": word["end"],
                                    "confidence": word.get("confidence", 1.0),
                                }
                            )
                        else:
                            # Handle missing keys
                            missing_keys = [
                                k for k in ["word", "start", "end"] if k not in word
                            ]
                            print(
                                f"Warning: Word missing required keys: {missing_keys}. Available keys: {list(word.keys())}"
                            )
                            # Use available keys or defaults
                            words_with_timestamps.append(
                                {
                                    "word": word.get("word", "[UNKNOWN]"),
                                    "start": word.get(
                                        "start", segment.get("start", 0.0)
                                    ),
                                    "end": word.get("end", segment.get("end", 0.0)),
                                    "confidence": word.get("confidence", 0.5),
                                }
                            )
                else:
                    # If word-level alignment is not available, use segment-level timing
                    if "text" in segment and "start" in segment and "end" in segment:
                        words_with_timestamps.append(
                            {
                                "word": segment["text"],
                                "start": segment["start"],
                                "end": segment["end"],
                                "confidence": segment.get("confidence", 1.0),
                            }
                        )
                    else:
                        # Handle missing segment keys
                        missing_keys = [
                            k for k in ["text", "start", "end"] if k not in segment
                        ]
                        print(
                            f"Warning: Segment missing required keys: {missing_keys}. Available keys: {list(segment.keys())}"
                        )
                        # Use available keys or defaults
                        words_with_timestamps.append(
                            {
                                "word": segment.get("text", "[UNKNOWN]"),
                                "start": segment.get("start", 0.0),
                                "end": segment.get("end", 0.0),
                                "confidence": segment.get("confidence", 0.5),
                            }
                        )
            except Exception as e:
                print(f"Warning: Error processing segment: {e}")
                # Continue with next segment

        return words_with_timestamps
