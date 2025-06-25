import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from sonata.models.whisperx import WhisperX
from sonata.models.korean_asr import KoreanASRModel


class ASRModel:
    def __init__(self, model_name: str = "large-v3", device: str = None):
        """Initialize the ASR model.

        Args:
            model_name: Name of the Whisper model to use
            device: Device to run the model on. If None, will use CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.whisperx = WhisperX(model_name=model_name, device=device)
        self.korean_asr = None

    def _get_korean_model(self):
        """Lazy load Korean ASR model."""
        if self.korean_asr is None:
            try:
                self.korean_asr = KoreanASRModel(device=self.device)
            except Exception as e:
                print(f"Warning: Could not load Korean ASR model: {str(e)}")
                print("Falling back to WhisperX for Korean language")
                self.korean_asr = False
        return self.korean_asr

    def _is_korean_language(self, language: str) -> bool:
        """Check if the language is Korean."""
        return language.lower() in ["ko", "kor", "korean", "kr"]

    def transcribe(self, audio_file: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file using appropriate ASR model.

        Args:
            audio_file: Path to the audio file
            language: Language code for transcription

        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if self._is_korean_language(language):
            korean_model = self._get_korean_model()
            if korean_model and korean_model is not False:
                print("Using Korean Conformer Transducer model for Korean language")
                return korean_model.transcribe(audio_file, language=language)
            else:
                print("Korean ASR model not available, using WhisperX")

        result = self.whisperx.transcribe(audio_file, language=language)

        if not isinstance(result, dict):
            print(
                f"Warning: Expected dict result, got {type(result)}. Converting to dict."
            )
            result = {"text": str(result), "segments": []}

        if "segments" not in result:
            print(
                "Warning: 'segments' key missing in transcription result. Adding empty list."
            )
            result["segments"] = []

        if "text" not in result:
            print(
                "Warning: 'text' key missing in transcription result. Adding empty string."
            )
            result["text"] = ""

        return result

    def get_word_timestamps(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from transcription result.

        Args:
            result: Transcription result from transcribe method

        Returns:
            List of words with start and end times
        """
        words = []

        if "segments" not in result:
            print("Warning: No 'segments' key in result for word timestamp extraction")
            return words

        if "word_segments" in result:
            print("Using word_segments for timestamp extraction")
            for word_segment in result["word_segments"]:
                words.append(
                    {
                        "word": word_segment.get("word", ""),
                        "start": word_segment.get("start", 0.0),
                        "end": word_segment.get("end", 0.0),
                        "score": word_segment.get("score", 0.0),
                    }
                )
            return words

        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    try:
                        word_entry = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "score": word.get("score", 0.0),
                        }
                        words.append(word_entry)
                    except Exception as e:
                        print(f"Error processing word: {str(e)}, word data: {word}")
                        continue

        return words
