import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from sonata.models.whisperx import WhisperX


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

    def transcribe(self, audio_file: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file using WhisperX.

        Args:
            audio_file: Path to the audio file
            language: Language code for transcription

        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        result = self.whisperx.transcribe(audio_file, language=language)

        # If result is not a dictionary, wrap it in one
        if not isinstance(result, dict):
            print(
                f"Warning: Expected dict result, got {type(result)}. Converting to dict."
            )
            result = {"text": str(result), "segments": []}

        # Ensure result contains expected keys
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

        # Check if segments exist
        if "segments" not in result:
            print("Warning: No 'segments' key in result for word timestamp extraction")
            return words

        # Check for word_segments which is used in newer WhisperX versions
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

        # Extract words from each segment
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
