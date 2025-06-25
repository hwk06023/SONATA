import os
import torch
import numpy as np
import librosa
from typing import Dict, List, Optional, Any, Union

try:
    import nemo.collections.asr as nemo_asr

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print(
        "Warning: NeMo toolkit not available. Korean ASR functionality will be limited."
    )


class KoreanASRModel:
    def __init__(
        self,
        model_name: str = "eesungkim/stt_kr_conformer_transducer_large",
        device: str = None,
    ):
        """Initialize the Korean ASR model.

        Args:
            model_name: Name of the Korean Conformer Transducer model to use
            device: Device to run the model on. If None, will use CUDA if available.
        """
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NeMo toolkit is required for Korean ASR. Please install with: pip install nemo-toolkit[all]>=1.23.0"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Korean Conformer Transducer model from HuggingFace."""
        try:
            print(f"Loading Korean ASR model: {self.model_name}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            print("Korean ASR model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Korean ASR model: {str(e)}")

    def transcribe(self, audio_file: str, language: str = "ko") -> Dict[str, Any]:
        """Transcribe audio file using Korean Conformer Transducer.

        Args:
            audio_file: Path to the audio file
            language: Language code (should be 'ko' for Korean)

        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if self.model is None:
            self._load_model()

        try:
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)

            transcriptions = self.model.transcribe([audio_file])

            if not transcriptions or len(transcriptions) == 0:
                return {"text": "", "segments": []}

            text = (
                transcriptions[0]
                if isinstance(transcriptions, list)
                else str(transcriptions)
            )

            duration = len(audio) / sr

            result = {
                "text": text,
                "segments": [
                    {
                        "start": 0.0,
                        "end": duration,
                        "text": text,
                        "words": self._extract_word_timestamps(text, duration),
                    }
                ],
                "language": language,
            }

            return result

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return {"text": "", "segments": []}

    def _extract_word_timestamps(
        self, text: str, duration: float
    ) -> List[Dict[str, Any]]:
        """Extract approximate word timestamps from transcribed text.

        Note: This is a simplified approach. The Korean Conformer model doesn't provide
        word-level timestamps by default, so we estimate them based on text length.
        """
        words = text.split()
        if not words:
            return []

        word_timestamps = []
        time_per_word = duration / len(words)

        for i, word in enumerate(words):
            start_time = i * time_per_word
            end_time = (i + 1) * time_per_word

            word_timestamps.append(
                {"word": word, "start": start_time, "end": end_time, "score": 1.0}
            )

        return word_timestamps

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

        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    try:
                        word_entry = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "score": word.get("score", 1.0),
                        }
                        words.append(word_entry)
                    except Exception as e:
                        print(f"Error processing word: {str(e)}, word data: {word}")
                        continue

        return words

    def is_korean_language(self, language: str) -> bool:
        """Check if the language is Korean."""
        return language.lower() in ["ko", "kor", "korean", "kr"]
