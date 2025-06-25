from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional
from sonata.constants import LanguageCode
from sonata.utils.text import clean_text_for_language
from sonata.models.model_loader import transcribe_with_model
from sonata.models.korean_asr import KoreanASRModel
import os
import ssl
import io
import sys
import logging
import warnings
import subprocess

# Base environment variables
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check current root logger level
root_logger = logging.getLogger()
current_level = root_logger.level

# Suppress warnings only at ERROR level
if current_level >= logging.ERROR:
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::DeprecationWarning"
    warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
    warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")


class ASRProcessor:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        """Initialize the ASR processor with default model parameters.

        Args:
            device: The device to use for inference ('cpu' or 'cuda')
            compute_type: The compute type for the model
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.align_model = None
        self.align_metadata = None
        self.current_language = None
        self.diarize_model = None
        self.diarize_model_type = None
        self.embedding_model_name = None
        self.clustering_method = None
        self.speaker_embeddings = {}
        self.vad_model = None
        self.scd_model = None
        self.speaker_embedding_model = None
        self.logger = logging.getLogger(__name__)
        self.asr_model = None
        self.korean_asr = None

    def _is_korean_language(self, language: str) -> bool:
        """Check if the language is Korean."""
        return language.lower() in ["ko", "kor", "korean", "kr"]

    def _get_korean_model(self):
        """Lazy load Korean ASR model."""
        if self.korean_asr is None:
            try:
                self.korean_asr = KoreanASRModel(device=self.device)
            except Exception as e:
                print(f"Warning: Could not load Korean ASR model: {str(e)}")
                print("Falling back to standard transcription model")
                self.korean_asr = False
        return self.korean_asr

    def process_audio(
        self,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        show_progress: bool = True,
    ) -> Dict:
        """Process audio file with appropriate ASR model to get transcription with timestamps."""

        if self.asr_model is None or self.current_language != language:
            if show_progress:
                print(
                    f"[ASR] Loading ASR model for language: {language}...",
                    flush=True,
                )
            self.current_language = language

        if not ".wav" in audio_path:
            err_msg = f"Audio file must be a .wav file: {audio_path}"
            self.logger.error(err_msg)
            return {
                "error": err_msg,
                "integrated_transcript": {"plain_text": "", "rich_text": []},
            }

        if show_progress:
            print(f"[ASR] Running speech recognition...", flush=True)
            sys.stdout.flush()

        if self._is_korean_language(language):
            korean_model = self._get_korean_model()
            if korean_model and korean_model is not False:
                if show_progress:
                    print("Using Korean Conformer Transducer model for Korean language")

                try:
                    result = korean_model.transcribe(audio_path, language=language)

                    result_segments = []
                    if "segments" in result:
                        for segment in result["segments"]:
                            if "words" in segment:
                                for word in segment["words"]:
                                    result_segments.append(
                                        {
                                            "start": word.get("start", 0.0),
                                            "end": word.get("end", 0.0),
                                            "content": word.get("word", ""),
                                            "type": "voice",
                                        }
                                    )
                            else:
                                result_segments.append(
                                    {
                                        "start": segment.get("start", 0.0),
                                        "end": segment.get("end", 0.0),
                                        "content": segment.get("text", ""),
                                        "type": "voice",
                                    }
                                )

                    if show_progress:
                        print(
                            f"[ASR] Korean transcription complete with {len(result_segments)} segments."
                        )

                    return result_segments

                except Exception as e:
                    print(f"Error using Korean ASR model: {str(e)}")
                    print("Falling back to standard transcription")

        transcription = transcribe_with_model(
            audio_path, device=self.device, language=language
        )

        if show_progress:
            print(f"[ASR] Transcription complete.", flush=True)

        text = transcription.get("text", "")
        segments = transcription.get("segments", [])

        result_segments = []

        if segments:
            for segment in segments:
                if "words" in segment:
                    for word in segment["words"]:
                        result_segments.append(
                            {
                                "start": word.get("start", 0.0),
                                "end": word.get("end", 0.0),
                                "content": word.get("word", ""),
                                "type": "voice",
                            }
                        )
                else:
                    result_segments.append(
                        {
                            "start": segment.get("start", 0.0),
                            "end": segment.get("end", 0.0),
                            "content": segment.get("text", ""),
                            "type": "voice",
                        }
                    )
        else:
            clean_words = clean_text_for_language(text, language)
            duration = 0.0
            try:
                import librosa

                audio_data, sr = librosa.load(audio_path, sr=None)
                duration = len(audio_data) / sr
            except:
                duration = 10.0

            time_per_word = duration / len(clean_words) if clean_words else 0.0

            for i, word in enumerate(clean_words):
                result_segments.append(
                    {
                        "start": i * time_per_word,
                        "end": (i + 1) * time_per_word,
                        "content": word,
                        "type": "voice",
                    }
                )

        return result_segments
