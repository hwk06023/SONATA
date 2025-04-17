import os
import numpy as np
import torch
import whisperx
import ssl
import io
import sys
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from typing import Dict, List, Union, Tuple, Optional
from sonata.constants import LanguageCode
from tqdm import tqdm

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

    for logger_name in ["pytorch_lightning", "whisperx", "pyannote.audio"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


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

        # Current logging level is irrelevant when loading models
        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Create context managers for filtering stderr/stdout
        redirect_context = redirect_stdout(stdout_buffer)
        redirect_err_context = redirect_stderr(stderr_buffer)

        # Create context manager for filtering warnings
        warning_context = warnings.catch_warnings()

        try:
            # Temporarily set all logging to ERROR level
            logging.getLogger().setLevel(logging.ERROR)

            # Filter warnings
            warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
            warnings.filterwarnings("ignore", message=".*set_stage.*")
            warnings.filterwarnings(
                "ignore", message=".*Trying to infer the `batch_size`.*"
            )

            # Run all context managers
            with redirect_context, redirect_err_context, warning_context:
                # Load model
                self.model = whisperx.load_model(
                    self.model_name,
                    self.device,
                    compute_type=self.compute_type,
                    language=language_code,  # Pass language parameter directly
                )
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

        # Ensure preset_language is set
        if hasattr(self.model, "preset_language"):
            self.model.preset_language = language_code

        try:
            # Reset warning filtering
            warning_context = warnings.catch_warnings()

            try:
                # Temporarily set all logging to ERROR level
                logging.getLogger().setLevel(logging.ERROR)

                # Filter warnings
                warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
                warnings.filterwarnings("ignore", message=".*set_stage.*")
                warnings.filterwarnings(
                    "ignore", message=".*Trying to infer the `batch_size`.*"
                )

                # Run all context managers
                with redirect_stdout(stdout_buffer), redirect_stderr(
                    stderr_buffer
                ), warning_context:
                    self.align_model, self.align_metadata = whisperx.load_align_model(
                        language_code=language_code, device=self.device
                    )
                self.current_language = language_code
            finally:
                # Restore original logging level
                logging.getLogger().setLevel(original_level)
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
        show_progress: bool = True,
    ) -> Dict:
        """Process audio file with WhisperX to get transcription with timestamps.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            batch_size: Batch size for processing
            show_progress: Whether to show progress indicators

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
            if show_progress:
                print(f"[ASR] Loading models for language: {language}...", flush=True)

            try:
                self.load_models(language_code=language)
                if show_progress:
                    print(f"[ASR] Models loaded successfully.", flush=True)
            except Exception as e:
                print(
                    f"Warning: Could not load alignment model for {language}. Falling back to transcription without alignment."
                )
                if self.model is None:
                    # Set up comprehensive warning suppression
                    original_level = logging.getLogger().level
                    stdout_buffer = io.StringIO()
                    stderr_buffer = io.StringIO()

                    try:
                        # Temporarily suppress all logging
                        logging.getLogger().setLevel(logging.ERROR)

                        # Redirect both stdout and stderr
                        with redirect_stdout(stdout_buffer), redirect_stderr(
                            stderr_buffer
                        ):
                            if show_progress:
                                print(f"[ASR] Loading base model...", flush=True)

                            self.model = whisperx.load_model(
                                self.model_name,
                                self.device,
                                compute_type=self.compute_type,
                            )

                            if show_progress:
                                print(
                                    f"[ASR] Base model loaded successfully.", flush=True
                                )
                    finally:
                        # Restore original logging level
                        logging.getLogger().setLevel(original_level)

        # Print parameters for debugging
        print(
            f"Transcribing with parameters - language: {language}, batch_size: {batch_size}"
        )

        # Transcribe with whisperx
        if show_progress:
            print(f"[ASR] Loading audio: {audio_path}", flush=True)

        audio = whisperx.load_audio(audio_path)

        if show_progress:
            print(f"[ASR] Running speech recognition...", flush=True)
            sys.stdout.flush()

        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,  # Explicitly pass language parameter
        )

        if show_progress:
            print(
                f"[ASR] Transcription complete. Processing {len(result.get('segments', []))} segments.",
                flush=True,
            )

        # Align timestamps if alignment model is available
        if self.align_model is not None:
            try:
                if show_progress:
                    print(f"[ASR] Aligning timestamps...", flush=True)

                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                )

                if show_progress:
                    print(f"[ASR] Alignment complete.", flush=True)
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
