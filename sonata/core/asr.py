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
        self.diarize_model = None

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

    def load_diarize_model(
        self, hf_token: Optional[str] = None, show_progress: bool = True
    ):
        """Load the speaker diarization model.

        Args:
            hf_token: Hugging Face token for model access
            show_progress: Whether to display progress messages
        """
        if self.diarize_model is None:
            if show_progress:
                print(f"[ASR] Loading diarization model...", flush=True)

            # Suppress warnings and logging during model loading
            original_level = logging.getLogger().level
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                # Temporarily set all logging to ERROR level
                logging.getLogger().setLevel(logging.ERROR)

                # Redirect both stdout and stderr
                with redirect_stdout(stdout_buffer), redirect_stderr(
                    stderr_buffer
                ), warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=hf_token, device=self.device
                    )

                if show_progress:
                    print(f"[ASR] Diarization model loaded successfully.", flush=True)
            except Exception as e:
                print(f"Warning: Could not load diarization model. Error: {str(e)}")
                self.diarize_model = None
            finally:
                # Restore original logging level
                logging.getLogger().setLevel(original_level)

    def process_audio(
        self,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        batch_size: int = 16,
        show_progress: bool = True,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hf_token: Optional[str] = None,
    ) -> Dict:
        """Process audio file with WhisperX to get transcription with timestamps.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            batch_size: Batch size for processing
            show_progress: Whether to show progress indicators
            diarize: Whether to perform speaker diarization
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization
            hf_token: HuggingFace token for diarization model (required if diarize=True)

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

        # Perform speaker diarization if requested
        if diarize:
            if show_progress:
                print(f"[ASR] Performing speaker diarization...", flush=True)

            # Load diarization model if not already loaded
            if self.diarize_model is None:
                self.load_diarize_model(hf_token=hf_token, show_progress=show_progress)

            if self.diarize_model is not None:
                try:
                    # Perform diarization
                    diarize_segments = self.diarize_model(
                        audio, min_speakers=min_speakers, max_speakers=max_speakers
                    )

                    # Assign speakers to segments
                    result = whisperx.assign_word_speakers(diarize_segments, result)

                    if show_progress:
                        print(f"[ASR] Speaker diarization complete.", flush=True)
                except Exception as e:
                    print(f"Warning: Speaker diarization failed. Error: {str(e)}")
            else:
                print(
                    f"Warning: Speaker diarization was requested but the model couldn't be loaded."
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
            # Check for word-level information
            if "words" in segment:
                for word_data in segment["words"]:
                    word_with_time = {
                        "word": word_data["word"],
                        "start": word_data["start"],
                        "end": word_data["end"],
                    }
                    if "score" in word_data:
                        word_with_time["score"] = word_data["score"]
                    if "speaker" in word_data:
                        word_with_time["speaker"] = word_data["speaker"]
                    words_with_timestamps.append(word_with_time)
            else:
                # Fallback if no word-level data (shouldn't happen with alignment)
                words_with_timestamps.append(
                    {
                        "word": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                    }
                )

        return words_with_timestamps
