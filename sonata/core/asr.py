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
        """Initialize the ASR processor with default model parameters.

        Args:
            model_name: The Whisper model to use
            device: The device to use for inference ('cpu' or 'cuda')
            compute_type: The compute type for the model
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.current_language = None
        self.diarize_model = None
        self.logger = logging.getLogger(__name__)

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
                logging.getLogger().level = logging.ERROR

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
                logging.getLogger().level = original_level
        except Exception as e:
            print(
                f"Warning: Could not load alignment model for {language_code}. Falling back to transcription without alignment."
            )
            self.align_model = None
            self.align_metadata = None
            self.current_language = language_code

    def load_diarize_model(
        self,
        hf_token: Optional[str] = None,
        show_progress: bool = True,
        offline_mode: bool = False,
        offline_config_path: Optional[str] = None,
    ):
        """Load the speaker diarization model.

        Args:
            hf_token: Hugging Face token for model access
            show_progress: Whether to display progress messages
            offline_mode: Whether to use offline mode
            offline_config_path: Path to offline config.yaml file
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

                    if offline_mode and offline_config_path:
                        # For offline mode, use local config directly
                        from pyannote.audio import Pipeline

                        # Expand user directory if needed (e.g., ~ to /home/user)
                        if offline_config_path.startswith("~"):
                            offline_config_path = os.path.expanduser(
                                offline_config_path
                            )

                        if show_progress:
                            print(
                                f"[ASR] Using offline diarization model from {offline_config_path}",
                                flush=True,
                            )

                        # Load directly from config file path, no token needed
                        self.diarize_model = Pipeline.from_pretrained(
                            checkpoint_path=offline_config_path,
                        )
                    else:
                        # For online mode, use standard HuggingFace loading
                        if not hf_token:
                            raise ValueError(
                                "HuggingFace token is required for online diarization"
                            )

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

    def diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            num_speakers: Fixed number of speakers (takes precedence over min/max)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            show_progress: Whether to show progress indicators

        Returns:
            List of diarization segments with speaker IDs and timestamps
        """
        if self.diarize_model is None:
            raise RuntimeError("Diarization model is not loaded")

        if show_progress:
            print(f"[ASR] Processing audio for diarization...", flush=True)

        # Load audio for diarization
        try:
            # Load audio using whisperx utility
            audio = whisperx.load_audio(audio_path)

            # Perform diarization
            if hasattr(self.diarize_model, "__call__"):
                # Direct Pipeline (offline mode)
                if show_progress:
                    print(
                        f"[ASR] Extracting speaker embeddings with ResNet...",
                        flush=True,
                    )
                    # The PyAnnote pipeline has internal steps including ResNet embedding extraction
                    from tqdm import tqdm
                    import time
                    import warnings

                    # Create progress bar for ResNet embedding
                    with tqdm(total=100, desc="Speaker embedding", unit="%") as pbar:
                        # Start in a separate thread to show progress while model runs
                        start_time = time.time()

                        # Execute diarization
                        progress_percent = 0
                        diarize_segments = None

                        # Run in the main thread but update progress bar periodically
                        import threading

                        def update_progress():
                            nonlocal progress_percent
                            # Update progress bar incrementally until we reach ~90%
                            # The final 10% will be filled when the process completes
                            while progress_percent < 90 and diarize_segments is None:
                                elapsed = time.time() - start_time
                                # Update more frequently at the beginning, then slow down
                                if elapsed > 0.5:
                                    increment = max(1, min(5, int(elapsed / 2)))
                                    if progress_percent + increment <= 90:
                                        pbar.update(increment)
                                        progress_percent += increment
                                time.sleep(0.5)

                        # Start progress updater thread
                        progress_thread = threading.Thread(target=update_progress)
                        progress_thread.daemon = True
                        progress_thread.start()

                        try:
                            # Run actual diarization - suppress warnings that cause the process to die
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore", message=".*degrees of freedom is <= 0.*"
                                )
                                warnings.filterwarnings("ignore", category=UserWarning)

                                # Prepare diarization parameters
                                diarization_params = {}
                                if num_speakers is not None:
                                    diarization_params["num_speakers"] = num_speakers
                                else:
                                    if min_speakers is not None:
                                        diarization_params[
                                            "min_speakers"
                                        ] = min_speakers
                                    if max_speakers is not None:
                                        diarization_params[
                                            "max_speakers"
                                        ] = max_speakers

                                diarize_segments = self.diarize_model(
                                    audio_path,  # Pipeline expects path, not audio data
                                    **diarization_params,  # Pass conditional parameters
                                )
                            # Complete the progress bar
                            pbar.update(100 - progress_percent)
                        except Exception as e:
                            # Complete the progress bar even if there's an error
                            pbar.update(100 - progress_percent)
                            raise e
                else:
                    # Suppress warnings in non-progress mode too
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*degrees of freedom is <= 0.*"
                        )
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)

                        # Prepare diarization parameters
                        diarization_params = {}
                        if num_speakers is not None:
                            diarization_params["num_speakers"] = num_speakers
                        else:
                            if min_speakers is not None:
                                diarization_params["min_speakers"] = min_speakers
                            if max_speakers is not None:
                                diarization_params["max_speakers"] = max_speakers

                        diarize_segments = self.diarize_model(
                            audio_path,  # Pipeline expects path, not audio data
                            **diarization_params,  # Pass conditional parameters
                        )

                # Convert output format to match whisperx format
                result = []
                for segment, track, label in diarize_segments.itertracks(
                    yield_label=True
                ):
                    # Ensure the speaker label is a string (SPEAKER_00, SPEAKER_01, etc.)
                    # Some diarization models might return non-string values
                    if isinstance(label, str):
                        speaker_label = label
                    else:
                        # Convert to string format expected by whisperX
                        speaker_label = f"SPEAKER_{str(label).zfill(2)}"

                    result.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "speaker": speaker_label,
                        }
                    )
                return result
            else:
                # WhisperX DiarizationPipeline
                if show_progress:
                    print(
                        f"[ASR] Extracting speaker embeddings with ResNet...",
                        flush=True,
                    )
                    from tqdm import tqdm
                    import time
                    import warnings

                    # Create progress bar for ResNet embedding
                    with tqdm(total=100, desc="Speaker embedding", unit="%") as pbar:
                        # Start in a separate thread to show progress while model runs
                        start_time = time.time()

                        # Execute diarization
                        progress_percent = 0
                        result = None

                        # Run in the main thread but update progress bar periodically
                        import threading

                        def update_progress():
                            nonlocal progress_percent
                            # Update progress bar incrementally until we reach ~90%
                            # The final 10% will be filled when the process completes
                            while progress_percent < 90 and result is None:
                                elapsed = time.time() - start_time
                                # Update more frequently at the beginning, then slow down
                                if elapsed > 0.5:
                                    increment = max(1, min(5, int(elapsed / 2)))
                                    if progress_percent + increment <= 90:
                                        pbar.update(increment)
                                        progress_percent += increment
                                time.sleep(0.5)

                        # Start progress updater thread
                        progress_thread = threading.Thread(target=update_progress)
                        progress_thread.daemon = True
                        progress_thread.start()

                        try:
                            # Run actual diarization - suppress warnings that cause the process to die
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore", message=".*degrees of freedom is <= 0.*"
                                )
                                warnings.filterwarnings("ignore", category=UserWarning)

                                # Prepare diarization parameters
                                diarization_params = {}
                                if num_speakers is not None:
                                    diarization_params["num_speakers"] = num_speakers
                                else:
                                    if min_speakers is not None:
                                        diarization_params[
                                            "min_speakers"
                                        ] = min_speakers
                                    if max_speakers is not None:
                                        diarization_params[
                                            "max_speakers"
                                        ] = max_speakers

                                result = self.diarize_model(
                                    audio,
                                    **diarization_params,  # Pass conditional parameters
                                )
                            # Complete the progress bar
                            pbar.update(100 - progress_percent)

                            # Ensure speaker labels are strings
                            if result:
                                for i in range(len(result)):
                                    if "speaker" in result[i] and not isinstance(
                                        result[i]["speaker"], str
                                    ):
                                        result[i][
                                            "speaker"
                                        ] = f"SPEAKER_{str(result[i]['speaker']).zfill(2)}"

                            return result
                        except Exception as e:
                            # Complete the progress bar even if there's an error
                            pbar.update(100 - progress_percent)
                            raise e
                else:
                    # Suppress warnings in non-progress mode too
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*degrees of freedom is <= 0.*"
                        )
                        warnings.filterwarnings("ignore", category=UserWarning)

                        # Prepare diarization parameters
                        diarization_params = {}
                        if num_speakers is not None:
                            diarization_params["num_speakers"] = num_speakers
                        else:
                            if min_speakers is not None:
                                diarization_params["min_speakers"] = min_speakers
                            if max_speakers is not None:
                                diarization_params["max_speakers"] = max_speakers

                        return self.diarize_model(
                            audio, **diarization_params  # Pass conditional parameters
                        )
        except Exception as e:
            print(f"Warning: Diarization failed. Error: {str(e)}")
            return []

    def process_audio(
        self,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        batch_size: int = 16,
        show_progress: bool = True,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
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
            num_speakers: Fixed number of speakers (takes precedence over min/max)
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
                    diarize_segments = self.diarize_audio(
                        audio_path=audio_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        show_progress=show_progress,
                    )

                    # Debug information
                    if show_progress and diarize_segments and len(diarize_segments) > 0:
                        self.logger.debug(
                            f"Speaker segment sample: {diarize_segments[0]}"
                        )
                        self.logger.debug(
                            f"Total speaker segments: {len(diarize_segments)}"
                        )
                        self.logger.debug(
                            f"Speaker labels: {set(s.get('speaker', 'unknown') for s in diarize_segments)}"
                        )

                    # Check if any segments contain numeric speaker IDs (problematic)
                    for seg in diarize_segments:
                        if "speaker" in seg and isinstance(
                            seg["speaker"], (int, float)
                        ):
                            seg["speaker"] = f"SPEAKER_{str(seg['speaker']).zfill(2)}"

                    # Ensure all segments have string 'speaker' keys
                    for i, seg in enumerate(diarize_segments):
                        if "speaker" not in seg:
                            if show_progress:
                                self.logger.debug(
                                    f"Adding missing speaker label to segment {i}"
                                )
                            seg["speaker"] = f"SPEAKER_UNKNOWN"

                    # Use our custom implementation instead of direct whisperx call
                    result = self._assign_word_speakers(diarize_segments, result)

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
            self.logger.debug(
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
                    # Check if required keys exist
                    if (
                        "word" not in word_data
                        or "start" not in word_data
                        or "end" not in word_data
                    ):
                        self.logger.debug(
                            f"Warning: Word data does not contain required keys. Skipping word: {word_data}"
                        )
                        continue

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

    def _assign_word_speakers(self, diarize_segments, result):
        """Custom implementation of whisperX's assign_word_speakers function to avoid index errors.

        This implementation ensures all speaker labels are treated correctly.
        """
        if len(diarize_segments) == 0:
            self.logger.debug("Warning: No diarization segments provided.")
            return result

        # Create mapping of speaker segments for quick lookup
        # Each segment is [start_time, end_time, speaker_id]
        speaker_segments = []
        for segment in diarize_segments:
            if not all(k in segment for k in ["start", "end", "speaker"]):
                self.logger.debug(f"Warning: Invalid diarization segment: {segment}")
                continue

            # Ensure speaker is a string
            speaker = segment["speaker"]
            if not isinstance(speaker, str):
                speaker = f"SPEAKER_{str(speaker).zfill(2)}"

            speaker_segments.append((segment["start"], segment["end"], speaker))

        # Sort by start time
        speaker_segments.sort(key=lambda x: x[0])

        # Check if result has the expected structure
        if "segments" not in result:
            self.logger.debug("Warning: Result does not have 'segments' key")
            return result

        # For each segment in the result
        for segment_idx, segment in enumerate(result["segments"]):
            # Skip segments without words
            if "words" not in segment:
                continue

            # For each word in the segment
            for word_idx, word in enumerate(segment["words"]):
                # Skip words without timestamps
                if "start" not in word or "end" not in word:
                    continue

                word_start = word["start"]
                word_end = word["end"]

                # Find the speaker who was talking during this word
                # Strategy: find the speaker segment with the most overlap
                best_speaker = None
                max_overlap = 0

                for start, end, speaker in speaker_segments:
                    # Check for overlap
                    overlap_start = max(start, word_start)
                    overlap_end = min(end, word_end)
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker

                # Assign the speaker to the word
                if best_speaker is not None:
                    result["segments"][segment_idx]["words"][word_idx][
                        "speaker"
                    ] = best_speaker

        # Now assign speaker to each segment based on majority of words
        for segment_idx, segment in enumerate(result["segments"]):
            if "words" not in segment or not segment["words"]:
                continue

            # Count speakers in words
            speaker_counts = {}
            for word in segment["words"]:
                if "speaker" in word:
                    speaker = word["speaker"]
                    if speaker not in speaker_counts:
                        speaker_counts[speaker] = 0
                    speaker_counts[speaker] += 1

            # Assign the majority speaker to the segment
            if speaker_counts:
                majority_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                result["segments"][segment_idx]["speaker"] = majority_speaker

        return result
