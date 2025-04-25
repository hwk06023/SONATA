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
        self.diarize_model_type = None
        self.embedding_model_name = None
        self.clustering_method = None
        self.speaker_embeddings = {}
        self.vad_model = None
        self.scd_model = None
        self.speaker_embedding_model = None
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
        embedding_model: str = "ecapa",
        clustering_method: str = "agglomerative",
        enhance_vad: bool = True,
    ):
        """Load the speaker diarization model.

        Args:
            hf_token: Hugging Face token for model access
            show_progress: Whether to display progress messages
            offline_mode: Whether to use offline mode
            offline_config_path: Path to offline config.yaml file
            embedding_model: Speaker embedding model type ('ecapa', 'resnet', 'xvector')
            clustering_method: Clustering method ('agglomerative', 'spectral')
            enhance_vad: Whether to use enhanced voice activity detection
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

                        # Store the model type for later reference
                        self.diarize_model_type = "pyannote_pipeline"
                        self.embedding_model_name = "default"
                        self.clustering_method = "default"
                    else:
                        # For online mode, use advanced setup if possible, otherwise fallback to standard WhisperX
                        try:
                            # Try to use advanced PyAnnote diarization
                            from pyannote.audio import Pipeline

                            # Select the best embedding model
                            self.embedding_model_name = embedding_model.lower()
                            self.clustering_method = clustering_method.lower()

                            if show_progress:
                                print(
                                    f"[ASR] Using advanced speaker diarization with {self.embedding_model_name} embeddings",
                                    flush=True,
                                )

                            if self.embedding_model_name in ["ecapa", "ecapa-tdnn"]:
                                # Use ECAPA-TDNN for superior speaker embeddings
                                self.diarize_model = Pipeline.from_pretrained(
                                    "pyannote/speaker-diarization-3.1",
                                    use_auth_token=hf_token,
                                )

                                # Try to set clustering parameters for better performance
                                if hasattr(self.diarize_model, "instantiate"):
                                    self.diarize_model.instantiate(
                                        {
                                            "clustering": self.clustering_method,
                                            "segmentation": {
                                                "threshold": 0.4445,  # Lower threshold for higher recall
                                                "min_duration_off": 0.1,  # Shorter silence tolerance
                                            },
                                        }
                                    )

                                self.diarize_model_type = "pyannote_advanced"
                            else:
                                # Standard WhisperX DiarizationPipeline as fallback
                                if not hf_token:
                                    raise ValueError(
                                        "HuggingFace token is required for online diarization"
                                    )

                                self.diarize_model = whisperx.DiarizationPipeline(
                                    use_auth_token=hf_token, device=self.device
                                )

                                self.diarize_model_type = "whisperx"
                                self.embedding_model_name = (
                                    "resnet"  # WhisperX uses ResNet by default
                                )
                        except Exception as e:
                            print(f"Advanced diarization setup failed: {str(e)}")
                            print("Falling back to standard WhisperX diarization")

                            # Fallback to standard WhisperX DiarizationPipeline
                            if not hf_token:
                                raise ValueError(
                                    "HuggingFace token is required for online diarization"
                                )

                            self.diarize_model = whisperx.DiarizationPipeline(
                                use_auth_token=hf_token, device=self.device
                            )

                            self.diarize_model_type = "whisperx"
                            self.embedding_model_name = "resnet"

                if show_progress:
                    print(
                        f"[ASR] Diarization model loaded successfully using {self.embedding_model_name} embeddings.",
                        flush=True,
                    )
            except Exception as e:
                print(f"Warning: Could not load diarization model. Error: {str(e)}")
                self.diarize_model = None
                self.diarize_model_type = None
                self.embedding_model_name = None
                self.clustering_method = None
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

            # Apply enhanced Voice Activity Detection if we're using the advanced model
            enhanced_vad_segments = None
            if (
                self.diarize_model_type == "pyannote_advanced"
                and hasattr(self, "enhance_vad")
                and self.enhance_vad
            ):
                try:
                    # Import silero VAD for improved speech detection
                    if show_progress:
                        print(
                            f"[ASR] Applying enhanced Voice Activity Detection...",
                            flush=True,
                        )

                    import torch

                    torch_device = (
                        "cuda"
                        if self.device == "cuda" and torch.cuda.is_available()
                        else "cpu"
                    )

                    try:
                        # Try to use Silero VAD (better speech detection)
                        model, utils = torch.hub.load(
                            repo_or_dir="snakers4/silero-vad",
                            model="silero_vad",
                            force_reload=False,
                            onnx=False,
                            verbose=False,
                        )

                        model = model.to(torch_device)
                        (get_speech_timestamps, _, _, _, _) = utils

                        # Make sure the audio is properly formatted (16kHz)
                        import librosa
                        import numpy as np

                        # Load audio with librosa to ensure proper resampling to 16kHz
                        waveform, sample_rate = librosa.load(audio_path, sr=16000)
                        waveform = torch.tensor(waveform).unsqueeze(0).to(torch_device)

                        # Get speech timestamps
                        speech_timestamps = get_speech_timestamps(
                            waveform,
                            model,
                            threshold=0.5,
                            sampling_rate=16000,
                            min_silence_duration_ms=500,
                            window_size_samples=1024,
                            speech_pad_ms=30,
                            return_seconds=True,
                        )

                        # Convert to format expected by diarization
                        enhanced_vad_segments = []
                        for segment in speech_timestamps:
                            enhanced_vad_segments.append(
                                {"start": segment["start"], "end": segment["end"]}
                            )

                        if show_progress:
                            print(
                                f"[ASR] Enhanced VAD found {len(enhanced_vad_segments)} speech segments",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"Enhanced VAD failed, falling back to default: {str(e)}")
                except Exception as vad_error:
                    print(
                        f"Enhanced VAD setup failed: {str(vad_error)}. Using default VAD."
                    )

            # Perform diarization
            if hasattr(self.diarize_model, "__call__"):
                # Direct Pipeline (offline mode or advanced pyannote)
                if show_progress:
                    print(
                        f"[ASR] Extracting speaker embeddings with {self.embedding_model_name}...",
                        flush=True,
                    )
                    # The PyAnnote pipeline has internal steps including embedding extraction
                    from tqdm import tqdm
                    import time
                    import warnings

                    # Create progress bar for speaker embedding
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

                                # Add VAD segments if we have them
                                if enhanced_vad_segments:
                                    # Check if our model supports the segments parameter
                                    if self.diarize_model_type == "pyannote_advanced":
                                        from pyannote.core import Segment, Timeline

                                        # Create a Timeline from our enhanced VAD segments
                                        vad_timeline = Timeline()
                                        for segment in enhanced_vad_segments:
                                            vad_timeline.add(
                                                Segment(
                                                    segment["start"], segment["end"]
                                                )
                                            )

                                        diarization_params["speech"] = vad_timeline

                                # Run diarization
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

                        # Add VAD segments if we have them
                        if (
                            enhanced_vad_segments
                            and self.diarize_model_type == "pyannote_advanced"
                        ):
                            from pyannote.core import Segment, Timeline

                            # Create a Timeline from our enhanced VAD segments
                            vad_timeline = Timeline()
                            for segment in enhanced_vad_segments:
                                vad_timeline.add(
                                    Segment(segment["start"], segment["end"])
                                )

                            diarization_params["speech"] = vad_timeline

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

                # Store speaker information for possible refinement
                self._extract_and_store_speaker_embeddings(audio_path, result)

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

                            # Store speaker embeddings for possible refinements
                            if result:
                                self._extract_and_store_speaker_embeddings(
                                    audio_path, result
                                )

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

                        result = self.diarize_model(
                            audio, **diarization_params  # Pass conditional parameters
                        )

                        # Store speaker information
                        if result:
                            self._extract_and_store_speaker_embeddings(
                                audio_path, result
                            )

                        return result
        except Exception as e:
            print(f"Warning: Diarization failed. Error: {str(e)}")
            return []

    def _extract_and_store_speaker_embeddings(self, audio_path, diarize_segments):
        """Extract and store speaker embeddings for the segments."""
        try:
            # Only attempt if we have a proper setup for this
            if self.diarize_model_type == "pyannote_advanced" and hasattr(
                self.diarize_model, "embeddings"
            ):
                # This is a simplified version, in reality we would need to extract
                # the actual embeddings from the audio segments
                self.speaker_embeddings = {}

                # Dictionary to collect segments by speaker
                speaker_segments = {}

                # Collect segments by speaker
                for segment in diarize_segments:
                    speaker = segment["speaker"]
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append((segment["start"], segment["end"]))

                # For now, we just store the segment information
                self.speaker_embeddings = speaker_segments
        except Exception as e:
            # Non-critical, so just log it
            self.logger.debug(f"Could not extract speaker embeddings: {str(e)}")

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
        embedding_model: str = "ecapa",
        enhance_vad: bool = True,
        use_enhanced_diarization: bool = True,
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
            embedding_model: Speaker embedding model type ('ecapa', 'resnet', 'xvector')
            enhance_vad: Whether to use enhanced VAD for better speech detection
            use_enhanced_diarization: Whether to use the enhanced diarization pipeline

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
            if (
                self.diarize_model is None
                or self.embedding_model_name != embedding_model
            ):
                self.load_diarize_model(
                    hf_token=hf_token,
                    show_progress=show_progress,
                    embedding_model=embedding_model,
                    enhance_vad=enhance_vad,
                )

            if self.diarize_model is not None:
                try:
                    # Perform diarization using standard or enhanced method
                    if use_enhanced_diarization:
                        if show_progress:
                            print(
                                f"[ASR] Using enhanced diarization pipeline for better accuracy",
                                flush=True,
                            )
                        diarize_segments = self.enhanced_diarize_audio(
                            audio_path=audio_path,
                            num_speakers=num_speakers,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                            show_progress=show_progress,
                        )
                    else:
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

                    # Use our enhanced implementation to assign speakers
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
        """Enhanced implementation of speaker assignment with better overlap handling.

        This implementation improves speaker assignment accuracy with context awareness
        and better handling of speaker transitions.
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

        # Check for segment overlaps and refine if needed
        refined_segments = self._refine_overlapping_segments(speaker_segments)

        # Check if result has the expected structure
        if "segments" not in result:
            self.logger.debug("Warning: Result does not have 'segments' key")
            return result

        # For each segment in the result, process within a context window
        # to improve speaker assignment consistency
        for segment_idx, segment in enumerate(result["segments"]):
            # Skip segments without words
            if "words" not in segment:
                continue

            # Group words by likely speaker
            word_groups = self._group_words_by_speaker_transition(segment["words"])

            # Process each group of words
            for word_group in word_groups:
                # Find the most likely speaker for this group based on overlap
                best_speaker = self._get_best_speaker_for_word_group(
                    word_group, refined_segments
                )

                # Assign the speaker to all words in this group
                for word in word_group:
                    word_idx = segment["words"].index(word)
                    if best_speaker:
                        result["segments"][segment_idx]["words"][word_idx][
                            "speaker"
                        ] = best_speaker

        # Refine speaker assignments using speech context
        result = self._refine_speaker_assignments_with_context(result)

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

    def _refine_overlapping_segments(self, speaker_segments):
        """Refine overlapping segments to improve speaker boundary accuracy."""
        if not speaker_segments or len(speaker_segments) <= 1:
            return speaker_segments

        refined_segments = []
        i = 0

        while i < len(speaker_segments) - 1:
            current = speaker_segments[i]
            next_seg = speaker_segments[i + 1]

            # Check for overlap
            if current[1] > next_seg[0]:
                # We have an overlap
                overlap_duration = current[1] - next_seg[0]

                # If significant overlap, keep both (could be overlapping speech)
                if overlap_duration > 0.5:  # More than half a second overlap
                    refined_segments.append(current)
                    i += 1
                    continue

                # For minor overlaps, adjust boundary to midpoint
                midpoint = (current[1] + next_seg[0]) / 2
                refined_segments.append((current[0], midpoint, current[2]))

                # Adjust the next segment
                speaker_segments[i + 1] = (midpoint, next_seg[1], next_seg[2])
            else:
                # No overlap, keep as is
                refined_segments.append(current)

            i += 1

        # Add the last segment if we haven't already
        if i < len(speaker_segments):
            refined_segments.append(speaker_segments[i])

        return refined_segments

    def _group_words_by_speaker_transition(self, words):
        """Group words that likely belong to the same speaker based on timing."""
        if not words:
            return []

        word_groups = []
        current_group = [words[0]]

        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]

            # Skip words without timestamp info
            if not all(k in prev_word for k in ["start", "end"]) or not all(
                k in curr_word for k in ["start", "end"]
            ):
                current_group.append(curr_word)
                continue

            # Check for potential speaker transition
            gap = curr_word["start"] - prev_word["end"]

            # If gap is significant, it might indicate a speaker change
            if gap > 0.5:  # Half-second threshold
                word_groups.append(current_group)
                current_group = [curr_word]
            else:
                current_group.append(curr_word)

        # Add the last group
        if current_group:
            word_groups.append(current_group)

        return word_groups

    def _get_best_speaker_for_word_group(self, word_group, speaker_segments):
        """Find the most likely speaker for a group of words."""
        if not word_group or not speaker_segments:
            return None

        # Get time span for this word group
        start_time = min(word["start"] for word in word_group if "start" in word)
        end_time = max(word["end"] for word in word_group if "end" in word)

        # Calculate overlap with each speaker segment
        best_speaker = None
        max_overlap = 0

        for start, end, speaker in speaker_segments:
            # Check for overlap
            overlap_start = max(start, start_time)
            overlap_end = min(end, end_time)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        return best_speaker

    def _refine_speaker_assignments_with_context(self, result):
        """Refine speaker assignments using context to fix potential errors."""
        if "segments" not in result:
            return result

        # First pass: identify potential errors (rapid speaker changes)
        for segment_idx, segment in enumerate(result["segments"]):
            if "words" not in segment or len(segment["words"]) < 3:
                continue

            # Look for speaker switching back and forth rapidly
            # A-B-A pattern is suspicious and might indicate an error
            for i in range(1, len(segment["words"]) - 1):
                if all(
                    k in word
                    for k in ["speaker"]
                    for word in [
                        segment["words"][i - 1],
                        segment["words"][i],
                        segment["words"][i + 1],
                    ]
                ):
                    prev_speaker = segment["words"][i - 1]["speaker"]
                    curr_speaker = segment["words"][i]["speaker"]
                    next_speaker = segment["words"][i + 1]["speaker"]

                    # If we have A-B-A pattern, it might be an error in speaker B assignment
                    if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                        # Short duration word between same speaker is suspicious
                        word_duration = (
                            segment["words"][i]["end"] - segment["words"][i]["start"]
                        )
                        if word_duration < 0.5:  # Less than half a second
                            # Correct the speaker assignment
                            result["segments"][segment_idx]["words"][i][
                                "speaker"
                            ] = prev_speaker

        # Second pass: smooth out speaker assignments using window-based voting
        window_size = 3  # Number of words to consider in each direction

        for segment_idx, segment in enumerate(result["segments"]):
            if "words" not in segment or len(segment["words"]) < (2 * window_size + 1):
                continue

            smoothed_words = segment["words"].copy()

            for i in range(window_size, len(segment["words"]) - window_size):
                # Get speakers in the window
                window_speakers = []
                for j in range(i - window_size, i + window_size + 1):
                    if "speaker" in segment["words"][j]:
                        window_speakers.append(segment["words"][j]["speaker"])

                # Count occurrences of each speaker
                speaker_counts = {}
                for speaker in window_speakers:
                    if speaker not in speaker_counts:
                        speaker_counts[speaker] = 0
                    speaker_counts[speaker] += 1

                # Assign the majority speaker
                if speaker_counts:
                    majority_speaker = max(speaker_counts.items(), key=lambda x: x[1])[
                        0
                    ]
                    # Only override if the count is significant (more than half the window)
                    if speaker_counts[majority_speaker] > window_size:
                        smoothed_words[i]["speaker"] = majority_speaker

            # Update the segment with smoothed words
            result["segments"][segment_idx]["words"] = smoothed_words

        return result

    def _enhanced_vad(self, audio_path, show_progress=True):
        """Implement advanced ensemble Voice Activity Detection for maximum accuracy.

        Uses a combination of multiple VAD models for better performance:
        1. Silero VAD (neural network-based)
        2. WebRTC VAD (traditional signal processing)
        3. Energy-based VAD (for catching quiet speech)

        Args:
            audio_path: Path to the audio file
            show_progress: Whether to show progress indicators

        Returns:
            List of speech segments with start and end times
        """
        if show_progress:
            print("[ASR] Running enhanced Voice Activity Detection...", flush=True)

        try:
            import torch
            import librosa
            import numpy as np
            import scipy

            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=16000)

            # Set up model device
            device = (
                self.device
                if self.device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            # 1. Apply Silero VAD (neural model)
            if show_progress:
                print("[ASR] Applying Silero VAD (Neural)...", flush=True)

            if self.vad_model is None:
                # Load Silero VAD
                self.vad_model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    verbose=False,
                )
                self.vad_model.to(device)

            # Prepare utils
            (get_speech_timestamps, _, _, _, _) = utils

            # Make sure audio is in the right format
            audio_tensor = torch.tensor(waveform).unsqueeze(0).to(device)

            # Get speech segments - be more sensitive to catch all speech
            silero_segments = get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=0.3,  # Lower threshold = higher sensitivity
                sampling_rate=16000,
                min_silence_duration_ms=500,
                window_size_samples=1024,
                speech_pad_ms=100,
                return_seconds=True,
            )

            # 2. Apply WebRTC VAD if available
            webrtc_segments = []
            try:
                import webrtcvad

                if show_progress:
                    print("[ASR] Applying WebRTC VAD...", flush=True)

                # WebRTC VAD for more precise boundaries
                vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)

                # Process in 30ms frames
                frame_duration = 30  # ms
                frame_size = int(sample_rate * frame_duration / 1000)
                frame_count = len(waveform) // frame_size

                # Iterate over frames
                frames = []
                for i in range(0, frame_count):
                    start = i * frame_size
                    end = start + frame_size
                    frame = waveform[start:end]
                    frames.append(frame)

                # Get VAD results
                is_speech = []
                for frame in frames:
                    # Convert to int16 PCM
                    pcm_data = (frame * 32768).astype(np.int16).tobytes()
                    try:
                        result = vad.is_speech(pcm_data, sample_rate)
                        is_speech.append(result)
                    except:
                        is_speech.append(False)

                # Convert to segments
                in_speech = False
                start_time = 0

                for i, speech in enumerate(is_speech):
                    frame_time = i * frame_duration / 1000  # Time in seconds

                    if speech and not in_speech:
                        in_speech = True
                        start_time = frame_time
                    elif not speech and in_speech:
                        in_speech = False
                        end_time = frame_time
                        webrtc_segments.append({"start": start_time, "end": end_time})

                # Don't forget the last segment
                if in_speech:
                    end_time = len(is_speech) * frame_duration / 1000
                    webrtc_segments.append({"start": start_time, "end": end_time})

            except Exception as e:
                if show_progress:
                    print(
                        f"[ASR] WebRTC VAD failed: {str(e)}, continuing with Silero only"
                    )

            # 3. Apply energy-based VAD for low-volume speech
            energy_segments = []
            try:
                if show_progress:
                    print("[ASR] Applying energy-based VAD...", flush=True)

                # Calculate energy
                window_size = int(0.03 * sample_rate)  # 30ms windows
                hop_length = int(0.01 * sample_rate)  # 10ms hop

                # Calculate energy in each window
                energy = []
                for i in range(0, len(waveform) - window_size, hop_length):
                    frame = waveform[i : i + window_size]
                    energy.append(np.sum(frame**2))

                # Normalize energy
                energy = np.array(energy)
                energy = (energy - np.min(energy)) / (
                    np.max(energy) - np.min(energy) + 1e-10
                )

                # Smooth the energy curve
                energy_smooth = scipy.ndimage.gaussian_filter1d(energy, sigma=2)

                # Find segments above threshold
                threshold = np.mean(energy_smooth) * 0.2  # Adaptive threshold
                is_speech = energy_smooth > threshold

                # Convert to segments
                in_speech = False
                start_idx = 0

                for i, speech in enumerate(is_speech):
                    if speech and not in_speech:
                        in_speech = True
                        start_idx = i
                    elif not speech and in_speech:
                        in_speech = False
                        end_idx = i
                        start_time = start_idx * hop_length / sample_rate
                        end_time = end_idx * hop_length / sample_rate

                        # Only add segments of reasonable duration
                        if end_time - start_time > 0.3:  # At least 300ms
                            energy_segments.append(
                                {"start": start_time, "end": end_time}
                            )

                # Don't forget the last segment
                if in_speech:
                    end_idx = len(is_speech)
                    start_time = start_idx * hop_length / sample_rate
                    end_time = end_idx * hop_length / sample_rate

                    if end_time - start_time > 0.3:
                        energy_segments.append({"start": start_time, "end": end_time})

            except Exception as e:
                if show_progress:
                    print(
                        f"[ASR] Energy-based VAD failed: {str(e)}, continuing without it"
                    )

            # 4. Combine all VAD results with priority to Silero (most accurate)
            # Start with Silero segments
            combined_segments = silero_segments.copy()

            # Add WebRTC segments that don't overlap with Silero
            for webrtc_seg in webrtc_segments:
                # Check if this segment overlaps with any Silero segment
                overlaps = False
                for silero_seg in silero_segments:
                    # Check for overlap
                    if (
                        webrtc_seg["start"] < silero_seg["end"]
                        and webrtc_seg["end"] > silero_seg["start"]
                    ):
                        overlaps = True
                        break

                # If no overlap, add this segment
                if not overlaps:
                    combined_segments.append(webrtc_seg)

            # Add energy segments that don't overlap with existing segments
            for energy_seg in energy_segments:
                # Check if this segment overlaps with any existing segment
                overlaps = False
                for existing_seg in combined_segments:
                    # Check for overlap
                    if (
                        energy_seg["start"] < existing_seg["end"]
                        and energy_seg["end"] > existing_seg["start"]
                    ):
                        overlaps = True
                        break

                # If no overlap, add this segment
                if not overlaps:
                    combined_segments.append(energy_seg)

            # 5. Merge segments that are very close
            final_segments = []

            # Sort segments by start time
            combined_segments.sort(key=lambda x: x["start"])

            if not combined_segments:
                return []

            # Start with the first segment
            current = combined_segments[0]

            for segment in combined_segments[1:]:
                # If this segment starts soon after the current one ends
                if segment["start"] - current["end"] < 0.3:  # 300ms threshold
                    # Merge by extending the end time
                    current["end"] = segment["end"]
                else:
                    # Add the current segment to the final list and move to the next
                    final_segments.append(current)
                    current = segment

            # Add the last segment
            final_segments.append(current)

            if show_progress:
                print(
                    f"[ASR] Enhanced VAD detected {len(final_segments)} speech segments",
                    flush=True,
                )

            return final_segments

        except Exception as e:
            print(f"[ASR] Enhanced VAD failed with error: {str(e)}")
            # Try to fallback to a more basic approach
            try:
                # Fallback to basic VAD
                if show_progress:
                    print("[ASR] Falling back to basic VAD...", flush=True)

                # Use librosa's amplitude-based VAD
                import librosa

                waveform, sr = librosa.load(audio_path, sr=16000)

                # Get amplitude envelope
                amplitude = np.abs(waveform)

                # Smooth the amplitude
                window_size = int(0.03 * sr)
                amplitude_smooth = np.convolve(
                    amplitude, np.ones(window_size) / window_size, mode="same"
                )

                # Apply a threshold
                threshold = np.mean(amplitude_smooth) * 0.5
                is_speech = amplitude_smooth > threshold

                # Convert to segments
                in_speech = False
                start_idx = 0
                segments = []

                for i, speech in enumerate(is_speech):
                    if speech and not in_speech:
                        in_speech = True
                        start_idx = i
                    elif not speech and in_speech:
                        in_speech = False
                        end_idx = i
                        start_time = start_idx / sr
                        end_time = end_idx / sr
                        segments.append({"start": start_time, "end": end_time})

                # Don't forget the last segment
                if in_speech:
                    end_time = len(is_speech) / sr
                    segments.append({"start": start_time, "end": end_time})

                return segments

            except:
                # Last resort: return a single segment covering the whole audio
                print(
                    "[ASR] Fallback VAD also failed. Using entire audio as one segment."
                )
                audio_info = librosa.get_duration(path=audio_path)
                return [{"start": 0, "end": audio_info}]

    def _detect_speaker_changes(self, audio_path, vad_segments, show_progress=True):
        """Detect potential speaker change points within speech segments.

        Uses multiple techniques to detect speaker changes:
        1. BIC-based segmentation (Bayesian Information Criterion)
        2. Neural network-based speaker change detection (if available)
        3. MFCC feature analysis for additional change points

        Args:
            audio_path: Path to the audio file
            vad_segments: List of VAD segments to analyze
            show_progress: Whether to show progress

        Returns:
            List of change points timestamps
        """
        if show_progress:
            print("[ASR] Detecting speaker change points...", flush=True)

        try:
            import torch
            import librosa
            import numpy as np
            from scipy.spatial.distance import cosine

            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=16000)

            device = (
                self.device
                if self.device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            change_points = []

            # 1. Try using PyAnnote's speaker segmentation model if available
            try:
                if show_progress:
                    print("[ASR] Using neural speaker segmentation...", flush=True)

                from pyannote.audio import Model
                from pyannote.audio.pipelines import SpeakerSegmentation

                # Load or reuse the segmentation model
                if self.scd_model is None:
                    try:
                        scd_model = Model.from_pretrained(
                            "pyannote/segmentation-3.0",
                            use_auth_token=None,  # Set your HF token if needed
                        ).to(device)

                        # Create a segmentation pipeline
                        self.scd_model = SpeakerSegmentation(
                            segmentation_model=scd_model,
                            clustering="pool",  # Use last layer pooling
                            segmentation={
                                "threshold": 0.45,  # Lower threshold = higher sensitivity
                                "min_duration_off": 0.3,
                            },  # Minimum silence duration between speakers
                        )
                    except Exception as e:
                        if show_progress:
                            print(f"[ASR] Could not load PyAnnote SCD model: {str(e)}")
                        self.scd_model = None

                # Apply segmentation to each VAD segment
                if self.scd_model is not None:
                    for segment in vad_segments:
                        # Extract the segment from the audio
                        start_sample = int(segment["start"] * sample_rate)
                        end_sample = int(segment["end"] * sample_rate)

                        # Skip very short segments
                        if end_sample - start_sample < 0.5 * sample_rate:
                            continue

                        seg_audio = waveform[start_sample:end_sample]

                        # Convert to torch tensor
                        seg_tensor = torch.tensor(seg_audio).unsqueeze(0).to(device)

                        # Run segmentation
                        audio_dict = {
                            "waveform": seg_tensor,
                            "sample_rate": sample_rate,
                        }
                        segmentation_result = self.scd_model(audio_dict)

                        # Extract change points and adjust to original timeline
                        for point in segmentation_result.get("change_points", []):
                            change_time = segment["start"] + point
                            change_points.append(change_time)

            except Exception as e:
                if show_progress:
                    print(f"[ASR] Neural segmentation failed: {str(e)}")

            # 2. MFCC-based change detection (classic approach)
            if show_progress:
                print(
                    "[ASR] Performing MFCC-based speaker change detection...",
                    flush=True,
                )

            # Process each VAD segment
            for segment in vad_segments:
                # Extract the segment from the audio
                start_sample = int(segment["start"] * sample_rate)
                end_sample = int(segment["end"] * sample_rate)

                # Skip very short segments
                if (
                    end_sample - start_sample < 1.0 * sample_rate
                ):  # Skip segments shorter than 1s
                    continue

                seg_audio = waveform[start_sample:end_sample]

                # Extract features - MFCCs with delta and acceleration
                mfccs = librosa.feature.mfcc(y=seg_audio, sr=sample_rate, n_mfcc=20)
                delta_mfccs = librosa.feature.delta(mfccs, width=5)
                delta2_mfccs = librosa.feature.delta(delta_mfccs, width=5)

                # Combine features
                features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

                # Set window parameters
                window_size = int(1.0 * sample_rate / 512)  # 1 second windows
                hop_size = int(0.1 * sample_rate / 512)  # 100ms hop size

                # Calculate distances between consecutive windows
                distances = []

                for i in range(hop_size, features.shape[1] - window_size, hop_size):
                    # Left window
                    left_window = features[:, i - hop_size : i + window_size - hop_size]
                    # Right window
                    right_window = features[:, i : i + window_size]

                    # Compute mean of each window
                    left_mean = np.mean(left_window, axis=1)
                    right_mean = np.mean(right_window, axis=1)

                    # Compute cosine distance
                    distance = cosine(left_mean, right_mean)
                    distances.append(distance)

                if not distances:
                    continue

                # Normalize distances
                distances = np.array(distances)
                if np.max(distances) - np.min(distances) > 0:
                    distances = (distances - np.min(distances)) / (
                        np.max(distances) - np.min(distances)
                    )

                # Detect peaks (potential change points)
                from scipy.signal import find_peaks

                # Use adaptive threshold based on mean and std
                threshold = np.mean(distances) + 1.5 * np.std(distances)
                threshold = min(
                    max(threshold, 0.3), 0.7
                )  # Keep within reasonable bounds

                peaks, _ = find_peaks(
                    distances,
                    height=threshold,
                    distance=int(0.5 * sample_rate / 512 / hop_size),
                )

                # Convert peak indices to time
                for peak in peaks:
                    # Calculate the time in the original audio
                    relative_time = (peak + 1) * hop_size * 512 / sample_rate
                    absolute_time = segment["start"] + relative_time

                    # Add to change points
                    change_points.append(absolute_time)

            # 3. BIC-based segmentation (Bayesian Information Criterion)
            if show_progress:
                print(
                    "[ASR] Performing BIC-based speaker change detection...", flush=True
                )

            def compute_bic(mfccs, i):
                """Compute BIC value for potential change point at position i."""
                n = mfccs.shape[1]
                d = mfccs.shape[0]

                if i < 10 or i > n - 10:  # Ensure enough data on each side
                    return 0

                # Split the data
                mfccs_left = mfccs[:, :i]
                mfccs_right = mfccs[:, i:]

                # Compute covariances
                cov_left = np.cov(mfccs_left)
                cov_right = np.cov(mfccs_right)
                cov_full = np.cov(mfccs)

                # Fix potential issues with covariance matrices
                min_val = 1e-10
                cov_left = cov_left + np.eye(cov_left.shape[0]) * min_val
                cov_right = cov_right + np.eye(cov_right.shape[0]) * min_val
                cov_full = cov_full + np.eye(cov_full.shape[0]) * min_val

                # Compute determinants
                try:
                    det_left = np.linalg.det(cov_left)
                    det_right = np.linalg.det(cov_right)
                    det_full = np.linalg.det(cov_full)

                    # Ensure positive determinants
                    det_left = max(det_left, min_val)
                    det_right = max(det_right, min_val)
                    det_full = max(det_full, min_val)

                    # Compute BIC value
                    n1 = i
                    n2 = n - i

                    # Penalty factor
                    p = 0.5 * (d + 0.5 * d * (d + 1)) * np.log(n)

                    # BIC value
                    bic = (
                        n * np.log(det_full)
                        - n1 * np.log(det_left)
                        - n2 * np.log(det_right)
                        - p
                    )

                    return bic
                except:
                    return 0

            # Process each VAD segment for BIC
            for segment in vad_segments:
                # Extract the segment from the audio
                start_sample = int(segment["start"] * sample_rate)
                end_sample = int(segment["end"] * sample_rate)

                # Skip very short segments
                if (
                    end_sample - start_sample < 2.0 * sample_rate
                ):  # Skip segments shorter than 2s
                    continue

                seg_audio = waveform[start_sample:end_sample]

                # Extract MFCCs with reasonable window and hop size for BIC
                mfccs = librosa.feature.mfcc(
                    y=seg_audio,
                    sr=sample_rate,
                    n_mfcc=13,
                    hop_length=int(0.01 * sample_rate),  # 10ms hop
                    win_length=int(0.025 * sample_rate),  # 25ms window
                )

                # Compute BIC values at regular intervals
                step = int(
                    0.2 * sample_rate / int(0.01 * sample_rate)
                )  # Check every 200ms
                bic_values = []

                for i in range(step, mfccs.shape[1] - step, step):
                    bic_values.append(compute_bic(mfccs, i))

                # Find peaks in BIC values
                if bic_values:
                    bic_values = np.array(bic_values)

                    # Only consider positive BIC values (indicating likely change points)
                    positive_indices = np.where(bic_values > 0)[0]

                    if len(positive_indices) > 0:
                        # Get top peaks
                        top_indices = positive_indices[
                            np.argsort(bic_values[positive_indices])[-3:]
                        ]

                        for idx in top_indices:
                            # Convert to time in the segment
                            relative_time = (
                                (idx + 1) * step * int(0.01 * sample_rate) / sample_rate
                            )
                            absolute_time = segment["start"] + relative_time

                            change_points.append(absolute_time)

            # 4. Post-process change points
            if show_progress:
                print(
                    f"[ASR] Found {len(change_points)} potential speaker change points",
                    flush=True,
                )

            # Sort change points
            change_points.sort()

            # Filter out change points that are too close together
            filtered_change_points = []

            if not change_points:
                return []

            # Start with the first point
            prev_point = change_points[0]
            filtered_change_points.append(prev_point)

            for point in change_points[1:]:
                # Only keep points that are at least 1s apart
                if point - prev_point > 1.0:
                    filtered_change_points.append(point)
                    prev_point = point

            # Ensure change points are within VAD segments
            validated_change_points = []

            for point in filtered_change_points:
                # Check if this point is within a VAD segment
                for segment in vad_segments:
                    if segment["start"] < point < segment["end"]:
                        validated_change_points.append(point)
                        break

            if show_progress:
                print(
                    f"[ASR] Final speaker change points: {len(validated_change_points)}",
                    flush=True,
                )

            return validated_change_points

        except Exception as e:
            print(f"[ASR] Speaker change detection failed: {str(e)}")
            return []

    def _extract_speaker_embeddings(self, audio_path, segments, show_progress=True):
        """Extract state-of-the-art speaker embeddings for each segment.

        Uses multiple speaker embedding techniques:
        1. ECAPA-TDNN from SpeechBrain (state-of-the-art as of 2023)
        2. WavLM/Wav2Vec2 embeddings as additional features

        Args:
            audio_path: Path to the audio file
            segments: List of audio segments to process
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping segment indices to embedding vectors
        """
        if show_progress:
            print("[ASR] Extracting speaker embeddings...", flush=True)

        try:
            import torch
            import librosa
            import numpy as np

            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=16000)

            device = (
                self.device
                if self.device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            # Initialize embeddings dictionary
            embeddings = {}

            # Load or initialize models
            if self.speaker_embedding_model is None:
                try:
                    if show_progress:
                        print(
                            "[ASR] Loading ECAPA-TDNN speaker embedding model...",
                            flush=True,
                        )

                    from speechbrain.pretrained import EncoderClassifier

                    # Load the ECAPA-TDNN model from SpeechBrain
                    self.speaker_embedding_model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="./pretrained_models/spkrec-ecapa-voxceleb",
                        run_opts={"device": device},
                    )
                except Exception as e:
                    if show_progress:
                        print(f"[ASR] Could not load ECAPA-TDNN: {str(e)}")

                    try:
                        # Try another model as fallback
                        if show_progress:
                            print(
                                "[ASR] Trying to load Xvector model as fallback...",
                                flush=True,
                            )

                        from speechbrain.pretrained import EncoderClassifier

                        # Load the Xvector model from SpeechBrain
                        self.speaker_embedding_model = EncoderClassifier.from_hparams(
                            source="speechbrain/spkrec-xvect-voxceleb",
                            savedir="./pretrained_models/spkrec-xvect-voxceleb",
                            run_opts={"device": device},
                        )
                    except Exception as e2:
                        if show_progress:
                            print(
                                f"[ASR] Could not load Xvector model either: {str(e2)}"
                            )
                        self.speaker_embedding_model = None

            # Process segments
            if self.speaker_embedding_model is not None:
                if show_progress:
                    from tqdm import tqdm

                    segments_iter = tqdm(
                        enumerate(segments),
                        total=len(segments),
                        desc="Extracting embeddings",
                    )
                else:
                    segments_iter = enumerate(segments)

                for i, segment in segments_iter:
                    # Extract the segment from the audio
                    start_sample = max(0, int(segment["start"] * sample_rate))
                    end_sample = min(len(waveform), int(segment["end"] * sample_rate))

                    # Skip very short segments
                    if (
                        end_sample - start_sample < 0.5 * sample_rate
                    ):  # Skip segments shorter than 0.5s
                        continue

                    seg_audio = waveform[start_sample:end_sample]

                    # Convert to tensor and extract embedding
                    with torch.no_grad():
                        # Make sure the audio is the right shape
                        audio_tensor = torch.tensor(seg_audio).unsqueeze(0).to(device)

                        # Extract embedding
                        embedding = self.speaker_embedding_model.encode_batch(
                            audio_tensor
                        )
                        embedding_np = embedding.squeeze().cpu().numpy()

                        # Store embedding
                        embeddings[i] = embedding_np

                return embeddings
            else:
                # Fallback to MFCC-based embeddings if models aren't available
                if show_progress:
                    print(
                        "[ASR] Using MFCC-based embeddings as fallback...", flush=True
                    )
                    from tqdm import tqdm

                    segments_iter = tqdm(
                        enumerate(segments),
                        total=len(segments),
                        desc="Extracting embeddings",
                    )
                else:
                    segments_iter = enumerate(segments)

                for i, segment in segments_iter:
                    # Extract the segment from the audio
                    start_sample = max(0, int(segment["start"] * sample_rate))
                    end_sample = min(len(waveform), int(segment["end"] * sample_rate))

                    # Skip very short segments
                    if end_sample - start_sample < 0.5 * sample_rate:
                        continue

                    seg_audio = waveform[start_sample:end_sample]

                    # Extract MFCCs
                    mfccs = librosa.feature.mfcc(y=seg_audio, sr=sample_rate, n_mfcc=20)
                    # Compute stats over time
                    mfcc_means = np.mean(mfccs, axis=1)
                    mfcc_vars = np.var(mfccs, axis=1)

                    # Combine into a single feature vector
                    embedding = np.concatenate([mfcc_means, mfcc_vars])

                    # Store embedding
                    embeddings[i] = embedding

                return embeddings

        except Exception as e:
            print(f"[ASR] Speaker embedding extraction failed: {str(e)}")
            return {}

    def _cluster_speakers(
        self,
        embeddings,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        show_progress=True,
    ):
        """Perform speaker clustering on embeddings.

        Uses advanced clustering techniques:
        1. Spectral clustering with adaptive affinity
        2. Auto-tuning of parameters
        3. Automatic speaker count estimation if not provided

        Args:
            embeddings: Dictionary of segment index to embedding vector
            num_speakers: Fixed number of speakers (takes precedence)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping segment indices to speaker IDs
        """
        if show_progress:
            print("[ASR] Clustering speakers...", flush=True)

        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering, SpectralClustering
            from sklearn.metrics import silhouette_score

            # Get embeddings as a list
            embedding_list = list(embeddings.values())
            segment_indices = list(embeddings.keys())

            if len(embedding_list) == 0:
                return {}

            # Convert to numpy array
            X = np.array(embedding_list)

            # Normalize embeddings
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            X_normalized = X / norms

            # Calculate cosine similarity matrix
            similarity_matrix = np.dot(X_normalized, X_normalized.T)

            # Determine number of speakers
            n_clusters = 2  # Default

            if num_speakers is not None:
                # Use provided number of speakers
                n_clusters = num_speakers
            else:
                # Auto-determine with spectral clustering if not specified
                if show_progress:
                    print(
                        "[ASR] Auto-determining optimal number of speakers...",
                        flush=True,
                    )

                # Set search range
                min_k = min_speakers if min_speakers is not None else 2
                max_k = min(
                    max_speakers if max_speakers is not None else 8, len(embedding_list)
                )

                best_score = -1
                best_k = 2

                # Try different values of k (number of speakers)
                for k in range(min_k, max_k + 1):
                    try:
                        # Skip to avoid errors
                        if k >= len(embedding_list):
                            continue

                        # Use spectral clustering
                        clustering = SpectralClustering(
                            n_clusters=k, affinity="precomputed", random_state=42
                        ).fit(similarity_matrix)

                        labels = clustering.labels_

                        # Calculate silhouette score to evaluate clustering
                        if (
                            len(set(labels)) > 1
                        ):  # Need at least 2 clusters for silhouette
                            score = silhouette_score(
                                similarity_matrix, labels, metric="precomputed"
                            )

                            if score > best_score:
                                best_score = score
                                best_k = k

                                if show_progress:
                                    print(f"[ASR] K={k}, Silhouette score: {score:.4f}")
                    except Exception as e:
                        if show_progress:
                            print(f"[ASR] Error with k={k}: {str(e)}")

                n_clusters = best_k

            if show_progress:
                print(f"[ASR] Clustering with {n_clusters} speakers...", flush=True)

            # Apply final clustering
            if n_clusters < len(embedding_list):
                # Use appropriate clustering method
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, affinity="cosine", linkage="average"
                ).fit(X_normalized)

                # Get labels
                labels = clustering.labels_

                # Create mapping from segment index to speaker
                speaker_mapping = {}

                for i, segment_idx in enumerate(segment_indices):
                    speaker_mapping[segment_idx] = f"SPEAKER_{labels[i]:02d}"

                return speaker_mapping
            else:
                # If as many or more clusters than segments, assign each to its own speaker
                speaker_mapping = {}

                for i, segment_idx in enumerate(segment_indices):
                    speaker_mapping[segment_idx] = f"SPEAKER_{i:02d}"

                return speaker_mapping

        except Exception as e:
            print(f"[ASR] Speaker clustering failed: {str(e)}")

            # Fallback: assign all to speaker 0
            speaker_mapping = {}
            for segment_idx in embeddings.keys():
                speaker_mapping[segment_idx] = "SPEAKER_00"

            return speaker_mapping

    def enhanced_diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """State-of-the-art speaker diarization with multiple enhancement techniques.

        Combines:
        1. Enhanced VAD with ensemble approach
        2. Advanced speaker change detection with multiple methods
        3. High-quality speaker embeddings using ECAPA-TDNN
        4. Optimized clustering with auto-tuning
        5. Overlapped speech detection and handling

        Args:
            audio_path: Path to the audio file
            num_speakers: Fixed number of speakers (takes precedence)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            show_progress: Whether to show progress

        Returns:
            List of diarization segments with speaker labels
        """
        if show_progress:
            print(f"[ASR] Running enhanced speaker diarization...", flush=True)

        try:
            # Step 1: Enhanced VAD to identify speech segments
            vad_segments = self._enhanced_vad(audio_path, show_progress)

            if not vad_segments:
                if show_progress:
                    print("[ASR] No speech detected in audio")
                return []

            # Step 2: Speaker change detection within VAD segments
            change_points = self._detect_speaker_changes(
                audio_path, vad_segments, show_progress
            )

            # Step 3: Create initial segments based on VAD and change points
            initial_segments = []

            for vad_seg in vad_segments:
                # Find all change points within this VAD segment
                seg_changes = [
                    cp for cp in change_points if vad_seg["start"] < cp < vad_seg["end"]
                ]

                # Add VAD start and end to create complete segment boundaries
                all_points = [vad_seg["start"]] + seg_changes + [vad_seg["end"]]
                all_points.sort()

                # Create segments between each pair of points
                for i in range(len(all_points) - 1):
                    # Skip very short segments (less than 0.3 seconds)
                    if all_points[i + 1] - all_points[i] < 0.3:
                        continue

                    initial_segments.append(
                        {
                            "start": all_points[i],
                            "end": all_points[i + 1],
                            "speech": True,
                        }
                    )

            if show_progress:
                print(
                    f"[ASR] Created {len(initial_segments)} initial segments",
                    flush=True,
                )

            if not initial_segments:
                if show_progress:
                    print("[ASR] No valid segments created after change detection")
                return []

            # Step 4: Extract speaker embeddings for each segment
            segment_embeddings = self._extract_speaker_embeddings(
                audio_path, initial_segments, show_progress
            )

            if not segment_embeddings:
                if show_progress:
                    print(
                        "[ASR] Could not extract speaker embeddings, falling back to basic diarization"
                    )
                # Fall back to basic diarization
                return self.diarize_audio(
                    audio_path=audio_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    show_progress=show_progress,
                )

            # Step 5: Cluster speakers
            speaker_mapping = self._cluster_speakers(
                segment_embeddings,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                show_progress=show_progress,
            )

            # Step 6: Detect and handle overlapped speech
            try:
                if show_progress:
                    print("[ASR] Detecting overlapped speech...", flush=True)

                import numpy as np
                import librosa

                # Load audio again for overlap detection
                waveform, sample_rate = librosa.load(audio_path, sr=16000)

                # Calculate features for overlap detection
                for i, segment in enumerate(initial_segments):
                    if i not in speaker_mapping:
                        continue

                    # Extract audio segment
                    start_sample = max(0, int(segment["start"] * sample_rate))
                    end_sample = min(len(waveform), int(segment["end"] * sample_rate))

                    if end_sample - start_sample < 0.5 * sample_rate:
                        continue

                    seg_audio = waveform[start_sample:end_sample]

                    # Calculate spectral flatness and other features
                    stft = np.abs(librosa.stft(seg_audio))
                    flatness = librosa.feature.spectral_flatness(S=stft)[0]
                    flatness_mean = np.mean(flatness)

                    # Lower flatness often indicates overlapped speech
                    if flatness_mean < 0.05:
                        # Check neighbors for different speakers
                        prev_speaker = None
                        next_speaker = None

                        if i > 0 and (i - 1) in speaker_mapping:
                            prev_speaker = speaker_mapping[i - 1]

                        if i < len(initial_segments) - 1 and (i + 1) in speaker_mapping:
                            next_speaker = speaker_mapping[i + 1]

                        # If neighbors have different speakers, this might be an overlap
                        if (
                            prev_speaker
                            and next_speaker
                            and prev_speaker != next_speaker
                            and prev_speaker != speaker_mapping[i]
                            and next_speaker != speaker_mapping[i]
                        ):
                            # Mark as potentially overlapped with both speakers
                            if show_progress:
                                print(
                                    f"[ASR] Detected potential overlap at {segment['start']:.2f}-{segment['end']:.2f}"
                                )

                            # For now, we'll keep the assigned speaker
                            segment["is_overlap"] = True
                            segment["overlap_speakers"] = [prev_speaker, next_speaker]
            except Exception as e:
                if show_progress:
                    print(f"[ASR] Overlap detection failed: {str(e)}")

            # Step 7: Create the final diarization result
            result = []

            for i, segment in enumerate(initial_segments):
                if i not in speaker_mapping:
                    continue

                # Get the assigned speaker
                speaker = speaker_mapping[i]

                # Create the segment
                diarize_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": speaker,
                }

                # Add overlap information if available
                if segment.get("is_overlap", False) and "overlap_speakers" in segment:
                    diarize_segment["overlap"] = True
                    diarize_segment["overlap_speakers"] = segment["overlap_speakers"]

                result.append(diarize_segment)

            if show_progress:
                print(
                    f"[ASR] Enhanced diarization complete with {len(set(s['speaker'] for s in result))} speakers",
                    flush=True,
                )

            return result

        except Exception as e:
            print(f"[ASR] Enhanced diarization failed with error: {str(e)}")

            # Fall back to original method
            print("[ASR] Falling back to standard diarization")
            return self.diarize_audio(
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                show_progress=show_progress,
            )
