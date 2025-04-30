import os
import numpy as np
import torch
import whisperx
import ssl
import io
import sys
import logging
import warnings
import scipy
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from typing import Dict, List, Union, Tuple, Optional
from sonata.constants import LanguageCode
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
import importlib.util
import librosa
import webrtcvad
import tempfile
import os
from pydub import AudioSegment

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
        if not isinstance(result, dict):
            self.logger.debug(
                f"Warning: WhisperX result is not a dictionary. Got {type(result)}"
            )
            return []

        if "segments" not in result:
            self.logger.debug(
                f"Warning: WhisperX result does not contain 'segments'. Keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}"
            )
            # Create a minimal output with the whole text if available
            if isinstance(result, dict) and "text" in result:
                return [
                    {
                        "word": result["text"],
                        "start": 0.0,
                        "end": 1.0,
                        "confidence": 1.0,
                    }
                ]
            return []

        # Check if segments is a list
        if not isinstance(result["segments"], list):
            self.logger.debug(
                f"Warning: 'segments' is not a list. Got {type(result['segments'])}"
            )
            return []

        for segment in result["segments"]:
            # Verify segment is a dictionary
            if not isinstance(segment, dict):
                self.logger.debug(
                    f"Warning: Segment is not a dictionary. Got {type(segment)}"
                )
                continue

            # Check for word-level information
            if "words" in segment and isinstance(segment["words"], list):
                for word_data in segment["words"]:
                    # Skip if not a dictionary
                    if not isinstance(word_data, dict):
                        self.logger.debug(
                            f"Warning: Word data is not a dictionary. Got {type(word_data)}"
                        )
                        continue

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
            elif "text" in segment and "start" in segment and "end" in segment:
                # Fallback if no word-level data (shouldn't happen with alignment)
                words_with_timestamps.append(
                    {
                        "word": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                    }
                )
            else:
                # Segment doesn't have either words or required text with timestamps
                self.logger.debug(
                    f"Warning: Segment missing both 'words' and required text fields: {segment.keys() if isinstance(segment, dict) else 'not a dict'}"
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
        """
        Enhanced Voice Activity Detection using multiple VAD methods and ensemble fusion.
        """
        vad_segments = []

        if show_progress:
            print("[ASR] Running enhanced Voice Activity Detection...")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 1. Silero VAD (Neural)
        silero_segments = []
        try:
            if show_progress:
                print("[ASR] Applying Silero VAD (Neural)...")

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            (get_speech_timestamps, _, _, _, _) = utils

            timestamps = get_speech_timestamps(
                torch.tensor(audio), model, sampling_rate=sr, threshold=0.5
            )

            for ts in timestamps:
                start = ts["start"] / sr
                end = ts["end"] / sr
                silero_segments.append({"start": start, "end": end})

        except Exception as e:
            if show_progress:
                print(f"[ASR] Silero VAD failed: {str(e)}")

        # 2. WebRTC VAD
        webrtc_segments = []
        try:
            import webrtcvad

            if show_progress:
                print("[ASR] Applying WebRTC VAD...")

            vad = webrtcvad.Vad()
            vad.set_mode(3)  # Aggressive mode

            # WebRTC VAD requires frame sizes of 10, 20, or 30 ms
            frame_duration = 30  # in ms
            frame_size = int(sr * frame_duration / 1000)

            # Pad audio to ensure it's divisible by frame_size
            padding = frame_size - (len(audio) % frame_size)
            padded_audio = np.pad(audio, (0, padding), "constant")

            # Detect speech frames
            speech_frames = []
            for i in range(0, len(padded_audio), frame_size):
                frame = padded_audio[i : i + frame_size]
                frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                is_speech = vad.is_speech(frame_bytes, sr)
                speech_frames.append(is_speech)

            # Group consecutive speech frames into segments
            in_speech = False
            start_frame = 0

            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    start_frame = i
                    in_speech = True
                elif not is_speech and in_speech:
                    end_frame = i
                    start_time = start_frame * frame_duration / 1000
                    end_time = end_frame * frame_duration / 1000
                    webrtc_segments.append({"start": start_time, "end": end_time})
                    in_speech = False

            # Handle the case where the audio ends during speech
            if in_speech:
                end_frame = len(speech_frames)
                start_time = start_frame * frame_duration / 1000
                end_time = end_frame * frame_duration / 1000
                webrtc_segments.append({"start": start_time, "end": end_time})

        except ImportError:
            if show_progress:
                print("[ASR] WebRTC VAD module not installed, skipping this step")
        except Exception as e:
            if show_progress:
                print(f"[ASR] WebRTC VAD failed: {str(e)}")

        # 3. Energy-based VAD
        energy_segments = []
        try:
            if show_progress:
                print("[ASR] Applying energy-based VAD...")

            # Calculate energy with multiple window sizes for better accuracy
            window_sizes = [
                int(0.01 * sr),
                int(0.03 * sr),
                int(0.05 * sr),
            ]
            hop_length = int(0.01 * sr)  # 10ms hop

            for window_size in window_sizes:
                # Calculate energy in each window
                energy = []
                for i in range(0, len(audio) - window_size, hop_length):
                    frame = audio[i : i + window_size]
                    energy.append(np.sum(frame**2))

                # Normalize energy
                energy = np.array(energy)
                if np.max(energy) - np.min(energy) > 0:
                    energy = (energy - np.min(energy)) / (
                        np.max(energy) - np.min(energy) + 1e-10
                    )

                # Adaptive threshold based on percentile
                percentile_20 = np.percentile(energy, 20)
                percentile_50 = np.percentile(energy, 50)
                threshold = percentile_20 + 0.1 * (percentile_50 - percentile_20)

                # Smooth the energy curve
                energy_smooth = scipy.ndimage.gaussian_filter1d(energy, sigma=2)

                # Find segments above threshold
                is_speech = energy_smooth > threshold

                # Apply hysteresis to avoid rapid switching
                hysteresis_up = threshold * 0.8  # Lower threshold for staying in speech
                hysteresis_down = threshold * 1.2  # Higher threshold for exiting speech

                in_speech = False
                start_idx = 0
                energy_window_segments = []

                for i, speech in enumerate(is_speech):
                    if not in_speech and energy_smooth[i] > threshold:
                        in_speech = True
                        start_idx = i
                    elif in_speech and energy_smooth[i] < hysteresis_down:
                        in_speech = False
                        end_idx = i
                        start_time = start_idx * hop_length / sr
                        end_time = end_idx * hop_length / sr

                        # Only add segments of reasonable duration
                        if end_time - start_time > 0.2:  # At least 200ms
                            energy_window_segments.append(
                                {"start": start_time, "end": end_time}
                            )

                # Don't forget the last segment
                if in_speech:
                    end_idx = len(is_speech)
                    start_time = start_idx * hop_length / sr
                    end_time = end_idx * hop_length / sr

                    if end_time - start_time > 0.2:
                        energy_window_segments.append(
                            {"start": start_time, "end": end_time}
                        )

                # Add non-overlapping segments from this window size
                for segment in energy_window_segments:
                    # Check if this segment overlaps significantly with any existing segment
                    should_add = True
                    for existing_seg in energy_segments:
                        overlap_start = max(segment["start"], existing_seg["start"])
                        overlap_end = min(segment["end"], existing_seg["end"])

                        if overlap_end > overlap_start:
                            # Calculate overlap percentage
                            segment_duration = segment["end"] - segment["start"]
                            overlap_duration = overlap_end - overlap_start

                            if overlap_duration / segment_duration > 0.7:
                                # More than 70% overlap, skip this segment
                                should_add = False
                                break

                    if should_add:
                        energy_segments.append(segment)

        except Exception as e:
            if show_progress:
                print(f"[ASR] Energy-based VAD failed: {str(e)}, continuing without it")

        # 4. PyAnnote VAD
        pyannote_segments = []
        try:
            if hasattr(self, "diarize_model") and self.diarize_model is not None:
                if show_progress:
                    print("[ASR] Applying PyAnnote VAD...")

                    # Use diarization pipeline to get VAD results
                    if hasattr(self.diarize_model, "__call__") and hasattr(
                        self.diarize_model, "get_vad"
                    ):
                        # Direct access to VAD in newer versions
                        vad_results = self.diarize_model.get_vad(audio_path)
                        for segment, _, label in vad_results.itertracks(
                            yield_label=True
                        ):
                            if label == "SPEECH":
                                pyannote_segments.append(
                                    {"start": segment.start, "end": segment.end}
                                )
                    else:
                        # Extract VAD from diarization result
                        diarize_result = self.diarize_model(audio_path)
                        # Collect all segments regardless of speaker
                        for segment, _, _ in diarize_result.itertracks(
                            yield_label=True
                        ):
                            pyannote_segments.append(
                                {"start": segment.start, "end": segment.end}
                            )
        except Exception as e:
            if show_progress:
                print(f"[ASR] PyAnnote VAD failed: {str(e)}")

        # 5. Combine all VAD results with priority weighting
        # Assign weights to different VAD methods
        vad_methods = [
            {"name": "Silero", "segments": silero_segments, "weight": 1.0},
            {"name": "WebRTC", "segments": webrtc_segments, "weight": 0.8},
            {"name": "Energy", "segments": energy_segments, "weight": 0.5},
            {"name": "PyAnnote", "segments": pyannote_segments, "weight": 0.9},
        ]

        # Start with an empty timeline
        combined_segments = []

        # Sort all segments by start time
        all_segments = []
        for method in vad_methods:
            for segment in method["segments"]:
                all_segments.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "method": method["name"],
                        "weight": method["weight"],
                    }
                )

        all_segments.sort(key=lambda x: x["start"])

        # Merge overlapping segments with weight-based voting
        if not all_segments:
            return []

        current = all_segments[0]
        voting_segments = [current]

        for segment in all_segments[1:]:
            # Check if this segment overlaps with current merged segment
            if segment["start"] <= current["end"] + 0.3:  # Allow small gaps (300ms)
                # Extend current segment if the incoming one has higher weight
                if segment["end"] > current["end"]:
                    if segment["weight"] >= current["weight"] * 0.8:
                        current["end"] = segment["end"]
                voting_segments.append(segment)
            else:
                # Create a new segment for non-overlapping parts
                # First, finalize the current segment with voting
                if len(voting_segments) > 1:
                    # Multiple methods detected this segment, use weighted voting
                    methods = set(s["method"] for s in voting_segments)

                    # If multiple methods agree, use their consensus boundaries
                    if len(methods) >= 2:
                        # Calculate weighted start time
                        starts = [(s["start"], s["weight"]) for s in voting_segments]
                        total_weight = sum(weight for _, weight in starts)
                        weighted_start = (
                            sum(start * weight for start, weight in starts)
                            / total_weight
                        )

                        # Calculate weighted end time
                        ends = [(s["end"], s["weight"]) for s in voting_segments]
                        total_weight = sum(weight for _, weight in ends)
                        weighted_end = (
                            sum(end * weight for end, weight in ends) / total_weight
                        )

                        combined_segments.append(
                            {"start": weighted_start, "end": weighted_end}
                        )
                    else:
                        # Single method, use as is
                        combined_segments.append(
                            {"start": current["start"], "end": current["end"]}
                        )
                else:
                    # Single method detected this segment
                    combined_segments.append(
                        {"start": current["start"], "end": current["end"]}
                    )

                # Move to the new segment
                current = segment
                voting_segments = [current]

        # Don't forget the last segment
        if voting_segments:
            if len(voting_segments) > 1:
                # Multiple methods detected this segment, use weighted voting
                methods = set(s["method"] for s in voting_segments)

                # If multiple methods agree, use their consensus boundaries
                if len(methods) >= 2:
                    # Calculate weighted start time
                    starts = [(s["start"], s["weight"]) for s in voting_segments]
                    total_weight = sum(weight for _, weight in starts)
                    weighted_start = (
                        sum(start * weight for start, weight in starts) / total_weight
                    )

                    # Calculate weighted end time
                    ends = [(s["end"], s["weight"]) for s in voting_segments]
                    total_weight = sum(weight for _, weight in ends)
                    weighted_end = (
                        sum(end * weight for end, weight in ends) / total_weight
                    )

                    combined_segments.append(
                        {"start": weighted_start, "end": weighted_end}
                    )
                else:
                    # Single method, use as is
                    combined_segments.append(
                        {"start": current["start"], "end": current["end"]}
                    )
            else:
                # Single method detected this segment
                combined_segments.append(
                    {"start": current["start"], "end": current["end"]}
                )

        # 6. Final cleanup and filtering
        final_segments = []

        # Sort by start time
        combined_segments.sort(key=lambda x: x["start"])

        # Merge segments that are very close or overlapping
        if not combined_segments:
            return []

        current = combined_segments[0]

        for segment in combined_segments[1:]:
            # If this segment starts soon after the current one ends
            if segment["start"] - current["end"] < 0.3:  # 300ms threshold for merging
                # Merge by extending the end time
                current["end"] = max(current["end"], segment["end"])
            else:
                # Add the current segment to the final list and move to the next
                final_segments.append(current)
                current = segment

        # Add the last segment
        final_segments.append(current)

        # Filter out very short segments (likely noise)
        filtered_segments = [s for s in final_segments if s["end"] - s["start"] > 0.3]

        if show_progress:
            counts = {}
            for method in vad_methods:
                counts[method["name"]] = len(method["segments"])

            print(
                f"[ASR] VAD detection counts - Silero: {counts['Silero']}, WebRTC: {counts['WebRTC']}, "
                f"Energy: {counts['Energy']}, PyAnnote: {counts['PyAnnote']}"
            )
            print(
                f"[ASR] Enhanced VAD detected {len(filtered_segments)} speech segments after ensemble fusion",
                flush=True,
            )

        return filtered_segments

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

                # Completely avoid direct SpeakerSegmentation import
                # Instead, use the diarize_model which is more reliable
                if hasattr(self, "diarize_model") and self.diarize_model is not None:
                    # Store reference to the diarize model to use in the nested function
                    diarize_model = self.diarize_model

                    # Create a segmentation function based on diarization results
                    def detect_speaker_changes_with_diarization(audio_file, segments):
                        detected_changes = []

                        for segment in segments:
                            try:
                                # Process this segment with diarization without start/end parameters
                                # Instead, extract the segment and process it
                                import soundfile as sf
                                from pydub import AudioSegment
                                import tempfile
                                import os

                                # Read the audio segment
                                audio_obj = AudioSegment.from_file(audio_file)
                                # Extract the segment (convert to milliseconds)
                                start_ms = int(segment["start"] * 1000)
                                end_ms = int(segment["end"] * 1000)
                                segment_audio = audio_obj[start_ms:end_ms]

                                # Save to temporary file
                                with tempfile.NamedTemporaryFile(
                                    suffix=".wav", delete=False
                                ) as tmp_file:
                                    temp_path = tmp_file.name

                                segment_audio.export(temp_path, format="wav")

                                # Process with diarization using model from outer scope
                                try:
                                    diar_result = diarize_model(temp_path)

                                    # Extract change points by detecting speaker changes
                                    prev_speaker = None
                                    for track, _, speaker in diar_result.itertracks(
                                        yield_label=True
                                    ):
                                        if (
                                            prev_speaker is not None
                                            and prev_speaker != speaker
                                        ):
                                            # Add the change point (adjusted to segment start time)
                                            change_time = segment["start"] + track.start
                                            detected_changes.append(change_time)
                                        prev_speaker = speaker
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.unlink(temp_path)

                            except Exception as e:
                                if show_progress:
                                    print(
                                        f"[ASR] Error processing segment {segment}: {str(e)}"
                                    )

                        return detected_changes

                    # Process segments and collect change points
                    for segment in vad_segments:
                        # Skip very short segments
                        duration = segment["end"] - segment["start"]
                        if duration < 1.0:  # Skip segments shorter than 1s
                            continue

                        segment_changes = detect_speaker_changes_with_diarization(
                            audio_path, [segment]
                        )

                        # Add the detected changes to our main list
                        change_points.extend(segment_changes)

                    if show_progress:
                        print(
                            f"[ASR] Detected {len(change_points)} change points using diarization"
                        )
                else:
                    # Fall back to using pyannote.audio's segmentation models
                    # Try to access the Model class to create a segmentation model
                    try:
                        from pyannote.audio import Model
                        import torch

                        # Check if segmentation model is available
                        segmentation_model_available = False

                        try:
                            # Try to instantiate a segmentation model
                            seg_model = Model.from_pretrained(
                                "pyannote/segmentation-3.0", use_auth_token=None
                            ).to(device)
                            segmentation_model_available = True
                        except Exception as model_error:
                            if show_progress:
                                print(
                                    f"[ASR] Could not load segmentation model: {str(model_error)}"
                                )

                        if segmentation_model_available:
                            # Process each VAD segment with the segmentation model
                            for segment in vad_segments:
                                # Extract the segment audio
                                start_sample = int(segment["start"] * sample_rate)
                                end_sample = int(segment["end"] * sample_rate)

                                # Skip very short segments
                                if end_sample - start_sample < 0.5 * sample_rate:
                                    continue

                                seg_audio = waveform[start_sample:end_sample]

                                # Process with segmentation model
                                with torch.no_grad():
                                    # Create input tensor
                                    audio_tensor = (
                                        torch.tensor(seg_audio).unsqueeze(0).to(device)
                                    )

                                    # Get embeddings and detect changes
                                    embeddings = seg_model(
                                        {
                                            "waveform": audio_tensor,
                                            "sample_rate": sample_rate,
                                        }
                                    )

                                    # Analyze embeddings to find change points
                                    # Implementation will depend on the model output format
                                    # Here is a simple approach using cosine distance:
                                    emb = embeddings.squeeze().cpu().numpy()

                                    if emb.ndim > 1 and emb.shape[0] > 1:
                                        from scipy.spatial.distance import cosine
                                        from scipy.signal import find_peaks

                                        # Calculate distances between consecutive embeddings
                                        distances = []
                                        for i in range(1, len(emb)):
                                            dist = cosine(emb[i - 1], emb[i])
                                            distances.append(dist)

                                        if distances:
                                            # Find peaks in distances (potential change points)
                                            distances = np.array(distances)
                                            # Adaptive threshold based on mean and std
                                            threshold = np.mean(
                                                distances
                                            ) + 1.0 * np.std(distances)
                                            peaks, _ = find_peaks(
                                                distances, height=threshold, distance=5
                                            )

                                            # Convert peak indices to time and add to change points
                                            frame_shift = (
                                                segment["end"] - segment["start"]
                                            ) / emb.shape[0]
                                            for peak in peaks:
                                                change_time = (
                                                    segment["start"]
                                                    + (peak + 1) * frame_shift
                                                )
                                                change_points.append(change_time)
                    except ImportError:
                        if show_progress:
                            print(
                                "[ASR] PyAnnote's segmentation models are not available"
                            )
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
            # Try to use the custom diarizer for better performance
            try:
                from sonata.core.speaker_diarization import SpeakerDiarizer

                speaker_diarizer = SpeakerDiarizer(device=self.device)

                # Use the enhanced speaker diarizer
                speaker_segments = speaker_diarizer.diarize(
                    audio_path=audio_path,
                    num_speakers=num_speakers,
                    show_progress=show_progress,
                )

                # Convert to standard format
                result = []
                for segment in speaker_segments:
                    diarize_segment = {
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": segment.speaker,
                    }

                    # Add overlap information if available
                    if segment.is_overlap and segment.overlap_speakers:
                        diarize_segment["overlap"] = True
                        diarize_segment["overlap_speakers"] = segment.overlap_speakers

                    result.append(diarize_segment)

                if result:
                    if show_progress:
                        print(
                            f"[ASR] Enhanced diarization using SpeakerDiarizer completed successfully with {len(set(s['speaker'] for s in result))} speakers",
                            flush=True,
                        )
                    return result
                else:
                    if show_progress:
                        print(
                            f"[ASR] SpeakerDiarizer returned no results, falling back to traditional method",
                            flush=True,
                        )

            except Exception as e:
                if show_progress:
                    print(
                        f"[ASR] Error using SpeakerDiarizer: {str(e)}, falling back to traditional method",
                        flush=True,
                    )

            # --- Original method as fallback starts here ---

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
            # Ensure we're using the fully qualified method name to avoid reference errors
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

                    # Enhanced overlap detection
                    # 1. Calculate spectral flatness
                    stft = np.abs(librosa.stft(seg_audio))
                    flatness = librosa.feature.spectral_flatness(S=stft)[0]
                    flatness_mean = np.mean(flatness)

                    # 2. Calculate harmonic-percussive separation
                    try:
                        harmonic, percussive = librosa.effects.hpss(seg_audio)
                        hp_ratio = np.mean(np.abs(harmonic)) / (
                            np.mean(np.abs(percussive)) + 1e-8
                        )
                    except:
                        hp_ratio = 1.0

                    # 3. Spectral centroid variation (high variation can indicate multiple speakers)
                    centroid = librosa.feature.spectral_centroid(
                        y=seg_audio, sr=sample_rate
                    )[0]
                    centroid_std = np.std(centroid)

                    # Compute "complexity score" - higher means more likely to be overlap
                    complexity_score = (
                        (centroid_std / 1000) * (1 - flatness_mean) * (1 + hp_ratio)
                    )

                    # Improved overlap detection condition
                    if flatness_mean < 0.07 and complexity_score > 0.4:
                        # Check neighbors for different speakers
                        prev_speaker = None
                        next_speaker = None

                        if i > 0 and (i - 1) in speaker_mapping:
                            prev_speaker = speaker_mapping[i - 1]

                        if i < len(initial_segments) - 1 and (i + 1) in speaker_mapping:
                            next_speaker = speaker_mapping[i + 1]

                        # If neighbors have different speakers, this might be an overlap
                        if (
                            prev_speaker is not None
                            and next_speaker is not None
                            and prev_speaker != next_speaker
                        ):
                            # Mark as potentially overlapped with both speakers
                            if show_progress:
                                print(
                                    f"[ASR] Detected potential overlap at {segment['start']:.2f}-{segment['end']:.2f}"
                                )

                            # Mark the segment as overlap
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

            # Step 8: Merge very short segments with the same speaker
            merged_result = []
            if result:
                current = result[0].copy()
                for next_seg in result[1:]:
                    # If same speaker and short gap
                    if (
                        next_seg["speaker"] == current["speaker"]
                        and next_seg["start"] - current["end"] < 0.3
                        and next_seg["start"] - current["end"] >= 0
                    ):
                        # Merge them
                        current["end"] = next_seg["end"]
                        # Preserve overlap info
                        if "overlap" in next_seg:
                            current["overlap"] = next_seg["overlap"]
                            if "overlap_speakers" in next_seg:
                                current["overlap_speakers"] = next_seg[
                                    "overlap_speakers"
                                ]
                    else:
                        # Add current segment to results and start new one
                        merged_result.append(current)
                        current = next_seg.copy()

                # Add the last segment
                merged_result.append(current)

            if show_progress:
                print(
                    f"[ASR] Enhanced diarization complete with {len(set(s['speaker'] for s in merged_result))} speakers",
                    flush=True,
                )

            return merged_result

        except Exception as e:
            print(f"[ASR] Enhanced diarization failed with error: {str(e)}")
            import traceback

            traceback.print_exc()  # 자세한 오류 추적을 위해 추가

            # Fall back to original method
            print("[ASR] Falling back to standard diarization")
            return self.diarize_audio(
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                show_progress=show_progress,
            )

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
            List of embedding vectors for each segment
        """
        if show_progress:
            print("[ASR] Extracting speaker embeddings...", flush=True)

        try:
            import torch
            import librosa
            import numpy as np
            import tempfile
            import os
            from pydub import AudioSegment
            from tqdm import tqdm

            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=16000)

            device = (
                self.device
                if hasattr(self, "device")
                and self.device == "cuda"
                and torch.cuda.is_available()
                else "cpu"
            )

            embeddings = []
            ecapa_embeddings = []
            wavlm_embeddings = []

            # Check if we have a speech embedding model from diarization
            has_embedding_extractor = (
                hasattr(self, "diarize_model")
                and self.diarize_model is not None
                and hasattr(self.diarize_model, "get_embeddings")
            )

            # Track if we have ECAPA-TDNN model
            has_ecapa_model = False

            # Set up progress tracking
            total_segments = len(segments)
            if show_progress and total_segments > 1:
                pbar = tqdm(total=total_segments, desc="Extracting embeddings")

            # Extract embeddings for each segment
            for i, segment in enumerate(segments):
                try:
                    start_time = segment["start"]
                    end_time = segment["end"]

                    # Skip segments that are too short
                    if end_time - start_time < 0.5:
                        if show_progress and total_segments > 1:
                            pbar.update(1)
                        continue

                    # 1. Try to get embeddings from diarization model
                    model_embedding = None
                    if has_embedding_extractor:
                        try:
                            # Extract the segment audio
                            start_sample = int(start_time * sample_rate)
                            end_sample = min(len(waveform), int(end_time * sample_rate))

                            if end_sample - start_sample < 0.1 * sample_rate:
                                # Skip too short segments
                                if show_progress and total_segments > 1:
                                    pbar.update(1)
                                continue

                            segment_audio = waveform[start_sample:end_sample]

                            # Get embedding using the diarization model's embedding extractor
                            with torch.no_grad():
                                segment_tensor = (
                                    torch.tensor(segment_audio).unsqueeze(0).to(device)
                                )
                                model_embedding = self.diarize_model.get_embeddings(
                                    {
                                        "waveform": segment_tensor,
                                        "sample_rate": sample_rate,
                                    }
                                )

                                # Convert to numpy array
                                model_embedding = (
                                    model_embedding.squeeze().cpu().numpy()
                                )
                                wavlm_embeddings.append(model_embedding)
                        except Exception as e:
                            model_embedding = None

                    # 2. Try SpeechBrain ECAPA-TDNN (superior speaker embeddings)
                    ecapa_embedding = None
                    try:
                        import speechbrain as sb

                        # Check if we already have the embedding model
                        if not hasattr(self, "embed_model") or self.embed_model is None:
                            # Load the ECAPA-TDNN model
                            self.embed_model = (
                                sb.pretrained.EncoderClassifier.from_hparams(
                                    source="speechbrain/spkrec-ecapa-voxceleb",
                                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                    run_opts={"device": device},
                                )
                            )
                            has_ecapa_model = True

                        # Extract the segment
                        audio_obj = AudioSegment.from_file(audio_path)
                        start_ms = int(start_time * 1000)
                        end_ms = int(end_time * 1000)
                        segment_audio = audio_obj[start_ms:end_ms]

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False
                        ) as tmp_file:
                            temp_path = tmp_file.name

                        segment_audio.export(temp_path, format="wav")

                        try:
                            # Extract embedding with SpeechBrain
                            with torch.no_grad():
                                signal, fs = self.embed_model.load_audio(temp_path)
                                batch = signal.unsqueeze(0).to(device)
                                ecapa_embedding = self.embed_model.encode_batch(batch)
                                ecapa_embedding = (
                                    ecapa_embedding.squeeze().cpu().numpy()
                                )
                                ecapa_embeddings.append(ecapa_embedding)
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    except Exception as e:
                        ecapa_embedding = None

                    # Decide which embedding to use (prefer ECAPA-TDNN if available)
                    if ecapa_embedding is not None:
                        embeddings.append(ecapa_embedding)
                    elif model_embedding is not None:
                        embeddings.append(model_embedding)
                    else:
                        # Last resort: Use MFCC features if everything else fails
                        try:
                            start_sample = int(start_time * sample_rate)
                            end_sample = min(len(waveform), int(end_time * sample_rate))
                            segment_audio = waveform[start_sample:end_sample]

                            # Extract enhanced MFCC features with spectral contrast
                            mfccs = librosa.feature.mfcc(
                                y=segment_audio, sr=sample_rate, n_mfcc=24
                            )
                            contrast = librosa.feature.spectral_contrast(
                                y=segment_audio, sr=sample_rate
                            )
                            # Create a feature vector
                            features = np.concatenate(
                                [
                                    np.mean(mfccs, axis=1),
                                    np.std(mfccs, axis=1),
                                    np.mean(contrast, axis=1),
                                ]
                            )

                            embeddings.append(features)
                        except Exception as mfcc_error:
                            print(
                                f"[ASR] Failed to extract any features: {str(mfcc_error)}"
                            )
                            # If all methods fail, skip this segment
                            if show_progress and total_segments > 1:
                                pbar.update(1)
                            continue
                except Exception as segment_error:
                    print(f"[ASR] Error processing segment {i}: {str(segment_error)}")
                    if show_progress and total_segments > 1:
                        pbar.update(1)
                    continue

                # Update progress bar
                if show_progress and total_segments > 1:
                    pbar.update(1)

            # Close progress bar
            if show_progress and total_segments > 1:
                pbar.close()

            # Provide information about which embeddings we're using
            if ecapa_embeddings and len(ecapa_embeddings) == len(embeddings):
                if show_progress:
                    print(
                        f"[ASR] Using ECAPA-TDNN embeddings ({len(embeddings)} segments)"
                    )
            elif wavlm_embeddings and len(wavlm_embeddings) == len(embeddings):
                if show_progress:
                    print(f"[ASR] Using WavLM embeddings ({len(embeddings)} segments)")
            else:
                if show_progress:
                    print(f"[ASR] Using mixed embeddings ({len(embeddings)} segments)")

            return embeddings

        except Exception as e:
            if show_progress:
                print(f"[ASR] Speaker embedding extraction failed: {str(e)}")
                import traceback

                traceback.print_exc()
            return []

    def _cluster_speakers(
        self,
        embeddings,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        show_progress=True,
    ):
        """
        Cluster speaker embeddings to identify unique speakers.

        Args:
            embeddings: List of speaker embeddings for each segment
            num_speakers: Optional fixed number of speakers
            min_speakers: Minimum number of speakers if auto-determining
            max_speakers: Maximum number of speakers if auto-determining
            show_progress: Whether to show progress information

        Returns:
            Dictionary mapping segment indices to speaker IDs
        """
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering, SpectralClustering
        from sklearn.metrics import silhouette_score

        # Handle edge cases
        if len(embeddings) <= 1:
            return {0: f"SPEAKER_00"} if len(embeddings) == 1 else {}

        # Convert embeddings to numpy array if needed
        X = np.array(embeddings)

        # Normalize embeddings for more robust clustering
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10  # Avoid division by zero
        X = X / norm

        # If number of speakers is provided, use it directly
        if num_speakers is not None:
            n_clusters = min(num_speakers, len(X) - 1)
            n_clusters = max(2, n_clusters)  # At least 2 speakers
        else:
            # Auto-determine number of speakers with eigenvalue analysis
            try:
                from scipy.spatial.distance import cosine

                # Compute similarity matrix
                similarity_matrix = 1 - np.array(
                    [[cosine(emb1, emb2) for emb2 in X] for emb1 in X]
                )

                # Apply adaptive threshold to create affinity matrix
                threshold = np.mean(similarity_matrix) * 0.5
                affinity_matrix = (similarity_matrix > threshold).astype(float)

                # Compute Laplacian
                from scipy import sparse

                if not sparse.issparse(affinity_matrix):
                    affinity_matrix = sparse.csr_matrix(affinity_matrix)

                # Get the Laplacian from SpectralClustering
                try:
                    from sklearn.cluster._spectral import spectral_embedding

                    eigenvalues = spectral_embedding(
                        affinity_matrix,
                        n_components=min(10, affinity_matrix.shape[0] - 1),
                        eigen_solver="arpack",
                        drop_first=True,
                        return_eigenvalues=True,
                    )[1]
                except (ImportError, AttributeError):
                    # Fallback for older scikit-learn versions
                    from scipy.sparse.linalg import eigsh

                    laplacian = sparse.csgraph.laplacian(affinity_matrix, normed=True)
                    eigenvalues, _ = eigsh(
                        laplacian, k=min(10, affinity_matrix.shape[0] - 1), which="SM"
                    )

                # Find the elbow point in eigenvalues
                eigenvalues = sorted(eigenvalues)
                diffs = np.diff(eigenvalues)

                # Find largest gap in eigenvalues
                largest_gap_idx = np.argmax(diffs) + 1

                # Estimate is the index of largest gap + 1
                estimated_speakers = largest_gap_idx + 1

                # Adjust with min/max constraints
                if min_speakers is not None:
                    estimated_speakers = max(min_speakers, estimated_speakers)
                if max_speakers is not None:
                    estimated_speakers = min(max_speakers, estimated_speakers)

                # Ensure reasonable bounds
                n_clusters = max(2, min(8, estimated_speakers))

                if show_progress:
                    print(
                        f"[ASR] Estimated {n_clusters} speakers using eigenvalue analysis"
                    )

            except Exception as e:
                if show_progress:
                    print(f"[ASR] Error estimating speaker count: {str(e)}")

                # Fallback estimation based on heuristics
                min_speakers = min_speakers or 2
                max_speakers = max_speakers or min(8, len(X) - 1)

                # Default estimate based on audio length (as approximated by segment count)
                n_clusters = max(
                    min_speakers, min(max_speakers, int(np.sqrt(len(X)) / 2))
                )

        # Try multiple clustering methods with proper version handling
        methods = []

        # Check scikit-learn version and use compatible parameters
        # 1. Agglomerative with cosine distance
        try:
            test_agg = AgglomerativeClustering(
                n_clusters=2, affinity="cosine", linkage="average"
            )
            methods.append(
                {
                    "name": "Agglomerative (cosine)",
                    "method": AgglomerativeClustering(
                        n_clusters=n_clusters, affinity="cosine", linkage="average"
                    ),
                }
            )
        except TypeError:
            # Fallback for older scikit-learn versions without affinity support
            methods.append(
                {
                    "name": "Agglomerative (basic)",
                    "method": AgglomerativeClustering(n_clusters=n_clusters),
                }
            )

        # 2. Spectral clustering with nearest neighbors
        try:
            methods.append(
                {
                    "name": "Spectral",
                    "method": SpectralClustering(
                        n_clusters=n_clusters,
                        affinity="nearest_neighbors",
                        n_neighbors=min(len(X) // 3, 10),
                        random_state=42,
                    ),
                }
            )
        except TypeError:
            # Basic spectral clustering fallback
            try:
                methods.append(
                    {
                        "name": "Spectral (basic)",
                        "method": SpectralClustering(
                            n_clusters=n_clusters, random_state=42
                        ),
                    }
                )
            except:
                pass

        # 3. Ward linkage clustering (generally available in all versions)
        try:
            methods.append(
                {
                    "name": "Agglomerative (ward)",
                    "method": AgglomerativeClustering(
                        n_clusters=n_clusters, linkage="ward"
                    ),
                }
            )
        except:
            pass

        best_score = -1
        best_labels = None
        best_method = None

        # Try each clustering method and evaluate results
        for method_info in methods:
            try:
                labels = method_info["method"].fit_predict(X)

                # Skip if only one cluster was found
                if len(set(labels)) <= 1:
                    continue

                # Evaluate clustering quality
                try:
                    # Try with different metrics depending on what's available
                    try:
                        from sklearn.metrics import (
                            silhouette_score,
                            calinski_harabasz_score,
                        )

                        # Silhouette score measures how well-separated the clusters are
                        sil_score = silhouette_score(X, labels, metric="cosine")

                        # Calinski-Harabasz measures the ratio of between-cluster to within-cluster dispersion
                        ch_score = calinski_harabasz_score(X, labels)

                        # Combined score (weighted average)
                        combined_score = (0.7 * sil_score) + (0.3 * (ch_score / 10000))
                    except (ImportError, AttributeError):
                        # Simplified version for older scikit-learn
                        sil_score = silhouette_score(X, labels)
                        combined_score = sil_score
                        ch_score = 0

                    if show_progress:
                        print(
                            f"[ASR] {method_info['name']}: silhouette={sil_score:.4f}, CH={ch_score:.2f}"
                        )

                    if combined_score > best_score:
                        best_score = combined_score
                        best_labels = labels
                        best_method = method_info["name"]
                except Exception as e:
                    # If evaluation fails, use these labels if we don't have any yet
                    if best_labels is None:
                        best_labels = labels
                        best_method = method_info["name"]
            except Exception as e:
                if show_progress:
                    print(f"[ASR] Error with {method_info['name']}: {str(e)}")

        # If all methods failed, use simple agglomerative clustering
        if best_labels is None:
            try:
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                best_labels = clustering.fit_predict(X)
                best_method = "Fallback Agglomerative"
            except Exception as e:
                if show_progress:
                    print(f"[ASR] All clustering methods failed. Final error: {str(e)}")
                # Create sequential labels if everything fails
                best_labels = np.zeros(len(X), dtype=int)
                for i in range(1, min(n_clusters, len(X))):
                    if i < len(X):
                        best_labels[i] = i % n_clusters
                best_method = "Emergency Fallback (Sequential Assignment)"

        if show_progress:
            print(f"[ASR] Selected {best_method} with {len(set(best_labels))} speakers")

        # Format speaker labels with SPEAKER_XX format
        mapping = {
            i: f"SPEAKER_{int(label):02d}" for i, label in enumerate(best_labels)
        }
        return mapping
