from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional
from sonata.constants import LanguageCode
from sonata.utils.text import clean_text_for_language
from textgrids import TextGrid
import os
import whisperx
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

        if not ".wav" in audio_path:
            err_msg = f"Audio file must be a .wav file: {audio_path}"
            self.logger.error(err_msg)
            return {
                "error": err_msg,
                "integrated_transcript": {"plain_text": "", "rich_text": []},
            }

        audio_name = audio_path.split("/")[-1].split(".wav")[0]
        data_directory = f"mfa/{audio_name}"
        align_directory = f"mfa/{audio_name}_align"
        os.makedirs(data_directory, exist_ok=True)
        os.makedirs(align_directory, exist_ok=True)
        subprocess.run(["cp", audio_path, f"{data_directory}/audio.wav"])
        if show_progress:
            print(f"[ASR] Running speech recognition...", flush=True)
            sys.stdout.flush()

        if not os.path.exists(f"{data_directory}/audio.lab"):
            # Transcribe audio with whisper into .lab format
            model = whisperx.load_audio(audio_path)
            transcription = model.transcribe(
                audio_path, batch_size=batch_size, language=language
            )

            if show_progress:
                print(
                    f"[ASR] Transcription complete. Processing {len(transcription.get('segments', []))} segments.",
                    flush=True,
                )

            text = ""
            for segment in transcription["segments"]:
                text += segment["text"]

            # Save .lab file
            with open(f"{data_directory}/audio.lab", "w") as f:
                f.write(text)
        else:
            with open(f"{data_directory}/audio.lab", "r") as f:
                text = f.read()

        if not os.path.exists(f"{align_directory}/audio.TextGrid"):
            # Validate the audio and text corpus first
            subprocess.run(["mfa", "validate", data_directory, "korean_mfa"])
            # Use MFA to get .TextGrid format
            subprocess.run(
                [
                    "mfa",
                    "align",
                    data_directory,
                    "korean_mfa",
                    "korean_mfa",
                    align_directory,
                ]
            )

        # Use TextGrid to get speaker segments
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        tg = TextGrid()
        tg.read(f"{align_directory}/audio.TextGrid")
        word_items = tg["words"]

        # Use TextGrid to get speaker segments
        # Clean the textgrid words
        word_len = len(word_items)
        words = []
        for i in range(word_len):
            if word_items[i].text:
                words.append(word_items[i])

        # Clean the text
        actual_words = clean_text_for_language(text, language)

        print("actual_words", actual_words)

        # Get word timestamps
        # TODO: Fix bug
        start_times = []
        end_times = []
        word_idx = 0
        for i, actual_word in enumerate(actual_words):
            cur_idx = 0
            start_times.append(words[word_idx].xmin)
            while cur_idx < len(actual_word):
                cur_idx += len(words[word_idx].text)
                word_idx += 1
            end_times.append(words[word_idx - 1].xmax)
        # Debug: Saving word audio
        # for i in range(len(actual_words)):
        #     word_audio = waveform[:, int(start_times[i] * sample_rate):int(end_times[i] * sample_rate)]
        #     torchaudio.save(f"temp/{actual_words[i]}.wav", word_audio, sample_rate)
        # Transcribe with whisper-large-v3
        result_segments = []
        for i in range(len(actual_words)):
            result_segments.append(
                {
                    "start": start_times[i],
                    "end": end_times[i],
                    "content": actual_words[i],
                    "type": "voice",
                }
            )
        return result_segments
