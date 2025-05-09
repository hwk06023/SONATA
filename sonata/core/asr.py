import os
import numpy as np
import torch
import logging
import warnings
from typing import Dict, List, Union, Tuple, Optional
from sonata.constants import LanguageCode
from sonata.core.utils.asr import (
    ASRModelManager,
    AudioTranscriber,
    VoiceActivityDetector,
    WordTimestampExtractor,
    SpeakerAssignmentProcessor,
    TextAligner,
)

# Set base environment variables
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check current root logger level and suppress warnings at ERROR level
root_logger = logging.getLogger()
current_level = root_logger.level

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

        # Initialize utility modules
        self.model_manager = ASRModelManager(model_name, device, compute_type)
        self.transcriber = AudioTranscriber(model_name, device, compute_type)
        self.vad = VoiceActivityDetector()
        self.word_extractor = WordTimestampExtractor()
        self.speaker_assigner = SpeakerAssignmentProcessor()
        self.text_aligner = TextAligner()

        # Diarization attributes
        self.diarize_model = None
        self.diarize_model_type = None
        self.embedding_model_name = None
        self.clustering_method = None
        self.speaker_embeddings = {}

        # Logging
        self.logger = logging.getLogger(__name__)

    def load_models(self, language_code: str = LanguageCode.ENGLISH.value):
        """Load WhisperX and alignment models for the specified language.

        Args:
            language_code: ISO language code (e.g., "en", "ko", "zh")
        """
        self.model_manager.get_models(language_code)

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
        # Import diarization modules here to avoid circular imports
        from sonata.core.utils.diarization import (
            SpeakerDiarizerLoader,
            SpeakerDiarizer,
        )

        if self.diarize_model is None:
            if show_progress:
                print(f"[ASR] Loading diarization model...", flush=True)

            # Use the diarization loader
            diarizer_loader = SpeakerDiarizerLoader()

            # Load the appropriate diarization model
            self.diarize_model, self.diarize_model_type = diarizer_loader.load_diarizer(
                embedding_model=embedding_model,
                clustering_method=clustering_method,
                hf_token=hf_token,
                offline_mode=offline_mode,
                offline_config_path=offline_config_path,
                show_progress=show_progress,
            )

            # Store the configuration for future reference
            self.embedding_model_name = embedding_model.lower()
            self.clustering_method = clustering_method.lower()

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
            num_speakers: Expected number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            show_progress: Whether to show progress information

        Returns:
            List of diarization segments with speaker information
        """
        # Import diarization modules here to avoid circular imports
        from sonata.core.utils.diarization import SpeakerDiarizer

        if self.diarize_model is None:
            raise ValueError(
                "Diarization model not loaded. Call load_diarize_model first."
            )

        # Create diarizer instance with loaded model
        diarizer = SpeakerDiarizer(
            model=self.diarize_model, model_type=self.diarize_model_type
        )

        # Perform diarization
        segments = diarizer.diarize(
            audio_path=audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            show_progress=show_progress,
        )

        return segments

    def enhanced_diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """Perform enhanced speaker diarization on an audio file.

        This method uses a more sophisticated approach with multiple steps:
        1. Voice activity detection
        2. Speaker change detection
        3. Speaker embedding extraction
        4. Speaker clustering

        Args:
            audio_path: Path to the audio file
            num_speakers: Expected number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            show_progress: Whether to show progress information

        Returns:
            List of diarization segments with speaker information
        """
        # Import diarization modules here to avoid circular imports
        from sonata.core.utils.diarization import (
            SpeakerDiarizer,
            VoiceActivityDetector as DiarizationVAD,
            ChangeDetector,
            EmbeddingExtractor,
            SpeakerClusterer,
            OverlapDetector,
            SegmentProcessor,
        )

        if show_progress:
            print("[ASR] Performing enhanced speaker diarization...", flush=True)

        try:
            # Try to use the dedicated diarizer for better performance
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
                            f"[ASR] SpeakerDiarizer returned no results, falling back to alternative method",
                            flush=True,
                        )
            except Exception as e:
                if show_progress:
                    print(
                        f"[ASR] Error using SpeakerDiarizer: {str(e)}, falling back to alternative method",
                        flush=True,
                    )

            # --- Alternative method using individual modules ---

            # Step 1: Voice Activity Detection
            vad_processor = DiarizationVAD()
            vad_segments = vad_processor.detect(audio_path, show_progress=show_progress)

            if not vad_segments:
                if show_progress:
                    print("[ASR] No speech detected in audio")
                return []

            # Step 2: Speaker Change Detection
            change_detector = ChangeDetector()
            initial_segments = change_detector.detect_changes(
                audio_path, vad_segments, show_progress=show_progress
            )

            if not initial_segments:
                if show_progress:
                    print("[ASR] No valid segments created after change detection")
                return []

            # Step 3: Extract Speaker Embeddings
            embedding_extractor = EmbeddingExtractor()
            segment_embeddings = embedding_extractor.extract_embeddings(
                audio_path,
                initial_segments,
                embedding_model=self.embedding_model_name,
                show_progress=show_progress,
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

            # Step 4: Cluster Speakers
            clusterer = SpeakerClusterer()
            speaker_labels = clusterer.cluster_speakers(
                segment_embeddings,
                num_speakers=num_speakers,
                show_progress=show_progress,
            )

            # Create segment timings
            segment_timings = [
                (segment["start"], segment["end"]) for segment in initial_segments
            ]

            # Step 5: Detect Overlapped Speech
            overlap_detector = OverlapDetector(device=self.device)
            import torch
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0)  # Convert to mono
            overlap_segments = overlap_detector.detect_overlapped_speech(
                waveform, sample_rate, segment_timings
            )

            # Step 6: Create Final Speaker Segments
            segment_processor = SegmentProcessor()
            speaker_segments = segment_processor.create_speaker_segments(
                segment_timings, speaker_labels
            )

            # Add overlap information
            for overlap_idx in overlap_segments:
                if overlap_idx < len(speaker_segments):
                    speaker_segments[overlap_idx].is_overlap = True

                    # Try to determine overlapping speakers
                    if overlap_idx > 0 and overlap_idx < len(speaker_segments) - 1:
                        prev_speaker = speaker_segments[overlap_idx - 1].speaker
                        next_speaker = speaker_segments[overlap_idx + 1].speaker

                        if prev_speaker != next_speaker:
                            speaker_segments[overlap_idx].overlap_speakers = [
                                prev_speaker,
                                next_speaker,
                            ]

            # Convert speaker segments to dictionary format
            result = []
            for segment in speaker_segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": segment.speaker,
                }

                if segment.is_overlap:
                    segment_dict["overlap"] = True
                    if segment.overlap_speakers:
                        segment_dict["overlap_speakers"] = segment.overlap_speakers

                result.append(segment_dict)

            if show_progress:
                print(
                    f"[ASR] Enhanced diarization complete with {len(set(s['speaker'] for s in result))} speakers",
                    flush=True,
                )

            return result

        except Exception as e:
            print(f"[ASR] Enhanced diarization failed with error: {str(e)}")
            import traceback

            traceback.print_exc()

            # Fall back to standard diarization
            print("[ASR] Falling back to standard diarization")
            return self.diarize_audio(
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                show_progress=show_progress,
            )

    def _capture_speaker_embeddings(self, embeddings, speaker_labels):
        """Store average embeddings for each speaker for later use"""
        unique_speakers = np.unique(speaker_labels)

        for speaker in unique_speakers:
            # Get indices for this speaker
            speaker_indices = np.where(speaker_labels == speaker)[0]

            if len(speaker_indices) > 0:
                # Get embeddings for this speaker
                speaker_embeds = [embeddings[i] for i in speaker_indices]

                # Store average embedding
                avg_embedding = np.mean(speaker_embeds, axis=0)
                self.speaker_embeddings[f"SPEAKER_{int(speaker):02d}"] = avg_embedding

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
        """Process audio file with transcription and optional diarization.

        Args:
            audio_path: Path to the audio file
            language: Language code (ISO format)
            batch_size: Batch size for transcription
            show_progress: Whether to show progress information
            diarize: Whether to perform speaker diarization
            num_speakers: Expected number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            hf_token: Hugging Face token for model access
            embedding_model: Speaker embedding model type
            enhance_vad: Whether to use enhanced voice activity detection
            use_enhanced_diarization: Whether to use enhanced diarization pipeline

        Returns:
            Dictionary with transcription and optional diarization results
        """
        # Step 1: Transcribe audio
        result = self.transcriber.transcribe_audio(
            audio_path=audio_path,
            language=language,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Extract word-level information
        words_with_times = self.word_extractor.extract_word_timestamps(result)

        # Step 2: Perform speaker diarization if requested
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

                    # Assign speakers to words in the transcription
                    if diarize_segments:
                        result = self.speaker_assigner.assign_speakers_to_words(
                            diarize_segments, result
                        )
                except Exception as e:
                    if show_progress:
                        print(f"[ASR] Diarization error: {str(e)}", flush=True)

        # Include words with timestamps in the result
        result["words"] = words_with_times

        return result

    def get_word_timestamps(self, result: Dict) -> List[Dict]:
        """Extract word-level timestamps from the result.

        Args:
            result: Transcription result from process_audio

        Returns:
            List of dictionaries with word, start time, end time, and optional speaker
        """
        return self.word_extractor.extract_word_timestamps(result)
