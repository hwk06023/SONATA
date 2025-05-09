import os
import logging
from typing import Dict, List, Union, Tuple, Optional
from sonata.constants import (
    AUDIO_EVENT_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_DEVICE,
    DEFAULT_COMPUTE_TYPE,
)

# Import our new modular structure
from sonata.core.utils.transcriber import IntegratedTranscriber as ModularTranscriber


class IntegratedTranscriber:
    def __init__(
        self,
        asr_model: str = DEFAULT_MODEL,
        audio_model_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        custom_audio_thresholds: Optional[Dict[str, float]] = None,
        deep_detect: bool = False,
        deep_detect_params: Optional[Dict] = None,
    ):
        """Initialize the integrated transcriber.

        Args:
            asr_model: WhisperX model name to use
            audio_model_path: Path to custom audio event detection model (optional)
            device: Compute device (cpu/cuda)
            compute_type: Compute precision (float32, float16, etc.)
            custom_audio_thresholds: Dictionary of custom thresholds for specific audio event types (optional)
            deep_detect: Whether to use multi-scale audio event detection
            deep_detect_params: Dictionary with window_sizes and hop_sizes for deep detection
        """
        self.logger = logging.getLogger(__name__)

        # Initialize the modular transcriber
        self.transcriber = ModularTranscriber(
            asr_model=asr_model,
            audio_model_path=audio_model_path,
            device=device,
            compute_type=compute_type,
            custom_audio_thresholds=custom_audio_thresholds,
            deep_detect=deep_detect,
            deep_detect_params=deep_detect_params,
        )

    def process_audio(
        self,
        audio_path: str,
        language: str = DEFAULT_LANGUAGE,
        audio_threshold: float = AUDIO_EVENT_THRESHOLD,
        batch_size: int = 16,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        save_diarization_steps: bool = False,
    ) -> Dict:
        """Process audio to get transcription with audio events integrated.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            audio_threshold: Detection threshold for audio events
            batch_size: Batch size for processing
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers for diarization (optional)
            save_diarization_steps: Whether to save intermediate outputs for each diarization step

        Returns:
            Dictionary containing the complete transcription results
        """
        return self.transcriber.process_audio(
            audio_path=audio_path,
            language=language,
            audio_threshold=audio_threshold,
            batch_size=batch_size,
            diarize=diarize,
            num_speakers=num_speakers,
            save_diarization_steps=save_diarization_steps,
        )

    def save_result(self, result: Dict, output_path: str):
        """Save results to a file."""
        self.transcriber.save_result(result, output_path)

    def get_formatted_transcript(
        self, result: Dict, format_type: str = "default"
    ) -> str:
        """Get formatted transcript for display."""
        return self.transcriber.get_formatted_transcript(result, format_type)

    def get_plain_transcript(self, result: Dict) -> str:
        """Get plain text transcript without formatting."""
        return self.transcriber.get_plain_transcript(result)

    # For backward compatibility
    def _assign_word_speakers(self, speaker_segments, asr_result):
        """Backward compatibility method."""
        return self.transcriber.speaker_assigner.assign_word_speakers(
            speaker_segments, asr_result
        )

    def _integrate_results(self, word_timestamps, audio_events):
        """Backward compatibility method."""
        return self.transcriber.event_integrator.integrate_results(
            word_timestamps, audio_events
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
