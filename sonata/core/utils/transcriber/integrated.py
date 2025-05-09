import os
import logging
import traceback
from typing import Dict, List, Optional
from sonata.core.asr import ASRProcessor
from sonata.core.audio_event_detector import AudioEventDetector
from sonata.core.speaker_diarization import SpeakerDiarizer
from .speaker import SpeakerAssigner
from .audio_event import AudioEventIntegrator
from .formatter import TranscriptFormatter
from sonata.constants import (
    AUDIO_EVENT_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_DEVICE,
    DEFAULT_COMPUTE_TYPE,
)


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
        self.device = device
        self.deep_detect = deep_detect
        self.deep_detect_params = deep_detect_params or {
            "window_sizes": [0.2, 1.0, 2.5],
            "hop_sizes": [0.1, 0.5, 1.0],
        }
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.asr = ASRProcessor(
            model_name=asr_model, device=device, compute_type=compute_type
        )

        self.audio_detector = AudioEventDetector(
            model_path=audio_model_path,
            device=device,
            threshold=AUDIO_EVENT_THRESHOLD,
            custom_thresholds=custom_audio_thresholds,
        )

        # Helper components
        self.speaker_assigner = SpeakerAssigner()
        self.event_integrator = AudioEventIntegrator()
        self.formatter = TranscriptFormatter()

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
        # Validate input parameters
        if not os.path.exists(audio_path):
            err_msg = f"Audio file not found: {audio_path}"
            self.logger.error(err_msg)
            return {
                "error": err_msg,
                "integrated_transcript": {"plain_text": "", "rich_text": []},
            }

        if not isinstance(batch_size, int) or batch_size <= 0:
            self.logger.warning(f"Invalid batch_size: {batch_size}. Using default (16)")
            batch_size = 16

        # Set threshold for the detector
        self.audio_detector.threshold = audio_threshold

        # Run ASR first
        self.logger.info("Running speech recognition...")
        try:
            asr_result = self.asr.process_audio(
                audio_path=audio_path,
                language=language,
                batch_size=batch_size,
                show_progress=True,
                diarize=False,
            )
        except Exception as e:
            error_msg = f"ASR processing failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            asr_result = {"error": error_msg, "segments": []}

        # Get word timestamps
        word_timestamps = []
        try:
            word_timestamps = self.asr.get_word_timestamps(asr_result)
        except Exception as e:
            error_msg = f"Failed to get word timestamps: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

        # Run audio event detection
        self.logger.info("\nRunning audio event detection...")
        try:
            if self.deep_detect:
                self.logger.info(
                    "Using multi-scale deep detection for better paralinguistic feature detection..."
                )
                window_sizes = self.deep_detect_params.get(
                    "window_sizes", [0.2, 1.0, 2.5]
                )
                hop_sizes = self.deep_detect_params.get("hop_sizes", [0.1, 0.5, 1.0])
                parallel = self.deep_detect_params.get("parallel", False)
                show_detailed_progress = self.deep_detect_params.get(
                    "show_progress", False
                )

                audio_events = self.audio_detector.detect_events_multi_scale(
                    audio=audio_path,
                    window_sizes=window_sizes,
                    hop_sizes=hop_sizes,
                    parallel=parallel,
                    show_progress=show_detailed_progress,
                )
            else:
                audio_events = self.audio_detector.detect_events(
                    audio=audio_path,
                    show_progress=True,
                )
        except Exception as e:
            error_msg = f"Audio event detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            audio_events = []

        # Handle diarization if requested
        result = asr_result
        if diarize:
            self.logger.info("Running speaker diarization...")
            try:
                diarizer = SpeakerDiarizer(device=self.device)

                diarize_segments = diarizer.diarize(
                    audio_path=audio_path,
                    num_speakers=num_speakers,
                    show_progress=True,
                    save_steps=save_diarization_steps,
                )

                # Convert segments to expected format
                speaker_segments = []
                for segment in diarize_segments:
                    speaker_segments.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "speaker": segment.speaker,
                            "score": segment.score,
                        }
                    )

                # Assign speakers to words
                result = self.speaker_assigner.assign_word_speakers(
                    speaker_segments, asr_result
                )
            except Exception as e:
                error_msg = f"Speaker diarization failed: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())

        # Integrate word timestamps with audio events
        try:
            integrated_result = self.event_integrator.integrate_results(
                word_timestamps, audio_events
            )
            result["integrated_transcript"] = integrated_result
            result["audio_events"] = [event.to_dict() for event in audio_events]
        except Exception as e:
            error_msg = f"Failed to integrate results: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            result["integrated_transcript"] = {"plain_text": "", "rich_text": []}
            result["audio_events"] = []

        return result

    def save_result(self, result: Dict, output_path: str):
        """Save results to a file."""
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Results saved to {output_path}")

    def get_formatted_transcript(
        self, result: Dict, format_type: str = "default"
    ) -> str:
        """Get formatted transcript for display."""
        return self.formatter.format_transcript(result, format_type)

    def get_plain_transcript(self, result: Dict) -> str:
        """Get plain text transcript without formatting."""
        return self.formatter.get_plain_text(result)
