import os
import json
import io
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Union, Tuple, Optional
import concurrent.futures
from sonata.core.asr import ASRProcessor
from sonata.core.audio_event_detector import AudioEventDetector, AudioEvent
from sonata.constants import (
    AUDIO_EVENT_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_DEVICE,
    DEFAULT_COMPUTE_TYPE,
    LanguageCode,
)
import whisperx


class IntegratedTranscriber:
    def __init__(
        self,
        asr_model: str = DEFAULT_MODEL,
        audio_model_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        offline_diarization: bool = False,
        offline_config_path: Optional[str] = None,
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
            offline_diarization: Whether to use offline diarization mode
            offline_config_path: Path to offline diarization config file
            custom_audio_thresholds: Dictionary of custom thresholds for specific audio event types (optional)
            deep_detect: Whether to use multi-scale audio event detection
            deep_detect_params: Dictionary with window_sizes and hop_sizes for deep detection
        """
        self.device = device
        self.offline_diarization = offline_diarization
        self.offline_config_path = offline_config_path
        self.deep_detect = deep_detect
        self.deep_detect_params = deep_detect_params or {
            "window_sizes": [0.2, 1.0, 2.5],
            "hop_sizes": [0.1, 0.5, 1.0],
        }
        self.logger = logging.getLogger(__name__)

        # Set up comprehensive warning suppression
        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Temporarily suppress all logging
            logging.getLogger().setLevel(logging.ERROR)

            # Redirect both stdout and stderr during initialization
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                self.asr = ASRProcessor(
                    model_name=asr_model, device=device, compute_type=compute_type
                )
                self.audio_detector = AudioEventDetector(
                    model_path=audio_model_path,
                    device=device,
                    threshold=AUDIO_EVENT_THRESHOLD,
                    custom_thresholds=custom_audio_thresholds,
                )
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

    def process_audio(
        self,
        audio_path: str,
        language: str = DEFAULT_LANGUAGE,
        audio_threshold: float = AUDIO_EVENT_THRESHOLD,
        batch_size: int = 16,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hf_token: Optional[str] = None,
    ) -> Dict:
        """Process audio to get transcription with audio events integrated.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            audio_threshold: Detection threshold for audio events
            batch_size: Batch size for processing
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers for diarization
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization
            hf_token: HuggingFace token for diarization model (may not be required if using offline mode)

        Returns:
            Dictionary containing the complete transcription results
        """
        # Set threshold for the detector
        self.audio_detector.threshold = audio_threshold

        # Run ASR first
        self.logger.info("Running speech recognition...")
        asr_result = self.asr.process_audio(
            audio_path=audio_path,
            language=language,
            batch_size=batch_size,
            show_progress=True,
            diarize=False,  # We'll handle diarization separately
        )

        # Then run audio event detection with progress indicators
        self.logger.info("\nRunning audio event detection...")

        if self.deep_detect:
            self.logger.info(
                "Using multi-scale deep detection for better paralinguistic feature detection..."
            )
            # Use custom window and hop sizes if provided
            window_sizes = self.deep_detect_params.get("window_sizes", [0.2, 1.0, 2.5])
            hop_sizes = self.deep_detect_params.get("hop_sizes", [0.1, 0.5, 1.0])
            parallel = self.deep_detect_params.get("parallel", False)
            show_detailed_progress = self.deep_detect_params.get("show_progress", False)

            print(f"Using {len(window_sizes)} window sizes: {window_sizes}")
            if parallel:
                print(f"Using parallel processing mode (ThreadPool)")
            if show_detailed_progress:
                print(f"Using detailed progress bars for scale monitoring")

            audio_events = self.audio_detector.detect_events_multi_scale(
                audio=audio_path,
                window_sizes=window_sizes,
                hop_sizes=hop_sizes,
                parallel=parallel,
                show_progress=show_detailed_progress,
            )
        else:
            # Use standard detection with single window size
            audio_events = self.audio_detector.detect_events(
                audio=audio_path,
                show_progress=True,
            )

        # Get word timestamps after ASR is done
        word_timestamps = self.asr.get_word_timestamps(asr_result)

        # Handle diarization if requested
        if diarize:
            self.logger.info("\nRunning speaker diarization...")

            # Load diarization model if needed
            if self.asr.diarize_model is None:
                self.asr.load_diarize_model(
                    hf_token=hf_token,
                    show_progress=True,
                    offline_mode=self.offline_diarization,
                    offline_config_path=self.offline_config_path,
                )

            if self.asr.diarize_model is not None:
                try:
                    # Run diarization
                    diarize_segments = self.asr.diarize_audio(
                        audio_path=audio_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        show_progress=True,
                    )

                    # Use whisperX's native assign_word_speakers function
                    try:
                        # Log the structure of a few segments to debug issues
                        if diarize_segments and len(diarize_segments) > 0:
                            self.logger.debug(
                                f"Speaker segment sample: {diarize_segments[0]}"
                            )
                            self.logger.debug(
                                f"Total speaker segments: {len(diarize_segments)}"
                            )
                            self.logger.debug(
                                f"Speaker labels: {set(s.get('speaker', 'unknown') for s in diarize_segments)}"
                            )

                        # Debug ASR structure
                        self.logger.debug(f"ASR result keys: {asr_result.keys()}")
                        if "segments" in asr_result:
                            self.logger.debug(
                                f"ASR segments count: {len(asr_result['segments'])}"
                            )
                            if len(asr_result["segments"]) > 0:
                                self.logger.debug(
                                    f"First ASR segment keys: {asr_result['segments'][0].keys()}"
                                )
                                if "words" in asr_result["segments"][0]:
                                    self.logger.debug(
                                        f"First word sample: {asr_result['segments'][0]['words'][0] if asr_result['segments'][0]['words'] else 'No words'}"
                                    )

                        # Check if any segments contain numeric speaker IDs (problematic)
                        has_numeric_speakers = False
                        for seg in diarize_segments:
                            if "speaker" in seg and isinstance(
                                seg["speaker"], (int, float)
                            ):
                                has_numeric_speakers = True
                                self.logger.debug(
                                    f"Found numeric speaker ID: {seg['speaker']}"
                                )

                        if has_numeric_speakers:
                            self.logger.debug(
                                "Converting numeric speaker IDs to strings..."
                            )
                            for seg in diarize_segments:
                                if "speaker" in seg and isinstance(
                                    seg["speaker"], (int, float)
                                ):
                                    seg[
                                        "speaker"
                                    ] = f"SPEAKER_{str(seg['speaker']).zfill(2)}"

                        # Ensure all segments have string 'speaker' keys
                        for i, seg in enumerate(diarize_segments):
                            if "speaker" not in seg:
                                self.logger.debug(
                                    f"Adding missing speaker label to segment {i}"
                                )
                                seg["speaker"] = f"SPEAKER_UNKNOWN"

                        # Now execute the actual assign_word_speakers
                        asr_result = self._custom_assign_word_speakers(
                            diarize_segments, asr_result
                        )

                        # Update word_timestamps with speaker information from the updated ASR result
                        word_timestamps = self.asr.get_word_timestamps(asr_result)
                    except Exception as e:
                        self.logger.error(
                            f"Error assigning speakers to words: {str(e)}"
                        )
                        self.logger.error(
                            f"Diarize segments type: {type(diarize_segments)}"
                        )
                        if isinstance(diarize_segments, list) and diarize_segments:
                            self.logger.error(f"First segment: {diarize_segments[0]}")
                        raise e  # Re-raise to be caught by the outer try/except

                    self.logger.info(
                        f"Speaker diarization complete with {len(set(s['speaker'] for s in diarize_segments))} speakers detected",
                        flush=True,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Warning: Speaker diarization failed. Error: {str(e)}"
                    )
            else:
                self.logger.warning(
                    f"Warning: Speaker diarization was requested but the model couldn't be loaded."
                )

        # Post-process audio events to filter out those below threshold
        filtered_audio_events = []
        for event in audio_events:
            # Get the appropriate threshold for this event type
            event_threshold = audio_threshold
            if event.type in self.audio_detector.class_thresholds:
                event_threshold = self.audio_detector.class_thresholds[event.type]

            # Only keep events that meet or exceed their threshold
            if event.confidence >= event_threshold:
                filtered_audio_events.append(event)

        self.logger.info(
            f"After threshold filtering: {len(filtered_audio_events)}/{len(audio_events)} audio events kept"
        )

        # Integrate transcription and audio events
        integrated_result = self._integrate_results(
            word_timestamps, filtered_audio_events
        )

        return {
            "raw_asr": asr_result,
            "audio_events": [e.to_dict() for e in filtered_audio_events],
            "integrated_transcript": integrated_result,
        }

    def _integrate_results(
        self, word_timestamps: List[Dict], audio_events: List[AudioEvent]
    ) -> Dict:
        """Integrate ASR results with audio events based on timestamps."""
        # Sort all elements by their timestamps
        sorted_elements = []

        # Add words
        for word in word_timestamps:
            element = {
                "type": "word",
                "content": word["word"],
                "start": word["start"],
                "end": word["end"],
                "score": word.get("score", 0.0),
            }
            # Add speaker information if available
            if "speaker" in word:
                element["speaker"] = word["speaker"]
            sorted_elements.append(element)

        # Add audio events
        for event in audio_events:
            sorted_elements.append(
                {
                    "type": "audio_event",
                    "content": event.to_tag(),
                    "event_type": event.type,
                    "start": event.start_time,
                    "end": event.end_time,
                    "confidence": event.confidence,
                }
            )

        # Sort by start time
        sorted_elements.sort(key=lambda x: x["start"])

        # Create integrated transcript
        plain_text = ""
        rich_text = []

        for element in sorted_elements:
            if element["type"] == "word":
                word_text = element["content"] + " "
                plain_text += word_text

                word_element = {
                    "type": "word",
                    "content": element["content"],
                    "start": element["start"],
                    "end": element["end"],
                    "score": element.get("score", 0.0),
                }
                # Add speaker information if available
                if "speaker" in element:
                    word_element["speaker"] = element["speaker"]
                rich_text.append(word_element)
            else:  # audio_event
                plain_text += element["content"] + " "
                rich_text.append(
                    {
                        "type": "audio_event",
                        "content": element["content"],
                        "event_type": element["event_type"],
                        "start": element["start"],
                        "end": element["end"],
                        "confidence": element.get("confidence", 0.0),
                    }
                )

        return {"plain_text": plain_text.strip(), "rich_text": rich_text}

    def save_result(self, result: Dict, output_path: str):
        """Save the transcription result to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def get_formatted_transcript(
        self, result: Dict, format_type: str = "default"
    ) -> str:
        """Get a formatted transcript based on the requested format.

        Args:
            result: The transcription result
            format_type: The format type ('concise', 'default', or 'extended')
                - concise: Text with integrated audio event tags
                - default: Text with timestamps (default format)
                - extended: Default format with confidence scores

        Returns:
            A formatted transcript string
        """
        rich_text = result["integrated_transcript"]["rich_text"]

        # Concise format: simple text with audio event tags integrated
        if format_type == "concise":
            text_parts = []
            current_sentence = []
            current_speaker = None

            for item in rich_text:
                # Handle speaker changes
                if item["type"] == "word" and "speaker" in item:
                    if current_speaker != item["speaker"]:
                        current_speaker = item["speaker"]
                        if current_sentence:  # If we have text, add a line break
                            text_parts.append("".join(current_sentence))
                            current_sentence = []
                        current_sentence.append(f"[{current_speaker}]: ")

                # Add content
                if item["type"] == "word":
                    # Add space before word if needed
                    word = item["content"]
                    if word not in [".", ",", "!", "?", ":", ";"] and current_sentence:
                        current_sentence.append(" ")
                    current_sentence.append(word)
                else:  # audio_event
                    current_sentence.append(f" {item['content']}")

            # Add final sentence
            if current_sentence:
                text_parts.append("".join(current_sentence))

            return "\n".join(text_parts)

        # Default format: with timestamps
        elif format_type == "default":
            formatted_lines = []
            current_speaker = None

            for item in rich_text:
                start_time = self._format_time(item["start"])

                # Check for speaker change
                if item["type"] == "word" and "speaker" in item:
                    if current_speaker != item["speaker"]:
                        current_speaker = item["speaker"]
                        formatted_lines.append(f"\n[{start_time}] [{current_speaker}]")

                if item["type"] == "word":
                    formatted_lines.append(f"[{start_time}] {item['content']}")
                else:  # audio_event
                    formatted_lines.append(f"[{start_time}] {item['content']}")

            return "\n".join(formatted_lines)

        # Extended format: with confidence scores
        elif format_type == "extended":
            formatted_lines = []
            current_speaker = None

            for item in rich_text:
                start_time = self._format_time(item["start"])

                # Check for speaker change
                if item["type"] == "word" and "speaker" in item:
                    if current_speaker != item["speaker"]:
                        current_speaker = item["speaker"]
                        formatted_lines.append(f"\n[{start_time}] [{current_speaker}]")

                if item["type"] == "word":
                    score = item.get("score", 0.0)
                    formatted_lines.append(
                        f"[{start_time}] {item['content']} (confidence: {score:.2f})"
                    )
                else:  # audio_event
                    confidence = item.get("confidence", 0.0)
                    formatted_lines.append(
                        f"[{start_time}] {item['content']} (confidence: {confidence:.2f})"
                    )

            return "\n".join(formatted_lines)

        # Default to standard format if invalid format type
        else:
            return self.get_formatted_transcript(result, format_type="default")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to MM:SS.sss format."""
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:06.3f}"

    def _custom_assign_word_speakers(self, diarize_segments, result):
        """Custom implementation of whisperX's assign_word_speakers function to avoid index errors.

        This implementation ensures all speaker labels are treated correctly.
        """
        if len(diarize_segments) == 0:
            self.logger.warning("Warning: No diarization segments provided.")
            return result

        # Create mapping of speaker segments for quick lookup
        # Each segment is [start_time, end_time, speaker_id]
        speaker_segments = []
        for segment in diarize_segments:
            if not all(k in segment for k in ["start", "end", "speaker"]):
                self.logger.warning(f"Warning: Invalid diarization segment: {segment}")
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
            self.logger.warning("Warning: Result does not have 'segments' key")
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
