import os
import json
import io
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Union, Tuple, Optional
import concurrent.futures
from sonata.core.asr import ASRProcessor
from sonata.core.audio_event_detector import AudioEventDetector, AudioEvent
from sonata.core.speaker_diarization import SpeakerDiarizer, SpeakerSegment
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
        self.device = device
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
    ) -> Dict:
        """Process audio to get transcription with audio events integrated.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            audio_threshold: Detection threshold for audio events
            batch_size: Batch size for processing
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers for diarization (optional)

        Returns:
            Dictionary containing the complete transcription results
        """
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
                diarize=False,  # We'll handle diarization separately
            )
        except Exception as e:
            error_msg = f"ASR processing failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            asr_result = {"error": error_msg, "segments": []}

        # Then run audio event detection with progress indicators
        self.logger.info("\nRunning audio event detection...")
        try:
            if self.deep_detect:
                self.logger.info(
                    "Using multi-scale deep detection for better paralinguistic feature detection..."
                )
                # Use custom window and hop sizes if provided
                window_sizes = self.deep_detect_params.get(
                    "window_sizes", [0.2, 1.0, 2.5]
                )
                hop_sizes = self.deep_detect_params.get("hop_sizes", [0.1, 0.5, 1.0])
                parallel = self.deep_detect_params.get("parallel", False)
                show_detailed_progress = self.deep_detect_params.get(
                    "show_progress", False
                )

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
        except Exception as e:
            error_msg = f"Audio event detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            audio_events = []

        # Get word timestamps after ASR is done
        word_timestamps = []
        try:
            word_timestamps = self.asr.get_word_timestamps(asr_result)
        except Exception as e:
            error_msg = f"Failed to get word timestamps: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

        # Handle diarization if requested
        if diarize:
            self.logger.info("Running speaker diarization...")
            try:
                # Use the speaker diarizer
                diarizer = SpeakerDiarizer(device=self.device)
                diarize_segments = diarizer.diarize(
                    audio_path=audio_path, num_speakers=num_speakers, show_progress=True
                )

                # Convert speaker segments to the format expected by assign_word_speakers
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
                result = self._assign_word_speakers(speaker_segments, asr_result)
            except Exception as e:
                error_msg = f"Speaker diarization failed: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                result = asr_result
        else:
            result = asr_result

        # Integrate word timestamps with audio events
        try:
            integrated_result = self._integrate_results(word_timestamps, audio_events)
            result["integrated_transcript"] = integrated_result
            result["audio_events"] = [event.to_dict() for event in audio_events]
        except Exception as e:
            error_msg = f"Failed to integrate results: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            result["integrated_transcript"] = {"plain_text": "", "rich_text": []}
            result["audio_events"] = []

        return result

    def _assign_word_speakers(self, speaker_segments, asr_result):
        """Assign speakers to words in the ASR result based on diarization segments.

        Args:
            speaker_segments: List of speaker segments from diarization
            asr_result: ASR result with segments containing words

        Returns:
            Updated ASR result with speaker information
        """
        result = asr_result.copy()

        # Skip if no speaker segments
        if not speaker_segments or "segments" not in result:
            return result

        # Process each segment
        for segment in result.get("segments", []):
            # Skip segments without words
            if "words" not in segment:
                continue

            # Process each word in the segment
            for word in segment["words"]:
                word_start = word.get("start", 0)
                word_end = word.get("end", 0)

                # Find matching speaker segment
                assigned_speaker = None
                max_overlap = 0

                for spk_segment in speaker_segments:
                    spk_start = spk_segment["start"]
                    spk_end = spk_segment["end"]

                    # Calculate overlap
                    overlap_start = max(word_start, spk_start)
                    overlap_end = min(word_end, spk_end)
                    overlap = max(0, overlap_end - overlap_start)

                    # Check if this speaker has more overlap
                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_speaker = spk_segment["speaker"]

                # Assign speaker to word
                if assigned_speaker:
                    word["speaker"] = assigned_speaker

        return result

    def _integrate_results(
        self, word_timestamps: List[Dict], audio_events: List[AudioEvent]
    ) -> Dict:
        """Integrate word timestamps with audio events."""
        # Create a timeline mapping timestamp -> action
        timeline = []

        # Sort events by start time for easier integration
        audio_events.sort(key=lambda x: x.start_time)

        # Add word timestamps to timeline
        for word in word_timestamps:
            start_time = word.get("start", 0)
            end_time = word.get("end", 0)
            speaker = word.get("speaker", None)

            # Add word start event
            timeline.append(
                {
                    "time": start_time,
                    "type": "word_start",
                    "content": word["word"],
                    "speaker": speaker,
                    "confidence": word.get("confidence", 1.0),
                }
            )

            # Add word end event
            timeline.append(
                {
                    "time": end_time,
                    "type": "word_end",
                    "content": word["word"],
                    "speaker": speaker,
                }
            )

        # Add audio events to timeline
        for event in audio_events:
            # Add audio event start
            timeline.append(
                {
                    "time": event.start_time,
                    "type": "audio_event_start",
                    "content": event.type,
                    "confidence": event.confidence,
                }
            )

            # Add audio event end
            timeline.append(
                {
                    "time": event.end_time,
                    "type": "audio_event_end",
                    "content": event.type,
                }
            )

        # Sort timeline by time
        timeline.sort(key=lambda x: x["time"])

        # Process timeline to create integrated transcript
        rich_text = []
        plain_text = ""
        current_text = ""
        open_audio_events = set()
        active_speaker = None

        for event in timeline:
            event_type = event["type"]

            if event_type == "word_start":
                word = event["content"]

                # Check if speaker changed
                if event.get("speaker") != active_speaker:
                    # If we have accumulated text, add it to rich_text
                    if current_text:
                        rich_text.append(
                            {
                                "content": current_text.strip(),
                                "start": prev_start,
                                "end": prev_end,
                                "speaker": active_speaker,
                                "audio_events": list(open_audio_events),
                            }
                        )

                        # Add speaker label to plain text if needed
                        if active_speaker and event.get("speaker"):
                            plain_text += f" [{active_speaker}] "

                        plain_text += current_text.strip() + " "
                        current_text = ""

                    active_speaker = event.get("speaker")
                    prev_start = event["time"]

                # Add word with proper spacing
                if current_text and not current_text.endswith(" "):
                    current_text += " "
                current_text += word
                prev_end = event.get("time", 0)

            elif event_type == "audio_event_start":
                open_audio_events.add(event["content"])

            elif event_type == "audio_event_end":
                if event["content"] in open_audio_events:
                    open_audio_events.remove(event["content"])

        # Add any remaining text
        if current_text:
            rich_text.append(
                {
                    "content": current_text.strip(),
                    "start": prev_start if "prev_start" in locals() else 0,
                    "end": prev_end if "prev_end" in locals() else 0,
                    "speaker": active_speaker,
                    "audio_events": list(open_audio_events),
                }
            )

            # Add final speaker label if needed
            if active_speaker:
                plain_text += f" [{active_speaker}] "

            plain_text += current_text.strip()

        return {
            "plain_text": plain_text.strip(),
            "rich_text": rich_text,
        }

    def save_result(self, result: Dict, output_path: str):
        """Save the transcription result to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def get_formatted_transcript(
        self, result: Dict, format_type: str = "default"
    ) -> str:
        """Get a formatted transcript based on the requested format.

        Args:
            result: The transcription result
            format_type: The format type ('concise', 'default', or 'extended')
                - concise: Text with speaker markers and audio event tags in one line
                - default: Text grouped by speaker with timestamps
                - extended: Default format with confidence/score values

        Returns:
            A formatted transcript string
        """
        if not result or "integrated_transcript" not in result:
            return "No transcript available."

        # Get word timestamps from the segments
        word_timestamps = []
        for segment in result.get("segments", []):
            if "words" in segment:
                for word in segment.get("words", []):
                    word_data = {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "speaker": word.get("speaker", None),
                        "score": word.get("score", 0.5),  # Use score directly from ASR
                    }
                    word_timestamps.append(word_data)

        # Get audio events
        audio_events = []
        for event in result.get("audio_events", []):
            audio_events.append(
                {
                    "type": event.get("type", "unknown"),
                    "start": event.get("start", 0),
                    "end": event.get("end", 0),
                    "confidence": event.get("confidence", 0.5),
                }
            )

        # Sort all items by start time
        word_timestamps.sort(key=lambda x: x["start"])
        audio_events.sort(key=lambda x: x["start"])

        # Concise format: all text in one line with speaker markers and audio events
        if format_type == "concise":
            formatted_lines = []
            current_speaker = None
            current_line = []
            all_items = []

            # Combine words and audio events and sort by time
            for word in word_timestamps:
                all_items.append(
                    {
                        "type": "word",
                        "content": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "speaker": word["speaker"],
                    }
                )

            for event in audio_events:
                all_items.append(
                    {
                        "type": "audio",
                        "content": event["type"],
                        "start": event["start"],
                        "end": event["end"],
                    }
                )

            all_items.sort(key=lambda x: x["start"])

            # Process all items in time order
            for item in all_items:
                if item["type"] == "word":
                    # If speaker changes, start a new line
                    if item["speaker"] != current_speaker:
                        # Add current line to output if it exists
                        if current_line:
                            formatted_lines.append(" ".join(current_line))
                            current_line = []

                        current_speaker = item["speaker"]
                        if current_speaker:
                            current_line.append(f"[{current_speaker}]")

                    current_line.append(item["content"])
                else:  # audio event
                    current_line.append(f"[{item['content']}]")

            # Add the last line if it exists
            if current_line:
                formatted_lines.append(" ".join(current_line))

            return "\n".join(formatted_lines)

        # Default and Extended formats: Chronologically ordered events with timestamps
        elif format_type in ["default", "extended"]:
            # Merge words and audio events into a single timeline
            all_events = []

            # Add words
            for word in word_timestamps:
                all_events.append(
                    {
                        "type": "word",
                        "start": word["start"],
                        "content": word["word"],
                        "speaker": word["speaker"],
                        "score": word["score"],
                    }
                )

            # Add audio events
            for event in audio_events:
                all_events.append(
                    {
                        "type": "audio",
                        "start": event["start"],
                        "content": event["type"],
                        "confidence": event["confidence"],
                    }
                )

            # Sort all events chronologically
            all_events.sort(key=lambda x: x["start"])

            formatted_lines = []
            current_speaker = None

            # Process all events in chronological order
            for event in all_events:
                start_time = self._format_time(event["start"])

                if event["type"] == "word":
                    # Check for speaker change
                    if event["speaker"] != current_speaker:
                        if formatted_lines:  # Add a blank line between speakers
                            formatted_lines.append("")
                        current_speaker = event["speaker"]
                        if current_speaker:
                            formatted_lines.append(f"[{current_speaker}]")

                    # Add word with timestamp (and score for extended format)
                    if format_type == "extended":
                        score_str = (
                            f"{event['score']:.2f}"
                            if isinstance(event["score"], float)
                            else event["score"]
                        )
                        formatted_lines.append(
                            f"[{start_time}] {event['content']} (score: {score_str})"
                        )
                    else:
                        formatted_lines.append(f"[{start_time}] {event['content']}")

                else:  # audio event
                    # Check for speaker change to AUDIO
                    if current_speaker != "AUDIO":
                        if formatted_lines:  # Add a blank line between speakers
                            formatted_lines.append("")
                        current_speaker = "AUDIO"
                        formatted_lines.append(f"[{current_speaker}]")

                    # Add audio event with timestamp (and confidence for extended format)
                    if format_type == "extended":
                        confidence_str = (
                            f"{event['confidence']:.2f}"
                            if isinstance(event["confidence"], float)
                            else event["confidence"]
                        )
                        formatted_lines.append(
                            f"[{start_time}] [{event['content']}] (confidence: {confidence_str})"
                        )
                    else:
                        formatted_lines.append(f"[{start_time}] [{event['content']}]")

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

    def get_plain_transcript(self, result: Dict) -> str:
        """Get a plain transcript with word-level details and audio events.

        Format:
        - Word: 'text' (SPEAKER) at start_times to end_times
        - Audio event: [event_type] at start_times to end_times

        Args:
            result: The transcription result

        Returns:
            Plain transcript with word-level details
        """
        if not result:
            return "No transcript available."

        # Collect all words with timestamps and speakers
        words = []
        for segment in result.get("segments", []):
            if "words" in segment:
                for word in segment.get("words", []):
                    words.append(
                        {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0),
                            "end": word.get("end", 0),
                            "speaker": word.get("speaker", "UNKNOWN"),
                        }
                    )

        # Collect all audio events
        audio_events = []
        for event in result.get("audio_events", []):
            audio_events.append(
                {
                    "type": event.get("type", "unknown"),
                    "start": event.get("start", 0),
                    "end": event.get("end", 0),
                }
            )

        # Sort everything by start time
        all_items = []

        for word in words:
            all_items.append(
                {
                    "type": "word",
                    "content": word["word"],
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                }
            )

        for event in audio_events:
            all_items.append(
                {
                    "type": "audio_event",
                    "content": event["type"],
                    "start": event["start"],
                    "end": event["end"],
                }
            )

        all_items.sort(key=lambda x: x["start"])

        # Format output
        output_lines = []

        for item in all_items:
            if item["type"] == "word":
                line = f"- Word: '{item['content']}' ({item['speaker']}) at {item['start']:.3f}s to {item['end']:.3f}s"
                output_lines.append(line)
            else:  # audio_event
                line = f"- Audio event: [{item['content']}] at {item['start']:.3f}s to {item['end']:.3f}s"
                output_lines.append(line)

        return "\n".join(output_lines)
