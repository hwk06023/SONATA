import os
import json
import io
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Union, Tuple, Optional
import concurrent.futures
from sonata.core.asr import ASRProcessor
from sonata.core.emotive_detector import EmotiveDetector, EmotiveEvent
from sonata.constants import (
    EMOTIVE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_DEVICE,
    DEFAULT_COMPUTE_TYPE,
    LanguageCode,
)


class IntegratedTranscriber:
    def __init__(
        self,
        asr_model: str = DEFAULT_MODEL,
        emotive_model_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ):
        """Initialize the integrated transcriber.

        Args:
            asr_model: WhisperX model name to use
            emotive_model_path: Path to custom emotive detection model (optional)
            device: Compute device (cpu/cuda)
            compute_type: Compute precision (float32, float16, etc.)
        """
        self.device = device

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
                self.emotive_detector = EmotiveDetector(
                    model_path=emotive_model_path,
                    device=device,
                    threshold=EMOTIVE_THRESHOLD,
                )
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

    def process_audio(
        self,
        audio_path: str,
        language: str = DEFAULT_LANGUAGE,
        emotive_threshold: float = EMOTIVE_THRESHOLD,
        batch_size: int = 16,
    ) -> Dict:
        """Process audio to get transcription with emotive events integrated.

        Args:
            audio_path: Path to the audio file
            language: ISO language code (e.g., "en", "ko")
            emotive_threshold: Detection threshold for emotive events
            batch_size: Batch size for processing

        Returns:
            Dictionary containing the complete transcription results
        """
        # Set threshold for the detector
        self.emotive_detector.threshold = emotive_threshold

        # Run ASR first
        print("Running speech recognition...", flush=True)
        asr_result = self.asr.process_audio(
            audio_path=audio_path,
            language=language,
            batch_size=batch_size,
            show_progress=True,
        )

        # Then run emotive detection with progress indicators
        print("\nRunning emotive sound detection...", flush=True)
        emotive_events = self.emotive_detector.detect_events(
            audio=audio_path, show_progress=True
        )

        # Get word timestamps after ASR is done
        word_timestamps = self.asr.get_word_timestamps(asr_result)

        # Integrate transcription and emotive events
        integrated_result = self._integrate_results(word_timestamps, emotive_events)

        return {
            "raw_asr": asr_result,
            "emotive_events": [e.to_dict() for e in emotive_events],
            "integrated_transcript": integrated_result,
        }

    def _integrate_results(
        self, word_timestamps: List[Dict], emotive_events: List[EmotiveEvent]
    ) -> Dict:
        """Integrate ASR results with emotive events based on timestamps."""
        # Sort all elements by their timestamps
        sorted_elements = []

        # Add words
        for word in word_timestamps:
            sorted_elements.append(
                {
                    "type": "word",
                    "content": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "score": word.get("score", 0.0),
                }
            )

        # Add emotive events
        for event in emotive_events:
            sorted_elements.append(
                {
                    "type": "emotive",
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
                plain_text += element["content"] + " "
                rich_text.append(
                    {
                        "type": "word",
                        "content": element["content"],
                        "start": element["start"],
                        "end": element["end"],
                        "score": element.get("score", 0.0),
                    }
                )
            else:  # emotive
                plain_text += element["content"] + " "
                rich_text.append(
                    {
                        "type": "emotive",
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
                - concise: Text with integrated emotive tags
                - default: Text with timestamps (default format)
                - extended: Default format with confidence scores

        Returns:
            A formatted transcript string
        """
        rich_text = result["integrated_transcript"]["rich_text"]

        # Concise format: simple text with emotive tags integrated
        if format_type == "concise":
            text_parts = []
            current_sentence = []

            for item in rich_text:
                if item["type"] == "word":
                    # Add space before word if needed
                    word = item["content"]
                    if word not in [".", ",", "!", "?", ":", ";"] and current_sentence:
                        current_sentence.append(" ")
                    current_sentence.append(word)
                else:  # emotive
                    current_sentence.append(f" {item['content']}")

            text = "".join(current_sentence)
            return text

        # Default format: with timestamps
        elif format_type == "default":
            formatted_lines = []
            for item in rich_text:
                start_time = self._format_time(item["start"])
                if item["type"] == "word":
                    formatted_lines.append(f"[{start_time}] {item['content']}")
                else:  # emotive
                    formatted_lines.append(f"[{start_time}] {item['content']}")
            return "\n".join(formatted_lines)

        # Extended format: with confidence scores
        elif format_type == "extended":
            formatted_lines = []
            for item in rich_text:
                start_time = self._format_time(item["start"])
                if item["type"] == "word":
                    formatted_lines.append(f"[{start_time}] {item['content']}")
                else:  # emotive
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
