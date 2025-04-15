import os
import json
from typing import Dict, List, Union, Tuple, Optional
from sonata.core.asr import ASRProcessor
from sonata.core.emotive_detector import EmotiveDetector, EmotiveEvent


class IntegratedTranscriber:
    def __init__(
        self,
        asr_model: str = "large-v3",
        emotive_model_path: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        self.device = device
        self.asr = ASRProcessor(
            model_name=asr_model, device=device, compute_type=compute_type
        )
        self.emotive_detector = EmotiveDetector(
            model_path=emotive_model_path, device=device
        )

    def process_audio(
        self, audio_path: str, language: str = "en", emotive_threshold: float = 0.5
    ) -> Dict:
        """Process audio to get transcription with emotive events integrated."""
        # Get transcription with word-level timestamps
        asr_result = self.asr.process_audio(audio_path, language=language)
        word_timestamps = self.asr.get_word_timestamps(asr_result)

        # Detect emotive events
        emotive_events = self.emotive_detector.detect_events(audio_path)

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
                    }
                )

        return {"plain_text": plain_text.strip(), "rich_text": rich_text}

    def save_result(self, result: Dict, output_path: str):
        """Save the transcription result to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def get_formatted_transcript(self, result: Dict) -> str:
        """Get a nicely formatted transcript with timestamps."""
        rich_text = result["integrated_transcript"]["rich_text"]
        formatted_lines = []

        for item in rich_text:
            start_time = self._format_time(item["start"])

            if item["type"] == "word":
                formatted_lines.append(f"[{start_time}] {item['content']}")
            else:  # emotive
                formatted_lines.append(f"[{start_time}] {item['content']}")

        return "\n".join(formatted_lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to MM:SS.sss format."""
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:06.3f}"
