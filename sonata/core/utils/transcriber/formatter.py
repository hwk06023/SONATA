import logging
from typing import Dict, List, Optional, Any


class TranscriptFormatter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def format_transcript(self, result: Dict, format_type: str = "default") -> str:
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
            return self.format_transcript(result, format_type="default")

    def get_plain_text(self, result: Dict) -> str:
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

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to MM:SS.sss format."""
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:06.3f}"
