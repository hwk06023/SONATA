import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from sonata.core.audio_event_detector import AudioEvent


class AudioEventIntegrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def integrate_results(
        self, word_timestamps: List[Dict], audio_events: List[AudioEvent]
    ) -> Dict:
        """Integrate word timestamps with audio events.

        Args:
            word_timestamps: List of word timestamps
            audio_events: List of audio events

        Returns:
            Dictionary with integrated transcript information
        """
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
