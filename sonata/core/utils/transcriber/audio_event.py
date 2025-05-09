import logging
from typing import Dict, List, Optional, Any, Tuple
from sonata.core.utils.audio_event import AudioEvent


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
        # Create result structure
        result = {
            "plain_text": "",
            "rich_text": [],
        }

        # Return early if no words
        if not word_timestamps:
            return result

        # Sort word timestamps by start time
        sorted_words = sorted(word_timestamps, key=lambda x: x.get("start", 0))

        # Sort audio events by start time
        sorted_events = sorted(audio_events, key=lambda x: x.start_time)

        # Interleave words and audio events based on timing
        current_text = ""
        rich_text = []

        word_idx = 0
        event_idx = 0

        while word_idx < len(sorted_words) or event_idx < len(sorted_events):
            # Process next word
            if word_idx < len(sorted_words):
                word = sorted_words[word_idx]

                # Check if an event comes before this word
                if event_idx < len(sorted_events) and sorted_events[
                    event_idx
                ].start_time <= word.get("start", float("inf")):
                    # Add accumulated text as a segment
                    if current_text:
                        rich_text.append(
                            {
                                "type": "text",
                                "content": current_text,
                            }
                        )
                        current_text = ""

                    # Add the event tag
                    event = sorted_events[event_idx]
                    rich_text.append(
                        {
                            "type": "tag",
                            "content": f"[{event.type}]",
                            "event": event.to_dict(),
                        }
                    )

                    event_idx += 1
                else:
                    # Add the word
                    if "text" in word:
                        current_text += word["text"]
                        if word.get("is_last_word_in_segment", False):
                            current_text += " "
                    word_idx += 1
            else:
                # Only events left
                if event_idx < len(sorted_events):
                    # Add accumulated text as a segment
                    if current_text:
                        rich_text.append(
                            {
                                "type": "text",
                                "content": current_text,
                            }
                        )
                        current_text = ""

                    # Add the event tag
                    event = sorted_events[event_idx]
                    rich_text.append(
                        {
                            "type": "tag",
                            "content": f"[{event.type}]",
                            "event": event.to_dict(),
                        }
                    )

                    event_idx += 1

        # Add any remaining text
        if current_text:
            rich_text.append(
                {
                    "type": "text",
                    "content": current_text,
                }
            )

        # Generate plain text from rich text
        plain_text = ""
        for item in rich_text:
            plain_text += item["content"]

        # Final result
        result["plain_text"] = plain_text
        result["rich_text"] = rich_text

        return result
