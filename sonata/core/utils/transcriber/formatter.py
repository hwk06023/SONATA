import logging
from typing import Dict, List, Optional, Any


class TranscriptFormatter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def format_transcript(self, result: Dict, format_type: str = "default") -> str:
        """Format transcript based on specified format type.

        Args:
            result: Transcription result dictionary
            format_type: Type of formatting to apply

        Returns:
            Formatted transcript text
        """
        if not result or "segments" not in result:
            return ""

        if format_type == "plain":
            return self.get_plain_text(result)
        elif format_type == "srt":
            return self._format_as_srt(result)
        elif format_type == "words_with_speakers":
            return self._format_words_with_speakers(result)
        elif format_type == "segments_with_speakers":
            return self._format_segments_with_speakers(result)
        elif format_type == "events_only":
            return self._format_events_only(result)
        else:
            # Default format
            return self._format_default(result)

    def get_plain_text(self, result: Dict) -> str:
        """Get plain text from result, without any formatting.

        Args:
            result: Transcription result

        Returns:
            Plain text transcript
        """
        if (
            "integrated_transcript" in result
            and "plain_text" in result["integrated_transcript"]
        ):
            return result["integrated_transcript"]["plain_text"]

        if "segments" not in result:
            return ""

        text = ""
        for segment in result["segments"]:
            if "text" in segment:
                text += segment["text"] + " "

        return text.strip()

    def _format_default(self, result: Dict) -> str:
        """Format transcript with default formatting.

        Args:
            result: Transcription result

        Returns:
            Formatted transcript
        """
        # Use rich text format if available
        if (
            "integrated_transcript" in result
            and "rich_text" in result["integrated_transcript"]
        ):
            rich_text = result["integrated_transcript"]["rich_text"]

            formatted_text = ""
            for item in rich_text:
                if item["type"] == "tag":
                    # Format tags in bold
                    formatted_text += f"\033[1m{item['content']}\033[0m"
                else:
                    formatted_text += item["content"]

            return formatted_text

        # Fall back to plain text
        return self.get_plain_text(result)

    def _format_as_srt(self, result: Dict) -> str:
        """Format transcript as SRT subtitles.

        Args:
            result: Transcription result

        Returns:
            SRT formatted transcript
        """
        if "segments" not in result:
            return ""

        srt_text = ""
        for i, segment in enumerate(result["segments"], 1):
            if "start" in segment and "end" in segment and "text" in segment:
                start_time = self._format_time(segment["start"])
                end_time = self._format_time(segment["end"])

                # Add speaker if available
                if "speaker" in segment:
                    text = f"[{segment['speaker']}]: {segment['text']}"
                else:
                    text = segment["text"]

                srt_text += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

        return srt_text.strip()

    def _format_words_with_speakers(self, result: Dict) -> str:
        """Format transcript with speaker information for each word.

        Args:
            result: Transcription result

        Returns:
            Formatted transcript with word-level speaker info
        """
        if "segments" not in result:
            return ""

        formatted_text = ""
        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    if "text" in word:
                        if "speaker" in word:
                            formatted_text += f"[{word['speaker']}] {word['text']} "
                        else:
                            formatted_text += f"{word['text']} "

        return formatted_text.strip()

    def _format_segments_with_speakers(self, result: Dict) -> str:
        """Format transcript with speaker information for each segment.

        Args:
            result: Transcription result

        Returns:
            Formatted transcript with segment-level speaker info
        """
        if "segments" not in result:
            return ""

        formatted_text = ""
        for segment in result["segments"]:
            if "text" in segment:
                if "speaker" in segment:
                    start_time = self._format_time(segment.get("start", 0))
                    formatted_text += (
                        f"[{start_time}] [{segment['speaker']}]: {segment['text']}\n"
                    )
                else:
                    formatted_text += f"{segment['text']}\n"

        return formatted_text.strip()

    def _format_events_only(self, result: Dict) -> str:
        """Format just the audio events.

        Args:
            result: Transcription result

        Returns:
            Formatted transcript with just audio events
        """
        if "audio_events" not in result or not result["audio_events"]:
            return "No audio events detected."

        formatted_text = "Detected Audio Events:\n"
        for i, event in enumerate(result["audio_events"], 1):
            start_time = self._format_time(event.get("start", 0))
            end_time = self._format_time(event.get("end", 0))
            formatted_text += (
                f"{i}. [{start_time}-{end_time}] {event.get('type', 'Unknown')} "
            )
            formatted_text += f"(Confidence: {event.get('confidence', 0):.2f})\n"

        return formatted_text

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
