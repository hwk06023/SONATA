import logging
from typing import Dict, List, Optional, Any


class SpeakerAssigner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assign_word_speakers(
        self, speaker_segments: List[Dict], asr_result: Dict
    ) -> Dict:
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
        for i, segment in enumerate(result["segments"]):
            if "words" not in segment:
                continue

            # Process each word in the segment
            for j, word in enumerate(segment["words"]):
                if "start" not in word or "end" not in word:
                    continue

                word_start = word["start"]
                word_end = word["end"]

                # Find the speaker segment that contains this word
                speaker_found = False
                for spk_segment in speaker_segments:
                    # Check if word is within this speaker segment
                    if (
                        word_start >= spk_segment["start"]
                        and word_end <= spk_segment["end"]
                    ):
                        # Assign speaker to word
                        result["segments"][i]["words"][j]["speaker"] = spk_segment[
                            "speaker"
                        ]
                        speaker_found = True
                        break
                    # Check for substantial overlap (>50% of word)
                    elif (
                        word_start < spk_segment["end"]
                        and word_end > spk_segment["start"]
                    ):
                        overlap = min(word_end, spk_segment["end"]) - max(
                            word_start, spk_segment["start"]
                        )
                        word_duration = word_end - word_start
                        if word_duration > 0 and overlap / word_duration > 0.5:
                            result["segments"][i]["words"][j]["speaker"] = spk_segment[
                                "speaker"
                            ]
                            speaker_found = True
                            break

                # If no speaker was found, assign a default
                if not speaker_found:
                    result["segments"][i]["words"][j]["speaker"] = "UNKNOWN"

        # Update segment-level speaker info based on majority of words
        self._update_segment_speakers(result)

        return result

    def _update_segment_speakers(self, result: Dict) -> None:
        """Update segment-level speaker information based on words.

        Args:
            result: ASR result with speaker-assigned words
        """
        if "segments" not in result:
            return

        for i, segment in enumerate(result["segments"]):
            if "words" not in segment or not segment["words"]:
                continue

            # Count speakers in this segment
            speaker_counts = {}
            for word in segment["words"]:
                if "speaker" in word:
                    speaker = word["speaker"]
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            # Find most common speaker
            if speaker_counts:
                max_count = 0
                majority_speaker = "UNKNOWN"
                for speaker, count in speaker_counts.items():
                    if count > max_count:
                        max_count = count
                        majority_speaker = speaker

                # Assign to segment
                result["segments"][i]["speaker"] = majority_speaker
