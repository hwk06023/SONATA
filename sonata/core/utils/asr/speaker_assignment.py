import logging
from typing import Dict, List, Tuple, Optional


class SpeakerAssignmentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assign_speakers_to_words(
        self, diarize_segments: List[Dict], result: Dict
    ) -> Dict:
        """Assign speakers to transcribed words based on diarization segments

        Args:
            diarize_segments: List of diarization segments with speaker info
            result: Transcription result from WhisperX

        Returns:
            Updated transcription with speaker assignments
        """
        if len(diarize_segments) == 0:
            self.logger.debug("Warning: No diarization segments provided.")
            return result

        # Create mapping of speaker segments for quick lookup
        # Each segment is (start_time, end_time, speaker_id)
        speaker_segments = []
        for segment in diarize_segments:
            if not all(k in segment for k in ["start", "end", "speaker"]):
                self.logger.debug(f"Warning: Invalid diarization segment: {segment}")
                continue

            # Ensure speaker is a string
            speaker = segment["speaker"]
            if not isinstance(speaker, str):
                speaker = f"SPEAKER_{str(speaker).zfill(2)}"

            speaker_segments.append((segment["start"], segment["end"], speaker))

        # Sort by start time
        speaker_segments.sort(key=lambda x: x[0])

        # Check for segment overlaps and refine if needed
        refined_segments = self._refine_overlapping_segments(speaker_segments)

        # Check if result has the expected structure
        if "segments" not in result:
            self.logger.debug("Warning: Result does not have 'segments' key")
            return result

        # For each segment in the result, process within a context window
        # to improve speaker assignment consistency
        for segment_idx, segment in enumerate(result["segments"]):
            # Skip segments without words
            if "words" not in segment:
                continue

            # Group words by likely speaker
            word_groups = self._group_words_by_speaker_transition(segment["words"])

            # Process each group of words
            for word_group in word_groups:
                # Find the most likely speaker for this group based on overlap
                best_speaker = self._get_best_speaker_for_word_group(
                    word_group, refined_segments
                )

                # Assign the speaker to all words in this group
                for word in word_group:
                    word_idx = segment["words"].index(word)
                    if best_speaker:
                        result["segments"][segment_idx]["words"][word_idx][
                            "speaker"
                        ] = best_speaker

        # Refine speaker assignments using speech context
        result = self._refine_speaker_assignments_with_context(result)

        return result

    def _refine_overlapping_segments(
        self, speaker_segments: List[Tuple]
    ) -> List[Tuple]:
        """Refine speaker segments to handle overlapping regions

        Args:
            speaker_segments: List of tuples (start, end, speaker)

        Returns:
            Refined list of non-overlapping segments
        """
        if not speaker_segments:
            return []

        # Sort by start time
        sorted_segments = sorted(speaker_segments, key=lambda x: x[0])
        refined = []

        # Iterate through segments and handle overlaps
        for i, (start, end, speaker) in enumerate(sorted_segments):
            # Skip segments with invalid times
            if start >= end:
                continue

            # Check for overlaps with previous segments
            is_overlapped = False
            for j, (prev_start, prev_end, prev_speaker) in enumerate(refined):
                # Check for overlap
                if start < prev_end and end > prev_start:
                    # Calculate overlap region
                    overlap_start = max(start, prev_start)
                    overlap_end = min(end, prev_end)

                    # If significant overlap (>50% of current segment)
                    overlap_ratio = (overlap_end - overlap_start) / (end - start)

                    if overlap_ratio > 0.5:
                        is_overlapped = True
                        # Split the segment around the overlap
                        if start < prev_start:
                            # Add segment before overlap if significant
                            if prev_start - start > 0.1:  # At least 100ms
                                refined.append((start, prev_start, speaker))

                        if end > prev_end:
                            # Add segment after overlap if significant
                            if end - prev_end > 0.1:  # At least 100ms
                                refined.append((prev_end, end, speaker))
                    else:
                        # Small overlap, adjust boundary
                        if start < prev_start:
                            refined.append((start, prev_start, speaker))
                        elif end > prev_end:
                            refined.append((prev_end, end, speaker))

            # If no overlap, add the segment as is
            if not is_overlapped:
                refined.append((start, end, speaker))

        # Sort again after refinement
        return sorted(refined, key=lambda x: x[0])

    def _group_words_by_speaker_transition(self, words: List[Dict]) -> List[List[Dict]]:
        """Group words by likely speaker transitions

        Args:
            words: List of word dictionaries

        Returns:
            List of word groups likely spoken by the same speaker
        """
        if not words:
            return []

        groups = []
        current_group = [words[0]]

        for i in range(1, len(words)):
            current_word = words[i]
            previous_word = words[i - 1]

            # Check if this might be a speaker transition
            time_gap = current_word["start"] - previous_word["end"]

            # Use a threshold to detect potential speaker transitions
            # A pause longer than 0.8 seconds might indicate a speaker change
            if time_gap > 0.8:
                # End current group and start a new one
                groups.append(current_group)
                current_group = [current_word]
            else:
                # Continue current group
                current_group.append(current_word)

        # Add the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _get_best_speaker_for_word_group(
        self, word_group: List[Dict], speaker_segments: List[Tuple]
    ) -> Optional[str]:
        """Find the most likely speaker for a group of words

        Args:
            word_group: List of words in a group
            speaker_segments: List of speaker segments

        Returns:
            Speaker ID or None if no match
        """
        if not word_group or not speaker_segments:
            return None

        # Get time span for this word group
        start_time = min(word["start"] for word in word_group if "start" in word)
        end_time = max(word["end"] for word in word_group if "end" in word)

        # Calculate overlap with each speaker segment
        best_speaker = None
        max_overlap = 0

        for start, end, speaker in speaker_segments:
            # Check for overlap
            overlap_start = max(start, start_time)
            overlap_end = min(end, end_time)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        return best_speaker

    def _refine_speaker_assignments_with_context(self, result: Dict) -> Dict:
        """Refine speaker assignments using context to improve consistency

        This handles issues like isolated words with different speaker assignments
        by using the surrounding context.

        Args:
            result: Transcription result with initial speaker assignments

        Returns:
            Updated result with refined speaker assignments
        """
        # Check if result has the expected structure
        if "segments" not in result:
            return result

        for segment_idx, segment in enumerate(result["segments"]):
            # Skip segments without words
            if "words" not in segment or not segment["words"]:
                continue

            words = segment["words"]

            # First pass: identify isolated words with different speakers
            for i in range(1, len(words) - 1):
                # Skip words that don't have speaker info
                if not all(
                    "speaker" in word for word in [words[i - 1], words[i], words[i + 1]]
                ):
                    continue

                # If current word has different speaker from both neighbors
                if (
                    words[i]["speaker"] != words[i - 1]["speaker"]
                    and words[i]["speaker"] != words[i + 1]["speaker"]
                    and words[i - 1]["speaker"] == words[i + 1]["speaker"]
                ):
                    # Small gap between words (less than 0.5s on either side)
                    if (
                        words[i]["start"] - words[i - 1]["end"] < 0.5
                        and words[i + 1]["start"] - words[i]["end"] < 0.5
                    ):
                        # Correct the isolated word to match its neighbors
                        result["segments"][segment_idx]["words"][i]["speaker"] = words[
                            i - 1
                        ]["speaker"]

        # Second pass: majority vote for small groups of words
        for segment_idx, segment in enumerate(result["segments"]):
            # Skip segments without words
            if "words" not in segment or len(segment["words"]) < 3:
                continue

            words = segment["words"]
            window_size = 5

            for i in range(len(words)):
                # Skip words that don't have speaker info
                if "speaker" not in words[i]:
                    continue

                # Get surrounding words as context window
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(words), i + window_size // 2 + 1)
                window = words[start_idx:end_idx]

                # Count speakers in window
                speaker_counts = {}
                for word in window:
                    if "speaker" in word:
                        speaker = word["speaker"]
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

                # If we have speaker counts, find the majority
                if speaker_counts:
                    majority_speaker = max(speaker_counts.items(), key=lambda x: x[1])[
                        0
                    ]

                    # If current speaker is different from majority and not dominant
                    current_speaker = words[i]["speaker"]
                    current_count = speaker_counts.get(current_speaker, 0)
                    majority_count = speaker_counts.get(majority_speaker, 0)

                    # Use majority if it's significantly more common
                    if majority_count > current_count * 2:
                        result["segments"][segment_idx]["words"][i][
                            "speaker"
                        ] = majority_speaker

        return result
