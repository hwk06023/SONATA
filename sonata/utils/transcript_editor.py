import json
import os
from typing import Dict, List, Tuple, Optional


class TranscriptEditor:
    def __init__(self, transcript_path: str):
        self.transcript_path = transcript_path
        self.transcript = self._load_transcript()

    def _load_transcript(self) -> Dict:
        if not os.path.exists(self.transcript_path):
            raise FileNotFoundError(
                f"Transcript file not found: {self.transcript_path}"
            )

        with open(self.transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_transcript(self, output_path: Optional[str] = None) -> str:
        save_path = output_path or self.transcript_path
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.transcript, f, ensure_ascii=False, indent=2)
        return save_path

    def get_speakers(self) -> List[str]:
        speakers = set()

        if (
            "integrated_transcript" in self.transcript
            and "rich_text" in self.transcript["integrated_transcript"]
        ):
            for item in self.transcript["integrated_transcript"]["rich_text"]:
                if item["type"] == "word" and "speaker" in item:
                    speakers.add(item["speaker"])

        if "raw_asr" in self.transcript and "segments" in self.transcript["raw_asr"]:
            for segment in self.transcript["raw_asr"]["segments"]:
                if "speaker" in segment:
                    speakers.add(segment["speaker"])
                if "words" in segment:
                    for word in segment["words"]:
                        if "speaker" in word:
                            speakers.add(word["speaker"])

        return sorted(list(speakers))

    def rename_speaker(self, old_speaker: str, new_speaker: str) -> int:
        count = 0

        # Update in integrated transcript
        if (
            "integrated_transcript" in self.transcript
            and "rich_text" in self.transcript["integrated_transcript"]
        ):
            for item in self.transcript["integrated_transcript"]["rich_text"]:
                if (
                    item["type"] == "word"
                    and "speaker" in item
                    and item["speaker"] == old_speaker
                ):
                    item["speaker"] = new_speaker
                    count += 1

        # Update in raw ASR output
        if "raw_asr" in self.transcript and "segments" in self.transcript["raw_asr"]:
            for segment in self.transcript["raw_asr"]["segments"]:
                if "speaker" in segment and segment["speaker"] == old_speaker:
                    segment["speaker"] = new_speaker
                    count += 1

                if "words" in segment:
                    for word in segment["words"]:
                        if "speaker" in word and word["speaker"] == old_speaker:
                            word["speaker"] = new_speaker
                            count += 1

        return count

    def reassign_speaker_by_time(
        self, start_time: float, end_time: float, new_speaker: str
    ) -> int:
        count = 0

        # Update in integrated transcript
        if (
            "integrated_transcript" in self.transcript
            and "rich_text" in self.transcript["integrated_transcript"]
        ):
            for item in self.transcript["integrated_transcript"]["rich_text"]:
                if (
                    item["type"] == "word"
                    and "start" in item
                    and "end" in item
                    and item["start"] >= start_time
                    and item["end"] <= end_time
                ):
                    if "speaker" not in item or item["speaker"] != new_speaker:
                        item["speaker"] = new_speaker
                        count += 1

        # Update in raw ASR output
        if "raw_asr" in self.transcript and "segments" in self.transcript["raw_asr"]:
            # Process words first to later calculate majority speaker for segments
            for segment in self.transcript["raw_asr"]["segments"]:
                # Check if the segment's words are within the time range
                if "words" in segment:
                    for word in segment["words"]:
                        if (
                            "start" in word
                            and "end" in word
                            and word["start"] >= start_time
                            and word["end"] <= end_time
                        ):
                            if "speaker" not in word or word["speaker"] != new_speaker:
                                word["speaker"] = new_speaker
                                count += 1

            # Now update segments based on majority speaker
            for i, segment in enumerate(self.transcript["raw_asr"]["segments"]):
                if "words" in segment and segment["words"]:
                    # Count speakers in words
                    speaker_counts = {}
                    for word in segment["words"]:
                        if "speaker" in word:
                            speaker = word["speaker"]
                            if speaker not in speaker_counts:
                                speaker_counts[speaker] = 0
                            speaker_counts[speaker] += 1

                    # Assign majority speaker to the segment
                    if speaker_counts:
                        majority_speaker = max(
                            speaker_counts.items(), key=lambda x: x[1]
                        )[0]
                        if (
                            "speaker" not in segment
                            or segment["speaker"] != majority_speaker
                        ):
                            segment["speaker"] = majority_speaker
                            count += 1

        return count

    def reassign_speaker_by_segments(
        self, segment_indices: List[int], new_speaker: str
    ) -> int:
        count = 0

        if "raw_asr" in self.transcript and "segments" in self.transcript["raw_asr"]:
            segments = self.transcript["raw_asr"]["segments"]
            for idx in segment_indices:
                if 0 <= idx < len(segments):
                    segment = segments[idx]

                    # Update the segment speaker
                    if "speaker" not in segment or segment["speaker"] != new_speaker:
                        segment["speaker"] = new_speaker
                        count += 1

                    # Update words in segment
                    if "words" in segment:
                        for word in segment["words"]:
                            if "speaker" not in word or word["speaker"] != new_speaker:
                                word["speaker"] = new_speaker
                                count += 1

                    # Update corresponding words in integrated transcript
                    if (
                        "integrated_transcript" in self.transcript
                        and "rich_text" in self.transcript["integrated_transcript"]
                    ):
                        if "start" in segment and "end" in segment:
                            start_time = segment["start"]
                            end_time = segment["end"]

                            for item in self.transcript["integrated_transcript"][
                                "rich_text"
                            ]:
                                if (
                                    item["type"] == "word"
                                    and "start" in item
                                    and "end" in item
                                    and item["start"] >= start_time
                                    and item["end"] <= end_time
                                ):
                                    if (
                                        "speaker" not in item
                                        or item["speaker"] != new_speaker
                                    ):
                                        item["speaker"] = new_speaker
                                        count += 1

                else:
                    print(f"Warning: Segment index {idx} is out of range")

        return count

    def merge_speakers(self, speakers_to_merge: List[str], new_speaker: str) -> int:
        count = 0
        for speaker in speakers_to_merge:
            count += self.rename_speaker(speaker, new_speaker)
        return count

    def get_transcript_structure(self) -> Dict:
        structure = {}
        structure["has_raw_asr"] = "raw_asr" in self.transcript
        structure["has_integrated"] = "integrated_transcript" in self.transcript

        if structure["has_raw_asr"]:
            structure["segments_count"] = len(
                self.transcript["raw_asr"].get("segments", [])
            )

        if structure["has_integrated"]:
            structure["words_count"] = sum(
                1
                for item in self.transcript["integrated_transcript"].get(
                    "rich_text", []
                )
                if item["type"] == "word"
            )

        structure["speakers"] = self.get_speakers()

        return structure
