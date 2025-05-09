import logging
from typing import Dict, List, Optional


class WordTimestampExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_word_timestamps(self, result: Dict) -> List[Dict]:
        """Extract word-level timestamps from WhisperX result

        Args:
            result: WhisperX transcription result

        Returns:
            List of dictionaries with word, start time, end time, and optional fields
        """
        words_with_timestamps = []

        if "segments" not in result:
            self.logger.debug("No segments found in transcription result")
            return words_with_timestamps

        for segment in result["segments"]:
            # Verify segment is a dictionary
            if not isinstance(segment, dict):
                self.logger.debug(
                    f"Warning: Segment is not a dictionary. Got {type(segment)}"
                )
                continue

            # Check for word-level information
            if "words" in segment and isinstance(segment["words"], list):
                for word_data in segment["words"]:
                    # Skip if not a dictionary
                    if not isinstance(word_data, dict):
                        self.logger.debug(
                            f"Warning: Word data is not a dictionary. Got {type(word_data)}"
                        )
                        continue

                    # Check if required keys exist
                    if (
                        "word" not in word_data
                        or "start" not in word_data
                        or "end" not in word_data
                    ):
                        self.logger.debug(
                            f"Warning: Word data does not contain required keys. Skipping word: {word_data}"
                        )
                        continue

                    word_with_time = {
                        "word": word_data["word"],
                        "start": word_data["start"],
                        "end": word_data["end"],
                    }
                    if "score" in word_data:
                        word_with_time["score"] = word_data["score"]
                    if "speaker" in word_data:
                        word_with_time["speaker"] = word_data["speaker"]
                    words_with_timestamps.append(word_with_time)
            elif "text" in segment and "start" in segment and "end" in segment:
                # Fallback if no word-level data (shouldn't happen with alignment)
                words_with_timestamps.append(
                    {
                        "word": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                    }
                )
            else:
                # Segment doesn't have either words or required text with timestamps
                self.logger.debug(
                    f"Warning: Segment missing both 'words' and required text fields: {segment.keys() if isinstance(segment, dict) else 'not a dict'}"
                )

        return words_with_timestamps

    def merge_words_to_sentences(self, words_with_timestamps: List[Dict]) -> List[Dict]:
        """Merge word-level timestamps into sentence-level timestamps

        Args:
            words_with_timestamps: List of words with timing information

        Returns:
            List of sentences with timing and speaker information
        """
        if not words_with_timestamps:
            return []

        sentences = []
        current_sentence = {
            "text": "",
            "start": words_with_timestamps[0]["start"],
            "end": words_with_timestamps[0]["end"],
            "words": [],
        }

        # Keep track of current speaker
        current_speaker = words_with_timestamps[0].get("speaker", None)
        if current_speaker:
            current_sentence["speaker"] = current_speaker

        for word_data in words_with_timestamps:
            word = word_data["word"]
            word_speaker = word_data.get("speaker", None)

            # Check if we need to start a new sentence (different speaker or long pause)
            if (
                word_speaker != current_speaker
                or (word_data["start"] - current_sentence["end"])
                > 1.0  # 1 second pause
            ):
                # Finalize current sentence
                if current_sentence["text"].strip():
                    sentences.append(current_sentence)

                # Start new sentence
                current_sentence = {
                    "text": "",
                    "start": word_data["start"],
                    "end": word_data["end"],
                    "words": [],
                }

                # Update current speaker
                current_speaker = word_speaker
                if current_speaker:
                    current_sentence["speaker"] = current_speaker

            # Add space if needed
            if current_sentence["text"] and not (
                current_sentence["text"].endswith(" ")
                or word.startswith(" ")
                or word in [",", ".", "!", "?", ":", ";"]
            ):
                current_sentence["text"] += " "

            # Add word to current sentence
            current_sentence["text"] += word
            current_sentence["end"] = word_data["end"]
            current_sentence["words"].append(word_data)

        # Add the last sentence if not empty
        if current_sentence["text"].strip():
            sentences.append(current_sentence)

        return sentences
