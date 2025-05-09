import logging
import whisperx
import warnings
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from typing import Dict, List, Optional


class TextAligner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def align_transcription(
        self,
        segments: List[Dict],
        align_model,
        align_metadata,
        audio_path: str,
        show_progress: bool = True,
    ) -> Dict:
        """Align transcription with WhisperX alignment model

        Args:
            segments: List of transcription segments
            align_model: WhisperX alignment model
            align_metadata: Alignment metadata
            audio_path: Path to audio file
            show_progress: Whether to show progress information

        Returns:
            Aligned transcription result
        """
        if show_progress:
            print(f"[ASR] Aligning transcription...", flush=True)

        # Set up contexts for suppressing output based on show_progress
        stdout_context = nullcontext() if show_progress else redirect_stdout(None)
        stderr_context = nullcontext() if show_progress else redirect_stderr(None)
        warning_context = warnings.catch_warnings()

        try:
            # Filter warnings regardless of show_progress
            warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
            warnings.filterwarnings(
                "ignore", message=".*Trying to infer the `batch_size`.*"
            )

            # Run alignment with appropriate logging
            with stdout_context, stderr_context, warning_context:
                result_aligned = whisperx.align(
                    segments,
                    align_model,
                    align_metadata,
                    audio_path,
                    return_char_alignments=False,
                )

            return result_aligned
        except Exception as e:
            self.logger.error(f"WhisperX alignment error: {str(e)}")
            # Return original segments if alignment fails
            return {"segments": segments}

    def align_diarization_with_transcription(
        self,
        diarization_segments: List[Dict],
        transcription_segments: List[Dict],
    ) -> List[Dict]:
        """Align diarization segments with transcription segments

        Args:
            diarization_segments: List of speaker diarization segments
            transcription_segments: List of transcription segments

        Returns:
            List of aligned segments with speaker and text information
        """
        aligned_segments = []

        # Ensure segments are sorted by start time
        diarization_segments.sort(key=lambda x: x["start"])

        for trans_segment in transcription_segments:
            # Get start and end times for this segment
            if "start" not in trans_segment or "end" not in trans_segment:
                continue

            start_time = trans_segment["start"]
            end_time = trans_segment["end"]

            # Find overlapping diarization segments
            overlapping_speakers = []

            for diar_segment in diarization_segments:
                # Check for overlap between transcription and diarization segments
                overlap_start = max(start_time, diar_segment["start"])
                overlap_end = min(end_time, diar_segment["end"])

                # If segments overlap significantly
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    segment_duration = end_time - start_time

                    # Only include if overlap is significant (>30% of transcription segment)
                    if overlap_duration > segment_duration * 0.3:
                        overlapping_speakers.append(
                            {
                                "speaker": diar_segment["speaker"],
                                "overlap_duration": overlap_duration,
                                "overlap_ratio": overlap_duration / segment_duration,
                            }
                        )

            # Choose the speaker with the most overlap
            speaker = None
            if overlapping_speakers:
                # Sort by overlap duration in descending order
                overlapping_speakers.sort(
                    key=lambda x: x["overlap_duration"], reverse=True
                )
                speaker = overlapping_speakers[0]["speaker"]

            # Create aligned segment with speaker information
            aligned_segment = {
                "start": start_time,
                "end": end_time,
                "text": trans_segment.get("text", ""),
            }

            # Add speaker if available
            if speaker:
                aligned_segment["speaker"] = speaker

            # Add words if available
            if "words" in trans_segment:
                aligned_segment["words"] = trans_segment["words"]

            aligned_segments.append(aligned_segment)

        return aligned_segments
