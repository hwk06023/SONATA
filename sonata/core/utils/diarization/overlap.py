import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    score: float = 1.0
    is_overlap: bool = False
    overlap_speakers: List[str] = None


class OverlapDetector:
    def __init__(self, device="cpu"):
        self.device = device

    def detect_overlapped_speech(self, waveform, sample_rate, segments) -> List[int]:
        """
        Detect segments with potential overlapped speech

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            segments: List of (start, end) tuples in seconds

        Returns:
            List of segment indices with potential overlap
        """
        # Basic approach - too complex to fully implement here without specialized models
        # In a real implementation, this would use a neural model trained to detect overlapped speech

        # This is a simplified energy-based heuristic
        overlapped_indices = []
        for i, (start, end) in enumerate(segments):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            # Skip if out of bounds
            if start_sample >= len(waveform) or start_sample >= end_sample:
                continue

            # Ensure end sample is within bounds
            end_sample = min(end_sample, len(waveform))

            segment_waveform = waveform[start_sample:end_sample]

            # Convert tensor to numpy if needed
            if isinstance(segment_waveform, torch.Tensor):
                segment_waveform = segment_waveform.cpu().numpy()

            # Simplified energy-based detection
            # Higher energy or rapid energy changes might indicate overlapped speech
            # (This is just a basic heuristic and would be replaced with a proper model)
            if len(segment_waveform) > 0:
                energy = np.mean(np.abs(segment_waveform))
                energy_variance = np.var(np.abs(segment_waveform))

                # Empirical threshold - would be calibrated in a real system
                if energy > 0.15 and energy_variance > 0.03:
                    overlapped_indices.append(i)

        return overlapped_indices


class SegmentProcessor:
    def __init__(self):
        pass

    def create_speaker_segments(
        self, segment_timings, speaker_labels
    ) -> List[SpeakerSegment]:
        """
        Create speaker segments from timings and labels

        Args:
            segment_timings: List of (start, end) tuples
            speaker_labels: Array of speaker labels matching segments

        Returns:
            List of SpeakerSegment objects
        """
        if len(segment_timings) == 0 or len(speaker_labels) == 0:
            return []

        # Create initial speaker segments
        speaker_segments = []
        for i, ((start, end), label) in enumerate(zip(segment_timings, speaker_labels)):
            segment = SpeakerSegment(
                start=start,
                end=end,
                speaker=f"SPEAKER_{str(label).zfill(2)}",
                score=1.0,
                is_overlap=False,
            )
            speaker_segments.append(segment)

        # Merge adjacent segments from same speaker
        merged_segments = []
        if speaker_segments:
            current = speaker_segments[0]

            for next_segment in speaker_segments[1:]:
                # If same speaker and close in time, merge
                if (
                    current.speaker == next_segment.speaker
                    and next_segment.start - current.end < 0.5
                ):
                    current.end = next_segment.end
                else:
                    merged_segments.append(current)
                    current = next_segment

            # Add the last segment
            merged_segments.append(current)

        return merged_segments
