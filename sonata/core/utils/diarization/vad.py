import torch
import torchaudio
from typing import List, Tuple
from tqdm import tqdm


class VADProcessor:
    def __init__(self, vad_model, vad_get_speech_timestamps, device="cpu"):
        self.vad_model = vad_model
        self.vad_get_speech_timestamps = vad_get_speech_timestamps
        self.device = device

    def get_vad_segments(
        self, waveform, sample_rate, show_progress=True
    ) -> List[Tuple[float, float]]:
        """Get voice activity segments using Silero VAD with enhanced parameters"""
        if show_progress:
            print("Running voice activity detection...")

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # Use more sensitive parameters for better recall
        speech_timestamps = self.vad_get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=200,  # Reduced from 250ms
            min_silence_duration_ms=400,  # Reduced from 500ms
            window_size_samples=512,
            speech_pad_ms=200,  # Increased padding
            threshold=0.3,  # Lower threshold for better recall
        )

        segments = []
        for seg in speech_timestamps:
            start = seg["start"] / sample_rate
            end = seg["end"] / sample_rate
            segments.append((start, end))

        # Merge segments that are very close
        merged_segments = self.merge_close_segments(segments, gap_threshold=0.5)

        if show_progress:
            print(f"Found {len(merged_segments)} speech segments after merging")

        return merged_segments

    def merge_close_segments(
        self, segments, gap_threshold=0.5
    ) -> List[Tuple[float, float]]:
        """Merge segments that are separated by small gaps"""
        if not segments:
            return []

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x[0])

        merged = []
        current_start, current_end = sorted_segments[0]

        for start, end in sorted_segments[1:]:
            # If this segment starts soon after the previous one ends
            if start - current_end <= gap_threshold:
                # Extend the current segment
                current_end = end
            else:
                # Add the current segment to results and start a new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last segment
        merged.append((current_start, current_end))

        return merged
