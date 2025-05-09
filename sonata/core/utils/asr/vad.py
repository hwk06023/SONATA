import os
import torch
import numpy as np
import webrtcvad
import tempfile
import librosa
from typing import List, Dict, Optional
import logging
from scipy import signal
from pydub import AudioSegment


class VoiceActivityDetector:
    def __init__(self):
        self.vad_model = None
        self.logger = logging.getLogger(__name__)

    def enhanced_vad(self, audio_path: str, show_progress: bool = True) -> List[Dict]:
        """Enhanced Voice Activity Detection with multiple methods combined

        Args:
            audio_path: Path to audio file
            show_progress: Whether to show progress information

        Returns:
            List of VAD segments with start and end times
        """
        if show_progress:
            print("[VAD] Performing enhanced voice activity detection...")

        try:
            # Load audio with librosa
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)

            # Normalize audio
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)

            # Use WebRTC VAD for initial segmentation
            webrtc_segments = self._apply_webrtc_vad(
                waveform, sample_rate, show_progress
            )

            # Apply energy-based VAD
            energy_segments = self._apply_energy_vad(waveform, sample_rate)

            # Merge segments from both methods
            combined_segments = self._merge_vad_segments(
                webrtc_segments, energy_segments, show_progress
            )

            # Make sure segments are returned in order of start time
            combined_segments.sort(key=lambda x: x["start"])

            # Fill in any small gaps (less than 0.3 seconds)
            merged_segments = self._merge_close_segments(
                combined_segments, gap_threshold=0.3
            )

            return merged_segments

        except Exception as e:
            self.logger.error(f"Error in enhanced VAD: {str(e)}")
            # Return empty list on error
            return []

    def _apply_webrtc_vad(self, waveform, sample_rate, show_progress=True):
        """Apply WebRTC Voice Activity Detection"""
        if show_progress:
            print("[VAD] Applying WebRTC VAD...")

        # Create a WebRTC VAD instance with aggressiveness 2 (0-3, higher is more aggressive)
        vad = webrtcvad.Vad(2)

        # Resample if needed (WebRTC only supports 8kHz, 16kHz, 32kHz, 48kHz)
        supported_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in supported_rates:
            target_rate = 16000  # Default to 16kHz
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=target_rate
            )
            sample_rate = target_rate

        # Get frame length (30ms) in samples
        frame_length = int(0.03 * sample_rate)

        # Prepare audio for WebRTC VAD (int16, 30ms frames)
        audio_int16 = (waveform * 32767).astype(np.int16)

        segments = []
        is_speech = False
        start_time = 0

        # Process audio in 30ms frames
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i : i + frame_length]

            # Convert to bytes in the right format for WebRTC VAD
            frame_bytes = frame.tobytes()

            # Skip frames that are too short
            if len(frame_bytes) != 2 * frame_length:
                continue

            try:
                # Check if frame contains speech
                frame_is_speech = vad.is_speech(frame_bytes, sample_rate)

                # State transition: non-speech to speech
                if frame_is_speech and not is_speech:
                    is_speech = True
                    start_time = i / sample_rate

                # State transition: speech to non-speech
                elif not frame_is_speech and is_speech:
                    is_speech = False
                    end_time = i / sample_rate

                    # Add segment if it's long enough (at least 100ms)
                    if end_time - start_time >= 0.1:
                        segments.append(
                            {
                                "start": start_time,
                                "end": end_time,
                                "method": "webrtc",
                                "weight": 1.0,
                            }
                        )
            except Exception as e:
                self.logger.debug(f"WebRTC VAD error on frame: {str(e)}")
                continue

        # Handle case where last segment extends to the end
        if is_speech:
            end_time = len(audio_int16) / sample_rate
            if end_time - start_time >= 0.1:
                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "method": "webrtc",
                        "weight": 1.0,
                    }
                )

        return segments

    def _apply_energy_vad(self, waveform, sample_rate):
        """Apply energy-based Voice Activity Detection"""
        # Compute signal energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.01 * sample_rate)  # 10ms hop

        # Compute energy in each frame
        energy = np.array(
            [
                np.sum(waveform[i : i + frame_length] ** 2)
                for i in range(0, len(waveform) - frame_length, hop_length)
            ]
        )

        # Normalize energy
        energy = energy / (np.max(energy) + 1e-10)

        # Compute adaptive threshold based on energy distribution
        sorted_energy = np.sort(energy)
        # Use 20th percentile as noise level estimate
        noise_threshold = sorted_energy[int(len(sorted_energy) * 0.2)]
        # Set threshold above noise level
        threshold = noise_threshold * 2

        # Find segments above threshold
        is_speech = energy > threshold
        segments = []
        in_segment = False
        start_idx = 0

        for i, speech in enumerate(is_speech):
            # State transition: non-speech to speech
            if speech and not in_segment:
                in_segment = True
                start_idx = i

            # State transition: speech to non-speech
            elif not speech and in_segment:
                in_segment = False
                # Convert frame indices to time
                start_time = start_idx * hop_length / sample_rate
                end_time = i * hop_length / sample_rate

                # Add segment if it's long enough (at least 100ms)
                if end_time - start_time >= 0.1:
                    segments.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "method": "energy",
                            "weight": 0.8,  # Lower weight than WebRTC
                        }
                    )

        # Handle case where last segment extends to the end
        if in_segment:
            start_time = start_idx * hop_length / sample_rate
            end_time = len(waveform) / sample_rate
            if end_time - start_time >= 0.1:
                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "method": "energy",
                        "weight": 0.8,
                    }
                )

        return segments

    def _merge_vad_segments(self, webrtc_segments, energy_segments, show_progress=True):
        """Merge segments from different VAD methods"""
        if show_progress:
            print("[VAD] Merging segments from multiple detection methods...")

        # Combine all segments
        all_segments = webrtc_segments + energy_segments

        # Sort by start time
        all_segments.sort(key=lambda x: x["start"])

        if not all_segments:
            return []

        # Merge overlapping segments
        merged_segments = []
        current = all_segments[0].copy()
        voting_segments = [current]

        for segment in all_segments[1:]:
            # Check if this segment overlaps with current merged segment
            if segment["start"] <= current["end"] + 0.3:  # Allow small gaps (300ms)
                # Extend current segment if the incoming one has higher weight
                if segment["end"] > current["end"]:
                    if segment["weight"] >= current["weight"] * 0.8:
                        current["end"] = segment["end"]
                voting_segments.append(segment)
            else:
                # Create a new segment for non-overlapping parts
                # First, finalize the current segment with voting
                if len(voting_segments) > 1:
                    # Multiple methods detected this segment, use weighted voting
                    methods = set(s["method"] for s in voting_segments)

                    # If multiple methods agree, use their consensus boundaries
                    if len(methods) >= 2:
                        # Calculate weighted start time
                        starts = [(s["start"], s["weight"]) for s in voting_segments]
                        total_weight = sum(weight for _, weight in starts)
                        weighted_start = (
                            sum(start * weight for start, weight in starts)
                            / total_weight
                        )

                        # Calculate weighted end time
                        ends = [(s["end"], s["weight"]) for s in voting_segments]
                        total_weight = sum(weight for _, weight in ends)
                        weighted_end = (
                            sum(end * weight for end, weight in ends) / total_weight
                        )

                        merged_segments.append(
                            {"start": weighted_start, "end": weighted_end}
                        )
                    else:
                        # Single method, use as is
                        merged_segments.append(
                            {"start": current["start"], "end": current["end"]}
                        )
                else:
                    # Only one detection method, keep as is
                    merged_segments.append(
                        {"start": current["start"], "end": current["end"]}
                    )

                # Start a new current segment
                current = segment.copy()
                voting_segments = [current]

        # Add the last segment
        if voting_segments:
            if len(voting_segments) > 1:
                # Multiple methods detected this segment, use weighted voting
                methods = set(s["method"] for s in voting_segments)

                # If multiple methods agree, use their consensus boundaries
                if len(methods) >= 2:
                    # Calculate weighted boundaries
                    starts = [(s["start"], s["weight"]) for s in voting_segments]
                    total_weight = sum(weight for _, weight in starts)
                    weighted_start = (
                        sum(start * weight for start, weight in starts) / total_weight
                    )

                    ends = [(s["end"], s["weight"]) for s in voting_segments]
                    total_weight = sum(weight for _, weight in ends)
                    weighted_end = (
                        sum(end * weight for end, weight in ends) / total_weight
                    )

                    merged_segments.append(
                        {"start": weighted_start, "end": weighted_end}
                    )
                else:
                    # Single method, use as is
                    merged_segments.append(
                        {"start": current["start"], "end": current["end"]}
                    )
            else:
                # Only one detection method, keep as is
                merged_segments.append(
                    {"start": current["start"], "end": current["end"]}
                )

        return merged_segments

    def _merge_close_segments(self, segments, gap_threshold=0.3):
        """Merge segments that are close together (separated by small gaps)"""
        if not segments:
            return []

        # Sort by start time (should already be sorted, but ensure it)
        segments.sort(key=lambda x: x["start"])

        merged = []
        current = segments[0].copy()

        for segment in segments[1:]:
            # If gap between segments is small, merge them
            if segment["start"] - current["end"] <= gap_threshold:
                current["end"] = segment["end"]
            else:
                # Gap is too large, start a new segment
                merged.append(current)
                current = segment.copy()

        # Add the last segment
        merged.append(current)

        return merged
