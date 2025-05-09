import torch
import numpy as np
from typing import List, Tuple, Callable


class VADProcessor:
    def __init__(self, vad_model, get_speech_timestamps_func, device="cpu"):
        self.vad_model = vad_model
        self.get_speech_timestamps = get_speech_timestamps_func
        self.device = device

        # Ensure model is on the correct device
        if self.vad_model is not None and hasattr(self.vad_model, "to"):
            self.vad_model = self.vad_model.to(self.device)

    def get_vad_segments(
        self, waveform, sample_rate, show_progress=False
    ) -> List[Tuple[float, float]]:
        """Get voice activity detection segments

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            show_progress: Whether to show progress

        Returns:
            List of (start, end) tuples in seconds
        """
        if self.vad_model is None or self.get_speech_timestamps is None:
            print("VAD model or function not initialized")
            return []

        try:
            # Make sure waveform is the right type
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform).float()

            # Ensure the waveform is on the correct device
            waveform = waveform.to(self.device)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import torchaudio

                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sample_rate, 16000
                ).squeeze(0)
                sample_rate = 16000

            # Apply VAD
            speech_timestamps = self.get_speech_timestamps(
                waveform, self.vad_model, sampling_rate=sample_rate
            )

            # Convert to seconds
            segments = []
            for segment in speech_timestamps:
                start_time = segment["start"] / sample_rate
                end_time = segment["end"] / sample_rate
                segments.append((start_time, end_time))

            return segments

        except Exception as e:
            print(f"Error in VAD processing: {str(e)}")
            return []

    def detect(self, audio_path, show_progress=False) -> List[Tuple[float, float]]:
        """Load and process audio file to get VAD segments

        Args:
            audio_path: Path to audio file
            show_progress: Whether to show progress

        Returns:
            List of (start, end) tuples in seconds
        """
        try:
            import torchaudio

            if show_progress:
                print("Applying voice activity detection...")

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)

            # Process with VAD
            return self.get_vad_segments(waveform, sample_rate, show_progress)

        except Exception as e:
            print(f"Error loading audio for VAD: {str(e)}")
            return []
