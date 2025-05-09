import numpy as np
import torch
import librosa
import logging
from typing import List, Dict, Tuple, Optional, Any
import scipy.signal as signal
from tqdm import tqdm


class AudioProcessor:
    """Utility class for audio processing functions."""

    def __init__(self):
        """Initialize the audio processor."""
        self.logger = logging.getLogger(__name__)

    def load_audio(
        self, audio_path: str, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """Load audio file at specified sample rate.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sr = librosa.load(audio_path, sr=target_sr)
            return audio_data, sr
        except Exception as e:
            self.logger.error(f"Failed to load audio file: {str(e)}")
            raise

    @staticmethod
    def compute_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCC features from audio signal.

        Args:
            y: Audio signal
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc

    @staticmethod
    def compute_delta_features(mfcc_features: np.ndarray) -> np.ndarray:
        """Compute delta features from MFCC features.

        Args:
            mfcc_features: MFCC features

        Returns:
            Delta features
        """
        return librosa.feature.delta(mfcc_features)

    @staticmethod
    def lowpass_filter(
        sig: np.ndarray, filter_order: int = 2, cutoff: float = 0.01
    ) -> np.ndarray:
        """Apply a low-pass filter to a signal.

        Args:
            sig: Input signal
            filter_order: Filter order
            cutoff: Cutoff frequency (normalized to Nyquist)

        Returns:
            Filtered signal
        """
        B, A = signal.butter(filter_order, cutoff, output="ba")
        return signal.filtfilt(B, A, sig)

    @staticmethod
    def segment_audio(
        audio_path: str,
        window_size: float = 1.0,
        hop_size: float = 0.5,
        show_progress: bool = True,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Segment audio into overlapping windows for analysis.

        Args:
            audio_path: Path to audio file
            window_size: Size of each window in seconds
            hop_size: Hop size between windows in seconds
            show_progress: Whether to show a progress bar

        Returns:
            List of tuples (start_time, end_time, audio_segment)
        """
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)

            segments = []
            window_samples = int(window_size * sr)
            hop_samples = int(hop_size * sr)

            total_segments = (len(y) - window_samples) // hop_samples + 1

            iterator = range(0, len(y) - window_samples + 1, hop_samples)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    desc="Segmenting audio",
                    unit="segments",
                    total=total_segments,
                )

            for start_sample in iterator:
                start_time = start_sample / sr
                end_time = start_time + window_size
                if end_time > duration:
                    end_time = duration

                segment = y[start_sample : start_sample + window_samples]
                segments.append((start_time, end_time, segment))

                if end_time >= duration:
                    break

            return segments
        except Exception as e:
            logging.error(f"Audio segmentation failed: {str(e)}")
            return []

    @staticmethod
    def extract_features(
        audio_path: str, show_progress: bool = True
    ) -> Optional[torch.Tensor]:
        """Extract mel spectrogram features from audio for model input.

        Args:
            audio_path: Path to audio file
            show_progress: Whether to show progress information

        Returns:
            Tensor containing mel spectrogram features
        """
        try:
            if show_progress:
                print("Loading audio file...")

            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050)

            if show_progress:
                print("Extracting mel spectrogram...")

            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

            if show_progress:
                print("Processing spectrogram...")

            # Convert to decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
                mel_spec_db.max() - mel_spec_db.min()
            )

            # Ensure the feature has a consistent size for the model
            # Assuming the model expects a 128x128 mel spectrogram
            target_length = 128
            if mel_spec_db.shape[1] < target_length:
                # Pad if too short
                padding = np.zeros(
                    (mel_spec_db.shape[0], target_length - mel_spec_db.shape[1])
                )
                mel_spec_db = np.hstack((mel_spec_db, padding))
            elif mel_spec_db.shape[1] > target_length:
                # Trim if too long
                mel_spec_db = mel_spec_db[:, :target_length]

            # Reshape for CNN input (batch_size, channels, height, width)
            mel_spec_db = mel_spec_db.reshape(
                1, 1, mel_spec_db.shape[0], mel_spec_db.shape[1]
            )

            # Convert to tensor
            features = torch.FloatTensor(mel_spec_db)

            if show_progress:
                print("Feature extraction complete.")

            return features
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            return None

    @staticmethod
    def extract_segment_features(
        segment: np.ndarray, sr: int = 22050, show_progress: bool = False
    ) -> Dict[str, float]:
        """Extract comprehensive features from audio segment for classification.

        Args:
            segment: Audio segment
            sr: Sample rate
            show_progress: Whether to show progress information

        Returns:
            Dictionary of extracted features
        """
        try:
            if show_progress:
                print("Extracting segment features...")

            # Time-domain features
            rms = np.sqrt(np.mean(segment**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment))

            # Spectral features
            spec_centroid = np.mean(
                librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            )
            spec_bandwidth = np.mean(
                librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
            )
            spec_contrast = np.mean(
                librosa.feature.spectral_contrast(y=segment, sr=sr), axis=1
            )
            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
            spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))

            # Rhythm features
            tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)

            # MFCC features - important for many audio sounds
            mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1)

            # Onset features - useful for detecting abrupt sounds like cough, sneeze
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            onset_density = np.mean(onset_env)

            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(segment)
            harmonic_rms = np.sqrt(np.mean(y_harmonic**2))
            percussive_rms = np.sqrt(np.mean(y_percussive**2))

            # Energy features and distribution
            energy = np.sum(segment**2) / len(segment)
            energy_entropy = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[
                0
            ].std()

            # Additional temporal dynamics
            # Amplitude envelope
            frames = librosa.util.frame(segment, frame_length=2048, hop_length=512)
            amp_envelope = np.sqrt(np.mean(frames**2, axis=0))
            amp_envelope_std = np.std(amp_envelope)

            # Return features dictionary
            features = {
                "rms": float(rms),
                "zcr": float(zcr),
                "centroid": float(spec_centroid),
                "bandwidth": float(spec_bandwidth),
                "flatness": float(spec_flatness),
                "rolloff": float(spec_rolloff),
                "tempo": float(tempo),
                "energy": float(energy),
                "onset_density": float(onset_density),
                "harmonic_rms": float(harmonic_rms),
                "percussive_rms": float(percussive_rms),
                "energy_entropy": float(energy_entropy),
                "amp_envelope_std": float(amp_envelope_std),
            }

            # Add contrast features
            for i, contrast in enumerate(spec_contrast):
                features[f"contrast_{i}"] = float(contrast)

            # Add MFCC features
            for i, mfcc in enumerate(mfccs):
                features[f"mfcc_{i}"] = float(mfcc)

            if show_progress:
                print("Feature extraction complete.")

            return features
        except Exception as e:
            logging.error(f"Feature extraction for segment failed: {str(e)}")
            return {}
