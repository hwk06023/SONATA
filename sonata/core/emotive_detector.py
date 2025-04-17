import os
import numpy as np
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional, Any
import sys
import logging
from pathlib import Path
from sonata.models.model_loader import load_audioset
from scipy.special import softmax
from sonata.constants import EMOTIVE_THRESHOLD, EmotiveEventType, EMOTIVE_CLASS_MAPPING

# Temporary - Set up debug logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from dataclasses import dataclass
import scipy.signal as signal
import tempfile
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


@dataclass
class EmotiveEvent:
    type: str
    start_time: float
    end_time: float
    confidence: float

    def to_dict(self):
        return {
            "type": self.type,
            "start": self.start_time,
            "end": self.end_time,
            "confidence": self.confidence,
        }

    def to_tag(self):
        return f"[{self.type}]"


class EmotiveCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotiveCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)


class AudioProcessor:
    """Utility class for audio processing functions."""

    @staticmethod
    def compute_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCC features from audio signal."""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc

    @staticmethod
    def compute_delta_features(mfcc_features: np.ndarray) -> np.ndarray:
        """Compute delta features from MFCC features."""
        return librosa.feature.delta(mfcc_features)

    @staticmethod
    def lowpass_filter(
        sig: np.ndarray, filter_order: int = 2, cutoff: float = 0.01
    ) -> np.ndarray:
        """Apply a low-pass filter to a signal."""
        B, A = signal.butter(filter_order, cutoff, output="ba")
        return signal.filtfilt(B, A, sig)

    @staticmethod
    def segment_audio(
        audio_path: str, window_size: float = 1.0, hop_size: float = 0.5
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Segment audio into overlapping windows for analysis."""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)

            segments = []
            window_samples = int(window_size * sr)
            hop_samples = int(hop_size * sr)

            for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
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
    def extract_features(audio_path: str) -> Optional[torch.Tensor]:
        """Extract mel spectrogram features from audio for model input."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050)

            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

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
            return features
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            return None

    @staticmethod
    def extract_segment_features(
        segment: np.ndarray, sr: int = 22050
    ) -> Dict[str, float]:
        """Extract comprehensive features from audio segment for classification."""
        try:
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

            # MFCC features - important for many emotive sounds
            mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1)

            # Onset features - useful for detecting abrupt sounds like cough, sneeze
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            onset_density = np.mean(onset_env)

            # Harmonic and percussive components - useful for distinguishing between types
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

            return features
        except Exception as e:
            logging.error(f"Feature extraction for segment failed: {str(e)}")
            return {}


class EmotiveClassifier:
    """Base class for different emotive classifiers."""

    def classify(
        self, features: Dict[str, float], threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify based on features. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement classify method")


class RuleBasedClassifier(EmotiveClassifier):
    """Rule-based classifier for emotive events."""

    def classify(
        self, features: Dict[str, float], threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify segment based on extracted features using rule-based approach."""
        results = []

        # 1. Laugh detection
        laugh_score = 0.0
        if (
            features.get("rms", 0) > 0.08
            and features.get("zcr", 0) > 0.08
            and features.get("amp_envelope_std", 0) > 0.01
        ):
            # High energy, zero-crossing rate and amplitude variation indicate laughter
            laugh_base = 0.6
            laugh_intensity = min(1.0, features.get("rms", 0) * 5)
            laugh_score = laugh_base + (laugh_intensity * 0.4)

        if laugh_score > threshold:
            results.append(("laugh", laugh_score))

        # 2. Sigh detection
        sigh_score = 0.0
        if (
            features.get("rms", 0) < 0.06
            and features.get("centroid", 0) < 1200
            and features.get("mfcc_1", 0) < 0
            and features.get("harmonic_rms", 0) > features.get("percussive_rms", 0)
        ):
            # Sighs have low energy, lower spectral centroid, distinctive MFCC patterns,
            # and more harmonic than percussive content
            sigh_score = 0.7

        if sigh_score > threshold:
            results.append(("sigh", sigh_score))

        # 3. Surprise detection
        surprise_score = 0.0
        if (
            features.get("rms", 0) > 0.1
            and features.get("bandwidth", 0) > 3000
            and features.get("zcr", 0) > 0.1
            and features.get("onset_density", 0) > 0.2
        ):
            # Surprise expressions have high energy, wide bandwidth,
            # rapid changes, and strong onset
            surprise_score = 0.75

        if surprise_score > threshold:
            results.append(("surprise", surprise_score))

        # 4. Inhale detection
        inhale_score = 0.0
        if (
            features.get("zcr", 0) < 0.05
            and features.get("rms", 0) > 0.02
            and features.get("rolloff", 0) > 2000
            and features.get("flatness", 0) > 0.4
        ):
            # Inhales have low zero-crossing rate, modest energy,
            # higher rolloff, and relatively flat spectrum
            inhale_score = 0.68

        if inhale_score > threshold:
            results.append(("inhale", inhale_score))

        # 5. Groan detection
        groan_score = 0.0
        if (
            features.get("centroid", 0) < 1500
            and features.get("rms", 0) > 0.04
            and features.get("mfcc_3", 0) < -2
            and features.get("harmonic_rms", 0)
            > 1.5 * features.get("percussive_rms", 0)
        ):
            # Groans have low spectral centroid, medium energy,
            # distinctive MFCC patterns, and are highly harmonic
            groan_score = 0.72

        if groan_score > threshold:
            results.append(("groan", groan_score))

        # 6. Cough detection
        cough_score = 0.0
        if (
            features.get("percussive_rms", 0) > features.get("harmonic_rms", 0)
            and features.get("onset_density", 0) > 0.3
            and features.get("rms", 0) > 0.12
            and features.get("amp_envelope_std", 0) > 0.03
        ):
            # Coughs have strong percussive content, high onset density,
            # high energy, and strong amplitude variation
            cough_score = 0.78

        if cough_score > threshold:
            results.append(("cough", cough_score))

        # 7. Sneeze detection
        sneeze_score = 0.0
        if (
            features.get("rms", 0) > 0.15
            and features.get("zcr", 0) > 0.12
            and features.get("onset_density", 0) > 0.4
            and features.get("bandwidth", 0) > 4000
        ):
            # Sneezes have very high energy, high zero-crossing rate,
            # high onset density, and very wide bandwidth
            sneeze_score = 0.82

        if sneeze_score > threshold:
            results.append(("sneeze", sneeze_score))

        # 8. Sniffle detection
        sniffle_score = 0.0
        if (
            features.get("rms", 0) < 0.04
            and features.get("centroid", 0) > 3000
            and features.get("flatness", 0) < 0.1
            and features.get("zcr", 0) > 0.15
        ):
            # Sniffles have low energy, high spectral centroid,
            # low spectral flatness, and high zero-crossing rate
            sniffle_score = 0.68

        if sniffle_score > threshold:
            results.append(("sniffle", sniffle_score))

        return results


class ModelBasedClassifier(EmotiveClassifier):
    """Model-based classifier using a trained neural network."""

    def __init__(self, model: Any, emotive_types: List[str], device: torch.device):
        self.model = model
        self.emotive_types = emotive_types
        self.device = device

    def classify(
        self, features: torch.Tensor, threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify using the trained model."""
        results = []

        if features is None:
            return results

        features = features.to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            probabilities = outputs.cpu().numpy()[0]

        # Create results for each emotive type that exceeds the threshold
        for i, prob in enumerate(probabilities):
            if prob > threshold and i < len(self.emotive_types):
                results.append((self.emotive_types[i], float(prob)))

        return results


# AudioSet-based AST model for emotional sound detection
class AudiosetEmotiveDetector:
    """Class for detecting emotional sounds using AudioSet-based AST model"""

    def __init__(self, model_dir: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = load_audioset(model_dir, device)
        self.valid_classes = list(EMOTIVE_CLASS_MAPPING.keys())
        logging.info(
            f"Added valid emotion classes: {', '.join([str(c) for c in self.valid_classes])}"
        )

    def detect_events(
        self, audio: Union[str, torch.Tensor, np.ndarray], sr: int = 16000
    ) -> List[dict]:
        detections = []

        # Handle string path to audio file
        if isinstance(audio, str):
            # Load audio file
            try:
                logging.debug(f"Loading audio from file: {audio}")
                y, sr = librosa.load(audio, sr=sr)
                audio = y
            except Exception as e:
                logging.error(f"Failed to load audio file: {str(e)}")
                return []

        # Process the audio array
        probs = self.detect_from_array(audio, sr)

        if len(probs.shape) > 1:
            for i in range(probs.shape[0]):
                for cls_idx in self.valid_classes:
                    prob = probs[i, cls_idx]
                    if prob > 0.1:
                        emotive_type = EMOTIVE_CLASS_MAPPING[cls_idx]
                        detections.append({"type": emotive_type, "score": float(prob)})
        else:
            for cls_idx in self.valid_classes:
                prob = probs[cls_idx]
                if prob > 0.1:
                    emotive_type = EMOTIVE_CLASS_MAPPING[cls_idx]
                    detections.append({"type": emotive_type, "score": float(prob)})

        return detections

    def detect_from_array(
        self, audio: Union[torch.Tensor, np.ndarray], sr: int = 16000
    ) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        if audio.shape[1] != 1:
            audio = audio.unsqueeze(1)

        audio = audio.to(self.device)

        with torch.no_grad():
            logits = self.model(audio, sr)

        logits_np = logits.cpu().numpy()
        probs = softmax(logits_np, axis=-1)

        logging.debug(f"Model output logits shape: {logits.shape}")
        return probs


class EmotiveDetector(AudiosetEmotiveDetector):
    """
    Detector for emotive events in audio using AudioSet-based model.

    This class extends AudiosetEmotiveDetector with additional functionality
    specific to detecting emotive events like laughs, sighs, etc.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = EMOTIVE_THRESHOLD,
        device: str = None,
        emotive_types: Optional[List[str]] = None,
    ):
        """
        Initialize the emotive detector.

        Args:
            model_path: Path to a custom model (optional)
            threshold: Detection confidence threshold
            device: Compute device (cpu/cuda)
            emotive_types: List of emotive event types to detect (optional)
        """
        super().__init__(model_dir=model_path, device=device)
        self.threshold = threshold

        # Use the EmotiveEventType enum values for emotive types if not provided
        if emotive_types is None:
            self.emotive_types = [event_type.value for event_type in EmotiveEventType]
        else:
            self.emotive_types = emotive_types
