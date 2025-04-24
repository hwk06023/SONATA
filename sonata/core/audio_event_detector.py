import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import io
import sys
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from sonata.models.model_loader import load_audioset
from scipy.special import softmax
from sonata.constants import (
    AUDIO_EVENT_THRESHOLD,
    AudioEventType,
    AUDIOSET_CLASS_MAPPING,
    AUDIO_EVENT_THRESHOLDS,
)
from tqdm import tqdm

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
class AudioEvent:
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


class AudioCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioCNN, self).__init__()

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
        audio_path: str,
        window_size: float = 1.0,
        hop_size: float = 0.5,
        show_progress: bool = True,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Segment audio into overlapping windows for analysis."""
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
        """Extract mel spectrogram features from audio for model input."""
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
        """Extract comprehensive features from audio segment for classification."""
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

            if show_progress:
                print("Feature extraction complete.")

            return features
        except Exception as e:
            logging.error(f"Feature extraction for segment failed: {str(e)}")
            return {}


class AudioClassifier:
    """Base class for different audio classifiers."""

    def classify(
        self, features: Dict[str, float], threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify based on features. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement classify method")


class RuleBasedClassifier(AudioClassifier):
    """Rule-based classifier for audio events."""

    def classify(
        self, features: Dict[str, float], threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify segment based on extracted features using rule-based approach."""
        results = []

        # Rule-based classification has been removed as we now use the AST model
        # for comprehensive audio event detection

        return results


class ModelBasedClassifier(AudioClassifier):
    """Model-based classifier using a trained neural network."""

    def __init__(self, model: Any, class_types: List[str], device: torch.device):
        self.model = model
        self.class_types = class_types
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

        # Create results for each class type that exceeds the threshold
        for i, prob in enumerate(probabilities):
            if prob > threshold and i < len(self.class_types):
                results.append((self.class_types[i], float(prob)))

        return results


# AudioSet-based AST model for audio sound detection
class AudiosetClassifier:
    """Base class for detecting audio events in audio using Audioset."""

    def __init__(self, model_dir: Optional[str] = None, device: str = "cuda"):
        """Initialize the detector with a model."""

        # Set up comprehensive warning suppression
        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Temporarily suppress all logging
            logging.getLogger().setLevel(logging.ERROR)

            # Redirect both stdout and stderr during model loading
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Fix: load_audioset returns a single function, not a tuple
                self.model = load_audioset(device=device, model_dir=model_dir)
                # Initialize empty labels dictionary - will be populated later if needed
                self.labels = AUDIOSET_CLASS_MAPPING
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

        logging.info(f"Loaded Audioset model with {len(self.labels)} classes")
        self.device = device

    def detect_events(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        sr: int = 16000,
        show_progress: bool = True,
    ) -> List[AudioEvent]:
        detections = []
        audio_duration = None

        # Handle string path to audio file
        if isinstance(audio, str):
            # Load audio file
            try:
                if show_progress:
                    print(
                        f"[AudioDetector] Loading audio from file: {audio}",
                        flush=True,
                    )
                logging.debug(f"Loading audio from file: {audio}")
                y, sr = librosa.load(audio, sr=sr)
                audio_duration = len(y) / sr  # Calculate audio duration
                audio = y
                if show_progress:
                    print("[AudioDetector] Audio loaded successfully.", flush=True)
            except Exception as e:
                logging.error(f"Failed to load audio file: {str(e)}")
                return []
        elif isinstance(audio, (torch.Tensor, np.ndarray)):
            # If we have raw audio data, estimate its duration
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            # Make sure it's at least 1D
            if len(audio_np.shape) >= 1:
                audio_duration = audio_np.shape[-1] / sr
            else:
                audio_duration = 0

        # Process the audio array
        if show_progress:
            print("[AudioDetector] Processing audio through model...", flush=True)
        probs = self.detect_from_array(audio, sr, show_progress=show_progress)
        if show_progress:
            print("[AudioDetector] Audio processing complete.", flush=True)

        # Process the results
        if show_progress:
            print("[AudioDetector] Analyzing detection results...", flush=True)
            sys.stdout.flush()
            cls_items = list(self.labels.items())
            iterator = tqdm(
                cls_items,
                desc="[AudioDetector] Processing detections",
                unit="class",
                file=sys.stdout,
            )
        else:
            iterator = self.labels.items()

        for cls_idx, event_type in iterator:
            try:
                cls_idx_int = int(
                    cls_idx
                )  # Convert string indices to integers if needed

                # Check if we have enough dimensions and indices in bounds
                if len(probs.shape) > 1 and cls_idx_int < probs.shape[1]:
                    # For multiple segments/batches
                    for i in range(probs.shape[0]):
                        prob = probs[i, cls_idx_int]
                        if prob > 0.1:
                            # Create AudioEvent object instead of dictionary
                            # Use a reasonable time estimate for the entire audio
                            start_time = 0.0 if audio_duration is None else 0.0
                            end_time = 0.0 if audio_duration is None else audio_duration
                            detections.append(
                                AudioEvent(
                                    type=event_type,
                                    start_time=start_time,
                                    end_time=end_time,
                                    confidence=float(prob),
                                )
                            )
                elif cls_idx_int < len(probs):
                    # For single segment/batch
                    prob = probs[cls_idx_int]
                    if prob > 0.1:
                        # Create AudioEvent object instead of dictionary
                        start_time = 0.0 if audio_duration is None else 0.0
                        end_time = 0.0 if audio_duration is None else audio_duration
                        detections.append(
                            AudioEvent(
                                type=event_type,
                                start_time=start_time,
                                end_time=end_time,
                                confidence=float(prob),
                            )
                        )
            except (ValueError, IndexError, TypeError) as e:
                logging.warning(f"Error processing class index {cls_idx}: {str(e)}")
                continue

        if show_progress:
            print(
                f"[AudioDetector] Detection complete. Found {len(detections)} audio events.",
                flush=True,
            )

        return detections

    def detect_from_array(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sr: int = 16000,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Process audio through the model and return probabilities."""
        if show_progress:
            print("[AudioDetector] Preparing audio for model...", flush=True)

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure audio has the right shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension

        if show_progress:
            print("[AudioDetector] Running model inference...", flush=True)

        # The model function now handles both feature extraction and forward pass
        logits = self.model(audio, sr)

        # Convert logits to probabilities
        logits_np = logits.cpu().numpy()
        probs = softmax(logits_np, axis=-1)

        if show_progress:
            print("[AudioDetector] Model inference complete.", flush=True)

        logging.debug(f"Model output logits shape: {logits.shape}")
        return probs


class AudioEventDetector(AudiosetClassifier):
    """Detects audio events in audio"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = AUDIO_EVENT_THRESHOLD,
        device: str = None,
        event_types: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize the audio event detector.

        Args:
            model_path: Path to custom model (optional)
            threshold: Detection threshold (0.0-1.0)
            device: Computing device (cuda/cpu)
            event_types: List of event types to detect (defaults to all)
            custom_thresholds: Dictionary mapping event types to custom threshold values (optional)
        """
        # Default to CPU if no device specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set up comprehensive warning suppression
        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Temporarily suppress all logging
            logging.getLogger().setLevel(logging.ERROR)

            # Redirect both stdout and stderr during initialization
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Initialize parent class
                super().__init__(model_dir=model_path, device=device)
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

        self.threshold = threshold
        self.device = device

        # Map model outputs to event types
        self.event_class_map = AUDIOSET_CLASS_MAPPING

        # Filter event types if specified
        if event_types:
            # Keep only the requested event types
            self.event_class_map = {
                k: v for k, v in self.event_class_map.items() if v in event_types
            }

        # Load class-specific thresholds from constants
        self.class_thresholds = dict(AUDIO_EVENT_THRESHOLDS)

        # Apply custom thresholds if provided
        if custom_thresholds:
            # Update default thresholds with custom values
            self.class_thresholds.update(custom_thresholds)
            logging.info(
                f"Applied custom thresholds for {len(custom_thresholds)} event types"
            )

        # Use windowing for segmented analysis
        self.use_windowing = True
        self.window_size = 1.0  # seconds
        self.hop_size = 0.5  # seconds

    def detect_events(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        sr: int = 16000,
        show_progress: bool = True,
    ) -> List[AudioEvent]:
        """
        Detect audio events in the given audio with improved multi-event detection.

        This method uses windowing to better detect short events and employs
        class-specific thresholds for improved detection of secondary events.
        """
        audio_duration = None
        audio_data = None

        # Handle string path to audio file
        if isinstance(audio, str):
            # Load audio file
            try:
                if show_progress:
                    print(
                        f"[AudioDetector] Loading audio from file: {audio}", flush=True
                    )
                logging.debug(f"Loading audio from file: {audio}")
                y, sr = librosa.load(audio, sr=sr)
                audio_duration = len(y) / sr  # Calculate audio duration
                audio_data = y
                audio = y
                if show_progress:
                    print("[AudioDetector] Audio loaded successfully.", flush=True)
            except Exception as e:
                logging.error(f"Failed to load audio file: {str(e)}")
                return []
        elif isinstance(audio, (torch.Tensor, np.ndarray)):
            # If we have raw audio data, estimate its duration
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            audio_data = audio_np
            # Make sure it's at least 1D
            if len(audio_np.shape) >= 1:
                audio_duration = audio_np.shape[-1] / sr
            else:
                audio_duration = 0

        # Initialize detections list
        all_detections = []

        # Process the full audio first to get global events like "speech", "music", etc.
        if show_progress:
            print("[AudioDetector] Processing full audio through model...", flush=True)

        full_audio_probs = self.detect_from_array(
            audio, sr, show_progress=show_progress
        )

        # Get global events (those that span the entire audio)
        global_detections = self._process_probabilities(
            full_audio_probs,
            audio_duration=audio_duration,
            segment_start=0.0,
            segment_end=audio_duration if audio_duration else 0.0,
            show_progress=show_progress,
        )

        if show_progress:
            print(
                f"[AudioDetector] Found {len(global_detections)} global audio events.",
                flush=True,
            )

        # Add global detections to our list
        all_detections.extend(global_detections)

        # Process with windowing for better short event detection
        if (
            self.use_windowing
            and audio_data is not None
            and audio_duration > self.window_size * 2
        ):
            # Only use windowing for longer audio clips
            if show_progress:
                print(
                    "[AudioDetector] Performing windowed analysis for short events...",
                    flush=True,
                )

            # Segment the audio into overlapping windows
            window_samples = int(self.window_size * sr)
            hop_samples = int(self.hop_size * sr)

            # Determine total number of segments
            total_segments = max(
                1, int(np.ceil((len(audio_data) - window_samples) / hop_samples)) + 1
            )

            if show_progress:
                iterator = tqdm(
                    range(0, len(audio_data) - window_samples + 1, hop_samples),
                    desc="[AudioDetector] Processing segments",
                    unit="segment",
                    total=total_segments,
                )
            else:
                iterator = range(0, len(audio_data) - window_samples + 1, hop_samples)

            # Process each window
            window_detections = []

            for i, start_sample in enumerate(iterator):
                # Extract segment
                segment = audio_data[start_sample : start_sample + window_samples]

                # Calculate segment timestamps
                segment_start = start_sample / sr
                segment_end = min(segment_start + self.window_size, audio_duration)

                # Process segment through model
                segment_probs = self.detect_from_array(segment, sr, show_progress=False)

                # Process segment probabilities with segment-specific timestamps
                segment_events = self._process_probabilities(
                    segment_probs,
                    audio_duration=self.window_size,
                    segment_start=segment_start,
                    segment_end=segment_end,
                    show_progress=False,
                    use_lower_thresholds=True,  # Use lower thresholds for short events
                )

                # Add to window detections
                window_detections.extend(segment_events)

            # Filter window detections to avoid duplicates
            if window_detections:
                # Group events by type
                event_groups = {}
                for event in window_detections:
                    if event.type not in event_groups:
                        event_groups[event.type] = []
                    event_groups[event.type].append(event)

                # For each event type, merge overlapping events and keep the highest confidence
                for event_type, events in event_groups.items():
                    # Sort by start time
                    events.sort(key=lambda e: e.start_time)

                    i = 0
                    while i < len(events):
                        # Start with current event
                        current = events[i]

                        # Look for overlapping events to merge
                        j = i + 1
                        while j < len(events):
                            if events[j].start_time <= current.end_time:
                                # Overlapping event found
                                # Expand time range if needed
                                current.start_time = min(
                                    current.start_time, events[j].start_time
                                )
                                current.end_time = max(
                                    current.end_time, events[j].end_time
                                )
                                # Take highest confidence
                                current.confidence = max(
                                    current.confidence, events[j].confidence
                                )
                                # Remove the merged event
                                events.pop(j)
                            else:
                                j += 1

                        i += 1

                    # Add merged events to final detection list
                    all_detections.extend(events)

        # Remove duplicate global vs window detections
        final_detections = self._remove_duplicate_events(all_detections)

        if show_progress:
            print(
                f"[AudioDetector] Detection complete. Found {len(final_detections)} audio events.",
                flush=True,
            )

        # Optional: refine event timestamps if we have audio data
        if audio_data is not None and len(final_detections) > 0:
            final_detections = self._refine_event_timestamps(
                audio_data, sr, final_detections
            )

        return final_detections

    def _process_probabilities(
        self,
        probs: np.ndarray,
        audio_duration: Optional[float],
        segment_start: float = 0.0,
        segment_end: float = 0.0,
        show_progress: bool = False,
        use_lower_thresholds: bool = False,
    ) -> List[AudioEvent]:
        """Process model probabilities to extract audio events with appropriate timestamps."""
        detections = []

        if show_progress:
            print("[AudioDetector] Analyzing detection results...", flush=True)
            sys.stdout.flush()
            cls_items = list(self.labels.items())
            iterator = tqdm(
                cls_items,
                desc="[AudioDetector] Processing detections",
                unit="class",
                file=sys.stdout,
            )
        else:
            iterator = self.labels.items()

        for cls_idx, event_type in iterator:
            try:
                cls_idx_int = int(
                    cls_idx
                )  # Convert string indices to integers if needed

                # Get appropriate threshold for this event type
                event_threshold = self.threshold
                if use_lower_thresholds and event_type in self.class_thresholds:
                    event_threshold = self.class_thresholds.get(
                        event_type, self.threshold
                    )

                # Check if we have enough dimensions and indices in bounds
                if len(probs.shape) > 1 and cls_idx_int < probs.shape[1]:
                    # For multiple segments/batches (batch processing)
                    for i in range(probs.shape[0]):
                        prob = probs[i, cls_idx_int]
                        if prob > event_threshold:
                            # Create AudioEvent with segment timestamps
                            if audio_duration and probs.shape[0] > 1:
                                # Divide the segment duration by the number of sub-segments
                                sub_segment_duration = (
                                    segment_end - segment_start
                                ) / probs.shape[0]
                                sub_start_time = segment_start + (
                                    i * sub_segment_duration
                                )
                                sub_end_time = segment_start + (
                                    (i + 1) * sub_segment_duration
                                )
                                start_time = sub_start_time
                                end_time = sub_end_time
                            else:
                                # Use segment boundaries
                                start_time = segment_start
                                end_time = segment_end

                            detections.append(
                                AudioEvent(
                                    type=event_type,
                                    start_time=start_time,
                                    end_time=end_time,
                                    confidence=float(prob),
                                )
                            )
                elif cls_idx_int < len(probs):
                    # For single segment/batch
                    prob = probs[cls_idx_int]
                    if prob > event_threshold:
                        # Event spans the segment
                        detections.append(
                            AudioEvent(
                                type=event_type,
                                start_time=segment_start,
                                end_time=segment_end,
                                confidence=float(prob),
                            )
                        )
            except (ValueError, IndexError, TypeError) as e:
                logging.warning(f"Error processing class index {cls_idx}: {str(e)}")
                continue

        return detections

    def _remove_duplicate_events(self, events: List[AudioEvent]) -> List[AudioEvent]:
        """Remove duplicate events, keeping the highest confidence version."""
        # Group events by type
        event_map = {}

        for event in events:
            event_type = event.type

            # Create new entry for this type if it doesn't exist
            if event_type not in event_map:
                event_map[event_type] = []

            # Add this event to its type group
            event_map[event_type].append(event)

        # Process each event type to remove/merge duplicates
        final_events = []

        for event_type, type_events in event_map.items():
            # If this event type only has one instance, keep it
            if len(type_events) == 1:
                final_events.append(type_events[0])
                continue

            # Sort by confidence (highest first)
            type_events.sort(key=lambda e: e.confidence, reverse=True)

            # Apply same overlap detection to all event types (including speech)
            # For other events, take non-overlapping ones or higher confidence ones
            events_to_keep = []
            for event in type_events:
                # Check if this event overlaps with any already kept event
                overlaps = False
                for kept_event in events_to_keep:
                    # Check for significant overlap
                    if (
                        event.start_time < kept_event.end_time
                        and event.end_time > kept_event.start_time
                    ):
                        # If they overlap significantly, keep the higher confidence one
                        overlap_duration = min(
                            event.end_time, kept_event.end_time
                        ) - max(event.start_time, kept_event.start_time)
                        event_duration = event.end_time - event.start_time

                        # If overlap is significant and current event is lower confidence
                        if (
                            overlap_duration / event_duration > 0.5
                            and event.confidence <= kept_event.confidence
                        ):
                            overlaps = True
                            break

                if not overlaps:
                    events_to_keep.append(event)

            final_events.extend(events_to_keep)

        return final_events

    def _refine_event_timestamps(
        self, audio: np.ndarray, sr: int, events: List[AudioEvent]
    ) -> List[AudioEvent]:
        """
        Refines timestamps for detected events based on audio features.

        This uses energy-based segmentation to try to locate when certain
        audio events might occur, particularly for short, high-energy events.
        """
        # Only process if we have meaningful audio data
        if audio is None or len(audio) == 0:
            return events

        try:
            # Calculate audio duration
            audio_duration = len(audio) / sr

            # These events can be better localized through energy/spectral analysis
            energy_based_events = {
                "laughter",
                "baby_laughter",
                "giggle",
                "chuckle",
                "clapping",
                "bark",
                "slam",
                "explosion",
                "gunshot",
                "bang",
                "crash",
                "breaking",
                "knock",
                "footsteps",
                "cough",
                "sneeze",
                "throat_clearing",
            }

            # These need onset detection (they typically have sharp attacks)
            onset_based_events = {
                "cough",
                "sneeze",
                "throat_clearing",
                "clapping",
                "knock",
                "slam",
                "bang",
                "crash",
                "explosion",
            }

            # These need a different approach - they're more tonal
            tonal_events = {
                "sigh",
                "breathing",
                "wheeze",
                "whistle",
                "music",
                "singing",
            }

            # Calculate overall energy envelope
            frame_length = int(sr * 0.025)  # 25ms frames
            hop_length = int(sr * 0.010)  # 10ms hop

            # Energy-based features
            energy = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]

            # Onset strength for onset-based events
            onset_env = librosa.onset.onset_strength(
                y=audio, sr=sr, hop_length=hop_length
            )

            # Convert frames to time
            frames_time = librosa.frames_to_time(
                np.arange(len(energy)), sr=sr, hop_length=hop_length
            )

            # Automatically adapt thresholds based on energy distribution
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            energy_threshold = energy_mean + 1.5 * energy_std

            onset_mean = np.mean(onset_env)
            onset_std = np.std(onset_env)
            onset_threshold = onset_mean + 2.0 * onset_std

            # Find regions above threshold
            energy_is_above = energy > energy_threshold
            onset_is_above = onset_env > onset_threshold

            # Energy-based regions
            energy_transitions = np.diff(energy_is_above.astype(int))
            energy_onsets = np.where(energy_transitions == 1)[0]
            energy_offsets = np.where(energy_transitions == -1)[0]

            # Onset-based regions
            onset_peaks = librosa.util.peak_pick(
                onset_env,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.5,
                wait=10,
            )

            # Create energy-based time regions
            energy_regions = []
            for i, onset in enumerate(energy_onsets):
                offset_idx = len(energy_offsets)
                # Find the next offset after this onset
                for j, offset in enumerate(energy_offsets):
                    if offset > onset:
                        offset_idx = j
                        break

                if offset_idx < len(energy_offsets):
                    offset = energy_offsets[offset_idx]
                    onset_time = frames_time[onset]
                    offset_time = frames_time[offset]

                    # Only consider significant regions
                    if offset_time - onset_time > 0.1:  # At least 100ms
                        region_energy = np.mean(energy[onset:offset])
                        energy_regions.append((onset_time, offset_time, region_energy))

            # Create onset-based time markers
            onset_times = frames_time[onset_peaks]
            onset_values = onset_env[onset_peaks]
            onset_regions = []

            for i, time in enumerate(onset_times):
                # Create a short region around each onset
                onset_regions.append(
                    (
                        max(0, time - 0.1),  # Start 100ms before peak
                        min(audio_duration, time + 0.3),  # End 300ms after peak
                        onset_values[i],  # Onset strength as "energy" measure
                    )
                )

            # Process each event to refine timestamps
            refined_events = []

            for event in events:
                event_type = event.type

                # Skip processing for continuous events like speech, music
                if event_type in ["speech", "male_speech", "female_speech", "music"]:
                    refined_events.append(event)
                    continue

                # For energy-based events
                if event_type in energy_based_events and energy_regions:
                    # Find regions that overlap with this event's current timespan
                    relevant_regions = []
                    for start, end, region_energy in energy_regions:
                        if end > event.start_time and start < event.end_time:
                            relevant_regions.append((start, end, region_energy))

                    # If we found relevant regions
                    if relevant_regions:
                        # Sort by energy (highest first)
                        relevant_regions.sort(key=lambda r: r[2], reverse=True)

                        # Take the highest energy region
                        best_region = relevant_regions[0]

                        # Create a refined event
                        refined_events.append(
                            AudioEvent(
                                type=event_type,
                                start_time=best_region[0],
                                end_time=best_region[1],
                                confidence=event.confidence,
                            )
                        )
                    else:
                        # Keep original event
                        refined_events.append(event)

                # For onset-based events, additional processing
                elif event_type in onset_based_events and onset_regions:
                    # Find onset regions that overlap with this event's timespan
                    relevant_onsets = []
                    for start, end, onset_strength in onset_regions:
                        if end > event.start_time and start < event.end_time:
                            relevant_onsets.append((start, end, onset_strength))

                    if relevant_onsets:
                        # Sort by strength (highest first)
                        relevant_onsets.sort(key=lambda r: r[2], reverse=True)

                        # Take the strongest onset
                        best_onset = relevant_onsets[0]

                        # Create refined event
                        refined_events.append(
                            AudioEvent(
                                type=event_type,
                                start_time=best_onset[0],
                                end_time=best_onset[1],
                                confidence=event.confidence,
                            )
                        )
                    else:
                        # Keep original event
                        refined_events.append(event)
                else:
                    # Keep the original event timing
                    refined_events.append(event)

            return refined_events
        except Exception as e:
            logging.warning(f"Error refining event timestamps: {str(e)}")
            # Return original events if refinement fails
            return events
