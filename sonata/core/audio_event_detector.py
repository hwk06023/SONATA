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
from typing import Dict, Any, List, Tuple, Optional, Union, Set
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
import concurrent.futures

# Temporary - Set up debug logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from dataclasses import dataclass
import scipy.signal as signal
import tempfile
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Import our new modular structure
from sonata.core.utils.audio_event import (
    AudioEvent as ModularAudioEvent,
    AudioEventDetector as ModularAudioEventDetector,
)


# Keep the original AudioEvent class for backward compatibility
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

    @classmethod
    def from_modular(cls, event: ModularAudioEvent) -> "AudioEvent":
        return cls(
            type=event.event_type,
            start_time=event.start_time,
            end_time=event.end_time,
            confidence=event.confidence,
        )


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


# Wrapper class that uses the modular implementation internally
class AudioEventDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = AUDIO_EVENT_THRESHOLD,
        device: str = None,
        event_types: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
        window_size: float = 1.0,
        hop_size: float = 0.5,
    ):
        """Initialize the audio event detector.

        Args:
            model_path: Path to the model
            threshold: Default detection threshold
            device: Device to use for inference
            event_types: List of event types to detect
            custom_thresholds: Custom thresholds for specific event types
            window_size: Size of processing window in seconds
            hop_size: Hop size between windows in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold

        # Use device if provided, otherwise use CUDA if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.event_types = event_types
        self.custom_thresholds = custom_thresholds or AUDIO_EVENT_THRESHOLDS
        self.window_size = window_size
        self.hop_size = hop_size

        # Initialize the modular detector
        self.detector = ModularAudioEventDetector(
            model_path=model_path,
            device=self.device,
        )

        self.logger.info(f"Initialized AudioEventDetector using {self.device}")

    def detect_events(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        sr: int = 16000,
        show_progress: bool = True,
    ) -> List[AudioEvent]:
        """Detect audio events in the input audio.

        Args:
            audio: Audio input as file path, tensor, or array
            sr: Sample rate of the audio
            show_progress: Whether to show progress information

        Returns:
            List of detected AudioEvent objects
        """
        try:
            # Initialize event list
            detected_events = []

            # Convert set of event types if provided
            target_events = set(self.event_types) if self.event_types else None

            # Handle different input types
            if isinstance(audio, str):
                # Detect events using modular detector
                modular_events = self.detector.detect_events(
                    audio_file=audio,
                    threshold=self.threshold,
                    segment_size=self.window_size,
                    hop_size=self.hop_size,
                    target_sr=sr,
                    target_events=target_events,
                )

                # Convert modular events to original format
                for event in modular_events:
                    # Apply custom thresholds if applicable
                    if (
                        self.custom_thresholds
                        and event.event_type in self.custom_thresholds
                        and event.confidence < self.custom_thresholds[event.event_type]
                    ):
                        continue

                    detected_events.append(AudioEvent.from_modular(event))

            else:
                # For tensor/array inputs, convert to file first
                # This is a limitation of the current implementation
                self.logger.warning(
                    "Direct tensor/array input is deprecated. "
                    "Please use file paths for better performance."
                )

                import tempfile
                import soundfile as sf

                # Convert to numpy array if needed
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.cpu().numpy()
                else:
                    audio_np = audio

                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_np, sr)
                    tmp_path = tmp.name

                try:
                    # Detect events using the file path
                    return self.detect_events(
                        audio=tmp_path, sr=sr, show_progress=show_progress
                    )
                finally:
                    # Clean up temporary file
                    import os

                    os.unlink(tmp_path)

            return detected_events

        except Exception as e:
            self.logger.error(f"Error detecting events: {str(e)}")
            return []

    def detect_events_multi_scale(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        sr: int = 16000,
        window_sizes: List[float] = [0.2, 1.0, 2.5],
        hop_sizes: List[float] = [0.1, 0.5, 1.0],
        parallel: bool = False,
        show_progress: bool = True,
    ) -> List[AudioEvent]:
        """Detect audio events using multiple window sizes.

        Args:
            audio: Audio input as file path, tensor, or array
            sr: Sample rate of the audio
            window_sizes: List of window sizes to use
            hop_sizes: List of hop sizes to use
            parallel: Whether to process in parallel
            show_progress: Whether to show progress information

        Returns:
            List of detected AudioEvent objects
        """
        if len(window_sizes) != len(hop_sizes):
            raise ValueError("window_sizes and hop_sizes must have the same length")

        all_events = []

        if parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                for i, (window_size, hop_size) in enumerate(
                    zip(window_sizes, hop_sizes)
                ):
                    # Save the original values
                    original_window = self.window_size
                    original_hop = self.hop_size

                    # Set new values
                    self.window_size = window_size
                    self.hop_size = hop_size

                    # Submit the task
                    futures.append(
                        executor.submit(
                            self.detect_events, audio=audio, sr=sr, show_progress=False
                        )
                    )

                    # Restore original values
                    self.window_size = original_window
                    self.hop_size = original_hop

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    events = future.result()
                    all_events.extend(events)
        else:
            # Sequential processing
            for i, (window_size, hop_size) in enumerate(zip(window_sizes, hop_sizes)):
                if show_progress:
                    scale_name = f"Scale {i+1}/{len(window_sizes)}"
                    print(
                        f"Processing with {scale_name}: window={window_size}s, hop={hop_size}s"
                    )

                # Save the original values
                original_window = self.window_size
                original_hop = self.hop_size

                # Set new values
                self.window_size = window_size
                self.hop_size = hop_size

                # Detect events
                events = self.detect_events(
                    audio=audio, sr=sr, show_progress=show_progress
                )

                all_events.extend(events)

                # Restore original values
                self.window_size = original_window
                self.hop_size = original_hop

        # Merge events using non-maximum suppression
        merged_events = self._merge_events_nms(all_events)

        return merged_events

    def _merge_events_nms(
        self, events: List[AudioEvent], iou_threshold: float = 0.5
    ) -> List[AudioEvent]:
        """Merge events using non-maximum suppression.

        Args:
            events: List of events
            iou_threshold: Intersection over union threshold

        Returns:
            List of merged events
        """
        if not events:
            return []

        # Group events by type
        event_groups = {}
        for event in events:
            if event.type not in event_groups:
                event_groups[event.type] = []
            event_groups[event.type].append(event)

        merged_events = []

        # Process each group
        for event_type, group in event_groups.items():
            # Sort by confidence
            sorted_events = sorted(group, key=lambda e: e.confidence, reverse=True)

            kept_events = []

            while sorted_events:
                # Take the highest confidence event
                best_event = sorted_events.pop(0)
                kept_events.append(best_event)

                # Filter out events with high overlap
                remaining_events = []
                for event in sorted_events:
                    # Calculate IoU
                    intersection_start = max(best_event.start_time, event.start_time)
                    intersection_end = min(best_event.end_time, event.end_time)

                    if intersection_end <= intersection_start:
                        # No overlap
                        remaining_events.append(event)
                        continue

                    intersection = intersection_end - intersection_start
                    best_duration = best_event.end_time - best_event.start_time
                    event_duration = event.end_time - event.start_time
                    union = best_duration + event_duration - intersection

                    iou = intersection / union if union > 0 else 0

                    if iou < iou_threshold:
                        remaining_events.append(event)

                sorted_events = remaining_events

            merged_events.extend(kept_events)

        # Sort by start time
        return sorted(merged_events, key=lambda e: e.start_time)
