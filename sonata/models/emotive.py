import os
import numpy as np
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional, Any
import logging
import tempfile
import soundfile as sf
from dataclasses import dataclass
from sonata.core.emotive_detector import EmotiveEvent, EmotiveCNN


class EmotiveDetector:
    # Default list of supported emotive types
    EMOTIVE_TYPES = [
        "laugh",
        "sigh",
        "yawn",
        "surprise",
        "inhale",
        "groan",
        "cough",
        "sneeze",
        "sniffle",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device: str = None,
        emotive_types: Optional[List[str]] = None,
    ):
        """Initialize EmotiveDetector.

        Args:
            model_path: Path to pre-trained model
            threshold: Confidence threshold for detection
            device: Device to run inference on ('cpu' or 'cuda')
            emotive_types: List of emotive types to detect
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set emotive types
        self.emotive_types = emotive_types or self.EMOTIVE_TYPES

        # Set threshold
        self.threshold = threshold

        # Initialize model
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a pre-trained emotive detection model.

        Args:
            model_path: Path to model file
        """
        try:
            # Initialize model
            self.model = EmotiveCNN(num_classes=len(self.emotive_types))

            # Load weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Set model to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            return True
        except Exception as e:
            logging.error(f"Failed to load emotive model: {str(e)}")
            self.model = None
            return False

    def detect_events(self, audio_path: str) -> List[EmotiveEvent]:
        """Detect emotive events in an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of detected emotive events
        """
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return []

        # Use rule-based detection if no model is loaded
        if self.model is None:
            logging.info("No model loaded, using rule-based detection")
            # Note: We would implement rule-based detection here
            return []

        # Extract features from audio segments
        from sonata.core.emotive_detector import AudioProcessor

        segments = AudioProcessor.segment_audio(audio_path)

        events = []
        for start_time, end_time, segment in segments:
            # Save segment to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, segment, 22050)

                # Extract features
                features = AudioProcessor.extract_features(tmp.name)

                if features is not None:
                    # Run inference
                    with torch.no_grad():
                        features = features.to(self.device)
                        outputs = self.model(features)

                        # Get predictions
                        probs = outputs.cpu().numpy()[0]

                        # Create events for each emotive type above threshold
                        for i, prob in enumerate(probs):
                            if prob >= self.threshold:
                                event = EmotiveEvent(
                                    type=self.emotive_types[i],
                                    start_time=start_time,
                                    end_time=end_time,
                                    confidence=float(prob),
                                )
                                events.append(event)

        return events

    def detect_from_array(
        self, audio_array: np.ndarray, sr: int = 22050
    ) -> List[EmotiveEvent]:
        """Detect emotive events from an audio array.

        Args:
            audio_array: Audio data as numpy array
            sr: Sample rate

        Returns:
            List of detected emotive events
        """
        # Save array to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_array, sr)
            return self.detect_events(tmp.name)
