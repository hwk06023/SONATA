import os
import numpy as np
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional


class EmotiveEvent:
    def __init__(
        self,
        type_name: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
    ):
        self.type = type_name
        self.start = start_time
        self.end = end_time
        self.confidence = confidence

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }

    def to_tag(self) -> str:
        return f"({self.type})"


class EmotiveCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(EmotiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EmotiveDetector:
    EMOTIVE_TYPES = ["laughter", "sigh", "yawn", "surprise", "inhale"]

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self.sample_rate = 16000
        self.window_size = 1.0  # 1 second window
        self.hop_length = 0.5  # 0.5 second hop

    def load_model(self):
        """Load or initialize the model."""
        if self.model_path and os.path.exists(self.model_path):
            self.model = EmotiveCNN(len(self.EMOTIVE_TYPES))
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            # For now, we'll use a placeholder model
            # In a real project, you would train this on emotive sound data
            self.model = EmotiveCNN(len(self.EMOTIVE_TYPES))

        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, audio_path: str) -> Tuple[np.ndarray, List[float]]:
        """Extract features from audio for emotive detection."""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr

        # Calculate window and hop in samples
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_length * sr)

        # Prepare timestamps and features
        timestamps = []
        features = []

        for i in range(0, len(y) - window_samples + 1, hop_samples):
            window = y[i : i + window_samples]
            start_time = i / sr
            timestamps.append(start_time)

            # Extract mel spectrogram for this window
            mel_spec = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Resize to fixed dimensions for CNN
            mel_spec_resized = librosa.util.fix_length(mel_spec_db, size=64, axis=1)
            features.append(mel_spec_resized)

        # Stack features
        if features:
            features = np.stack(features)
        else:
            features = np.array([])

        return features, timestamps

    def detect_events(
        self, audio_path: str, threshold: float = 0.5
    ) -> List[EmotiveEvent]:
        """Detect emotive events in audio file."""
        if self.model is None:
            self.load_model()

        features, timestamps = self.extract_features(audio_path)
        if len(features) == 0:
            return []

        # Convert to torch tensor and reshape for CNN
        features_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

        events = []
        with torch.no_grad():
            for i in range(len(features_tensor)):
                output = self.model(features_tensor[i : i + 1])
                probabilities = F.softmax(output, dim=1)[0]

                # Find the highest probability class
                max_prob, predicted_class = torch.max(probabilities, 0)

                if max_prob.item() > threshold:
                    event_type = self.EMOTIVE_TYPES[predicted_class.item()]
                    start_time = timestamps[i]
                    end_time = start_time + self.window_size
                    confidence = max_prob.item()

                    events.append(
                        EmotiveEvent(event_type, start_time, end_time, confidence)
                    )

        return events
