import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Union, Optional
from scipy.special import softmax


class AudioClassifier:
    """Base class for different audio classifiers."""

    def __init__(self):
        """Initialize the audio classifier."""
        self.logger = logging.getLogger(__name__)

    def classify(
        self, features: Dict[str, float], threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify based on features. To be implemented by subclasses.

        Args:
            features: Dictionary of audio features
            threshold: Classification threshold

        Returns:
            List of (class_name, confidence) tuples
        """
        raise NotImplementedError("Subclasses must implement classify method")


class RuleBasedClassifier(AudioClassifier):
    """Rule-based classifier for audio events."""

    def __init__(self):
        """Initialize the rule-based classifier."""
        super().__init__()

    def classify(
        self, features: Dict[str, float], threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify segment based on extracted features using rule-based approach.

        Args:
            features: Dictionary of audio features
            threshold: Classification threshold

        Returns:
            List of (class_name, confidence) tuples
        """
        results = []

        # Rule-based classification has been removed as we now use the AST model
        # for comprehensive audio event detection

        return results


class ModelBasedClassifier(AudioClassifier):
    """Model-based classifier using a trained neural network."""

    def __init__(self, model: Any, class_types: List[str], device: torch.device):
        """Initialize the model-based classifier.

        Args:
            model: Trained neural network model
            class_types: List of class names
            device: Device to run inference on
        """
        super().__init__()
        self.model = model
        self.class_types = class_types
        self.device = device

    def classify(
        self, features: torch.Tensor, threshold: float
    ) -> List[Tuple[str, float]]:
        """Classify using the trained model.

        Args:
            features: Tensor of audio features
            threshold: Classification threshold

        Returns:
            List of (class_name, confidence) tuples
        """
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


class AudiosetClassifier:
    """Base class for detecting audio events in audio using Audioset."""

    def __init__(self, model: Any, labels: Dict[str, int], device: str = "cuda"):
        """Initialize the classifier with a model.

        Args:
            model: Trained AudioSet model
            labels: Dictionary mapping class names to indices
            device: Device to run inference on
        """
        self.model = model
        self.labels = labels
        self.device = device
        self.logger = logging.getLogger(__name__)

    def detect_from_array(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sr: int = 16000,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Process audio array through model to get class probabilities.

        Args:
            audio: Audio data as tensor or array
            sr: Sample rate
            show_progress: Whether to show progress information

        Returns:
            Array of class probabilities
        """
        try:
            # Convert numpy array to tensor if needed
            if isinstance(audio, np.ndarray):
                # Make sure it's float32, normalized to [-1, 1]
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                # Ensure it's in the range [-1, 1]
                if np.max(np.abs(audio)) > 1.0:
                    audio = audio / np.max(np.abs(audio))

                # Convert to tensor
                audio_tensor = torch.FloatTensor(audio)
            else:
                audio_tensor = audio

            # Make it at least 2D: [batch_size, time]
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Move to device
            audio_tensor = audio_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                # AudioSet models typically return logits
                model_output = self.model(audio_tensor, sample_rate=sr)

                # Some models might return probabilities directly
                if isinstance(model_output, tuple):
                    # Handle the case where model returns multiple outputs
                    logits = model_output[0]
                else:
                    logits = model_output

                # Apply softmax if the model returns logits
                if isinstance(logits, torch.Tensor):
                    probs = torch.nn.functional.sigmoid(logits).cpu().numpy()
                else:
                    probs = softmax(logits, axis=-1)

            return probs
        except Exception as e:
            self.logger.error(f"Error in detect_from_array: {str(e)}")
            # Return empty probabilities as fallback
            return np.zeros((1, len(self.labels)))
