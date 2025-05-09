import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any
from sonata.models.model_loader import load_audioset
from sonata.constants import AUDIOSET_CLASS_MAPPING


class AudioCNN(nn.Module):
    """Simple CNN for audio classification."""

    def __init__(self, num_classes=8):
        """Initialize the CNN model.

        Args:
            num_classes: Number of output classes
        """
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
        """Forward pass through the network.

        Args:
            x: Input tensor with shape [batch_size, channels, height, width]

        Returns:
            Output tensor with shape [batch_size, num_classes]
        """
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


class AudiosetModelLoader:
    """Loader for the AudioSet AST (Audio Spectrogram Transformer) model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_path: Optional[str] = None,
        device: str = "cuda",
        use_vggish: bool = False,
    ):
        """Initialize the model loader.

        Args:
            model_path: Path to the model weights (optional)
            label_path: Path to the label mapping (optional)
            device: Device to load the model on ('cpu' or 'cuda')
            use_vggish: Whether to use VGGish model instead of AST
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.label_path = label_path
        self.device = device
        self.use_vggish = use_vggish
        self.model = None
        self.labels = None

    def load_model(self) -> Any:
        """Load the AudioSet model.

        Returns:
            The loaded model
        """
        # Set up comprehensive warning suppression
        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Temporarily suppress all logging
            logging.getLogger().setLevel(logging.ERROR)

            # Redirect both stdout and stderr during model loading
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Load the AudioSet model
                model = load_audioset(device=self.device, model_dir=self.model_path)
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

        self.model = model
        self.logger.info(f"Loaded Audioset model")
        return model

    def load_labels(self) -> Dict[str, int]:
        """Load the label mapping.

        Returns:
            Dictionary mapping class names to indices
        """
        # Use predefined class mapping from constants
        labels = AUDIOSET_CLASS_MAPPING
        self.labels = labels
        self.logger.info(f"Loaded {len(labels)} audio event classes")
        return labels
