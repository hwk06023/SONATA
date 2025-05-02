import torch
import logging
import numpy as np
import librosa
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def load_audioset(model_dir=None, device="cpu"):
    """
    Load AudioSet AST model for audio classification

    Args:
        model_dir: Path to model directory or model name on HuggingFace
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model
    """
    # If no model directory is provided, use the default model
    if model_dir is None:
        model_dir = "MIT/ast-finetuned-audioset-10-10-0.4593"

    logging.info(f"Loading AudioSet AST model from {model_dir}")

    try:
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        model = AutoModelForAudioClassification.from_pretrained(model_dir)

        # Move model to device
        model = model.to(device)
        model.eval()

        # Create a wrapper function to handle both feature extraction and model forward pass
        def model_fn(audio, sr=16000):
            # If audio is a file path, load it
            if isinstance(audio, str):
                audio, sr = librosa.load(audio, sr=sr)

            # Ensure audio is a numpy array
            if isinstance(audio, torch.Tensor):
                if audio.dim() == 2:
                    # If audio is [batch_size, seq_len], we keep it as is
                    audio_np = audio.cpu().numpy()
                elif audio.dim() == 3:
                    # If audio is [batch_size, channels, seq_len], we take the first channel
                    audio_np = audio.squeeze(1).cpu().numpy()
                else:
                    audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            # Ensure we have at least one dimension
            if not isinstance(audio_np, np.ndarray):
                raise ValueError(
                    f"Expected numpy array or tensor, got {type(audio_np)}"
                )

            # Make sure audio has correct dimensions
            if len(audio_np.shape) == 1:
                audio_np = np.expand_dims(audio_np, 0)

            # Ensure the audio is long enough for the feature extraction
            min_samples = 2 * sr // 100  # At least 20ms of audio
            for i in range(len(audio_np)):
                if len(audio_np[i]) < min_samples:
                    padding = np.zeros(min_samples - len(audio_np[i]))
                    audio_np[i] = np.concatenate([audio_np[i], padding])

            try:
                # Extract features
                inputs = feature_extractor(
                    audio_np, sampling_rate=sr, return_tensors="pt", padding=True
                ).to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = model(**inputs)

                return outputs.logits
            except Exception as e:
                logging.error(f"Error in feature extraction or model inference: {e}")
                # Return zero tensor with appropriate shape for number of classes
                num_classes = model.config.num_labels
                batch_size = len(audio_np) if isinstance(audio_np, list) else 1
                return torch.zeros((batch_size, num_classes), device=device)

        return model_fn
    except Exception as e:
        logging.error(f"Failed to load AudioSet model: {str(e)}")
        raise
