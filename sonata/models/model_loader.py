import torch
import logging
import numpy as np
import librosa
from pathlib import Path
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


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


def transcribe_with_model(audio_path, device="cpu", language=None):
    """
    Load Whisper Large V3 model for ASR and transcribe audio

    Args:
        audio_path: Path to the audio file to transcribe
        device: Device to load model on ('cpu' or 'cuda')
        language: Language code for transcription (e.g., 'en', 'ko'). If None, will use auto-detection.

    Returns:
        Transcription result
    """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # TODO: Add support for other models
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Set up parameters for transcription
    transcription_kwargs = {}

    # Add language parameter if specified - use direct language codes
    if language:
        transcription_kwargs["language"] = language

    # Add timestamp parameter
    transcription_kwargs["return_timestamps"] = True

    # Run inference
    result = pipe(audio_path, **transcription_kwargs)
    print("Transcription result:", result["text"])
    return result
