import os
import torch
import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union
import whisperx


# Filter out known dependency warnings
warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
warnings.filterwarnings("ignore", message=".*pyannote.audio.*")
warnings.filterwarnings("ignore", message=".*torch.*")


class WhisperX:
    def __init__(self, model_name: str = "large-v3", device: str = "cpu"):
        """Initialize the WhisperX model.

        Args:
            model_name: Name of the Whisper model to use
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.align_model = None
        self._load_models()

    def _load_models(self):
        """Load the Whisper and alignment models."""
        # Check if CUDA is available when device is set to cuda
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        # Load the ASR model
        self.model = whisperx.load_model(
            self.model_name, self.device, compute_type="float32"
        )

        # Load alignment model (HuBERT)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

    def transcribe(self, audio_file: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file using WhisperX with word-level timestamps.

        Args:
            audio_file: Path to the audio file
            language: Language code for transcription

        Returns:
            Dictionary containing transcription results with segments and word timestamps
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if self.model is None:
            self._load_models()

        try:
            # Transcribe with silero VAD for better segment boundary detection
            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, language=language, batch_size=16)

            print("Initial transcription result keys:", list(result.keys()))

            # Align to get word-level timestamps
            if self.align_model is not None:
                try:
                    aligned_result = whisperx.align(
                        result["segments"],
                        self.align_model,
                        self.align_metadata,
                        audio,
                        self.device,
                        return_char_alignments=False,
                    )
                    print("Aligned result keys:", list(aligned_result.keys()))
                    result = aligned_result
                except Exception as align_error:
                    print(f"Error during alignment: {str(align_error)}")
                    # Continue with unaligned result
                    if "segments" not in result:
                        print("Warning: No segments found in result")
                        result["segments"] = []
                    if "text" not in result:
                        print("Warning: No text found in result")
                        result["text"] = ""

            # Ensure the result has the required keys
            if "text" not in result:
                print("Adding missing 'text' key to result")
                result["text"] = " ".join(
                    [seg.get("text", "") for seg in result.get("segments", [])]
                )

            if "segments" not in result:
                print("Adding missing 'segments' key to result")
                result["segments"] = []

            return result

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            # Return a minimal valid result structure
            return {"text": "", "segments": [], "language": language}
