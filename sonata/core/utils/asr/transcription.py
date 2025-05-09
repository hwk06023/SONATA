import os
import numpy as np
import torch
import whisperx
import logging
import warnings
from typing import Dict, List, Optional
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from sonata.constants import LanguageCode
from sonata.core.utils.asr.models import ASRModelManager


class AudioTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        self.model_manager = ASRModelManager(
            model_name=model_name, device=device, compute_type=compute_type
        )
        self.logger = logging.getLogger(__name__)

    def transcribe_audio(
        self,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> Dict:
        """Transcribe audio file using WhisperX

        Args:
            audio_path: Path to audio file
            language: Language code
            batch_size: Batch size for inference
            show_progress: Whether to show progress information

        Returns:
            Dictionary with transcription results
        """
        if show_progress:
            print(f"[ASR] Transcribing audio in {language}...", flush=True)

        # Ensure models are loaded for the specified language
        model, align_model, align_metadata = self.model_manager.get_models(language)

        if model is None:
            raise RuntimeError(f"Failed to load transcription model for {language}")

        # Transcribe with WhisperX
        result = self._transcribe_with_whisperx(
            model=model,
            audio_path=audio_path,
            language=language,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Align if alignment model is available
        if align_model is not None and align_metadata is not None:
            result = self._align_with_whisperx(
                result=result,
                audio_path=audio_path,
                align_model=align_model,
                align_metadata=align_metadata,
                show_progress=show_progress,
            )

        return result

    def _transcribe_with_whisperx(
        self,
        model,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> Dict:
        """Transcribe audio with WhisperX model"""
        whisperx_options = {
            "language": language,
            "batch_size": batch_size,
            "vad_filter": True,  # Apply VAD filter for better accuracy
        }

        # Set up contexts for suppressing output based on show_progress
        stdout_context = nullcontext() if show_progress else redirect_stdout(None)
        stderr_context = nullcontext() if show_progress else redirect_stderr(None)
        warning_context = warnings.catch_warnings()

        try:
            # Filter warnings regardless of show_progress
            warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
            warnings.filterwarnings(
                "ignore", message=".*Trying to infer the `batch_size`.*"
            )

            # Run transcription with appropriate logging
            with stdout_context, stderr_context, warning_context:
                result = model.transcribe(audio_path, **whisperx_options)

            return result
        except Exception as e:
            self.logger.error(f"WhisperX transcription error: {str(e)}")
            # Return empty result on error
            return {"segments": []}

    def _align_with_whisperx(
        self,
        result: Dict,
        audio_path: str,
        align_model,
        align_metadata,
        show_progress: bool = True,
    ) -> Dict:
        """Align transcription with WhisperX alignment model"""
        if show_progress:
            print(f"[ASR] Aligning transcription...", flush=True)

        # Set up contexts for suppressing output based on show_progress
        stdout_context = nullcontext() if show_progress else redirect_stdout(None)
        stderr_context = nullcontext() if show_progress else redirect_stderr(None)
        warning_context = warnings.catch_warnings()

        try:
            # Filter warnings regardless of show_progress
            warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
            warnings.filterwarnings(
                "ignore", message=".*Trying to infer the `batch_size`.*"
            )

            # Run alignment with appropriate logging
            with stdout_context, stderr_context, warning_context:
                result_aligned = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio_path,
                    return_char_alignments=False,
                )

            return result_aligned
        except Exception as e:
            self.logger.error(f"WhisperX alignment error: {str(e)}")
            # Return original result if alignment fails
            return result
