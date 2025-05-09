import os
import torch
import whisperx
import ssl
import io
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Tuple, Optional
from sonata.constants import LanguageCode


class ASRModelManager:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.current_language = None
        self.logger = logging.getLogger(__name__)

    def load_transcription_model(self, language_code: str = LanguageCode.ENGLISH.value):
        ssl._create_default_https_context = ssl._create_unverified_context
        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        redirect_context = redirect_stdout(stdout_buffer)
        redirect_err_context = redirect_stderr(stderr_buffer)
        warning_context = warnings.catch_warnings()

        try:
            logging.getLogger().setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
            warnings.filterwarnings("ignore", message=".*set_stage.*")
            warnings.filterwarnings(
                "ignore", message=".*Trying to infer the `batch_size`.*"
            )

            with redirect_context, redirect_err_context, warning_context:
                self.model = whisperx.load_model(
                    self.model_name,
                    self.device,
                    compute_type=self.compute_type,
                    language=language_code,
                )
        finally:
            logging.getLogger().setLevel(original_level)

        if hasattr(self.model, "preset_language"):
            self.model.preset_language = language_code

        return self.model

    def load_alignment_model(self, language_code: str = LanguageCode.ENGLISH.value):
        if self.current_language == language_code and self.align_model is not None:
            return self.align_model, self.align_metadata

        original_level = logging.getLogger().level
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        warning_context = warnings.catch_warnings()

        try:
            logging.getLogger().level = logging.ERROR
            warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
            warnings.filterwarnings("ignore", message=".*set_stage.*")
            warnings.filterwarnings(
                "ignore", message=".*Trying to infer the `batch_size`.*"
            )

            with redirect_stdout(stdout_buffer), redirect_stderr(
                stderr_buffer
            ), warning_context:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code, device=self.device
                )
            self.current_language = language_code
        except Exception as e:
            self.logger.warning(
                f"Could not load alignment model for {language_code}. Falling back to transcription without alignment."
            )
            self.align_model = None
            self.align_metadata = None
            self.current_language = language_code

        return self.align_model, self.align_metadata

    def get_models(self, language_code: str = LanguageCode.ENGLISH.value) -> Tuple:
        if self.model is None or self.current_language != language_code:
            self.load_transcription_model(language_code)
            self.load_alignment_model(language_code)

        return self.model, self.align_model, self.align_metadata
