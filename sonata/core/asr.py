from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional
from sonata.constants import LanguageCode
from sonata.utils.text import clean_text_for_language
from sonata.models.model_loader import transcribe_with_model
from textgrids import TextGrid
import os
import ssl
import io
import sys
import logging
import warnings
import subprocess

# Base environment variables
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check current root logger level
root_logger = logging.getLogger()
current_level = root_logger.level

# Suppress warnings only at ERROR level
if current_level >= logging.ERROR:
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::DeprecationWarning"
    warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
    warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")


class ASRProcessor:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
    ):
        """Initialize the ASR processor with default model parameters.

        Args:
            device: The device to use for inference ('cpu' or 'cuda')
            compute_type: The compute type for the model
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.align_model = None
        self.align_metadata = None
        self.current_language = None
        self.diarize_model = None
        self.diarize_model_type = None
        self.embedding_model_name = None
        self.clustering_method = None
        self.speaker_embeddings = {}
        self.vad_model = None
        self.scd_model = None
        self.speaker_embedding_model = None
        self.logger = logging.getLogger(__name__)
        self.asr_model = None

    def process_audio(
        self,
        audio_path: str,
        language: str = LanguageCode.ENGLISH.value,
        show_progress: bool = True,
    ) -> Dict:
        """Process audio file with Whisper Large V3 & mfa to get transcription with timestamps."""

        # Always check if models need to be loaded or reloaded
        if self.asr_model is None or self.current_language != language:
            if show_progress:
                print(
                    f"[ASR] Loading Whisper model for language: {language}...",
                    flush=True,
                )
            # Initialize the ASR model if not already done
            self.current_language = language

        if not ".wav" in audio_path:
            err_msg = f"Audio file must be a .wav file: {audio_path}"
            self.logger.error(err_msg)
            return {
                "error": err_msg,
                "integrated_transcript": {"plain_text": "", "rich_text": []},
            }

        audio_name = audio_path.split("/")[-1].split(".wav")[0]
        data_directory = f"mfa/{audio_name}"
        align_directory = f"mfa/{audio_name}_align"
        os.makedirs(data_directory, exist_ok=True)
        os.makedirs(align_directory, exist_ok=True)
        if show_progress:
            print(f"[ASR] Running speech recognition...", flush=True)
            sys.stdout.flush()

        if not os.path.exists(f"{data_directory}/audio.lab"):
            # Transcribe audio with Whisper into .lab format
            transcription = transcribe_with_model(
                audio_path, device=self.device, language=language
            )

            if show_progress:
                print(
                    f"[ASR] Transcription complete.",
                    flush=True,
                )

            text = transcription["text"]

            # Save .lab file
            with open(f"{data_directory}/audio.lab", "w") as f:
                f.write(text)
        else:
            with open(f"{data_directory}/audio.lab", "r") as f:
                text = f.read()

        if not os.path.exists(f"{align_directory}/audio.TextGrid"):
            # Validate the audio and text corpus first
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    f"{data_directory}/audio.wav",
                ]
            )
            subprocess.run(["mfa", "validate", data_directory, "korean_mfa"])
            print("mfa validate done")
            # Use MFA to get .TextGrid format
            subprocess.run(
                [
                    "mfa",
                    "align",
                    data_directory,
                    "korean_mfa",
                    "korean_mfa",
                    align_directory,
                ]
            )
            print("mfa align done")
        # Use TextGrid to get speaker segments
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        tg = TextGrid()
        tg.read(f"{align_directory}/audio.TextGrid")
        word_items = tg["words"]

        # Use TextGrid to get speaker segments
        # Clean the textgrid words
        word_len = len(word_items)
        words = []
        for i in range(word_len):
            if word_items[i].text:
                words.append(word_items[i])

        # Clean the text
        actual_words = clean_text_for_language(text, language)

        # Get word timestamps
        # TODO: Fix bug
        start_times = []
        end_times = []
        word_idx = 0
        for i, actual_word in enumerate(actual_words):
            cur_idx = 0
            start_times.append(words[word_idx].xmin)
            while cur_idx < len(actual_word):
                cur_idx += len(words[word_idx].text)
                word_idx += 1
            end_times.append(words[word_idx - 1].xmax)
        # Debug: Saving word audio
        # for i in range(len(actual_words)):
        #     word_audio = waveform[:, int(start_times[i] * sample_rate):int(end_times[i] * sample_rate)]
        #     torchaudio.save(f"temp/{actual_words[i]}.wav", word_audio, sample_rate)
        # Transcribe with whisper-large-v3
        result_segments = []
        for i in range(len(actual_words)):
            result_segments.append(
                {
                    "start": start_times[i],
                    "end": end_times[i],
                    "content": actual_words[i],
                    "type": "voice",
                }
            )
        return result_segments
