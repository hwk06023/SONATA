import torch
import numpy as np
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from typing import List, Dict, Optional, Tuple, Union
import os
import logging
import warnings
import speechbrain as sb
from dataclasses import dataclass

# Configure speechbrain to suppress debug logs
sb_logger = logging.getLogger("speechbrain")
sb_logger.setLevel(logging.WARNING)

# Import utility modules
from sonata.core.utils.diarization.vad import VADProcessor
from sonata.core.utils.diarization.change_detection import ChangeDetector
from sonata.core.utils.diarization.embeddings import EmbeddingExtractor
from sonata.core.utils.diarization.clustering import SpeakerClusterer
from sonata.core.utils.diarization.overlap import (
    OverlapDetector,
    SegmentProcessor,
)
from sonata.core.utils.diarization.io import DiarizationIO


# Filter PyTorch transformer attention warnings
warnings.filterwarnings(
    "ignore", message="Support for mismatched key_padding_mask and attn_mask"
)


class SpeakerDiarizer:
    def __init__(self, device="cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

        # These will be initialized in _load_models
        self.vad_model = None
        self.vad_get_speech_timestamps = None
        self.wavlm_processor = None
        self.wavlm_model = None
        self.ecapa_model = None
        self.has_ecapa_model = False
        self.vad_processor = None
        self.change_detector = None
        self.embedding_extractor = None

        # Initialize utility processors that don't require models
        self.clusterer = SpeakerClusterer()
        self.overlap_detector = OverlapDetector(device=device)
        self.segment_processor = SegmentProcessor()
        self.io_handler = DiarizationIO()

        # Now load models and initialize remaining processors
        self._load_models()

    def _load_models(self):
        self.logger.info("Loading diarization models...")
        # 1. Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        self.vad_get_speech_timestamps = utils[0]
        self.vad_model.to(self.device)
        self.vad_processor = VADProcessor(
            self.vad_model, self.vad_get_speech_timestamps, device=self.device
        )

        # 2. WavLM XVector for speaker embeddings
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.wavlm_model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.wavlm_model.to(self.device)

        # 3. Load ECAPA-TDNN for better embeddings
        try:
            self.ecapa_model = sb.inference.EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
            self.has_ecapa_model = True
        except Exception as e:
            self.logger.warning(f"Could not load ECAPA-TDNN model: {str(e)}")
            self.has_ecapa_model = False

        # Initialize other processors that need models
        self.change_detector = ChangeDetector(
            self.ecapa_model if self.has_ecapa_model else None, device=self.device
        )
        self.embedding_extractor = EmbeddingExtractor(
            self.wavlm_model,
            self.wavlm_processor,
            self.ecapa_model if self.has_ecapa_model else None,
            device=self.device,
        )

    def diarize(
        self,
        audio_path,
        num_speakers=None,
        show_progress=True,
        save_steps=False,
    ):
        """Main diarization method with improved processing pipeline

        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers (estimated if None)
            show_progress: Whether to show progress information
            save_steps: Whether to save intermediate outputs
        """
        if show_progress:
            print(f"Starting enhanced diarization for: {audio_path}")

        output_dir = None
        if save_steps:
            # Create directory for saving step outputs
            output_dir = self.io_handler.create_output_directory(audio_path)

        # 1. Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if needed

        # 2. VAD to get speech segment
        vad_segments = self.vad_processor.get_vad_segments(
            waveform[0], sample_rate, show_progress
        )

        if save_steps:
            vad_txt = os.path.join(output_dir, "01_vad_segments.txt")
            self.io_handler.save_vad_segments(vad_segments, vad_txt)

        if len(vad_segments) == 0:
            self.logger.warning("No speech segments detected in audio")
            return []

        # 3. Create analysis segments based on input method
        change_points = self.change_detector.detect_speaker_changes(
            waveform[0], sample_rate, vad_segments, show_progress=show_progress
        )

        if save_steps:
            cp_txt = os.path.join(output_dir, "02_change_points.txt")
            cp_desc = "Speaker Change Points (in seconds)\nEach value represents a time point where one speaker changes to another."
            self.io_handler.save_to_txt(change_points, cp_txt, cp_desc)

        # Create analysis segments from VAD and change points
        all_boundaries = sorted(
            set(
                [s[0] for s in vad_segments]
                + [s[1] for s in vad_segments]
                + change_points
            )
        )
        analysis_segments = [
            (all_boundaries[i], all_boundaries[i + 1])
            for i in range(len(all_boundaries) - 1)
        ]

        if save_steps:
            seg_txt = os.path.join(output_dir, "03_analysis_segments.txt")
            seg_desc = "Analysis Segments\nFinal analysis segments created by combining VAD segments and speaker change points.\nFormat: start_time,end_time"
            with open(seg_txt, "w") as f:
                f.write(f"# {seg_desc}\n")
                f.write("#" + "-" * 50 + "\n")
                for start, end in analysis_segments:
                    f.write(f"{start},{end}\n")

        if show_progress:
            print(f"Created {len(analysis_segments)} analysis segments")

        # 4. Extract enhanced speaker embeddings
        embeddings, segment_timings = self.embedding_extractor.extract_embeddings(
            waveform[0], sample_rate, analysis_segments, show_progress
        )

        if save_steps:
            timing_txt = os.path.join(output_dir, "04_segment_timings.txt")
            timing_desc = "Embedding Segment Timings\nEach line represents the start and end time (in seconds) of audio segments used for speaker embedding extraction.\nFormat: start_time,end_time"
            with open(timing_txt, "w") as f:
                f.write(f"# {timing_desc}\n")
                f.write("#" + "-" * 50 + "\n")
                for start, end in segment_timings:
                    f.write(f"{start},{end}\n")

            # Just save embedding shape info since embeddings are large
            emb_txt = os.path.join(output_dir, "04_embeddings_shape.txt")
            self.io_handler.save_embedding_info(embeddings, segment_timings, emb_txt)

        if len(embeddings) == 0:
            self.logger.warning("Failed to extract any speaker embeddings")
            return []

        # 5. Enhanced clustering to determine speakers
        speaker_labels = self.clusterer.cluster_speakers(
            embeddings, num_speakers, show_progress
        )

        if save_steps:
            labels_txt = os.path.join(output_dir, "05_speaker_labels.txt")
            labels_desc = "Speaker Clustering Results\nEach line represents the speaker label for the corresponding segment.\nThese labels correspond to the segments in 04_segment_timings.txt."
            self.io_handler.save_to_txt(speaker_labels, labels_txt, labels_desc)

        # 6. Detect overlapped speech
        overlap_segments = self.overlap_detector.detect_overlapped_speech(
            waveform[0], sample_rate, segment_timings
        )

        if save_steps:
            overlap_txt = os.path.join(output_dir, "06_overlap_segments.txt")
            overlap_desc = "Overlapped Speech Segment Indices\nIndices of segments where multiple speakers are detected speaking simultaneously."
            self.io_handler.save_to_txt(overlap_segments, overlap_txt, overlap_desc)

        if show_progress and overlap_segments:
            print(f"Detected {len(overlap_segments)} potentially overlapped segments")

        # 7. Create final speaker segments with overlap information
        speaker_segments = self.segment_processor.create_speaker_segments(
            segment_timings, speaker_labels
        )

        if save_steps:
            segments_txt = os.path.join(output_dir, "07_speaker_segments.txt")
            segments_desc = "Final Speaker Segments (without overlap information)\nFormat: start_time,end_time,speaker_id,is_overlap"
            self.io_handler.save_segments(speaker_segments, segments_txt, segments_desc)

        # 8. Add overlap information to segments
        for overlap_idx in overlap_segments:
            if overlap_idx < len(speaker_segments):
                speaker_segments[overlap_idx].is_overlap = True

                # Try to detect which speakers are in the overlap
                if overlap_idx > 0 and overlap_idx < len(speaker_segments) - 1:
                    prev_speaker = speaker_segments[overlap_idx - 1].speaker
                    next_speaker = speaker_segments[overlap_idx + 1].speaker

                    if prev_speaker != next_speaker:
                        speaker_segments[overlap_idx].overlap_speakers = [
                            prev_speaker,
                            next_speaker,
                        ]

        if save_steps:
            final_txt = os.path.join(output_dir, "08_final_segments.txt")
            final_desc = "Final Speaker Segments with Overlap Information\nFormat: start_time,end_time,speaker_id,is_overlap,overlap_speakers"
            self.io_handler.save_segments(speaker_segments, final_txt, final_desc)

        if show_progress:
            print(
                f"Diarization complete: identified {len(speaker_segments)} speaker segments with {len(set(s.speaker for s in speaker_segments))} speakers"
            )

        return speaker_segments
