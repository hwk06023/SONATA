import torch
import numpy as np
import librosa
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore", message="Support for mismatched key_padding_mask and attn_mask"
)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    score: float = 1.0


class SpeakerDiarizer:
    def __init__(self, device="cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._load_models()

    def _load_models(self):
        self.logger.info("Loading speaker diarization models...")

        # Load Silero VAD for voice activity detection
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        self.vad_get_speech_timestamps = utils[0]
        self.vad_model.to(self.device)

        # Load WavLM for speaker embeddings
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.wavlm_model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.wavlm_model.to(self.device)

    def _get_vad_segments(self, waveform, sample_rate, show_progress=True):
        """Get voice activity segments using Silero VAD"""
        if show_progress:
            print("Running voice activity detection...")

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        speech_timestamps = self.vad_get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            window_size_samples=512,
            speech_pad_ms=150,
            threshold=0.5,
        )

        segments = []
        for seg in speech_timestamps:
            start = seg["start"] / sample_rate
            end = seg["end"] / sample_rate
            segments.append((start, end))

        if show_progress:
            print(f"Found {len(segments)} speech segments")

        return segments

    def _extract_embeddings(self, waveform, sample_rate, segments, show_progress=True):
        """Extract speaker embeddings using WavLM"""
        if show_progress:
            print("Extracting speaker embeddings...")

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # Convert to mono if stereo
        if isinstance(waveform, torch.Tensor) and waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)
        elif isinstance(waveform, np.ndarray) and waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)

        embeddings = []
        iterator = segments
        if show_progress:
            iterator = tqdm(segments, desc="Processing segments", unit="segment")

        for start, end in iterator:
            # Extract segment audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            # Skip segments that are too short
            if end_sample - start_sample < 0.5 * sample_rate:
                continue

            segment_waveform = waveform[start_sample:end_sample]

            # Get embedding from WavLM
            if isinstance(segment_waveform, np.ndarray):
                segment_waveform = torch.from_numpy(segment_waveform)

            segment_waveform = segment_waveform.to(self.device)

            # Prepare input for WavLM
            inputs = self.wavlm_processor(
                segment_waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).to(self.device)

            # Get speaker embedding
            with torch.no_grad():
                embeds = self.wavlm_model(**inputs).embeddings
                embedding = embeds.cpu().numpy().mean(axis=1)

            embeddings.append(
                {"start": start, "end": end, "embedding": embedding.flatten()}
            )

        return embeddings

    def _cluster_speakers(self, embeddings, num_speakers=None, show_progress=True):
        """Cluster embeddings to identify speakers"""
        if show_progress:
            print("Clustering speakers...")

        if not embeddings:
            return []

        # Extract embedding vectors
        embedding_vectors = np.array([e["embedding"] for e in embeddings])

        # Estimate number of speakers if not provided
        if num_speakers is None:
            # Simple estimation based on embedding distances
            from sklearn.metrics import silhouette_score

            max_speakers = min(len(embedding_vectors), 10)
            best_score = -1
            best_n = 2  # Default to 2 speakers

            for n in range(2, max_speakers + 1):
                if len(embedding_vectors) < n:
                    continue

                clustering = AgglomerativeClustering(n_clusters=n)
                labels = clustering.fit_predict(embedding_vectors)

                if len(np.unique(labels)) < 2:
                    continue

                try:
                    score = silhouette_score(embedding_vectors, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                except:
                    continue

            num_speakers = best_n
            print(f"Estimated number of speakers: {num_speakers}")

        # Perform speaker clustering
        clustering = AgglomerativeClustering(n_clusters=num_speakers)
        labels = clustering.fit_predict(embedding_vectors)

        # Assign speaker labels to segments
        speaker_segments = []
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            speaker_segments.append(
                SpeakerSegment(
                    start=emb["start"],
                    end=emb["end"],
                    speaker=f"SPEAKER_{label+1}",
                    score=1.0,
                )
            )

        return speaker_segments

    def diarize(self, audio_path, num_speakers=None, show_progress=True):
        """Perform speaker diarization on an audio file"""
        if show_progress:
            print(f"Diarizing audio file: {audio_path}")

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.dim() > 1:
            # Convert stereo to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True).squeeze(0)

        # Get voice activity segments
        vad_segments = self._get_vad_segments(waveform, sample_rate, show_progress)

        # Extract speaker embeddings
        embeddings = self._extract_embeddings(
            waveform, sample_rate, vad_segments, show_progress
        )

        # Cluster speakers
        speaker_segments = self._cluster_speakers(
            embeddings, num_speakers, show_progress
        )

        return speaker_segments
