import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from typing import List, Dict, Optional, Tuple, Union
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass
import logging
import os
from tqdm import tqdm


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    score: float = 1.0


class CustomDiarizer:
    def __init__(self, device="cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
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

        # 2. WavLM XVector for speaker embeddings
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
        )

        segments = []
        for seg in speech_timestamps:
            start = seg["start"] / sample_rate
            end = seg["end"] / sample_rate
            segments.append((start, end))

        if show_progress:
            print(f"Found {len(segments)} speech segments")

        return segments

    def _detect_speaker_changes(
        self,
        waveform,
        sample_rate,
        vad_segments,
        window_size=1.0,
        hop_size=0.5,
        show_progress=True,
    ):
        """Detect speaker changes within VAD segments"""
        if show_progress:
            print("Detecting speaker changes...")

        changes = []

        # Create iterator with progress bar if needed
        iterator = vad_segments
        if show_progress:
            iterator = tqdm(vad_segments, desc="Processing segments", unit="segment")

        for start, end in iterator:
            if end - start < window_size:
                continue

            # Extract segment waveform
            segment_start_sample = int(start * sample_rate)
            segment_end_sample = int(end * sample_rate)
            segment_waveform = waveform[segment_start_sample:segment_end_sample]

            # Calculate features
            if isinstance(segment_waveform, torch.Tensor):
                segment_waveform = segment_waveform.cpu().numpy()

            mfccs = librosa.feature.mfcc(y=segment_waveform, sr=sample_rate, n_mfcc=20)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            features = np.concatenate([mfccs, delta, delta2])

            # Bayesian Information Criterion for change detection
            for t in np.arange(window_size, end - start - window_size, hop_size):
                t_sample = int(t * sample_rate / sample_rate * features.shape[1])
                t_sample = min(t_sample, features.shape[1] - 1)

                bic_score = self._compute_bic(features, t_sample)
                if bic_score > 0:  # Positive BIC indicates change point
                    changes.append(start + t)

        if show_progress:
            print(f"Detected {len(changes)} speaker change points")

        return sorted(changes)

    def _compute_bic(self, features, change_point):
        """Compute Bayesian Information Criterion for change detection"""
        n_samples = features.shape[1]
        n_features = features.shape[0]

        # Ensure we have enough samples on each side
        if change_point < 2 or n_samples - change_point < 2:
            return -np.inf

        # Split features at change point
        part1 = features[:, :change_point]
        part2 = features[:, change_point:]

        try:
            # Calculate covariances
            cov = np.cov(features)
            cov1 = np.cov(part1)
            cov2 = np.cov(part2)

            # Add small constant to avoid singularity
            eps = 1e-6
            cov += eps * np.eye(cov.shape[0])
            cov1 += eps * np.eye(cov1.shape[0])
            cov2 += eps * np.eye(cov2.shape[0])

            # BIC calculation
            n1 = part1.shape[1]
            n2 = part2.shape[1]

            bic = 0.5 * (
                n_samples * np.log(np.linalg.det(cov))
                - (n1 * np.log(np.linalg.det(cov1)) + n2 * np.log(np.linalg.det(cov2)))
            )

            # Penalty term
            penalty = (
                0.5
                * (n_features + 0.5 * n_features * (n_features + 1))
                * np.log(n_samples)
            )

            return bic - penalty
        except:
            return -np.inf

    def _extract_embeddings(self, waveform, sample_rate, segments, show_progress=True):
        """Extract WavLM speaker embeddings for each segment"""
        if show_progress:
            print("Extracting speaker embeddings...")

        embeddings = []
        timings = []

        # Create iterator with progress bar if needed
        iterator = segments
        if show_progress:
            iterator = tqdm(segments, desc="Processing segments", unit="segment")

        for start, end in iterator:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            # Handle edge case
            if (
                start_sample >= end_sample
                or start_sample >= len(waveform)
                or end_sample > len(waveform)
            ):
                continue

            segment_waveform = waveform[start_sample:end_sample]

            # Resample if needed
            if sample_rate != 16000:
                if isinstance(segment_waveform, torch.Tensor):
                    segment_waveform = torchaudio.functional.resample(
                        segment_waveform, sample_rate, 16000
                    )
                else:
                    segment_waveform = librosa.resample(
                        segment_waveform, orig_sr=sample_rate, target_sr=16000
                    )

            # Process through WavLM
            try:
                if isinstance(segment_waveform, torch.Tensor):
                    segment_waveform = segment_waveform.cpu().numpy()

                inputs = self.wavlm_processor(
                    segment_waveform, sampling_rate=16000, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.wavlm_model(**inputs)
                    embedding = outputs.embeddings.cpu().numpy()

                embeddings.append(embedding.squeeze())
                timings.append((start, end))
            except Exception as e:
                self.logger.warning(
                    f"Failed to extract embedding for segment {start}-{end}: {str(e)}"
                )

        if show_progress:
            print(f"Extracted {len(embeddings)} speaker embeddings")

        return np.array(embeddings), timings

    def _cluster_speakers(self, embeddings, num_speakers=None, show_progress=True):
        """Cluster embeddings to determine speakers"""
        if show_progress:
            print("Clustering speaker embeddings...")

        if embeddings.size == 0:
            return []

        if num_speakers is None:
            # Estimate number of speakers if not provided
            num_speakers = max(2, min(8, int(np.sqrt(len(embeddings)) / 2)))

        if show_progress:
            print(f"Clustering with {num_speakers} speakers")

        # Normalize embeddings
        norm_embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Perform clustering with proper parameters
        try:
            # First try with sklearn's recommended parameters
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers, metric="cosine", linkage="average"
            )
            labels = clustering.fit_predict(norm_embeddings)
        except Exception as e:
            self.logger.warning(f"Cosine clustering failed: {str(e)}")

            try:
                # Try with connectivity matrix for better performance
                from sklearn.neighbors import kneighbors_graph

                connectivity = kneighbors_graph(
                    norm_embeddings,
                    n_neighbors=min(len(norm_embeddings) // 2, 10),
                    include_self=False,
                )

                clustering = AgglomerativeClustering(
                    n_clusters=num_speakers, connectivity=connectivity
                )
                labels = clustering.fit_predict(norm_embeddings)
            except Exception as e2:
                self.logger.warning(f"Connectivity clustering failed: {str(e2)}")

                # Final fallback to the most basic clustering
                clustering = AgglomerativeClustering(
                    n_clusters=num_speakers,
                    linkage="ward",  # The default linkage method (works with euclidean)
                )
                labels = clustering.fit_predict(norm_embeddings)

        return labels

    def _create_speaker_segments(self, segment_timings, speaker_labels):
        """Create final speaker segments from segment timings and clustering"""
        if len(segment_timings) == 0 or len(speaker_labels) == 0:
            return []

        segments = []

        for i, ((start, end), label) in enumerate(zip(segment_timings, speaker_labels)):
            speaker = f"SPEAKER_{int(label):02d}"
            segments.append(SpeakerSegment(start, end, speaker))

        return segments

    def diarize(self, audio_path, num_speakers=None, show_progress=True):
        """Main diarization method"""
        if show_progress:
            print(f"Starting diarization for: {audio_path}")

        # 1. Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if needed

        # 2. VAD to get speech segments
        vad_segments = self._get_vad_segments(waveform[0], sample_rate, show_progress)

        if len(vad_segments) == 0:
            self.logger.warning("No speech segments detected in audio")
            return []

        # 3. Detect speaker changes within VAD segments
        change_points = self._detect_speaker_changes(
            waveform[0], sample_rate, vad_segments, show_progress=show_progress
        )

        # 4. Create segment boundaries from VAD and change points
        all_boundaries = sorted(
            list(
                set(
                    [s[0] for s in vad_segments]
                    + [s[1] for s in vad_segments]
                    + change_points
                )
            )
        )

        # 5. Create analysis segments
        analysis_segments = []
        for i in range(len(all_boundaries) - 1):
            analysis_segments.append((all_boundaries[i], all_boundaries[i + 1]))

        # 6. Extract speaker embeddings for each segment
        embeddings, segment_timings = self._extract_embeddings(
            waveform[0], sample_rate, analysis_segments, show_progress
        )

        if len(embeddings) == 0:
            self.logger.warning("Failed to extract any speaker embeddings")
            return []

        # 7. Cluster to determine speakers
        speaker_labels = self._cluster_speakers(embeddings, num_speakers, show_progress)

        # 8. Create final speaker segments
        speaker_segments = self._create_speaker_segments(
            segment_timings, speaker_labels
        )

        if show_progress:
            print(
                f"Diarization complete: identified {len(speaker_segments)} speaker segments"
            )

        return speaker_segments
