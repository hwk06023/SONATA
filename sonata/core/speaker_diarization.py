import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from typing import List, Dict, Optional, Tuple, Union
import torchaudio
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from dataclasses import dataclass
import logging
import os
from tqdm import tqdm
import warnings
from scipy.spatial.distance import cosine
from scipy import signal
from sklearn.decomposition import PCA

# Filter PyTorch transformer attention warnings
warnings.filterwarnings(
    "ignore", message="Support for mismatched key_padding_mask and attn_mask"
)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    score: float = 1.0
    is_overlap: bool = False
    overlap_speakers: List[str] = None


class SpeakerDiarizer:
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

        # 3. Load ECAPA-TDNN as fallback
        try:
            import speechbrain as sb

            self.ecapa_model = sb.pretrained.EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
            self.has_ecapa_model = True
        except Exception as e:
            self.logger.warning(f"Could not load ECAPA-TDNN model: {str(e)}")
            self.has_ecapa_model = False

    def _get_vad_segments(self, waveform, sample_rate, show_progress=True):
        """Get voice activity segments using Silero VAD with WavLM-optimized parameters"""
        if show_progress:
            print("Running voice activity detection...")

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # Apply normalization (crucial for accurate VAD)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # WavLM-optimized VAD parameters
        speech_timestamps = self.vad_get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=250,  # Optimized for clear speech detection
            min_silence_duration_ms=300,  # Reduced for better segmentation
            window_size_samples=512,
            speech_pad_ms=150,  # Better boundary precision
            threshold=0.3,  # Better precision for WavLM
        )

        segments = []
        for seg in speech_timestamps:
            start = seg["start"] / sample_rate
            end = seg["end"] / sample_rate
            segments.append((start, end))

        # Merge segments that are very close
        if len(segments) > 0:
            # 400ms gap threshold works well with WavLM
            merged_segments = self._merge_close_segments(segments, gap_threshold=0.4)
        else:
            merged_segments = []

        if show_progress:
            print(f"Found {len(merged_segments)} speech segments")

        return merged_segments

    def _merge_close_segments(self, segments, gap_threshold=0.5):
        """Merge segments that are separated by small gaps"""
        if not segments:
            return []

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x[0])

        merged = []
        current_start, current_end = sorted_segments[0]

        for start, end in sorted_segments[1:]:
            # If this segment starts soon after the previous one ends
            if start - current_end <= gap_threshold:
                # Extend the current segment
                current_end = end
            else:
                # Add the current segment to results and start a new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last segment
        merged.append((current_start, current_end))

        return merged

    def _detect_speaker_changes(
        self,
        waveform,
        sample_rate,
        vad_segments,
        window_size=0.75,
        hop_size=0.35,
        show_progress=True,
        batch_size=8,
    ):
        """Detect speaker changes within VAD segments using WavLM embeddings"""
        if show_progress:
            print("Detecting speaker changes...")

        changes = []

        for start, end in vad_segments:
            # Skip if segment is too short for analysis
            if end - start < window_size * 2:
                continue

            # Extract segment waveform
            segment_start_sample = int(start * sample_rate)
            segment_end_sample = int(end * sample_rate)
            segment_waveform = waveform[segment_start_sample:segment_end_sample]

            # Apply efficient embedding-based change detection
            segment_changes = self._detect_changes_with_embeddings(
                segment_waveform, sample_rate, window_size, hop_size, batch_size
            )

            # Convert local changes to global timeline
            segment_changes = [start + t for t in segment_changes]
            changes.extend(segment_changes)

        # Final filtering to remove duplicate change points
        if len(changes) > 1:
            changes.sort()
            filtered_changes = [changes[0]]
            for change in changes[1:]:
                # Only add if sufficiently distant from the last added change point
                if change - filtered_changes[-1] > 0.5:  # 500ms minimum
                    filtered_changes.append(change)
            changes = filtered_changes

        if show_progress:
            print(f"Detected {len(changes)} speaker change points")

        return sorted(changes)

    def _detect_changes_with_embeddings(
        self, waveform, sample_rate, window_size, hop_size, batch_size=8
    ):
        """Detect speaker changes using embedding similarity with WavLM batch processing"""
        changes = []
        duration = len(waveform) / sample_rate

        # Skip if segment is too short
        if duration < window_size * 2:
            return []

        # Create sliding windows with 50% overlap for better detection
        windows = []
        for t in np.arange(0, duration - window_size, hop_size * 0.5):
            start_sample = int(t * sample_rate)
            end_sample = int((t + window_size) * sample_rate)
            if end_sample <= len(waveform):
                windows.append((t, t + window_size, waveform[start_sample:end_sample]))

        # Skip if too few windows
        if len(windows) < 3:
            return []

        # Prepare window audio data for batch processing
        window_waveforms = []

        for _, _, window_samples in windows:
            # Convert to numpy if tensor
            if isinstance(window_samples, torch.Tensor):
                window_samples_np = window_samples.cpu().numpy()
            else:
                window_samples_np = window_samples

            # Normalize audio (crucial for WavLM)
            window_samples_np = window_samples_np / (
                np.max(np.abs(window_samples_np)) + 1e-8
            )

            # Add padding for very short segments (improve embedding quality)
            if (
                len(window_samples_np) < 0.8 * sample_rate
                and len(window_samples_np) >= 0.3 * sample_rate
            ):
                pad_length = min(int(0.1 * sample_rate), len(window_samples_np) // 4)
                window_samples_np = np.pad(
                    window_samples_np, (pad_length, pad_length), mode="reflect"
                )

            # Resample if needed
            if sample_rate != 16000:
                window_samples_np = librosa.resample(
                    window_samples_np, orig_sr=sample_rate, target_sr=16000
                )

            window_waveforms.append(window_samples_np)

        # Process in batches to improve performance
        embeddings = [None] * len(windows)  # Pre-allocate with None values
        batch_count = (len(window_waveforms) + batch_size - 1) // batch_size

        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(window_waveforms))
            batch = window_waveforms[start_idx:end_idx]

            try:
                # Process batch with WavLM
                inputs = self.wavlm_processor(
                    batch, sampling_rate=16000, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.wavlm_model(**inputs)
                    # Get embeddings and normalize
                    batch_embeddings = outputs.embeddings
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, dim=-1
                    )
                    batch_embeddings = batch_embeddings.cpu().numpy()

                    # Store embeddings at correct positions
                    for i in range(len(batch)):
                        embeddings[start_idx + i] = batch_embeddings[i]
            except Exception as e:
                # Process individually on failure
                for i, waveform_np in enumerate(batch):
                    try:
                        inputs = self.wavlm_processor(
                            waveform_np, sampling_rate=16000, return_tensors="pt"
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.wavlm_model(**inputs)
                            embedding = outputs.embeddings
                            embedding = torch.nn.functional.normalize(embedding, dim=-1)
                            embedding = embedding.cpu().numpy().squeeze()
                            embeddings[start_idx + i] = embedding
                    except Exception:
                        pass  # Keep None at this position

        # Check for distance between adjacent windows using cosine similarity
        if len(embeddings) > 2:
            distances = []
            for i in range(len(embeddings) - 1):
                if embeddings[i] is not None and embeddings[i + 1] is not None:
                    # Compute cosine distance between normalized embeddings
                    distances.append(cosine(embeddings[i], embeddings[i + 1]))

            # Use fixed threshold optimized for WavLM (Microsoft recommendation is ~0.86 for similarity)
            # Since cosine returns distance (1-similarity), threshold is 1-0.86 = 0.14
            threshold = 0.14  # Optimized for WavLM

            # Simple and effective change point detection
            changes = []
            for i in range(1, len(embeddings) - 1):
                if embeddings[i - 1] is None or embeddings[i + 1] is None:
                    continue

                # Compute distances
                prev_dist = cosine(embeddings[i - 1], embeddings[i])
                next_dist = cosine(embeddings[i], embeddings[i + 1])

                # Change point detection using Microsoft's recommended threshold
                if prev_dist > threshold and next_dist > threshold:
                    # Calculate midpoint time for the change point
                    midpoint = (windows[i][0] + windows[i][1]) / 2
                    changes.append(midpoint)

            # Apply simple filtering to avoid duplicate change points
            if len(changes) > 1:
                filtered_changes = [changes[0]]
                for change in changes[1:]:
                    # Only add if sufficiently distant from the last added change point
                    if change - filtered_changes[-1] > 0.5:  # 500ms minimum spacing
                        filtered_changes.append(change)
                changes = filtered_changes

        return changes

    def _cluster_change_points(self, change_points, threshold=0.35):
        """Cluster change points that are close to each other"""
        if not change_points:
            return []

        # Sort change points
        sorted_changes = sorted(change_points)

        # Cluster close change points
        clusters = []
        current_cluster = [sorted_changes[0]]

        for point in sorted_changes[1:]:
            if point - current_cluster[-1] < threshold:
                current_cluster.append(point)
            else:
                # Add the average of the current cluster
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [point]

        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

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

            # BIC calculation with improved penalty weight
            n1 = part1.shape[1]
            n2 = part2.shape[1]

            bic = 0.5 * (
                n_samples * np.log(np.linalg.det(cov))
                - (n1 * np.log(np.linalg.det(cov1)) + n2 * np.log(np.linalg.det(cov2)))
            )

            # Modified penalty term with tuned lambda factor
            lambda_factor = 1.0  # Can be tuned between 0.5-1.5
            penalty = (
                lambda_factor
                * 0.5
                * (n_features + 0.5 * n_features * (n_features + 1))
                * np.log(n_samples)
            )

            return bic - penalty
        except:
            return -np.inf

    def _extract_embeddings_batch(
        self, waveform, sample_rate, segments, show_progress=True, batch_size=8
    ):
        """Extract speaker embeddings using WavLM with efficient batch processing"""
        if show_progress:
            print("Extracting speaker embeddings in batches...")

        embeddings = []
        timings = []

        # Prepare batches of segments
        valid_segments = []
        segment_waveforms = []

        for start, end in segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            # Handle edge cases
            if (
                start_sample >= end_sample
                or start_sample >= len(waveform)
                or end_sample > len(waveform)
            ):
                continue

            segment_waveform = waveform[start_sample:end_sample]
            duration = (end_sample - start_sample) / sample_rate

            # Skip segments that are too short
            if duration < 0.3:  # Minimum 300ms
                continue

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

            # Normalize audio
            if isinstance(segment_waveform, torch.Tensor):
                segment_waveform_np = segment_waveform.cpu().numpy()
            else:
                segment_waveform_np = segment_waveform

            segment_waveform_np = segment_waveform_np / (
                np.max(np.abs(segment_waveform_np)) + 1e-8
            )

            segment_waveforms.append(segment_waveform_np)
            valid_segments.append((start, end))

        if not valid_segments:
            return np.array([]), []

        # Process in batches
        total_segments = len(valid_segments)
        batch_count = (
            total_segments + batch_size - 1
        ) // batch_size  # Ceiling division

        if show_progress:
            print(f"Processing {total_segments} segments in {batch_count} batches")

        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_segments)

            batch_waveforms = segment_waveforms[start_idx:end_idx]

            try:
                # Process batch with WavLM
                inputs = self.wavlm_processor(
                    batch_waveforms,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.wavlm_model(**inputs)
                    # Get embeddings and normalize
                    batch_embeddings = outputs.embeddings
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, dim=-1
                    )
                    batch_embeddings = batch_embeddings.cpu().numpy()

                    # Add embeddings and corresponding timings
                    for i in range(len(batch_waveforms)):
                        embeddings.append(batch_embeddings[i])
                        timings.append(valid_segments[start_idx + i])
            except Exception as e:
                self.logger.warning(f"Failed to process batch {batch_idx}: {str(e)}")
                # Process segments individually as fallback
                for i, waveform_np in enumerate(batch_waveforms):
                    try:
                        inputs = self.wavlm_processor(
                            waveform_np, sampling_rate=16000, return_tensors="pt"
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.wavlm_model(**inputs)
                            embedding = outputs.embeddings
                            embedding = torch.nn.functional.normalize(embedding, dim=-1)
                            embedding = embedding.cpu().numpy().squeeze()
                            embeddings.append(embedding)
                            timings.append(valid_segments[start_idx + i])
                    except Exception as e:
                        self.logger.warning(
                            f"Failed individual embedding extraction: {str(e)}"
                        )

        if show_progress:
            print(f"Successfully extracted {len(embeddings)} speaker embeddings")

        return np.array(embeddings), timings

    def diarize(self, audio_path, num_speakers=None, show_progress=True, batch_size=8):
        """Speaker diarization optimized for WavLM embeddings with batch processing"""
        if show_progress:
            print(f"Starting WavLM-optimized diarization for: {audio_path}")

        # 1. Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if needed

        # 2. Resample to 16kHz if needed (WavLM expects 16kHz)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # 3. Normalize audio volume (important for WavLM)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # 4. VAD to get speech segments
        vad_segments = self._get_vad_segments(waveform[0], sample_rate, show_progress)

        if len(vad_segments) == 0:
            self.logger.warning("No speech segments detected in audio")
            return []

        # 5. Speaker change detection
        change_points = self._detect_speaker_changes(
            waveform[0], sample_rate, vad_segments, show_progress=show_progress
        )

        # 6. Create segment boundaries from VAD and change points
        all_boundaries = sorted(
            list(
                set(
                    [s[0] for s in vad_segments]
                    + [s[1] for s in vad_segments]
                    + change_points
                )
            )
        )

        # 7. Create analysis segments
        analysis_segments = []
        for i in range(len(all_boundaries) - 1):
            analysis_segments.append((all_boundaries[i], all_boundaries[i + 1]))

        # 8. Extract speaker embeddings using WavLM with batch processing
        embeddings, segment_timings = self._extract_embeddings_batch(
            waveform[0],
            sample_rate,
            analysis_segments,
            show_progress,
            batch_size=batch_size,
        )

        if len(embeddings) == 0:
            self.logger.warning("Failed to extract any speaker embeddings")
            return []

        # 9. Determine speakers through clustering
        speaker_labels = self._cluster_speakers(embeddings, num_speakers, show_progress)

        # 10. Create final speaker segments
        speaker_segments = self._create_speaker_segments(
            segment_timings, speaker_labels
        )

        if show_progress:
            print(
                f"Diarization complete: identified {len(speaker_segments)} speaker segments with {len(set(s.speaker for s in speaker_segments))} speakers"
            )

        return speaker_segments

    def _cluster_speakers(self, embeddings, num_speakers=None, show_progress=True):
        """Speaker clustering optimized for WavLM embeddings"""
        if show_progress:
            print("Clustering speaker embeddings...")

        if embeddings.size == 0:
            return []

        # Estimate number of speakers if not provided
        if num_speakers is None:
            # More sophisticated estimation based on eigenvalues
            estimated_speakers = self._estimate_num_speakers(embeddings, show_progress)
            num_speakers = estimated_speakers
            if show_progress:
                print(f"Estimated {num_speakers} speakers based on eigenvalue analysis")

        # Cap number of speakers to reasonable range
        num_speakers = max(2, min(num_speakers, min(8, len(embeddings) // 2)))

        if show_progress:
            print(f"Clustering with {num_speakers} speakers")

        # For WavLM embeddings, the vectors should already be normalized
        # Apply simple but effective clustering specifically optimized for WavLM
        try:
            # Use Agglomerative Clustering with cosine distance - ideal for WavLM
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                affinity="cosine",
                linkage="average",  # Average linkage works best with WavLM embeddings
            )
            labels = clustering.fit_predict(embeddings)
            clustering_method = "Agglomerative (Cosine)"
        except Exception as e:
            # Fallback to basic clustering if cosine fails
            try:
                clustering = AgglomerativeClustering(n_clusters=num_speakers)
                labels = clustering.fit_predict(embeddings)
                clustering_method = "Agglomerative (Basic)"
            except Exception as last_error:
                if show_progress:
                    print(
                        f"All clustering methods failed. Last error: {str(last_error)}"
                    )
                # Create simple labels if everything fails
                labels = np.zeros(len(embeddings), dtype=int)
                for i in range(1, min(num_speakers, len(embeddings))):
                    if i < len(embeddings):
                        labels[i] = i % num_speakers
                clustering_method = "Emergency Fallback"

        if show_progress:
            print(f"Speaker clustering complete using {clustering_method}")

        # Create speaker labels with proper format
        speaker_labels = [f"SPEAKER_{int(label):02d}" for label in labels]

        return speaker_labels

    def _estimate_num_speakers(self, embeddings, show_progress=True):
        """Estimate number of speakers using eigenvalue analysis"""
        try:
            # Normalize embeddings
            norm_embeddings = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            )

            # Compute similarity matrix
            similarity_matrix = 1 - np.array(
                [
                    [cosine(emb1, emb2) for emb2 in norm_embeddings]
                    for emb1 in norm_embeddings
                ]
            )

            # Apply adaptive threshold to create affinity matrix
            threshold = np.mean(similarity_matrix) * 0.5
            affinity_matrix = (similarity_matrix > threshold).astype(float)

            # Compute Laplacian
            from sklearn.cluster import SpectralClustering
            from scipy import sparse

            if not sparse.issparse(affinity_matrix):
                affinity_matrix = sparse.csr_matrix(affinity_matrix)

            laplacian = SpectralClustering(
                n_clusters=2, affinity="precomputed"
            )._get_laplacian(affinity_matrix)

            # Get eigenvalues
            from scipy.sparse.linalg import eigsh

            eigenvalues, _ = eigsh(
                laplacian, k=min(10, laplacian.shape[0] - 1), which="SM"
            )

            # Find the elbow point in eigenvalues
            eigenvalues = sorted(eigenvalues)
            diffs = np.diff(eigenvalues)

            # Find largest gap in eigenvalues
            largest_gap_idx = np.argmax(diffs) + 1

            # Estimate is the index of largest gap + 1 (since we're looking at gaps)
            estimated_speakers = largest_gap_idx + 1

            # Ensure we're within reasonable bounds
            estimated_speakers = max(2, min(8, estimated_speakers))

            return estimated_speakers
        except Exception as e:
            if show_progress:
                print(f"Error estimating speaker count: {str(e)}")
            # Default fallback
            return max(2, min(3, len(embeddings) // 20))

    def _detect_overlapped_speech(self, waveform, sample_rate, segments):
        """Detect segments with overlapped speech"""
        overlap_segments = []

        # Skip if too little data
        if len(segments) < 3:
            return []

        try:
            # Extract features for each segment
            for i, (start, end) in enumerate(segments):
                # Skip if segment is too short
                if end - start < 0.5:
                    continue

                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)

                if end_sample <= start_sample or end_sample > len(waveform):
                    continue

                segment_audio = waveform[start_sample:end_sample]

                # Convert to numpy if needed
                if isinstance(segment_audio, torch.Tensor):
                    segment_audio = segment_audio.cpu().numpy()

                # Calculate spectral flatness
                stft = np.abs(librosa.stft(segment_audio))
                flatness = librosa.feature.spectral_flatness(S=stft)[0]
                flatness_mean = np.mean(flatness)

                # Calculate harmonic-percussive separation (useful for overlap detection)
                harmonic, percussive = librosa.effects.hpss(segment_audio)
                hp_ratio = np.mean(np.abs(harmonic)) / (
                    np.mean(np.abs(percussive)) + 1e-8
                )

                # Spectral centroid variation
                centroid = librosa.feature.spectral_centroid(
                    y=segment_audio, sr=sample_rate
                )[0]
                centroid_std = np.std(centroid)

                # Compute "complexity score" - higher means more likely to have overlaps
                complexity_score = (
                    (centroid_std / 1000) * (1 - flatness_mean) * (1 + hp_ratio)
                )

                # Segments with high complexity and low flatness are often overlaps
                if flatness_mean < 0.08 and complexity_score > 0.5:
                    overlap_segments.append(i)
        except Exception as e:
            self.logger.warning(f"Overlap detection failed: {str(e)}")

        return overlap_segments

    def _create_speaker_segments(self, segment_timings, speaker_labels):
        """Create final speaker segments from segment timings and clustering"""
        if len(segment_timings) == 0 or len(speaker_labels) == 0:
            return []

        segments = []

        for i, ((start, end), label) in enumerate(zip(segment_timings, speaker_labels)):
            speaker = label if isinstance(label, str) else f"SPEAKER_{int(label):02d}"
            segments.append(SpeakerSegment(start, end, speaker))

        # Sort segments by start time
        segments = sorted(segments, key=lambda s: s.start)

        # Merge very short segments with the same speaker
        merged_segments = []
        if len(segments) > 1:
            current = segments[0]

            for next_seg in segments[1:]:
                # If same speaker and short gap
                if (
                    next_seg.speaker == current.speaker
                    and next_seg.start - current.end < 0.3
                    and next_seg.start - current.end >= 0
                ):
                    # Merge them
                    current.end = next_seg.end
                else:
                    # If significant gap or different speaker, add current and start new
                    merged_segments.append(current)
                    current = next_seg

            # Add the last segment
            merged_segments.append(current)
        else:
            merged_segments = segments

        return merged_segments
