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

        # 3. Load ECAPA-TDNN for better embeddings
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
        """Get voice activity segments using Silero VAD with enhanced parameters"""
        if show_progress:
            print("Running voice activity detection...")

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # Use more sensitive parameters for better recall
        speech_timestamps = self.vad_get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=200,  # Reduced from 250ms
            min_silence_duration_ms=400,  # Reduced from 500ms
            window_size_samples=512,
            speech_pad_ms=200,  # Increased padding
            threshold=0.3,  # Lower threshold for better recall
        )

        segments = []
        for seg in speech_timestamps:
            start = seg["start"] / sample_rate
            end = seg["end"] / sample_rate
            segments.append((start, end))

        # Merge segments that are very close
        merged_segments = self._merge_close_segments(segments, gap_threshold=0.5)

        if show_progress:
            print(f"Found {len(merged_segments)} speech segments after merging")

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
        window_size=0.75,  # Reduced for more precision
        hop_size=0.35,  # Reduced for more granularity
        show_progress=True,
    ):
        """Detect speaker changes within VAD segments with improved algorithms"""
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

            # Enhanced feature extraction with more coefficients and deltas
            mfccs = librosa.feature.mfcc(y=segment_waveform, sr=sample_rate, n_mfcc=24)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            features = np.concatenate([mfccs, delta, delta2])

            # Add spectral features for better discrimination
            spec_contrast = librosa.feature.spectral_contrast(
                y=segment_waveform, sr=sample_rate
            )
            features = np.concatenate([features, spec_contrast])

            # Multi-algorithm approach: BIC + Divergence + Embeddings
            # 1. BIC-based change detection
            bic_changes = []
            for t in np.arange(window_size, end - start - window_size, hop_size):
                t_sample = int(t * sample_rate / sample_rate * features.shape[1])
                t_sample = min(t_sample, features.shape[1] - 1)

                bic_score = self._compute_bic(features, t_sample)
                if bic_score > 0:  # Positive BIC indicates change point
                    bic_changes.append(start + t)

            # 2. Embedding-based change detection for longer segments
            if end - start > window_size * 3:
                emb_changes = self._detect_changes_with_embeddings(
                    segment_waveform, sample_rate, window_size, hop_size
                )

                # Convert local changes to global timeline
                emb_changes = [start + t for t in emb_changes]

                # Combine both methods with weighting
                all_changes = bic_changes + emb_changes

                # Cluster close change points to avoid duplicates
                changes.extend(self._cluster_change_points(all_changes, threshold=0.35))
            else:
                changes.extend(bic_changes)

        # Filter out changes too close to segment boundaries
        filtered_changes = []
        min_boundary_dist = 0.3

        for change in changes:
            is_near_boundary = False
            for start, end in vad_segments:
                if (
                    abs(change - start) < min_boundary_dist
                    or abs(change - end) < min_boundary_dist
                ):
                    is_near_boundary = True
                    break
            if not is_near_boundary:
                filtered_changes.append(change)

        if show_progress:
            print(f"Detected {len(filtered_changes)} speaker change points")

        return sorted(filtered_changes)

    def _detect_changes_with_embeddings(
        self, waveform, sample_rate, window_size, hop_size
    ):
        """Detect speaker changes using embedding similarity"""
        if not self.has_ecapa_model:
            return []

        changes = []
        duration = len(waveform) / sample_rate

        # Skip if segment is too short
        if duration < window_size * 2:
            return []

        # Create sliding windows
        windows = []
        for t in np.arange(0, duration - window_size, hop_size):
            start_sample = int(t * sample_rate)
            end_sample = int((t + window_size) * sample_rate)
            if end_sample <= len(waveform):
                windows.append((t, t + window_size, waveform[start_sample:end_sample]))

        # Skip if too few windows
        if len(windows) < 3:
            return []

        # Extract embeddings for each window
        embeddings = []
        for start_time, end_time, window_samples in windows:
            try:
                # Convert to tensor
                if not isinstance(window_samples, torch.Tensor):
                    window_tensor = torch.tensor(window_samples).float()
                else:
                    window_tensor = window_samples

                # Make mono and apply correct shape
                if len(window_tensor.shape) == 1:
                    window_tensor = window_tensor.unsqueeze(0)

                # Resample if needed
                if sample_rate != 16000:
                    window_tensor = torchaudio.functional.resample(
                        window_tensor, sample_rate, 16000
                    )

                with torch.no_grad():
                    embedding = self.ecapa_model.encode_batch(
                        window_tensor.to(self.device)
                    )
                    embedding = embedding.squeeze().cpu().numpy()
                    embeddings.append(embedding)
            except Exception as e:
                # If extraction fails, use a dummy embedding to maintain indices
                embeddings.append(None)

        # Check for distance between adjacent windows
        for i in range(1, len(windows) - 1):
            if embeddings[i - 1] is None or embeddings[i + 1] is None:
                continue

            # Compute distances
            prev_dist = cosine(embeddings[i - 1], embeddings[i])
            next_dist = cosine(embeddings[i], embeddings[i + 1])

            # Check if this is a likely change point
            if prev_dist > 0.15 and next_dist > 0.15:  # Tuned threshold
                midpoint = (windows[i][0] + windows[i][1]) / 2
                changes.append(midpoint)

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

    def _extract_embeddings(self, waveform, sample_rate, segments, show_progress=True):
        """Extract speaker embeddings for each segment with multiple models"""
        if show_progress:
            print("Extracting speaker embeddings...")

        embeddings = []
        timings = []
        wavlm_embeddings = []
        ecapa_embeddings = []

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

            # Skip segments that are too short
            duration = (end_sample - start_sample) / sample_rate
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

            wavlm_embedding = None
            ecapa_embedding = None

            # 1. Process with WavLM
            try:
                if isinstance(segment_waveform, torch.Tensor):
                    segment_waveform_np = segment_waveform.cpu().numpy()
                else:
                    segment_waveform_np = segment_waveform

                inputs = self.wavlm_processor(
                    segment_waveform_np, sampling_rate=16000, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.wavlm_model(**inputs)
                    wavlm_embedding = outputs.embeddings.cpu().numpy().squeeze()
                    wavlm_embeddings.append(wavlm_embedding)
            except Exception as e:
                self.logger.warning(
                    f"Failed to extract WavLM embedding for segment {start}-{end}: {str(e)}"
                )

            # 2. Process with ECAPA-TDNN if available
            if self.has_ecapa_model:
                try:
                    if not isinstance(segment_waveform, torch.Tensor):
                        segment_tensor = torch.tensor(segment_waveform).float()
                    else:
                        segment_tensor = segment_waveform

                    # Make mono and apply correct shape
                    if len(segment_tensor.shape) == 1:
                        segment_tensor = segment_tensor.unsqueeze(0)

                    with torch.no_grad():
                        ecapa_embedding = self.ecapa_model.encode_batch(
                            segment_tensor.to(self.device)
                        )
                        ecapa_embedding = ecapa_embedding.squeeze().cpu().numpy()
                        ecapa_embeddings.append(ecapa_embedding)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract ECAPA embedding for segment {start}-{end}: {str(e)}"
                    )

            # If either embedding was extracted, add the timing
            if wavlm_embedding is not None or ecapa_embedding is not None:
                timings.append((start, end))

        # Determine which embeddings to use based on availability
        if self.has_ecapa_model and len(ecapa_embeddings) == len(timings):
            # Prefer ECAPA-TDNN embeddings
            embeddings = ecapa_embeddings
            if show_progress:
                print(f"Using ECAPA-TDNN embeddings for {len(embeddings)} segments")
        elif len(wavlm_embeddings) == len(timings):
            # Fall back to WavLM embeddings
            embeddings = wavlm_embeddings
            if show_progress:
                print(f"Using WavLM embeddings for {len(embeddings)} segments")
        else:
            # If counts don't match, use available embeddings and adjust timings
            if len(ecapa_embeddings) > len(wavlm_embeddings):
                embeddings = ecapa_embeddings
                # Adjust timing list to match
                timings = timings[: len(embeddings)]
                if show_progress:
                    print(
                        f"Using partial ECAPA-TDNN embeddings ({len(embeddings)} out of {len(timings)} segments)"
                    )
            else:
                embeddings = wavlm_embeddings
                # Adjust timing list to match
                timings = timings[: len(embeddings)]
                if show_progress:
                    print(
                        f"Using partial WavLM embeddings ({len(embeddings)} out of {len(timings)} segments)"
                    )

        if show_progress:
            print(f"Extracted {len(embeddings)} speaker embeddings")

        return np.array(embeddings), timings

    def _cluster_speakers(self, embeddings, num_speakers=None, show_progress=True):
        """Enhanced clustering with multiple algorithms and automatic speaker count estimation"""
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

        # Normalize embeddings
        norm_embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Try multiple clustering methods with proper version handling
        clustering_methods = []

        # Check if scikit-learn supports all parameters (handle version compatibility)
        try:
            # Try creating with affinity and check if it raises an error
            test_clustering = AgglomerativeClustering(
                n_clusters=2, affinity="cosine", linkage="average"
            )
            # If no error, add the full method
            clustering_methods.append(
                {
                    "name": "Agglomerative (Cosine)",
                    "method": AgglomerativeClustering(
                        n_clusters=num_speakers, affinity="cosine", linkage="average"
                    ),
                }
            )
        except TypeError:
            # Fallback to simpler parameters
            clustering_methods.append(
                {
                    "name": "Agglomerative (Basic)",
                    "method": AgglomerativeClustering(n_clusters=num_speakers),
                }
            )

        # Try spectral clustering with similar version check
        try:
            clustering_methods.append(
                {
                    "name": "Spectral",
                    "method": SpectralClustering(
                        n_clusters=num_speakers,
                        affinity="nearest_neighbors",
                        n_neighbors=min(len(norm_embeddings) // 3, 10),
                        random_state=42,
                    ),
                }
            )
        except TypeError:
            # Fallback to simpler spectral clustering
            try:
                clustering_methods.append(
                    {
                        "name": "Spectral (Basic)",
                        "method": SpectralClustering(
                            n_clusters=num_speakers, random_state=42
                        ),
                    }
                )
            except:
                # Skip if not available
                pass

        best_labels = None
        best_score = -1
        best_method = None

        # Try each clustering method
        for method_info in clustering_methods:
            try:
                # Apply clustering
                labels = method_info["method"].fit_predict(norm_embeddings)

                # Skip if only one cluster was found
                if len(set(labels)) <= 1:
                    continue

                # Evaluate clustering quality
                try:
                    from sklearn.metrics import (
                        silhouette_score,
                        calinski_harabasz_score,
                    )

                    sil_score = silhouette_score(
                        norm_embeddings, labels, metric="cosine"
                    )
                    ch_score = calinski_harabasz_score(norm_embeddings, labels)

                    # Combined score (weighted average)
                    combined_score = (0.7 * sil_score) + (0.3 * (ch_score / 10000))

                    if show_progress:
                        print(
                            f"{method_info['name']}: silhouette={sil_score:.4f}, CH={ch_score:.4f}"
                        )

                    if combined_score > best_score:
                        best_score = combined_score
                        best_labels = labels
                        best_method = method_info["name"]
                except Exception as e:
                    if show_progress:
                        print(
                            f"Error evaluating clusters for {method_info['name']}: {str(e)}"
                        )
                    # Use these labels if we don't have any yet
                    if best_labels is None:
                        best_labels = labels
                        best_method = method_info["name"]
            except Exception as e:
                if show_progress:
                    print(f"Error with {method_info['name']} clustering: {str(e)}")

        if best_labels is None:
            # Fallback to simplest clustering
            try:
                # Most basic form that should work with any scikit-learn version
                clustering = AgglomerativeClustering(n_clusters=num_speakers)
                best_labels = clustering.fit_predict(norm_embeddings)
                best_method = "Fallback Agglomerative"
            except Exception as last_error:
                if show_progress:
                    print(
                        f"All clustering methods failed. Last error: {str(last_error)}"
                    )
                # Create simple labels if everything fails
                best_labels = np.zeros(len(norm_embeddings), dtype=int)
                for i in range(1, min(num_speakers, len(norm_embeddings))):
                    if i < len(norm_embeddings):
                        best_labels[i] = i % num_speakers
                best_method = "Emergency Fallback (Sequential Assignment)"

        if show_progress:
            print(f"Selected clustering method: {best_method}")

        # Create speaker labels with proper format
        labels = [f"SPEAKER_{int(label):02d}" for label in best_labels]

        # Return in the format expected by create_speaker_segments
        return labels

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

    def diarize(self, audio_path, num_speakers=None, show_progress=True):
        """Main diarization method with improved processing pipeline"""
        if show_progress:
            print(f"Starting enhanced diarization for: {audio_path}")

        # 1. Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if needed

        # 2. Enhanced VAD to get speech segments
        vad_segments = self._get_vad_segments(waveform[0], sample_rate, show_progress)

        if len(vad_segments) == 0:
            self.logger.warning("No speech segments detected in audio")
            return []

        # 3. Improved speaker change detection
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

        # 6. Extract enhanced speaker embeddings
        embeddings, segment_timings = self._extract_embeddings(
            waveform[0], sample_rate, analysis_segments, show_progress
        )

        if len(embeddings) == 0:
            self.logger.warning("Failed to extract any speaker embeddings")
            return []

        # 7. Enhanced clustering to determine speakers
        speaker_labels = self._cluster_speakers(embeddings, num_speakers, show_progress)

        # 8. Detect overlapped speech
        overlap_segments = self._detect_overlapped_speech(
            waveform[0], sample_rate, segment_timings
        )

        if show_progress and overlap_segments:
            print(f"Detected {len(overlap_segments)} potentially overlapped segments")

        # 9. Create final speaker segments with overlap information
        speaker_segments = self._create_speaker_segments(
            segment_timings, speaker_labels
        )

        # 10. Add overlap information to segments
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

        if show_progress:
            print(
                f"Diarization complete: identified {len(speaker_segments)} speaker segments with {len(set(s.speaker for s in speaker_segments))} speakers"
            )

        return speaker_segments
