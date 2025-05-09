import torch
import numpy as np
import torchaudio
from typing import List, Tuple
from tqdm import tqdm


class ChangeDetector:
    def __init__(self, ecapa_model=None, device="cpu"):
        self.ecapa_model = ecapa_model
        self.device = device
        self.has_ecapa_model = ecapa_model is not None

    def detect_speaker_changes(
        self,
        waveform,
        sample_rate,
        vad_segments,
        window_size=0.75,
        hop_size=0.35,
        show_progress=True,
    ) -> List[float]:
        """Detect speaker changes within VAD segments using embedding similarity"""
        if show_progress:
            print("Detecting speaker changes...")

        changes = []

        # Create iterator with progress bar if needed
        iterator = vad_segments
        if show_progress:
            iterator = tqdm(vad_segments, desc="Processing segments", unit="segment")

        for start, end in iterator:
            if end - start < window_size * 3:
                continue

            # Extract segment waveform
            segment_start_sample = int(start * sample_rate)
            segment_end_sample = int(end * sample_rate)
            segment_waveform = waveform[segment_start_sample:segment_end_sample]

            # Convert tensor to numpy if needed
            if isinstance(segment_waveform, torch.Tensor):
                segment_waveform = segment_waveform.cpu().numpy()

            # Use embedding-based change detection for segments
            emb_changes = self.detect_changes_with_embeddings(
                segment_waveform, sample_rate, window_size, hop_size
            )

            # Convert local changes to global timeline
            emb_changes = [start + t for t in emb_changes]

            # Cluster close change points to avoid duplicates
            changes.extend(self.cluster_change_points(emb_changes, threshold=0.35))

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

    def detect_changes_with_embeddings(
        self, waveform, sample_rate, window_size, hop_size
    ) -> List[float]:
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
                print(f"Error extracting embedding: {str(e)}")
                continue

        if len(embeddings) < 3:
            return []

        # Compute similarity between adjacent windows
        similarities = []
        times = []
        for i in range(1, len(embeddings)):
            try:
                # Normalize embeddings
                emb1 = embeddings[i - 1] / np.linalg.norm(embeddings[i - 1])
                emb2 = embeddings[i] / np.linalg.norm(embeddings[i])

                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2)
                similarities.append(similarity)

                # Time point at the boundary between windows
                boundary_time = windows[i][0]  # Start time of current window
                times.append(boundary_time)
            except Exception as e:
                print(f"Error computing similarity: {str(e)}")
                continue

        if not similarities:
            return []

        # Find change points where similarity drops significantly
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Adaptive threshold based on statistics
        threshold = max(0.5, mean_sim - 1.5 * std_sim)

        # Find local minima below threshold
        for i in range(1, len(similarities) - 1):
            if (
                similarities[i] < threshold
                and similarities[i] < similarities[i - 1]
                and similarities[i] < similarities[i + 1]
            ):
                changes.append(times[i])

        return changes

    def cluster_change_points(self, change_points, threshold=0.35) -> List[float]:
        """Group change points that are close to each other and return centroids."""
        if not change_points:
            return []

        # Sort change points
        sorted_points = sorted(change_points)

        clusters = []
        current_cluster = [sorted_points[0]]

        # Group points by proximity
        for point in sorted_points[1:]:
            if point - current_cluster[-1] < threshold:
                # Add to current cluster
                current_cluster.append(point)
            else:
                # Finalize current cluster and start a new one
                clusters.append(current_cluster)
                current_cluster = [point]

        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        # Return the mean of each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
