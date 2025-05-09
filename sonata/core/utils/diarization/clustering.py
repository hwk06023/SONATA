import numpy as np
from typing import List, Optional
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from tqdm import tqdm


class SpeakerClusterer:
    def __init__(self):
        pass

    def cluster_speakers(
        self, embeddings, num_speakers=None, show_progress=True
    ) -> np.ndarray:
        """Cluster speaker embeddings to identify speakers

        Args:
            embeddings: Speaker embeddings array
            num_speakers: Number of speakers (estimated if None)
            show_progress: Whether to show progress information

        Returns:
            Array of speaker labels
        """
        if show_progress:
            print("Clustering speakers...")

        if len(embeddings) == 0:
            return np.array([])

        # Handle case with only one segment
        if len(embeddings) == 1:
            return np.array([0])

        # Estimate number of speakers if not provided
        if num_speakers is None:
            num_speakers = self.estimate_num_speakers(embeddings, show_progress)
            if show_progress:
                print(f"Estimated number of speakers: {num_speakers}")

        # Ensure at least one and at most the number of embeddings
        num_speakers = max(1, min(num_speakers, len(embeddings)))

        # Create distance matrix
        distance_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = cosine(embeddings[i], embeddings[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Try different clustering methods
        labels = None

        # First try Agglomerative Clustering with version-compatible parameters
        try:
            if show_progress:
                print("Trying agglomerative clustering...")

            # Try first with sklearn 0.24+ syntax
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=num_speakers, affinity="precomputed", linkage="average"
                )
                labels = clustering.fit_predict(distance_matrix)
            except TypeError:
                # Fallback for older scikit-learn versions
                clustering = AgglomerativeClustering(
                    n_clusters=num_speakers, linkage="average"
                )
                labels = clustering.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Agglomerative clustering failed: {str(e)}")
            labels = None

        # If agglomerative fails, try spectral clustering
        if labels is None:
            try:
                if show_progress:
                    print("Trying spectral clustering...")

                # Transform distances to similarities for spectral clustering
                similarity_matrix = 1 - distance_matrix

                # Try first with newer parameter format
                try:
                    clustering = SpectralClustering(
                        n_clusters=num_speakers,
                        affinity="precomputed",
                        assign_labels="discretize",
                        random_state=42,
                    )
                    labels = clustering.fit_predict(similarity_matrix)
                except TypeError:
                    # Fallback for older scikit-learn versions
                    clustering = SpectralClustering(
                        n_clusters=num_speakers, random_state=42
                    )
                    labels = clustering.fit_predict(similarity_matrix)
            except Exception as e:
                print(f"Spectral clustering failed: {str(e)}")

                # Fallback to simple assignment if both methods fail
                labels = np.zeros(len(embeddings), dtype=int)
                for i in range(min(num_speakers, len(embeddings))):
                    if i < len(embeddings):
                        labels[i] = i

        # Relabel to ensure consistently ascending order
        unique_labels = np.unique(labels)
        relabeled = np.zeros_like(labels)
        for i, label in enumerate(unique_labels):
            relabeled[labels == label] = i

        return relabeled

    def estimate_num_speakers(self, embeddings, show_progress=True) -> int:
        """Estimate the number of speakers using clustering metrics"""
        if len(embeddings) <= 1:
            return 1

        if show_progress:
            print("Estimating number of speakers...")

        # Create distance matrix
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = cosine(embeddings[i], embeddings[j])
                distances.append(distance)

        distances = np.array(distances)

        # Simple heuristic for very small embeddings sets
        if len(embeddings) < 4:
            # If any pair is very different, assume 2 speakers
            if np.max(distances) > 0.6:
                return 2
            else:
                return 1

        # Auto-tuned threshold based on distance statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Use distance distribution to estimate speaker count
        if mean_dist < 0.25:
            # Very similar embeddings - likely 1-2 speakers
            estimated = 1
        elif mean_dist < 0.4:
            # Moderately similar - likely 2-3 speakers
            estimated = 2
        elif mean_dist < 0.55:
            # Moderately different - likely 3-4 speakers
            estimated = 3
        else:
            # Very different - likely 4+ speakers
            estimated = 4

        # Adjust based on standard deviation
        if std_dist > 0.15:
            # High variance suggests more speakers
            estimated += 1

        # Cap based on number of embeddings
        max_speakers = min(8, max(2, len(embeddings) // 2))
        estimated = min(estimated, max_speakers)

        return estimated
