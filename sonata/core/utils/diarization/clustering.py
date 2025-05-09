import numpy as np
from typing import List, Optional, Dict, Any
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
        num_speakers = max(2, min(num_speakers, min(8, len(embeddings) // 2)))

        if show_progress:
            print(f"Clustering with {num_speakers} speakers")

        # Normalize embeddings
        norm_embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Create distance matrix for methods that use it
        distance_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = cosine(embeddings[i], embeddings[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Try multiple clustering methods with proper version handling
        clustering_methods = []

        # Check if scikit-learn supports all parameters (handle version compatibility)
        try:
            # Try creating with affinity and check if it raises an error
            test_clustering = AgglomerativeClustering(
                n_clusters=2, metric="cosine", linkage="average"
            )
            # If no error, add the full method
            clustering_methods.append(
                {
                    "name": "Agglomerative (Cosine)",
                    "method": AgglomerativeClustering(
                        n_clusters=num_speakers, metric="cosine", linkage="average"
                    ),
                    "input": norm_embeddings,
                }
            )

            # Add agglomerative with different linkages
            clustering_methods.append(
                {
                    "name": "Agglomerative (Ward)",
                    "method": AgglomerativeClustering(
                        n_clusters=num_speakers, metric="euclidean", linkage="ward"
                    ),
                    "input": norm_embeddings,
                }
            )

            clustering_methods.append(
                {
                    "name": "Agglomerative (Complete)",
                    "method": AgglomerativeClustering(
                        n_clusters=num_speakers, metric="cosine", linkage="complete"
                    ),
                    "input": norm_embeddings,
                }
            )

            # Add precomputed distance matrix version
            clustering_methods.append(
                {
                    "name": "Agglomerative (Precomputed)",
                    "method": AgglomerativeClustering(
                        n_clusters=num_speakers,
                        affinity="precomputed",
                        linkage="average",
                    ),
                    "input": distance_matrix,
                }
            )
        except TypeError:
            # Fallback to simpler parameters
            clustering_methods.append(
                {
                    "name": "Agglomerative (Basic)",
                    "method": AgglomerativeClustering(n_clusters=num_speakers),
                    "input": norm_embeddings,
                }
            )

        # Try spectral clustering with similar version check
        try:
            clustering_methods.append(
                {
                    "name": "Spectral (Nearest Neighbors)",
                    "method": SpectralClustering(
                        n_clusters=num_speakers,
                        affinity="nearest_neighbors",
                        n_neighbors=min(len(norm_embeddings) // 3, 10),
                        random_state=42,
                    ),
                    "input": norm_embeddings,
                }
            )

            # Add spectral with different affinity
            clustering_methods.append(
                {
                    "name": "Spectral (RBF)",
                    "method": SpectralClustering(
                        n_clusters=num_speakers,
                        affinity="rbf",
                        gamma=0.1,
                        random_state=42,
                    ),
                    "input": norm_embeddings,
                }
            )

            # Add precomputed version with similarity matrix
            similarity_matrix = 1 - distance_matrix
            clustering_methods.append(
                {
                    "name": "Spectral (Precomputed)",
                    "method": SpectralClustering(
                        n_clusters=num_speakers,
                        affinity="precomputed",
                        assign_labels="discretize",
                        random_state=42,
                    ),
                    "input": similarity_matrix,
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
                        "input": norm_embeddings,
                    }
                )
            except:
                # Skip if not available
                pass

        # Try K-means
        try:
            from sklearn.cluster import KMeans

            clustering_methods.append(
                {
                    "name": "K-Means",
                    "method": KMeans(
                        n_clusters=num_speakers,
                        init="k-means++",
                        n_init=10,
                        random_state=42,
                        max_iter=300,
                    ),
                    "input": norm_embeddings,
                }
            )
        except Exception:
            pass

        # Try Gaussian Mixture Model
        try:
            from sklearn.mixture import GaussianMixture

            clustering_methods.append(
                {
                    "name": "Gaussian Mixture",
                    "method": GaussianMixture(
                        n_components=num_speakers,
                        covariance_type="full",
                        random_state=42,
                        max_iter=100,
                    ),
                    "input": norm_embeddings,
                }
            )
        except Exception:
            pass

        # Try BIRCH
        try:
            from sklearn.cluster import Birch

            clustering_methods.append(
                {
                    "name": "BIRCH",
                    "method": Birch(
                        n_clusters=num_speakers, threshold=0.1, branching_factor=50
                    ),
                    "input": norm_embeddings,
                }
            )
        except Exception:
            pass

        best_labels = None
        best_score = -1
        best_method = None

        # Try each clustering method
        for method_info in clustering_methods:
            try:
                # Apply clustering
                method = method_info["method"]
                input_data = method_info["input"]

                # Handle methods like GMM that use fit_predict vs fit and predict
                if hasattr(method, "fit_predict"):
                    labels = method.fit_predict(input_data)
                elif hasattr(method, "fit") and hasattr(method, "predict"):
                    method.fit(input_data)
                    labels = method.predict(input_data)
                else:
                    continue

                # Skip if only one cluster was found
                if len(set(labels)) <= 1:
                    continue

                # If not enough clusters, skip
                if len(set(labels)) < num_speakers // 2:
                    continue

                # Evaluate clustering quality
                try:
                    from sklearn.metrics import (
                        silhouette_score,
                        calinski_harabasz_score,
                    )

                    # For some methods, using cosine distance for silhouette is more appropriate
                    sil_score = silhouette_score(
                        norm_embeddings, labels, metric="cosine"
                    )
                    ch_score = calinski_harabasz_score(norm_embeddings, labels)

                    # Combined score (weighted average)
                    combined_score = (0.7 * sil_score) + (0.3 * (ch_score / 10000))

                    if show_progress:
                        print(
                            f"{method_info['name']}: silhouette={sil_score:.4f}, CH={ch_score:.4f}, clusters={len(set(labels))}"
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

        # Ensure the return is in proper format
        if not isinstance(best_labels, np.ndarray):
            best_labels = np.array(best_labels)

        return best_labels

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

        # Try to use eigenvalue analysis for larger sets
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

            if show_progress:
                print(f"Eigenvalue analysis suggests {estimated_speakers} speakers")

            return estimated_speakers
        except Exception as e:
            if show_progress:
                print(
                    f"Error in eigenvalue analysis: {str(e)}, falling back to heuristics"
                )

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
