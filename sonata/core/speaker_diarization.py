import torch
import numpy as np
import librosa
import logging
import os
import re
from typing import List, Dict, Optional, Tuple, Union
import torchaudio
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from tqdm import tqdm
import warnings
import speechbrain as sb
from konlpy.tag import Mecab

# NeMo toolkit import
try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print(
        "NeMo toolkit is not installed. Please install it using: pip install nemo_toolkit['all']"
    )
    nemo_asr = None

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
        print("Loading diarization models...")

        if nemo_asr is None:
            raise ImportError(
                "NeMo toolkit is required but not installed. Please install nemo_toolkit['all']"
            )

        try:
            print("Loading NVIDIA TitaNet model using NeMo toolkit...")
            # Load TitaNet model using NeMo toolkit for speaker embeddings
            self.titanet_model = (
                nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                    "nvidia/speakerverification_en_titanet_large"
                )
            )

            # Move model to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.titanet_model = self.titanet_model.cuda()
            else:
                self.titanet_model = self.titanet_model.cpu()

            print(f"Successfully loaded TitaNet model on {self.device}")
            print(f"TitaNet model type: {type(self.titanet_model)}")

        except Exception as e:
            print(f"Failed to load TitaNet model via NeMo: {str(e)}")
            raise RuntimeError(f"Failed to load TitaNet model: {str(e)}")

    def _extract_embeddings(self, waveform, sample_rate, segments, show_progress=True):
        """Extract speaker embeddings for each segment using NeMo TitaNet model"""
        if show_progress:
            print("Extracting speaker embeddings with TitaNet...")

        embeddings = []
        timings = []

        # Create iterator with progress bar if needed
        iterator = segments
        if show_progress:
            iterator = tqdm(segments, desc="Processing segments", unit="segment")

        errors_count = 0  # Track number of errors for reporting
        temp_dir = os.path.join(os.getcwd(), "temp_audio_segments")
        os.makedirs(temp_dir, exist_ok=True)

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

            # Skip very short segments as they might cause issues
            if len(segment_waveform) < 1600:  # Less than 0.1s at 16kHz
                if show_progress:
                    print(f"Skipping segment {start}-{end} as it's too short")
                continue

            try:
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

                # Convert segment to numpy if it's a tensor
                if isinstance(segment_waveform, torch.Tensor):
                    segment_waveform_np = segment_waveform.cpu().numpy()
                else:
                    segment_waveform_np = segment_waveform

                # NeMo TitaNet model requires a temporary WAV file
                temp_wav = os.path.join(temp_dir, f"segment_{start}_{end}.wav")

                # Save segment as a WAV file - ensure it's in the right format
                sample_rate_16k = 16000
                if segment_waveform_np.ndim == 1:
                    # Ensure audio is mono and has the right shape for torchaudio.save
                    audio_data = segment_waveform_np.reshape(1, -1)
                    torchaudio.save(temp_wav, torch.tensor(audio_data), sample_rate_16k)
                else:
                    torchaudio.save(
                        temp_wav, torch.tensor(segment_waveform_np), sample_rate_16k
                    )

                # Extract embedding using NeMo's TitaNet model
                with torch.no_grad():
                    embedding = self.titanet_model.get_embedding(temp_wav)

                    # Convert to numpy and normalize
                    embedding = embedding.cpu().numpy()
                    # Reshape embedding from (1, 192) to (192,) - needed to fix dimensionality
                    embedding = embedding.squeeze()
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

                    embeddings.append(embedding)
                    timings.append((start, end))

                # Clean up the temporary file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

            except Exception as e:
                errors_count += 1
                print(
                    f"Failed to extract TitaNet embedding for segment {start}-{end}: {str(e)}"
                )
                # Only log detailed error for the first few failures
                if errors_count <= 3:
                    print(f"Detailed error: {str(e)}")
                # Skip this segment and continue
                continue

        # Clean up temp directory if it's empty
        try:
            if len(os.listdir(temp_dir)) == 0:
                os.rmdir(temp_dir)
        except:
            pass

        if show_progress:
            print(
                f"Using TitaNet embeddings for {len(embeddings)} segments (errors: {errors_count})"
            )
            if len(embeddings) > 0:
                embeddings_array = np.array(embeddings)
                print(f"TitaNet embedding shape: {embeddings_array.shape}")
                print(f"TitaNet embedding type: {type(embeddings[0])}")
                print(f"TitaNet embedding sample (first 5 values): {embeddings[0][:5]}")
                print(
                    f"TitaNet embedding stats - min: {np.min(embeddings_array):.4f}, max: {np.max(embeddings_array):.4f}, mean: {np.mean(embeddings_array):.4f}"
                )

        if len(embeddings) == 0:
            print("Failed to extract any speaker embeddings")
            if errors_count > 0:
                print(f"Encountered {errors_count} errors during embedding extraction")

        return np.array(embeddings), timings

    def _cluster_speakers(self, embeddings, num_speakers=None, show_progress=True):
        """Enhanced clustering with multiple algorithms and automatic speaker count estimation"""
        if show_progress:
            print("Clustering speaker embeddings...")

        if embeddings.size == 0:
            return []

        # Check if embeddings is 3D and reshape to 2D if needed
        if len(embeddings.shape) == 3:
            print(f"Reshaping embeddings from {embeddings.shape} to 2D...")
            # If shape is (N, 1, D), reshape to (N, D)
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            print(f"New embeddings shape: {embeddings.shape}")

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

        # if len(norm_embeddings) > 50:
        #     pca = PCA(n_components=min(64, norm_embeddings.shape[1]), random_state=42)
        #     norm_embeddings = pca.fit_transform(norm_embeddings)

        # Try multiple clustering methods with proper version handling
        clustering_methods = []
        clustering_methods.append(
            {
                "name": "Agglomerative (Cosine)",
                "method": AgglomerativeClustering(
                    n_clusters=num_speakers, metric="cosine", linkage="average"
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "Agglomerative (Ward)",
                "method": AgglomerativeClustering(
                    n_clusters=num_speakers, metric="euclidean", linkage="ward"
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "Agglomerative (Complete)",
                "method": AgglomerativeClustering(
                    n_clusters=num_speakers, metric="cosine", linkage="complete"
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "Agglomerative (Basic)",
                "method": AgglomerativeClustering(n_clusters=num_speakers),
            }
        )
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
        clustering_methods.append(
            {
                "name": "Spectral (RBF)",
                "method": SpectralClustering(
                    n_clusters=num_speakers,
                    affinity="rbf",
                    gamma=0.15,
                    n_init=20,
                    assign_labels="kmeans",
                    random_state=42,
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "Spectral (Basic)",
                "method": SpectralClustering(
                    n_clusters=num_speakers,
                    assign_labels="discretize",
                    random_state=42,
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "Spectral (arpack)",
                "method": SpectralClustering(
                    n_clusters=num_speakers,
                    affinity="nearest_neighbors",
                    n_neighbors=min(max(5, len(norm_embeddings) // 4), 8),
                    assign_labels="discretize",
                    n_init=15,
                    eigen_solver="arpack",
                    random_state=42,
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "K-Means",
                "method": KMeans(
                    n_clusters=num_speakers,
                    init="k-means++",
                    n_init=20,
                    max_iter=500,
                    tol=1e-5,
                    random_state=42,
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "Gaussian Mixture",
                "method": GaussianMixture(
                    n_components=num_speakers,
                    covariance_type="full",
                    reg_covar=1e-5,
                    n_init=10,
                    random_state=42,
                    max_iter=200,
                ),
            }
        )
        clustering_methods.append(
            {
                "name": "BIRCH",
                "method": Birch(
                    n_clusters=num_speakers, threshold=0.1, branching_factor=50
                ),
            }
        )

        best_labels = None
        best_score = -1
        best_method = None

        # Try each clustering method
        for method_info in clustering_methods:
            # Apply clustering
            method = method_info["method"]

            # Handle methods like GMM that use fit_predict vs fit and predict
            if hasattr(method, "fit_predict"):
                labels = method.fit_predict(norm_embeddings)
            elif hasattr(method, "fit") and hasattr(method, "predict"):
                method.fit(norm_embeddings)
                labels = method.predict(norm_embeddings)
            else:
                continue

            # Skip if only one cluster was found
            if len(set(labels)) <= 1:
                continue

            # If not enough clusters, skip
            if len(set(labels)) < num_speakers // 2:
                continue

            sil_score = silhouette_score(norm_embeddings, labels, metric="cosine")
            ch_score = calinski_harabasz_score(norm_embeddings, labels)

            # Combined score (weighted average)
            combined_score = (0.9 * sil_score) + (0.1 * (ch_score / 10000))

            if show_progress:
                print(
                    f"{method_info['name']}: silhouette={sil_score:.4f}, CH={ch_score:.4f}, clusters={len(set(labels))}"
                )

            if combined_score > best_score:
                best_score = combined_score
                best_labels = labels
                best_method = method_info["name"]

        if best_labels is None:
            clustering = AgglomerativeClustering(n_clusters=num_speakers)
            best_labels = clustering.fit_predict(norm_embeddings)
            best_method = "Warning: best method not found, using Fallback Agglomerative"

        if show_progress:
            print(f"Selected clustering method: {best_method}")

        # Create speaker labels with proper format
        labels = [f"SPEAKER_{int(label):02d}" for label in best_labels]

        # Return in the format expected by create_speaker_segments
        return labels

    def _estimate_num_speakers(self, embeddings, show_progress=True):
        """Estimate number of speakers using eigenvalue analysis"""

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
        if not sparse.issparse(affinity_matrix):
            affinity_matrix = sparse.csr_matrix(affinity_matrix)

        laplacian = SpectralClustering(
            n_clusters=2, affinity="precomputed"
        )._get_laplacian(affinity_matrix)

        eigenvalues, _ = eigsh(laplacian, k=min(10, laplacian.shape[0] - 1), which="SM")

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

    def _save_to_txt(self, data, filename, description=""):
        """Save data to text file with description header"""
        with open(filename, "w") as f:
            # Add description header if provided
            if description:
                f.write(f"# {description}\n")
                f.write("#" + "-" * 50 + "\n")

            if isinstance(data, list):
                for item in data:
                    f.write(f"{item}\n")
            elif isinstance(data, np.ndarray):
                for item in data:
                    f.write(f"{item}\n")
            else:
                f.write(str(data))

    def _chunk_korean_segments(self, result_segments: List[Dict]) -> List[Dict]:
        """
        Groups Korean speech segments into grammatically connected chunks

        Args:
            result_segments: List containing word segments
                [{'start': float, 'end': float, 'content': str, 'type': str}, ...]

        Returns:
            Merged segment list
        """
        self.logger.info(f"Initial segments: {len(result_segments)} segments")
        voice_segments = [seg for seg in result_segments if seg.get("type") == "voice"]
        self.logger.info(f"Voice segments: {len(voice_segments)} segments")

        if not voice_segments:
            self.logger.info("No voice segments found, returning original segments")
            return result_segments

        mecab = Mecab()
        chunked_segments = []
        current_chunk = None
        pending_segments = []
        all_morphs = []

        for segment in voice_segments:
            content = segment["content"]
            morphs = mecab.pos(content)
            self.logger.debug(f"Morphemes for '{content}': {morphs}")
            all_morphs.append((segment, morphs))

        self.logger.info(f"Analyzing {len(all_morphs)} segments for chunking")
        i = 0
        while i < len(all_morphs):
            segment, morphs = all_morphs[i]

            if current_chunk is None:
                current_chunk = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "content": segment["content"],
                    "type": segment["type"],
                }
                self.logger.debug(f"Starting new chunk: {current_chunk}")
                i += 1
                continue

            should_merge = False

            if i + 1 < len(all_morphs):
                next_segment, next_morphs = all_morphs[i + 1]
                should_merge = self._check_grammatical_connection(morphs, next_morphs)
                self.logger.debug(
                    f"Checking connection between '{segment['content']}' and '{next_segment['content']}': {should_merge}"
                )

            if should_merge:
                old_content = current_chunk["content"]
                current_chunk["end"] = next_segment["end"]
                current_chunk["content"] += " " + next_segment["content"]
                self.logger.debug(
                    f"Merged chunk: '{old_content}' + '{next_segment['content']}' = '{current_chunk['content']}'"
                )
                i += 1
            else:
                self.logger.debug(f"Saving chunk: {current_chunk}")
                chunked_segments.append(current_chunk)
                current_chunk = None

        if current_chunk is not None:
            self.logger.debug(f"Saving final chunk: {current_chunk}")
            chunked_segments.append(current_chunk)

        non_voice_segments = [
            seg for seg in result_segments if seg.get("type") != "voice"
        ]
        self.logger.info(f"Non-voice segments: {len(non_voice_segments)} segments")
        chunked_segments.extend(non_voice_segments)

        chunked_segments.sort(key=lambda x: x["start"])
        self.logger.info(
            f"Final chunked segments: {len(chunked_segments)} segments (from original {len(result_segments)})"
        )

        return chunked_segments

    @staticmethod
    def _check_grammatical_connection(
        morphs1: List[Tuple[str, str]], morphs2: List[Tuple[str, str]]
    ) -> bool:
        """
        Checks if two morpheme sequences are grammatically connected

        Args:
            morphs1: First word's morpheme analysis result [(word, part_of_speech), ...]
            morphs2: Second word's morpheme analysis result [(word, part_of_speech), ...]

        Returns:
            True if words are grammatically connected, False otherwise
        """
        if not morphs1 or not morphs2:
            return False

        last_pos = morphs1[-1][1]
        first_pos = morphs2[0][1]

        if last_pos.startswith("N") and first_pos.startswith("J"):
            return True

        if last_pos.startswith("N") and (
            first_pos.startswith("V") or first_pos.startswith("VA")
        ):
            return True

        if last_pos.startswith("MM") and first_pos.startswith("N"):
            return True

        if last_pos.startswith("MA") and (
            first_pos.startswith("V") or first_pos.startswith("VA")
        ):
            return True

        if (
            last_pos.startswith("V") or last_pos.startswith("VA")
        ) and first_pos.startswith("E"):
            return True

        if last_pos == "SN" and first_pos.startswith("N"):
            return True

        if last_pos == "NNP" and first_pos == "NNP":
            return True

        return False

    def diarize(
        self,
        audio_path,
        num_speakers=None,
        show_progress=True,
        save_steps=False,
        result_segments=None,
        language=None,
    ):
        """Main diarization method with improved processing pipeline

        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers (estimated if None)
            show_progress: Whether to show progress information
            save_steps: Whether to save intermediate outputs
            word_timestamps: Optional word timestamps from ASR to skip change detection
            language: Language code for language-specific processing (e.g., 'ko' for Korean)
        """
        if show_progress:
            print(f"Starting enhanced diarization for: {audio_path}")

        output_dir = None
        if save_steps:
            # Create directory for saving step outputs
            audio_basename = os.path.basename(audio_path).split(".")[0]
            output_dir = f"{audio_basename}_steps"
            os.makedirs(output_dir, exist_ok=True)

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if needed

        # 0. Chunk Korean segments
        if language == "ko":
            result_segments = self._chunk_korean_segments(result_segments)

        # 1. Create analysis segments based on input method
        if result_segments:
            if show_progress:
                print("Using ASR word timestamps to create analysis segments")

            # Create analysis segments between consecutive word boundaries
            # but ensure they fall within VAD segments
            analysis_segments = []
            for word in result_segments:
                if word.get("type") == "voice":
                    analysis_segments.append((word.get("start", 0), word.get("end", 0)))
            analysis_segments = sorted(set(analysis_segments))

            if save_steps:
                seg_txt = os.path.join(output_dir, "01_analysis_segments_from_asr.txt")
                seg_desc = "Analysis Segments from ASR\nSegments created using word timestamps from ASR.\nFormat: start_time,end_time"
                with open(seg_txt, "w") as f:
                    f.write(f"# {seg_desc}\n")
                    f.write("#" + "-" * 50 + "\n")
                    for start, end in analysis_segments:
                        f.write(f"{start},{end}\n")

        if show_progress:
            print(f"Created {len(analysis_segments)} analysis segments")

        if show_progress:
            print(
                f"After filtering short segments: {len(analysis_segments)} analysis segments"
            )

        # 2. Extract enhanced speaker embeddings
        embeddings, segment_timings = self._extract_embeddings(
            waveform[0], sample_rate, analysis_segments, show_progress
        )

        if save_steps:
            timing_txt = os.path.join(output_dir, "02_segment_timings.txt")
            timing_desc = "Embedding Segment Timings\nEach line represents the start and end time (in seconds) of audio segments used for speaker embedding extraction.\nFormat: start_time,end_time"
            with open(timing_txt, "w") as f:
                f.write(f"# {timing_desc}\n")
                f.write("#" + "-" * 50 + "\n")
                for start, end in segment_timings:
                    f.write(f"{start},{end}\n")

            # Just save embedding shape info since embeddings are large
            emb_txt = os.path.join(output_dir, "03_embeddings_shape.txt")
            emb_desc = "Speaker Embedding Information\nEmbeddings are vectors representing the voice characteristics of speakers.\nOne embedding vector is generated for each audio segment."
            with open(emb_txt, "w") as f:
                f.write(f"# {emb_desc}\n")
                f.write("#" + "-" * 50 + "\n")
                f.write(f"Shape: {embeddings.shape}\n")
                f.write(f"Type: {embeddings.dtype}\n")
                f.write(f"Number of segments: {len(segment_timings)}\n")
                f.write(f"Embedding dimensions: {embeddings.shape[1]}\n")

        if len(embeddings) == 0:
            self.logger.warning("Failed to extract any speaker embeddings")
            return []

        # 3. Enhanced clustering to determine speakers
        speaker_labels = self._cluster_speakers(embeddings, num_speakers, show_progress)

        if save_steps:
            labels_txt = os.path.join(output_dir, "04_speaker_labels.txt")
            labels_desc = "Speaker Clustering Results\nEach line represents the speaker label for the corresponding segment.\nThese labels correspond to the segments in 02_segment_timings.txt."
            self._save_to_txt(speaker_labels, labels_txt, labels_desc)

        # 4. Detect overlapped speech
        overlap_segments = self._detect_overlapped_speech(
            waveform[0], sample_rate, segment_timings
        )

        if save_steps:
            overlap_txt = os.path.join(output_dir, "05_overlap_segments.txt")
            overlap_desc = "Overlapped Speech Segment Indices\nIndices of segments where multiple speakers are detected speaking simultaneously."
            self._save_to_txt(overlap_segments, overlap_txt, overlap_desc)

        if show_progress and overlap_segments:
            print(f"Detected {len(overlap_segments)} potentially overlapped segments")

        # 5. Create final speaker segments with overlap information
        speaker_segments = self._create_speaker_segments(
            segment_timings, speaker_labels
        )

        if save_steps:
            segments_txt = os.path.join(output_dir, "06_speaker_segments.txt")
            segments_desc = "Final Speaker Segments (without overlap information)\nFormat: start_time,end_time,speaker_id,is_overlap"
            with open(segments_txt, "w") as f:
                f.write(f"# {segments_desc}\n")
                f.write("#" + "-" * 50 + "\n")
                for seg in speaker_segments:
                    f.write(f"{seg.start},{seg.end},{seg.speaker},{seg.is_overlap}\n")

        # 6. Add overlap information to segments
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
            final_txt = os.path.join(output_dir, "07_final_segments.txt")
            final_desc = "Final Speaker Segments with Overlap Information\nFormat: start_time,end_time,speaker_id,is_overlap,overlap_speakers"
            with open(final_txt, "w") as f:
                f.write(f"# {final_desc}\n")
                f.write("#" + "-" * 50 + "\n")
                for seg in speaker_segments:
                    overlap_str = (
                        ",".join(seg.overlap_speakers) if seg.overlap_speakers else ""
                    )
                    f.write(
                        f"{seg.start},{seg.end},{seg.speaker},{seg.is_overlap},{overlap_str}\n"
                    )

        if show_progress:
            print(
                f"Diarization complete: identified {len(speaker_segments)} speaker segments with {len(set(s.speaker for s in speaker_segments))} speakers"
            )

        return speaker_segments
