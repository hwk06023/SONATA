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
import random

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
    def __init__(self, device="cpu", model_type="titanet"):
        self.device = device
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        self._load_models()

    def _load_models(self):
        print("Loading diarization models...")

        if self.model_type == "wavlm-base-plus-sv":
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModel

                print("Loading Microsoft WavLM-Base-Plus-SV model...")
                self.wavlm_processor = AutoFeatureExtractor.from_pretrained(
                    "microsoft/wavlm-base-plus-sv"
                )
                self.wavlm_model = AutoModel.from_pretrained(
                    "microsoft/wavlm-base-plus-sv"
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    self.wavlm_model = self.wavlm_model.cuda()
                else:
                    self.wavlm_model = self.wavlm_model.cpu()
                print(f"Successfully loaded WavLM model on {self.device}")
            except Exception as e:
                print(f"Failed to load WavLM model: {str(e)}")
                print("Falling back to TitaNet model")
                self.model_type = "titanet"

        if self.model_type == "titanet":
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
            print(f"Extracting speaker embeddings with {self.model_type}...")

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

                # Create temp WAV file
                temp_wav = os.path.join(temp_dir, f"segment_{start}_{end}.wav")
                sample_rate_16k = 16000
                if segment_waveform_np.ndim == 1:
                    audio_data = segment_waveform_np.reshape(1, -1)
                    torchaudio.save(temp_wav, torch.tensor(audio_data), sample_rate_16k)
                else:
                    torchaudio.save(
                        temp_wav, torch.tensor(segment_waveform_np), sample_rate_16k
                    )

                # Extract embedding using selected model
                if self.model_type == "wavlm-base-plus-sv":
                    # Use WavLM model
                    with torch.no_grad():
                        # 오디오 로드
                        audio_input, sr = torchaudio.load(temp_wav)

                        # 오디오 형식 수정 - WavLM에 맞게 차원 변환
                        if audio_input.dim() > 2:
                            audio_input = audio_input.squeeze()  # 불필요한 차원 제거

                        if audio_input.dim() == 1:
                            audio_input = audio_input.unsqueeze(0)  # [N] -> [1, N]

                        # 스테레오일 경우 모노로 변환
                        if audio_input.size(0) > 1:
                            audio_input = torch.mean(audio_input, dim=0, keepdim=True)

                        # 샘플링 레이트 확인
                        expected_sr = 16000
                        if sr != expected_sr:
                            audio_input = torchaudio.functional.resample(
                                audio_input, sr, expected_sr
                            )
                            sr = expected_sr

                        # 최소 길이 확인 (너무 짧은 오디오는 문제 발생)
                        min_samples = 1000  # 최소 샘플 수
                        if audio_input.size(-1) < min_samples:
                            padding = torch.zeros(1, min_samples - audio_input.size(-1))
                            audio_input = torch.cat([audio_input, padding], dim=1)

                        # WavLM 프로세서 적용
                        inputs = self.wavlm_processor(
                            audio_input, sampling_rate=sr, return_tensors="pt"
                        )

                        # 차원 수정 - input_values의 차원 조정
                        if (
                            "input_values" in inputs
                            and inputs["input_values"].dim() == 3
                        ):
                            inputs["input_values"] = inputs["input_values"].squeeze(1)

                        # 모델에 입력
                        if self.device == "cuda" and torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}

                        outputs = self.wavlm_model(**inputs)
                        embedding = torch.mean(outputs.last_hidden_state, dim=1)
                        embedding = embedding.cpu().numpy()
                        embedding = embedding.squeeze()
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                else:
                    # Use TitaNet model (default)
                    with torch.no_grad():
                        embedding = self.titanet_model.get_embedding(temp_wav)
                        embedding = embedding.cpu().numpy()
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
                    f"Failed to extract embedding for segment {start}-{end}: {str(e)}"
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
                f"Using {self.model_type} embeddings for {len(embeddings)} segments (errors: {errors_count})"
            )
            if len(embeddings) > 0:
                embeddings_array = np.array(embeddings)
                print(f"{self.model_type} embedding shape: {embeddings_array.shape}")
                print(f"{self.model_type} embedding type: {type(embeddings[0])}")
                print(
                    f"{self.model_type} embedding sample (first 5 values): {embeddings[0][:5]}"
                )

                # 모델별 임베딩 특성 정보
                if self.model_type == "wavlm-base-plus-sv":
                    expected_dim = 768
                    print(f"WavLM 모델 사용 중 - 예상 임베딩 차원: {expected_dim}")
                else:
                    expected_dim = 192
                    print(f"TitaNet 모델 사용 중 - 예상 임베딩 차원: {expected_dim}")

                print(
                    f"{self.model_type} embedding stats - min: {np.min(embeddings_array):.4f}, max: {np.max(embeddings_array):.4f}, mean: {np.mean(embeddings_array):.4f}"
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

        # 차원 축소 적용 - WavLM과 TitaNet 차원 차이 고려
        embedding_dim = norm_embeddings.shape[1]
        if len(norm_embeddings) > 50:
            if self.model_type == "wavlm-base-plus-sv":
                # WavLM은 차원이 크므로 더 적극적으로 차원 축소
                pca_components = min(128, embedding_dim)
                pca = PCA(n_components=pca_components, random_state=42)
                norm_embeddings = pca.fit_transform(norm_embeddings)
                if show_progress:
                    print(
                        f"Applied PCA for WavLM: {embedding_dim} -> {pca_components} dimensions"
                    )
            else:
                # TitaNet은 기존 방식 유지
                pca = PCA(n_components=min(64, embedding_dim), random_state=42)
                norm_embeddings = pca.fit_transform(norm_embeddings)
                if show_progress:
                    print(
                        f"Applied PCA for TitaNet: {embedding_dim} -> {min(64, embedding_dim)} dimensions"
                    )

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
                    n_neighbors=min(len(norm_embeddings) // 3, 10)
                    if self.model_type == "titanet"
                    else min(len(norm_embeddings) // 4, 15),
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
                    n_neighbors=min(max(5, len(norm_embeddings) // 4), 8)
                    if self.model_type == "titanet"
                    else min(max(5, len(norm_embeddings) // 5), 12),
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
        processed_indices = set()
        all_morphs = []

        for segment in voice_segments:
            content = segment["content"]
            morphs = mecab.pos(content)
            self.logger.debug(f"Morphemes for '{content}': {morphs}")
            all_morphs.append((segment, morphs))

        self.logger.info(f"Analyzing {len(all_morphs)} segments for chunking")

        i = 0
        while i < len(all_morphs):
            if i in processed_indices:
                i += 1
                continue

            segment, morphs = all_morphs[i]
            current_chunk = {
                "start": segment["start"],
                "end": segment["end"],
                "content": segment["content"],
                "type": segment["type"],
            }
            processed_indices.add(i)

            # Look ahead for segments to merge
            j = i + 1
            while j < len(all_morphs):
                if j in processed_indices:
                    j += 1
                    continue

                prev_segment, prev_morphs = all_morphs[j - 1]
                next_segment, next_morphs = all_morphs[j]

                # Check time gap between segments
                time_gap = next_segment["start"] - prev_segment["end"]
                if time_gap > 0.5:
                    print(
                        f"Skipping chunking due to large time gap (>{0.5}s): '{prev_segment['content']}' -> '{next_segment['content']}'"
                    )
                    break

                should_merge = self._check_grammatical_connection(
                    prev_morphs, next_morphs
                )

                if should_merge:
                    current_chunk["end"] = next_segment["end"]
                    current_chunk["content"] += " " + next_segment["content"]
                    processed_indices.add(j)
                    print(f"Merged: '{current_chunk['content']}'")
                    j += 1
                else:
                    break

            chunked_segments.append(current_chunk)
            i = j if j > i + 1 else i + 1

        # Add non-voice segments
        non_voice_segments = [
            seg for seg in result_segments if seg.get("type") != "voice"
        ]
        self.logger.info(f"Non-voice segments: {len(non_voice_segments)} segments")
        chunked_segments.extend(non_voice_segments)

        # Sort all segments by start time
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

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge segments shorter than 0.2 seconds with adjacent segments based on linguistic features
        Also handles demonstrative pronouns for short segments

        Args:
            segments: List containing speech segments
                [{'start': float, 'end': float, 'content': str, 'type': str}, ...]

        Returns:
            List of merged segments
        """
        if not segments or len(segments) <= 1:
            return segments

        print(f"Starting short segment merging: {len(segments)} segments")

        # Filter to voice segments only
        voice_segments = [
            seg for seg in segments if seg.get("type") == "voice" and "content" in seg
        ]
        non_voice_segments = [
            seg
            for seg in segments
            if seg.get("type") != "voice" or "content" not in seg
        ]

        # Sort by start time to ensure correct processing
        voice_segments.sort(key=lambda x: x["start"])

        # Initialize mecab for morphological analysis
        mecab = Mecab()

        # Define special words for handling
        demonstrative_pronouns = ["이거", "저거", "그거", "여기", "저기", "거기"]
        interjections = ["아", "어", "음", "응", "네", "예", "에", "엄", "흠", "헉", "아니", "맞아"]

        # Identify short segments
        short_segments_indices = []
        for i, segment in enumerate(voice_segments):
            if segment["end"] - segment["start"] <= 0.2:
                short_segments_indices.append(i)

        print(f"Found {len(short_segments_indices)} short segments to merge")

        # Process until no more short segments or no changes
        while short_segments_indices:
            # Start with the earliest short segment
            short_segments_indices.sort(key=lambda i: voice_segments[i]["start"])
            current_short_idx = short_segments_indices.pop(0)

            # Skip if this segment was already removed in a previous merge
            if current_short_idx >= len(voice_segments):
                continue

            current = voice_segments[current_short_idx]

            # Skip if segment duration now exceeds 0.2 seconds
            # (might happen due to previous merges)
            if current["end"] - current["start"] > 0.2:
                print(
                    f"Skipping segment {current_short_idx} as duration now exceeds 0.2s: '{current['content']}'"
                )
                continue

            current_morphs = mecab.pos(current["content"])
            current_content = current["content"].strip()
            current_duration = current["end"] - current["start"]

            # Special case: first segment
            if current_short_idx == 0 and len(voice_segments) > 1:
                next_seg = voice_segments[1]

                # Check for large time gap
                if next_seg["start"] - current["end"] >= 1.0:
                    print(
                        f"Skipping merge due to large time gap (≥1s): '{current_content}' -> '{next_seg['content']}'"
                    )
                    continue

                merged_seg = {
                    "start": current["start"],
                    "end": next_seg["end"],
                    "content": current["content"] + " " + next_seg["content"],
                    "type": "voice",
                }

                # Replace with merged segment
                voice_segments.pop(1)  # Remove next
                voice_segments[0] = merged_seg  # Replace current with merged

                # Update indices of remaining short segments
                short_segments_indices = [
                    i - 1 if i > 1 else i for i in short_segments_indices
                ]

                continue

            # Special case: last segment
            if current_short_idx == len(voice_segments) - 1 and len(voice_segments) > 1:
                prev_seg = voice_segments[current_short_idx - 1]

                # Check for large time gap
                if current["start"] - prev_seg["end"] >= 1.0:
                    print(
                        f"Skipping merge due to large time gap (≥1s): '{prev_seg['content']}' -> '{current_content}'"
                    )
                    continue

                merged_seg = {
                    "start": prev_seg["start"],
                    "end": current["end"],
                    "content": prev_seg["content"] + " " + current["content"],
                    "type": "voice",
                }

                # Replace with merged segment
                voice_segments.pop(current_short_idx)  # Remove current
                voice_segments[
                    current_short_idx - 1
                ] = merged_seg  # Replace prev with merged

                # Update indices of remaining short segments
                short_segments_indices = [
                    i if i < current_short_idx else i - 1
                    for i in short_segments_indices
                ]

                continue

            # Regular case: compare with previous and next
            if current_short_idx > 0 and current_short_idx < len(voice_segments) - 1:
                prev_seg = voice_segments[current_short_idx - 1]
                next_seg = voice_segments[current_short_idx + 1]

                # Calculate time gaps with adjacent segments
                prev_gap = current["start"] - prev_seg["end"]
                next_gap = next_seg["start"] - current["end"]

                # Initialize scores with a default value for very short segments
                # This encourages merging of very short segments even with weaker connections
                prev_score = (
                    0.5 if current_duration <= 0.1 else 0
                )  # Bias for very short segments
                next_score = (
                    0.5 if current_duration <= 0.1 else 0
                )  # Bias for very short segments

                # Only consider merging if gap is less than 1 second
                if prev_gap < 1.0:
                    prev_morphs = mecab.pos(prev_seg["content"])
                    prev_score += self._calculate_connection_score(
                        prev_morphs, current_morphs
                    )

                    # Boost score for continuous segments (no gap)
                    if abs(prev_gap) < 0.01:
                        prev_score += 1.0

                    # Additional score for similar duration segments
                    prev_duration = prev_seg["end"] - prev_seg["start"]
                    if abs(prev_duration - current_duration) < 0.1:
                        prev_score += 0.5
                else:
                    print(
                        f"Large gap (≥1s) with previous segment: '{prev_seg['content']}' -> '{current_content}'"
                    )

                if next_gap < 1.0:
                    next_morphs = mecab.pos(next_seg["content"])
                    next_score += self._calculate_connection_score(
                        current_morphs, next_morphs
                    )

                    # Boost score for continuous segments (no gap)
                    if abs(next_gap) < 0.01:
                        next_score += 1.0

                    # Additional score for similar duration segments
                    next_duration = next_seg["end"] - next_seg["start"]
                    if abs(next_duration - current_duration) < 0.1:
                        next_score += 0.5
                else:
                    print(
                        f"Large gap (≥1s) with next segment: '{current_content}' -> '{next_seg['content']}'"
                    )

                # Get content of segments
                prev_content = prev_seg["content"].strip()
                next_content = next_seg["content"].strip()

                # Special case: interjection + demonstrative pronoun combination
                # Example: "아" + "그거" should be merged, or "어" + "이거" should be merged
                if next_gap < 1.0 and (
                    current_content in interjections
                    and next_content.split()[0] in demonstrative_pronouns
                ):
                    next_score += 5.0
                    print(
                        f"Strongly boosting connection for interjection+demonstrative: '{current_content}' + '{next_content}'"
                    )
                elif prev_gap < 1.0 and (
                    current_content in demonstrative_pronouns
                    and prev_content.split()[-1] in interjections
                ):
                    prev_score += 5.0
                    print(
                        f"Strongly boosting connection for interjection+demonstrative: '{prev_content}' + '{current_content}'"
                    )

                # Additional heuristics for interjections
                elif next_gap < 1.0 and current_content in interjections:
                    # Interjections typically connect more with what follows
                    next_score += 2.0

                # Check for demonstrative pronouns in current segment
                elif next_gap < 1.0 and (
                    current_content in demonstrative_pronouns
                    or (
                        len(current_content.split()) > 0
                        and current_content.split()[-1] in demonstrative_pronouns
                    )
                ):
                    # Demonstrative pronouns connect strongly with the next segment
                    next_score += 4.0
                    print(
                        f"Boosting next connection for demonstrative: '{current_content}'"
                    )

                # Additional linguistic heuristics for common Korean patterns
                # Short endings like '요', '죠', '네' often connect with both sides
                if len(current_content) <= 3 and (
                    current_content.endswith("요")
                    or current_content.endswith("죠")
                    or current_content.endswith("네")
                ):
                    # Check previous for noun or adjective
                    if prev_gap < 1.0 and any(
                        pos.startswith("N") or pos.startswith("VA")
                        for word, pos in prev_morphs
                    ):
                        prev_score += 2.0
                        print(
                            f"Boosting prev connection for ending pattern: '{prev_content}' + '{current_content}'"
                        )

                # Very short segments (1-2 syllables) generally need merging
                if len(current_content) <= 2 and current_duration <= 0.15:
                    # Boost both scores to ensure it gets merged somewhere
                    base_boost = 1.5
                    prev_score += base_boost
                    next_score += base_boost
                    print(
                        f"Boosting both connections for very short segment: '{current_content}'"
                    )

                # Merge with segment that has higher linguistic connection
                if prev_score >= next_score and prev_score > 0:
                    # Merge with previous
                    merged_seg = {
                        "start": prev_seg["start"],
                        "end": current["end"],
                        "content": prev_seg["content"] + " " + current["content"],
                        "type": "voice",
                    }

                    print(
                        f"Merging with previous: '{prev_content}' + '{current_content}' (scores: prev={prev_score:.1f}, next={next_score:.1f})"
                    )

                    # Replace with merged segment
                    voice_segments.pop(current_short_idx)  # Remove current
                    voice_segments[
                        current_short_idx - 1
                    ] = merged_seg  # Replace prev with merged

                    # If the merged segment is still short, keep it in the list for further processing
                    if (
                        merged_seg["end"] - merged_seg["start"] <= 0.2
                        and current_short_idx - 1 not in short_segments_indices
                    ):
                        short_segments_indices.append(current_short_idx - 1)

                    # Update indices of remaining short segments
                    short_segments_indices = [
                        i if i < current_short_idx else i - 1
                        for i in short_segments_indices
                    ]
                elif next_score > 0:
                    # Merge with next
                    merged_seg = {
                        "start": current["start"],
                        "end": next_seg["end"],
                        "content": current["content"] + " " + next_seg["content"],
                        "type": "voice",
                    }

                    print(
                        f"Merging with next: '{current_content}' + '{next_content}' (scores: prev={prev_score:.1f}, next={next_score:.1f})"
                    )

                    # Replace with merged segment
                    voice_segments.pop(current_short_idx + 1)  # Remove next
                    voice_segments[
                        current_short_idx
                    ] = merged_seg  # Replace current with merged

                    # If the merged segment is still short, keep it in the list for further processing
                    if (
                        merged_seg["end"] - merged_seg["start"] <= 0.2
                        and current_short_idx not in short_segments_indices
                    ):
                        short_segments_indices.append(current_short_idx)

                    # Update indices of remaining short segments
                    short_segments_indices = [
                        i if i <= current_short_idx else i - 1
                        for i in short_segments_indices
                    ]
                else:
                    # Don't merge if both scores are 0 (large gaps on both sides)
                    print(
                        f"Not merging segment '{current_content}' due to large gaps on both sides"
                    )

        # Final pass: check for overlapping segments and remove duplicates
        i = 0
        while i < len(voice_segments) - 1:
            current = voice_segments[i]
            next_seg = voice_segments[i + 1]

            # Check for time overlap
            if current["end"] > next_seg["start"]:
                # Take the longer segment
                if (next_seg["end"] - next_seg["start"]) > (
                    current["end"] - current["start"]
                ):
                    voice_segments.pop(i)  # Remove current
                else:
                    voice_segments.pop(i + 1)  # Remove next
                # Don't increment i as we need to check the next pair
            else:
                i += 1

        # Combine voice and non-voice segments and sort by start time
        result_segments = voice_segments + non_voice_segments
        result_segments.sort(key=lambda x: x["start"])

        print(f"After merging short segments: {len(result_segments)} segments")
        return result_segments

    def _calculate_connection_score(
        self, morphs1: List[Tuple[str, str]], morphs2: List[Tuple[str, str]]
    ) -> float:
        """
        Calculate linguistic connection score between two morpheme sequences

        Args:
            morphs1: First sequence's morpheme analysis [(word, part_of_speech), ...]
            morphs2: Second sequence's morpheme analysis [(word, part_of_speech), ...]

        Returns:
            Connection score (higher means stronger connection)
        """
        if not morphs1 or not morphs2:
            return 0.0

        score = 0.0

        # Basic grammatical connection from existing function
        if self._check_grammatical_connection(morphs1, morphs2):
            score += 2.0

        # Additional scoring for specific Korean language patterns
        last_word = morphs1[-1][0] if morphs1 else ""
        last_pos = morphs1[-1][1] if morphs1 else ""
        first_pos = morphs2[0][1] if morphs2 else ""

        # Subject + Verb/Adjective connection is very strong
        if last_pos == "JKS" and (
            first_pos.startswith("V") or first_pos.startswith("VA")
        ):
            score += 3.0

        # Incomplete predicate ending + continuation
        if last_pos.startswith("E") and not last_pos == "EF":
            score += 2.5

        # Modifier + Noun connection
        if (last_pos == "MM" or last_pos == "ETM") and first_pos.startswith("N"):
            score += 2.0

        # Conjunction + any continuation
        if last_pos == "MAJ":
            score += 1.5

        # Noun + Postposition connection
        if last_pos.startswith("N") and first_pos.startswith("J"):
            score += 2.0

        # Named entity continuation
        if last_pos == "NNP" and first_pos == "NNP":
            score += 2.5

        # Quotation marker + quotation content
        if (last_pos == "JKQ" or last_pos == "VCP") and first_pos.startswith("V"):
            score += 1.5

        return score

    def diarize(
        self,
        audio_path,
        num_speakers=None,
        show_progress=True,
        save_steps=False,
        result_segments=None,
        language=None,
        model_type="titanet",
    ):
        """Main diarization method with improved processing pipeline

        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers (estimated if None)
            show_progress: Whether to show progress information
            save_steps: Whether to save intermediate outputs
            word_timestamps: Optional word timestamps from ASR to skip change detection
            language: Language code for language-specific processing (e.g., 'ko' for Korean)
            model_type: Model to use for speaker embeddings (default: titanet)
        """
        # Update model type if passed in
        if model_type != self.model_type:
            self.model_type = model_type
            self._load_models()

        if show_progress:
            print(
                f"Starting enhanced diarization for: {audio_path} using {self.model_type}"
            )

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
            result_segments = self._merge_short_segments(result_segments)

        print(f"After chunking and merging:{result_segments}")

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
