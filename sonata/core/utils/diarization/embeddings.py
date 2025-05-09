import torch
import numpy as np
import torchaudio
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


class EmbeddingExtractor:
    def __init__(
        self, wavlm_model=None, wavlm_processor=None, ecapa_model=None, device="cpu"
    ):
        self.wavlm_model = wavlm_model
        self.wavlm_processor = wavlm_processor
        self.ecapa_model = ecapa_model
        self.device = device
        self.has_ecapa_model = ecapa_model is not None

    def extract_embeddings(
        self, waveform, sample_rate, segments, show_progress=True
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Extract speaker embeddings from audio segments

        Args:
            waveform: Audio waveform
            sample_rate: Audio sample rate
            segments: List of (start, end) tuples in seconds
            show_progress: Whether to show progress information

        Returns:
            Tuple of (embeddings, segment_timings)
        """
        if show_progress:
            print("Extracting speaker embeddings...")

        if not segments:
            return np.array([]), []

        valid_segments = []
        embeddings = []

        # Process segments with progress bar if needed
        iterator = segments
        if show_progress:
            iterator = tqdm(segments, desc="Extracting embeddings", unit="segment")

        for start, end in iterator:
            # Skip segments that are too short
            if end - start < 0.5:
                continue

            # Extract segment audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            # Handle edge cases
            if start_sample >= len(waveform) or end_sample <= start_sample:
                continue

            end_sample = min(end_sample, len(waveform))
            segment_waveform = waveform[start_sample:end_sample]

            # Convert to tensor if needed
            if not isinstance(segment_waveform, torch.Tensor):
                segment_waveform = torch.tensor(segment_waveform).float()

            # Attempt to extract embedding with ECAPA-TDNN first (better quality)
            embedding = None
            if self.has_ecapa_model:
                try:
                    # Ensure correct shape for ECAPA
                    if len(segment_waveform.shape) == 1:
                        segment_waveform = segment_waveform.unsqueeze(0)

                    # Resample if needed
                    if sample_rate != 16000:
                        segment_waveform = torchaudio.functional.resample(
                            segment_waveform, sample_rate, 16000
                        )

                    with torch.no_grad():
                        embedding = self.ecapa_model.encode_batch(
                            segment_waveform.to(self.device)
                        )
                        embedding = embedding.squeeze().cpu().numpy()
                except Exception as e:
                    print(f"ECAPA embedding failed: {str(e)}")
                    embedding = None

            # Fallback to WavLM if ECAPA failed
            if embedding is None and self.wavlm_model is not None:
                try:
                    # Ensure correct shape and length for WavLM
                    if len(segment_waveform.shape) == 1:
                        segment_waveform = segment_waveform.unsqueeze(0)

                    # Truncate if too long (WavLM has input limits)
                    max_length = min(16000 * 10, len(segment_waveform[0]))
                    segment_waveform = segment_waveform[:, :max_length]

                    # Resample if needed
                    if sample_rate != 16000:
                        segment_waveform = torchaudio.functional.resample(
                            segment_waveform, sample_rate, 16000
                        )

                    # Extract WavLM embedding
                    inputs = self.wavlm_processor(
                        segment_waveform.squeeze().numpy(),
                        sampling_rate=16000,
                        return_tensors="pt",
                    ).to(self.device)

                    with torch.no_grad():
                        embedding = self.wavlm_model(**inputs).embeddings.cpu().numpy()
                        embedding = embedding.mean(axis=1).squeeze()
                except Exception as e:
                    print(f"WavLM embedding failed: {str(e)}")
                    embedding = None

            # If both methods failed, try a simple fallback using MFCCs
            if embedding is None:
                try:
                    import librosa

                    # Generate simple MFCC features as fallback
                    mfccs = librosa.feature.mfcc(
                        y=segment_waveform.squeeze().cpu().numpy(), sr=16000, n_mfcc=20
                    )
                    # Use mean across time as a simple embedding
                    embedding = np.mean(mfccs, axis=1)
                    print("Using MFCC fallback for embedding")
                except Exception as e:
                    print(f"MFCC fallback failed: {str(e)}")
                    continue

            # Skip if all methods failed
            if embedding is None:
                continue

            # Store valid results
            embeddings.append(embedding)
            valid_segments.append((start, end))

        if not embeddings:
            print("Warning: Failed to extract any valid embeddings")
            return np.array([]), []

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        return embeddings_array, valid_segments
