import os
import numpy as np
from typing import List, Dict, Any, Union


class DiarizationIO:
    def __init__(self):
        pass

    def save_to_txt(self, data, filename, description=""):
        """Save data to text file with description header

        Args:
            data: Data to save
            filename: Target filename
            description: Optional description for file header
        """
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

    def save_segments(self, segments, filename, description="Speaker Segments"):
        """Save speaker segments to file

        Args:
            segments: List of speaker segments
            filename: Target filename
            description: Optional description for file header
        """
        with open(filename, "w") as f:
            f.write(f"# {description}\n")
            f.write("#" + "-" * 50 + "\n")

            for seg in segments:
                overlap_str = (
                    ",".join(seg.overlap_speakers) if seg.overlap_speakers else ""
                )
                f.write(
                    f"{seg.start},{seg.end},{seg.speaker},{seg.is_overlap},{overlap_str}\n"
                )

    def save_vad_segments(self, vad_segments, filename):
        """Save VAD segments to file

        Args:
            vad_segments: List of (start, end) tuples
            filename: Target filename
        """
        description = "Voice Activity Detection (VAD) Segments\nEach line represents the start and end time (in seconds) of a detected speech segment.\nFormat: start_time,end_time"

        with open(filename, "w") as f:
            f.write(f"# {description}\n")
            f.write("#" + "-" * 50 + "\n")
            for start, end in vad_segments:
                f.write(f"{start},{end}\n")

    def save_embedding_info(self, embeddings, segment_timings, filename):
        """Save embedding information to file

        Args:
            embeddings: Array of embeddings
            segment_timings: List of segment timings
            filename: Target filename
        """
        description = "Speaker Embedding Information\nEmbeddings are vectors representing the voice characteristics of speakers.\nOne embedding vector is generated for each audio segment."

        with open(filename, "w") as f:
            f.write(f"# {description}\n")
            f.write("#" + "-" * 50 + "\n")
            f.write(f"Shape: {embeddings.shape}\n")
            f.write(f"Type: {embeddings.dtype}\n")
            f.write(f"Number of segments: {len(segment_timings)}\n")
            f.write(f"Embedding dimensions: {embeddings.shape[1]}\n")

    def create_output_directory(self, audio_path):
        """Create output directory for saving diarization steps

        Args:
            audio_path: Path to the audio file

        Returns:
            Path to created directory
        """
        audio_basename = os.path.basename(audio_path).split(".")[0]
        output_dir = f"{audio_basename}_steps"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
