import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple, Optional


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio from file with resampling."""
    y, sr_orig = librosa.load(file_path, sr=sr)
    return y, sr


def convert_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    target_format: str = "wav",
):
    """Convert audio to the desired format and sample rate."""
    # Determine the input file extension
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()[1:]  # Remove the '.' and make lowercase

    if ext in ["mp3", "m4a", "aac", "ogg", "flac", "wav"]:
        # Use pydub for conversion
        audio = AudioSegment.from_file(input_path)

        # Set the sample rate
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        # Set channels to mono if it's not
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Export to target format
        audio.export(output_path, format=target_format)
    else:
        # Fallback to librosa for other formats
        y, sr = librosa.load(input_path, sr=target_sr)
        sf.write(output_path, y, target_sr)


def split_audio(
    file_path: str, output_dir: str, segment_length: int = 30, overlap: int = 5
) -> list:
    """Split long audio file into smaller segments with overlap."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Convert seconds to samples
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)

    segments = []

    # Extract base filename without extension
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    start_sample = 0
    segment_idx = 0

    while start_sample < len(y):
        end_sample = min(start_sample + segment_samples, len(y))
        segment = y[start_sample:end_sample]

        # Save segment
        segment_path = os.path.join(
            output_dir, f"{base_filename}_segment_{segment_idx}.wav"
        )
        sf.write(segment_path, segment, sr)

        # Calculate start and end times
        start_time = start_sample / sr
        end_time = end_sample / sr

        segments.append(
            {
                "path": segment_path,
                "start_time": start_time,
                "end_time": end_time,
                "idx": segment_idx,
            }
        )

        # Move start position for next segment (with overlap)
        start_sample = end_sample - overlap_samples
        segment_idx += 1

    return segments


def trim_silence(file_path: str, output_path: Optional[str] = None, top_db: int = 20):
    """Trim silence from audio file."""
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Trim silence
    y_trimmed, idx = librosa.effects.trim(y, top_db=top_db)

    # Save trimmed audio if output path is provided
    if output_path:
        sf.write(output_path, y_trimmed, sr)
        return output_path
    else:
        # Otherwise, overwrite the original file
        sf.write(file_path, y_trimmed, sr)
        return file_path
