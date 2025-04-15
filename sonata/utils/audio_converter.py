import os
from pydub import AudioSegment


def convert_to_wav(input_path, output_path=None, sample_rate=16000):
    """
    Convert audio file to WAV format

    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file (if None, will use same name but .wav extension)
        sample_rate: Target sample rate

    Returns:
        Path to the converted WAV file
    """
    if output_path is None:
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"{filename}.wav")

    # Load the audio file
    audio = AudioSegment.from_file(input_path)

    # Convert to desired sample rate
    audio = audio.set_frame_rate(sample_rate)

    # Export as WAV
    audio.export(output_path, format="wav")

    print(f"Converted {input_path} to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio files to WAV format")
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("--output", "-o", help="Path to output WAV file")
    parser.add_argument(
        "--sample-rate",
        "-sr",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)",
    )

    args = parser.parse_args()
    convert_to_wav(args.input_path, args.output, args.sample_rate)
