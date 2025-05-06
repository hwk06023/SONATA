import os
import sys
import argparse

# Add the parent directory to the path so we can import sonata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonata.core.speaker_diarization import SpeakerDiarizer


def main():
    parser = argparse.ArgumentParser(
        description="Run speaker diarization and save step outputs"
    )
    parser.add_argument(
        "audio_file", type=str, help="Path to the audio file for diarization"
    )
    parser.add_argument(
        "--num_speakers", type=int, default=None, help="Number of speakers (if known)"
    )
    args = parser.parse_args()

    # Check if the audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        return

    # Initialize diarizer
    print("Initializing speaker diarizer...")
    diarizer = SpeakerDiarizer(device="cuda" if torch.cuda.is_available() else "cpu")

    # Run diarization with step saving enabled
    print(f"Running diarization on {args.audio_file} with step output saving")
    speaker_segments = diarizer.diarize(
        args.audio_file,
        num_speakers=args.num_speakers,
        show_progress=True,
        save_steps=True,
    )

    # Print results summary
    audio_basename = os.path.basename(args.audio_file).split(".")[0]
    output_dir = f"{audio_basename}_steps"
    print(f"\nDiarization complete! Found {len(speaker_segments)} segments")
    print(f"Step outputs saved to: {os.path.abspath(output_dir)}")
    print("\nSpeaker segments:")

    # Group by speaker
    speakers = {}
    for segment in speaker_segments:
        if segment.speaker not in speakers:
            speakers[segment.speaker] = []
        speakers[segment.speaker].append((segment.start, segment.end))

    # Print summary by speaker
    for speaker, segments in speakers.items():
        total_duration = sum(end - start for start, end in segments)
        print(f"{speaker}: {len(segments)} segments, {total_duration:.2f} seconds")


if __name__ == "__main__":
    import torch

    main()
