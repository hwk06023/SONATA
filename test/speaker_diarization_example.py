import argparse
import json
from sonata.core.speaker_diarization import SpeakerDiarizer
from sonata.core.transcriber import IntegratedTranscriber


def format_time(seconds):
    """Format time in seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def main():
    parser = argparse.ArgumentParser(description="SONATA Speaker Diarization Example")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument(
        "--num-speakers", type=int, help="Number of speakers (if known)"
    )
    parser.add_argument("--device", default="cpu", help="Compute device (cpu/cuda)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Also transcribe the audio with speakers",
    )
    parser.add_argument(
        "--language", default="en", help="Language code for ASR (if transcribing)"
    )
    args = parser.parse_args()

    # Option 1: Use standalone diarizer
    if not args.transcribe:
        print(f"Running speaker diarization on {args.audio_file}...")
        diarizer = SpeakerDiarizer(device=args.device)
        speaker_segments = diarizer.diarize(
            audio_path=args.audio_file,
            num_speakers=args.num_speakers,
            show_progress=True,
        )

        # Convert to dictionary for JSON serialization
        result = {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": segment.speaker,
                    "start_formatted": format_time(segment.start),
                    "end_formatted": format_time(segment.end),
                }
                for segment in speaker_segments
            ]
        }

        # Print results
        print("\nSpeaker Diarization Results:")
        for segment in result["segments"]:
            print(
                f"{segment['start_formatted']} - {segment['end_formatted']}: {segment['speaker']}"
            )

        # Save results if output path is specified
        if args.output:
            try:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {args.output}")
            except (IOError, OSError) as e:
                print(f"Error saving results to {args.output}: {e}")

    # Option 2: Use integrated transcriber with diarization
    else:
        print(
            f"Running integrated transcription with diarization on {args.audio_file}..."
        )
        transcriber = IntegratedTranscriber(device=args.device)
        result = transcriber.process_audio(
            audio_path=args.audio_file,
            language=args.language,
            diarize=True,
            num_speakers=args.num_speakers,
        )

        # Print transcription with speaker labels
        print("\nTranscription with Speaker Diarization:")
        if (
            "integrated_transcript" in result
            and "rich_text" in result["integrated_transcript"]
        ):
            for segment in result["integrated_transcript"]["rich_text"]:
                if "speaker" in segment:
                    speaker = segment.get("speaker", "UNKNOWN")
                    text = segment.get("text", "")
                    if text:
                        print(f"{speaker}: {text}")

        # Save results if output path is specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
