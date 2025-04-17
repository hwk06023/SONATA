#!/usr/bin/env python3
import os
import sys
import argparse
from sonata.utils.audio_converter import convert_to_wav
from sonata.core.transcriber import IntegratedTranscriber
from sonata.constants import EMOTIVE_THRESHOLD, LanguageCode


def main():
    parser = argparse.ArgumentParser(
        description="Convert and process audio with SONATA"
    )
    parser.add_argument("input_path", help="Path to input audio file (any format)")
    parser.add_argument(
        "--language",
        "-l",
        default=LanguageCode.KOREAN.value,
        choices=[lang.value for lang in LanguageCode],
        help=f"Language code (default: {LanguageCode.KOREAN.value})",
    )
    parser.add_argument("--output", "-o", help="Path to output JSON file")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=EMOTIVE_THRESHOLD,
        help=f"Threshold for emotive event detection (default: {EMOTIVE_THRESHOLD})",
    )

    args = parser.parse_args()

    # Convert audio to WAV format if it's not already
    if not args.input_path.lower().endswith(".wav"):
        print(f"Converting {args.input_path} to WAV format...")
        wav_path = convert_to_wav(args.input_path)
    else:
        wav_path = args.input_path

    # Process with SONATA
    print(f"Processing {wav_path} with SONATA...")
    transcriber = IntegratedTranscriber()
    result = transcriber.process_audio(
        audio_path=wav_path, language=args.language, emotive_threshold=args.threshold
    )

    # Set default output path if not specified
    if not args.output:
        # Create result directory if it doesn't exist
        if not os.path.exists("result"):
            os.makedirs("result")

        # Extract filename without extension
        base_filename = os.path.basename(args.input_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        args.output = f"result/{filename_without_ext}.json"

    # Save result
    transcriber.save_result(result, args.output)
    print(f"Results saved to {args.output}")

    # Print summary
    print("\n=== Transcription ===")
    print(result["integrated_transcript"]["plain_text"])

    print("\n=== Emotive Events ===")
    if result["emotive_events"]:
        for event in result["emotive_events"]:
            print(
                f"{event['start']:.2f}s - {event['end']:.2f}s: {event['type']} (confidence: {event['confidence']:.2f})"
            )
    else:
        print("No emotive events detected")

    print(
        f"\nUse lower threshold (--threshold) to detect more emotive events (current: {args.threshold})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
