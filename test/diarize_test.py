#!/usr/bin/env python3
import os
import argparse
import time
from sonata.core.transcriber import IntegratedTranscriber
from sonata.constants import DEFAULT_MODEL, DEFAULT_LANGUAGE, DEFAULT_DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="SONATA Speaker Diarization Test")

    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Path to output JSON file")
    parser.add_argument("-t", "--text-output", help="Path to output text file")
    parser.add_argument(
        "-l", "--language", default=DEFAULT_LANGUAGE, help="Language code"
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="ASR model name")
    parser.add_argument(
        "-d", "--device", default=DEFAULT_DEVICE, help="Device (cpu/cuda)"
    )
    parser.add_argument(
        "--hf-token", required=True, help="HuggingFace token for diarization models"
    )
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    parser.add_argument(
        "--format",
        choices=["concise", "default", "extended"],
        default="concise",
        help="Text output format",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1

    # Create default output paths if not specified
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_diarized.json"

    if not args.text_output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.text_output = f"{base_name}_diarized.txt"

    # Initialize transcriber
    print(f"Initializing transcriber with model {args.model} on {args.device}...")
    transcriber = IntegratedTranscriber(asr_model=args.model, device=args.device)

    # Start timing
    start_time = time.time()

    # Process audio with diarization
    print(f"Processing audio file: {args.input}")
    print(f"Language: {args.language}")
    print(
        f"Speaker range: {args.min_speakers or 'auto'} to {args.max_speakers or 'auto'}"
    )

    result = transcriber.process_audio(
        audio_path=args.input,
        language=args.language,
        diarize=True,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
    )

    # End timing
    elapsed_time = time.time() - start_time

    # Get speaker info
    speakers = set()
    for item in result["integrated_transcript"]["rich_text"]:
        if item["type"] == "word" and "speaker" in item:
            speakers.add(item["speaker"])

    # Save result to JSON
    print(f"Saving results to {args.output}")
    transcriber.save_result(result, args.output)

    # Save formatted transcript
    formatted_transcript = transcriber.get_formatted_transcript(result, args.format)
    with open(args.text_output, "w", encoding="utf-8") as f:
        f.write(formatted_transcript)
    print(f"Saved formatted transcript to {args.text_output}")

    # Print stats
    audio_duration = result["raw_asr"].get("duration", 0)
    print("\n--- Processing Statistics ---")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(
        f"Real-time factor: {elapsed_time / audio_duration if audio_duration > 0 else 'N/A':.2f}x"
    )
    print(f"Detected speakers: {len(speakers)}")
    print(f"Speaker IDs: {', '.join(sorted(speakers))}")

    # Print a sample of the transcript
    print("\n--- Transcript Sample ---")
    sample_lines = formatted_transcript.split("\n")[:20]
    for line in sample_lines:
        print(line)

    if len(sample_lines) < formatted_transcript.count("\n") + 1:
        print("...")

    print(f"\nFull transcript saved to: {args.text_output}")
    return 0


if __name__ == "__main__":
    exit(main())
