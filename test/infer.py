#!/usr/bin/env python
import os
import sys
import json
import argparse
from sonata.core.transcriber import IntegratedTranscriber


def parse_args():
    parser = argparse.ArgumentParser(description="SONATA Inference Tool")
    parser.add_argument("input", help="Path to input audio file (.wav format)")
    parser.add_argument("-o", "--output", help="Path to output JSON file")
    parser.add_argument(
        "-l", "--language", default="en", help="Language code (default: en)"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="large-v3",
        help="WhisperX model size (default: large-v3)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        help="Device to run models on (default: cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "-f", "--format", action="store_true", help="Also output a formatted text file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    # Create output filename if not specified
    if not args.output:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{input_basename}_transcript.json"

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize transcriber
    print(f"Initializing transcriber with {args.model} model on {args.device}...")
    transcriber = IntegratedTranscriber(asr_model=args.model, device=args.device)

    # Process audio
    print(f"Processing audio file: {args.input}")
    result = transcriber.process_audio(args.input, language=args.language)

    # Save results
    transcriber.save_result(result, args.output)
    print(f"Transcription saved to {args.output}")

    # Generate formatted text file if requested
    if args.format:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        formatted_output = f"{input_basename}_transcript.txt"
        formatted_text = transcriber.get_formatted_transcript(result)
        with open(formatted_output, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"Formatted transcript saved to {formatted_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
