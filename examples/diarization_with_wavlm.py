#!/usr/bin/env python3
"""
Example script showing SONATA diarization with WavLM embeddings
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sonata.main import process_file
from sonata.core.transcriber import IntegratedTranscriber


def main():
    parser = argparse.ArgumentParser(description="SONATA with WavLM diarization")
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument(
        "--output", "-o", help="Output file path (default: based on input name)"
    )
    parser.add_argument(
        "--model",
        choices=["titanet", "wavlm-base-plus-sv"],
        default="wavlm-base-plus-sv",
        help="Speaker embedding model to use",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Number of speakers (estimated if not provided)",
    )
    parser.add_argument("--language", "-l", default="en", help="Language of the audio")
    parser.add_argument(
        "--device", "-d", default="cpu", help="Device to use (cpu or cuda)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set default output path if not provided
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_wavlm.json"

    # Initialize transcriber
    transcriber = IntegratedTranscriber(device=args.device)

    # Use our custom argument structure but equivalent to sonata.main.process_file
    class Args:
        def __init__(self):
            self.diarize = True
            self.diarize_model = args.model
            self.num_speakers = args.num_speakers
            self.save_steps = True
            self.language = args.language
            self.audio = True
            self.threshold = 0.5
            self.text_output = True
            self.format = "default"

    temp_args = Args()

    # Process the file with diarization
    print(f"Processing: {args.input}")
    print(f"Using diarization model: {args.model}")

    # Process audio file
    result = transcriber.process_audio(
        audio_path=args.input,
        language=args.language,
        audio_threshold=0.5,
        diarize=True,
        num_speakers=args.num_speakers,
        save_diarization_steps=True,
        detect_audio_events=True,
        diarize_model=args.model,
    )

    # Save result to JSON
    transcriber.save_result(result, output_path)
    print(f"Transcript saved to: {output_path}")

    # Save text output
    base_name = os.path.splitext(output_path)[0]
    text_output = f"{base_name}.txt"
    text_content = transcriber.get_formatted_transcript(result)
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(text_content)
    print(f"Formatted transcript saved to: {text_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
