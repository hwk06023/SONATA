#!/usr/bin/env python
import os
import sys
import argparse
import tempfile
import subprocess
from sonata.core.transcriber import IntegratedTranscriber


def parse_args():
    parser = argparse.ArgumentParser(description="SONATA AST Inference Tool")
    parser.add_argument("input", help="Path to input audio file (.wav format)")
    parser.add_argument("-o", "--output", help="Path to output JSON file")
    parser.add_argument(
        "-l", "--language", help="Language code (e.g., en, ko, ja, etc.)"
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
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    return parser.parse_args()


def convert_to_wav(input_path):
    if input_path.lower().endswith(".wav"):
        return input_path
    output_path = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"Converted {input_path} to WAV format at {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting file to WAV: {e}")
        return input_path


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        logging.error(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    if not args.output:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{input_basename}_transcript.json"
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Initializing transcriber with {args.model} model on {args.device}...")
    transcriber = IntegratedTranscriber(asr_model=args.model, device=args.device)
    processed_file = convert_to_wav(args.input)
    temp_files = []
    if processed_file != args.input:
        temp_files.append(processed_file)
    kwargs = {}
    if args.language:
        kwargs["language"] = args.language
        print(f"Using language: {args.language}")
    kwargs["batch_size"] = args.batch_size
    print(f"Using batch size: {args.batch_size}")
    print(f"Processing audio file: {processed_file}")
    result = transcriber.process_audio(processed_file, **kwargs)
    transcriber.save_result(result, args.output)
    print(f"Transcription saved to {args.output}")
    if args.format:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        formatted_output = f"{input_basename}_transcript.txt"
        formatted_text = transcriber.get_formatted_transcript(result)
        with open(formatted_output, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"Formatted transcript saved to {formatted_output}")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
