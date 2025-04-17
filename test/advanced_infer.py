#!/usr/bin/env python
import os
import sys
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from sonata.core.transcriber import IntegratedTranscriber
from sonata.utils.audio import convert_audio_file, trim_silence


def parse_args():
    parser = argparse.ArgumentParser(description="SONATA Advanced Inference Tool")
    parser.add_argument(
        "input", help="Path to input audio file or directory of audio files"
    )
    parser.add_argument("-o", "--output-dir", help="Output directory for transcripts")
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
        "--format", action="store_true", help="Also output formatted text files"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess audio files (convert format and trim silence)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process multiple files in parallel"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of workers for parallel processing",
    )
    return parser.parse_args()


def process_file(input_file, args, output_dir):
    """Process a single audio file"""
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_json = os.path.join(output_dir, f"{input_basename}_transcript.json")

    # Preprocess audio if requested
    processed_file = input_file
    if args.preprocess:
        temp_wav = os.path.join(output_dir, f"{input_basename}_temp.wav")
        convert_audio_file(input_file, temp_wav)
        processed_file = trim_silence(temp_wav)

    # Initialize transcriber
    transcriber = IntegratedTranscriber(asr_model=args.model, device=args.device)

    # Process audio
    start_time = time.time()
    result = transcriber.process_audio(
        processed_file, language=args.language, batch_size=16
    )
    processing_time = time.time() - start_time

    # Save results
    transcriber.save_result(result, output_json)

    # Generate formatted text file if requested
    if args.format:
        formatted_output = os.path.join(output_dir, f"{input_basename}_transcript.txt")
        formatted_text = transcriber.get_formatted_transcript(result)
        with open(formatted_output, "w", encoding="utf-8") as f:
            f.write(formatted_text)

    # Cleanup temporary files
    if args.preprocess and processed_file != input_file:
        if os.path.exists(processed_file):
            os.remove(processed_file)

    return {
        "file": input_file,
        "output": output_json,
        "processing_time": processing_time,
        "word_count": len(result["integrated_transcript"]["rich_text"]),
    }


def main():
    args = parse_args()

    # Determine input files
    input_files = []
    if os.path.isdir(args.input):
        for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
            input_files.extend(list(Path(args.input).glob(f"**/*{ext}")))
    elif os.path.isfile(args.input):
        input_files = [args.input]
    else:
        print(f"Error: Input '{args.input}' does not exist or is not accessible.")
        return 1

    if not input_files:
        print(f"Error: No audio files found in '{args.input}'.")
        return 1

    # Create output directory
    output_dir = args.output_dir if args.output_dir else "sonata_output"
    os.makedirs(output_dir, exist_ok=True)

    # Process files
    results = []
    total_files = len(input_files)

    print(f"Found {total_files} audio files to process")

    if args.batch and total_files > 1:
        max_workers = min(args.max_workers, total_files)
        print(f"Processing {total_files} files using {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, file, args, output_dir): file
                for file in input_files
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    print(
                        f"[{i}/{total_files}] Processed: {os.path.basename(result['file'])} "
                        f"({result['processing_time']:.2f}s, {result['word_count']} words)"
                    )
                    results.append(result)
                except Exception as e:
                    file = futures[future]
                    print(
                        f"[{i}/{total_files}] Error processing {os.path.basename(file)}: {e}"
                    )
    else:
        print("Processing files sequentially")

        for i, file in enumerate(input_files, 1):
            try:
                print(f"[{i}/{total_files}] Processing: {os.path.basename(file)}")
                result = process_file(file, args, output_dir)
                print(
                    f"[{i}/{total_files}] Completed: {os.path.basename(file)} "
                    f"({result['processing_time']:.2f}s, {result['word_count']} words)"
                )
                results.append(result)
            except Exception as e:
                print(
                    f"[{i}/{total_files}] Error processing {os.path.basename(file)}: {e}"
                )

    # Save summary report
    summary = {
        "total_files": total_files,
        "processed_files": len(results),
        "total_processing_time": sum(r["processing_time"] for r in results),
        "total_words": sum(r["word_count"] for r in results),
        "results": results,
    }

    summary_path = os.path.join(output_dir, "transcription_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Processed {len(results)} of {total_files} files")
    print(f"Total processing time: {summary['total_processing_time']:.2f} seconds")
    print(f"Total words transcribed: {summary['total_words']}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary report: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
