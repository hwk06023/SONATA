import os
import argparse
import json
from sonata.core.transcriber import IntegratedTranscriber
from sonata.utils.audio import convert_audio_file, split_audio, trim_silence


def parse_args():
    parser = argparse.ArgumentParser(
        description="SONATA: SOund and Narrative Advanced Transcription Assistant"
    )

    parser.add_argument("input", help="Path to input audio file")
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
        "-d", "--device", default="cpu", help="Device to run models on (default: cpu)"
    )
    parser.add_argument("-e", "--emotive-model", help="Path to emotive detection model")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for emotive event detection (default: 0.5)",
    )
    parser.add_argument(
        "--format", action="store_true", help="Also output a formatted text file"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess audio (convert format and trim silence)",
    )
    parser.add_argument(
        "--split", action="store_true", help="Split long audio into segments"
    )
    parser.add_argument(
        "--split-length",
        type=int,
        default=30,
        help="Length of split segments in seconds (default: 30)",
    )
    parser.add_argument(
        "--split-overlap",
        type=int,
        default=5,
        help="Overlap between split segments in seconds (default: 5)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output filenames if not specified
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    if not args.output:
        args.output = f"{input_basename}_transcript.json"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preprocess audio if requested
    input_file = args.input
    if args.preprocess:
        print("Preprocessing audio...")
        # Convert to WAV format
        temp_wav = f"{input_basename}_temp.wav"
        convert_audio_file(input_file, temp_wav)

        # Trim silence
        input_file = trim_silence(temp_wav)
        print(f"Preprocessed audio saved to {input_file}")

    # Initialize the transcriber
    print(f"Initializing transcriber with {args.model} model on {args.device}...")
    transcriber = IntegratedTranscriber(
        asr_model=args.model, emotive_model_path=args.emotive_model, device=args.device
    )

    # Process audio
    if args.split and os.path.getsize(input_file) > 10 * 1024 * 1024:  # If file > 10MB
        print("Splitting large audio file...")
        split_dir = f"{input_basename}_splits"
        segments = split_audio(
            input_file,
            split_dir,
            segment_length=args.split_length,
            overlap=args.split_overlap,
        )

        # Process each segment
        print(f"Processing {len(segments)} segments...")
        all_results = []

        for i, segment in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)}...")
            segment_result = transcriber.process_audio(
                segment["path"],
                language=args.language,
                emotive_threshold=args.threshold,
            )

            # Adjust timestamps to account for segment start time
            offset = segment["start_time"]
            for word in segment_result["integrated_transcript"]["rich_text"]:
                word["start"] += offset
                word["end"] += offset

            all_results.append(segment_result)

        # Merge results
        merged_result = merge_segment_results(all_results)
        result = merged_result
    else:
        print("Processing audio...")
        result = transcriber.process_audio(
            input_file, language=args.language, emotive_threshold=args.threshold
        )

    # Save results
    transcriber.save_result(result, args.output)
    print(f"Transcription saved to {args.output}")

    # Generate formatted text file if requested
    if args.format:
        formatted_output = f"{input_basename}_transcript.txt"
        formatted_text = transcriber.get_formatted_transcript(result)
        with open(formatted_output, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"Formatted transcript saved to {formatted_output}")


def merge_segment_results(segment_results):
    """Merge results from multiple audio segments."""
    if not segment_results:
        return None

    # Start with the first segment
    merged_result = segment_results[0]

    # Merge rich text from all segments
    rich_text = merged_result["integrated_transcript"]["rich_text"]

    for segment in segment_results[1:]:
        rich_text.extend(segment["integrated_transcript"]["rich_text"])

    # Sort by start time
    rich_text.sort(key=lambda x: x["start"])

    # Regenerate plain text
    plain_text = " ".join([item["content"] for item in rich_text])

    # Update merged result
    merged_result["integrated_transcript"] = {
        "plain_text": plain_text,
        "rich_text": rich_text,
    }

    # Merge emotive events
    all_emotive_events = merged_result["emotive_events"]
    for segment in segment_results[1:]:
        all_emotive_events.extend(segment["emotive_events"])

    merged_result["emotive_events"] = all_emotive_events

    return merged_result


if __name__ == "__main__":
    main()
