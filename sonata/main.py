import os
import argparse
import sys
import json
from sonata.core.transcriber import IntegratedTranscriber
from sonata.utils.audio import convert_audio_file, split_audio, trim_silence
from sonata.constants import (
    AUDIO_EVENT_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    FORMAT_DEFAULT,
    FORMAT_CONCISE,
    FORMAT_EXTENDED,
    DEFAULT_SPLIT_LENGTH,
    DEFAULT_SPLIT_OVERLAP,
    LanguageCode,
    FormatType,
)
from sonata import __version__


def parse_args():
    parser = argparse.ArgumentParser(
        description="SONATA: SOund and Narrative Advanced Transcription Assistant"
    )

    parser.add_argument("input", nargs="?", help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Path to output JSON file")
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        choices=[lang.value for lang in LanguageCode],
        help=f"Language code (default: {DEFAULT_LANGUAGE}, options: {', '.join([lang.value for lang in LanguageCode])})",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"WhisperX model size (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Device to run models on (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "-e", "--audio-model", help="Path to audio event detection model"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=AUDIO_EVENT_THRESHOLD,
        help=f"Threshold for audio event detection (default: {AUDIO_EVENT_THRESHOLD})",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=[format_type.value for format_type in FormatType],
        default=FORMAT_DEFAULT,
        help=(
            "Format for text output: "
            "concise (simple text with audio event tags), "
            "default (text with timestamps), "
            "extended (with confidence scores)"
        ),
    )
    parser.add_argument(
        "--text-output",
        type=str,
        help="Path to save formatted transcript text file",
        default=None,
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
        default=DEFAULT_SPLIT_LENGTH,
        help=f"Length of split segments in seconds (default: {DEFAULT_SPLIT_LENGTH})",
    )
    parser.add_argument(
        "--split-overlap",
        type=int,
        default=DEFAULT_SPLIT_OVERLAP,
        help=f"Overlap between split segments in seconds (default: {DEFAULT_SPLIT_OVERLAP})",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show SONATA version and exit"
    )

    return parser.parse_args()


def show_usage_and_exit():
    print("SONATA: SOund and Narrative Advanced Transcription Assistant")
    print("\nBasic usage:")
    print("  sonata-asr path/to/audio.wav")
    print("\nCommon options:")
    print("  -o, --output [FILE]     Save transcript to specified JSON file")
    print("  -d, --device [DEVICE]   Use specified device (cpu/cuda)")
    print(
        f"  -l, --language [LANG]   Specify language code (default: {DEFAULT_LANGUAGE})"
    )
    print("  --preprocess            Convert and trim silence before processing")
    print("  --format [TYPE]         Choose transcript format:")
    print("                           - concise: Text with integrated audio event tags")
    print("                           - default: Text with timestamps")
    print("                           - extended: Includes confidence scores")
    print("  --text-output [FILE]    Save formatted transcript to specified text file")
    print("\nFor more options:")
    print("  sonata-asr --help")
    print("\nExamples:")
    print("  sonata-asr input.wav")
    print("  sonata-asr input.wav -o transcript.json")
    print("  sonata-asr input.wav -d cuda --preprocess")
    print("  sonata-asr input.wav --format concise --text-output transcript.txt")
    sys.exit(1)


def main():
    args = parse_args()

    # Show version if requested
    if args.version:
        # First check the package's own version
        print(f"SONATA v{__version__}")
        sys.exit(0)

    # If no input file is provided, show usage and exit
    if not args.input:
        show_usage_and_exit()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        show_usage_and_exit()

    # Create output filenames if not specified
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    if not args.output:
        args.output = f"{input_basename}_transcript.json"

    # Set up text output path
    text_output = args.text_output
    if text_output is None:
        text_output = f"{input_basename}_transcript.txt"

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
        asr_model=args.model, audio_model_path=args.audio_model, device=args.device
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
                audio_path=segment["path"],
                language=args.language,
                audio_threshold=args.threshold,
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
            audio_path=input_file,
            language=args.language,
            audio_threshold=args.threshold,
        )

    # Save results
    transcriber.save_result(result, args.output)
    print(f"Transcription saved to {args.output}")

    # Generate formatted text file
    print(f"Generating {args.format} format transcript...")
    formatted_transcript = transcriber.get_formatted_transcript(
        result, format_type=args.format
    )

    print(f"Saving formatted transcript to: {text_output}")
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(formatted_transcript)


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

    # Merge audio events
    all_audio_events = merged_result["audio_events"]
    for segment in segment_results[1:]:
        all_audio_events.extend(segment["audio_events"])

    merged_result["audio_events"] = all_audio_events

    return merged_result


if __name__ == "__main__":
    main()
