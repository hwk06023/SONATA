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
        "--custom-thresholds",
        type=str,
        help="Path to JSON file with custom audio event thresholds",
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
    # Speaker diarization options
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization to identify different speakers",
    )
    parser.add_argument(
        "--min-speakers", type=int, help="Minimum number of speakers for diarization"
    )
    parser.add_argument(
        "--max-speakers", type=int, help="Maximum number of speakers for diarization"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Number of speakers if known (disables min/max speakers)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token for accessing diarization models (required for online diarization)",
    )

    # Offline diarization options
    parser.add_argument(
        "--offline-diarize",
        action="store_true",
        help="Use offline diarization (no HuggingFace token required after setup)",
    )
    parser.add_argument(
        "--offline-config",
        type=str,
        help="Path to offline diarization config file",
        default="~/.sonata/models/offline_config.yaml",
    )
    parser.add_argument(
        "--setup-offline",
        action="store_true",
        help="Download and setup offline diarization models",
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
    print(
        "  --diarize               Enable speaker diarization to identify different speakers"
    )
    print("\nDiarization options:")
    print("  --hf-token TOKEN        HuggingFace token (for online diarization)")
    print(
        "  --offline-diarize       Use offline diarization (no token needed after setup)"
    )
    print(
        "  --setup-offline         Download and set up offline diarization models (one-time setup)"
    )
    print("\nFor more options:")
    print("  sonata-asr --help")
    print("\nExamples:")
    print("  sonata-asr input.wav")
    print("  sonata-asr input.wav -o transcript.json")
    print("  sonata-asr input.wav -d cuda --preprocess")
    print("  sonata-asr input.wav --format concise --text-output transcript.txt")
    print("  sonata-asr input.wav --diarize --hf-token YOUR_HF_TOKEN")
    print("  sonata-asr input.wav --diarize --offline-diarize")
    sys.exit(1)


def main():
    args = parse_args()

    # Show version if requested
    if args.version:
        # First check the package's own version
        print(f"SONATA v{__version__}")
        sys.exit(0)

    # Handle offline diarization setup
    if args.setup_offline:
        if not args.hf_token:
            print(
                "Error: HuggingFace token is required for initial setup of offline diarization"
            )
            print("Please provide a token with the --hf-token option")
            sys.exit(1)

        try:
            from sonata.core.offline_diarization import download_diarization_models

            print("Setting up offline diarization models...")
            save_dir = os.path.expanduser("~/.sonata/models")
            result = download_diarization_models(args.hf_token, save_dir)

            print(f"Offline diarization models setup complete!")
            print(f"Configuration saved to: {result['config_path']}")
            print(f"Model saved to: {result['model_path']}")
            print("\nTo use offline diarization, run with:")
            print(f"  sonata-asr path/to/audio.wav --diarize --offline-diarize")
            sys.exit(0)
        except Exception as e:
            print(f"Error setting up offline diarization models: {str(e)}")
            sys.exit(1)

    # If no input file is provided, show usage and exit
    if not args.input:
        show_usage_and_exit()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        show_usage_and_exit()

    # Check diarization token requirements
    if args.diarize and not args.offline_diarize and not args.hf_token:
        print("Error: Speaker diarization requires either:")
        print("  - A HuggingFace token (--hf-token) for online mode")
        print("  - Offline mode (--offline-diarize) with pre-configured models")
        print(
            "To set up offline mode, run: sonata-asr --setup-offline --hf-token YOUR_HF_TOKEN"
        )
        sys.exit(1)

    # Verify offline configuration if requested
    if args.offline_diarize:
        if not os.path.exists(os.path.expanduser(args.offline_config)):
            print(f"Error: Offline configuration file {args.offline_config} not found")
            print("Please run: sonata-asr --setup-offline --hf-token YOUR_HF_TOKEN")
            sys.exit(1)

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

    # Load custom thresholds if specified
    custom_thresholds = None
    if args.custom_thresholds:
        if not os.path.exists(args.custom_thresholds):
            print(
                f"Error: Custom thresholds file '{args.custom_thresholds}' does not exist."
            )
            sys.exit(1)

        try:
            with open(args.custom_thresholds, "r") as f:
                custom_thresholds = json.load(f)
            print(f"Loaded custom thresholds for {len(custom_thresholds)} event types")
        except Exception as e:
            print(f"Error loading custom thresholds: {str(e)}")
            sys.exit(1)

    # Initialize the transcriber
    print(f"Initializing transcriber with {args.model} model on {args.device}...")
    transcriber = IntegratedTranscriber(
        asr_model=args.model,
        audio_model_path=args.audio_model,
        device=args.device,
        offline_diarization=args.offline_diarize,
        offline_config_path=args.offline_config if args.offline_diarize else None,
        custom_audio_thresholds=custom_thresholds,
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
                diarize=args.diarize,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                hf_token=args.hf_token,
            )
            # Add segment info to result
            segment_result["segment_info"] = {
                "index": i,
                "start": segment["start"],
                "end": segment["end"],
                "overlap_start": segment["overlap_start"],
                "overlap_end": segment["overlap_end"],
            }
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
            diarize=args.diarize,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            hf_token=args.hf_token,
        )

    # Save results
    print(f"Saving transcript to {args.output}...")
    transcriber.save_result(result, args.output)

    # Save formatted text if requested
    formatted_transcript = transcriber.get_formatted_transcript(result, args.format)
    if args.text_output or text_output:
        output_path = args.text_output or text_output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)
        print(f"Saved formatted transcript to {output_path}")

    # Print a preview of the transcript
    print("\nTranscript preview:")
    plain_text = result["integrated_transcript"]["plain_text"]
    print_preview = plain_text[:500] + "..." if len(plain_text) > 500 else plain_text
    print(print_preview)

    # Print completion message
    print(f"\nProcessing complete. Full results saved to {args.output}")
    if args.text_output or text_output:
        print(f"Text transcript saved to {args.text_output or text_output}")

    return None


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
