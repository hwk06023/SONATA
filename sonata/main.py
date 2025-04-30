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
        "--text-output",
        action="store_true",
        help="Save formatted transcript to text file (default: input_name.txt)",
    )
    parser.add_argument(
        "--format",
        nargs="?",
        const=FORMAT_DEFAULT,
        choices=[FORMAT_CONCISE, FORMAT_DEFAULT, FORMAT_EXTENDED],
        help=f"Save formatted transcript with specified format (default: {FORMAT_DEFAULT})",
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

    # Updated diarization options
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable SOTA speaker diarization using Silero VAD and WavLM embeddings",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Number of speakers if known (estimated automatically if not provided)",
    )

    # Deep detection option
    parser.add_argument(
        "--deep-detect",
        action="store_true",
        help="Enable multi-scale audio event detection with parallel window sizes for better paralinguistic detection",
    )
    parser.add_argument(
        "--deep-detect-scales",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Number of scales to use for deep detection (default: 3, fewer scales = faster processing)",
    )
    parser.add_argument(
        "--deep-detect-window-sizes",
        type=str,
        default="0.2,1.0,2.5",
        help="Comma-separated list of window sizes in seconds for deep detection (default: 0.2,1.0,2.5)",
    )
    parser.add_argument(
        "--deep-detect-hop-sizes",
        type=str,
        default="0.1,0.5,1.0",
        help="Comma-separated list of hop sizes in seconds for deep detection (default: 0.1,0.5,1.0)",
    )
    parser.add_argument(
        "--deep-detect-parallel",
        action="store_true",
        help="Use parallel processing for multi-scale detection (automatically enables --deep-detect)",
    )
    parser.add_argument(
        "--deep-detect-progress",
        action="store_true",
        help="Show detailed progress bars for deep detection processing",
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
    print(
        "  --text-output            Save transcript to text file (defaults to input_name.txt)"
    )
    print(
        "  --format [TYPE]          Save formatted transcript (concise, default, extended)"
    )
    print(
        "  --diarize               Enable speaker diarization to identify different speakers"
    )
    print("\nDiarization options:")
    print("  --num-speakers [NUM]    Set exact number of speakers (optional)")
    print("\nFor more options:")
    print("  sonata-asr --help")
    print("\nExamples:")
    print("  sonata-asr input.wav")
    print("  sonata-asr input.wav -o transcript.json")
    print("  sonata-asr input.wav -d cuda --preprocess")
    print("  sonata-asr input.wav --text-output")
    print("  sonata-asr input.wav --format concise")
    print("  sonata-asr input.wav --diarize")
    print("  sonata-asr input.wav --diarize --num-speakers 3")
    sys.exit(1)


def main():
    args = parse_args()

    # Show version if requested
    if args.version:
        # First check the package's own version
        print(f"SONATA v{__version__}")
        sys.exit(0)

    # If deep-detect-parallel is set, automatically enable deep-detect
    if args.deep_detect_parallel:
        args.deep_detect = True

    # Show usage if no input file is provided
    if not args.input:
        show_usage_and_exit()

    # Set up deep detect parameters
    deep_detect_params = {}
    if args.deep_detect:
        deep_detect_params = {
            "window_sizes": [
                float(x.strip()) for x in args.deep_detect_window_sizes.split(",")
            ],
            "hop_sizes": [
                float(x.strip()) for x in args.deep_detect_hop_sizes.split(",")
            ],
            "parallel": args.deep_detect_parallel,
            "show_progress": args.deep_detect_progress,
        }

    # Custom audio event thresholds
    custom_thresholds = None
    if args.custom_thresholds:
        try:
            with open(args.custom_thresholds, "r") as f:
                custom_thresholds = json.load(f)
            print(
                f"Loaded custom thresholds for {len(custom_thresholds)} audio event types"
            )
        except Exception as e:
            print(f"Error loading custom thresholds: {str(e)}")
            sys.exit(1)

    # Initialize the transcriber
    transcriber = IntegratedTranscriber(
        asr_model=args.model,
        audio_model_path=args.audio_model,
        device=args.device,
        custom_audio_thresholds=custom_thresholds,
        deep_detect=args.deep_detect,
        deep_detect_params=deep_detect_params,
    )

    # Process file or directory
    if os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")

        if args.output and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, args.input)
                    base_name = os.path.splitext(rel_path)[0]

                    if args.output:
                        output_path = os.path.join(args.output, f"{base_name}.json")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_dir = os.path.dirname(input_path)
                        output_path = os.path.join(
                            output_dir, f"{os.path.basename(base_name)}.json"
                        )

                    process_file(
                        transcriber,
                        input_path,
                        output_path,
                        args,
                    )
    else:
        # Process a single file
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        # Set default output path if not provided
        output_path = args.output
        if not output_path:
            base_name = os.path.splitext(args.input)[0]
            output_path = f"{base_name}.json"

        # Process the audio file
        process_file(
            transcriber,
            args.input,
            output_path,
            args,
        )


def process_file(transcriber, input_path, output_path, args):
    """Process a single audio file"""
    print(f"Processing: {input_path}")

    # Preprocessing if requested
    processed_path = input_path
    if args.preprocess:
        print("Preprocessing audio...")
        # Convert to WAV
        wav_path = convert_audio_file(input_path)
        # Trim silence
        processed_path = trim_silence(wav_path)
        print(f"Preprocessed audio saved to: {processed_path}")

    # Set default text output path
    base_name = os.path.splitext(output_path)[0]
    text_output = f"{base_name}.txt"

    # For --format option, use a different filename
    if args.format:
        text_output = f"{base_name}_formatted.txt"

    # Handle splitting if requested
    if args.split:
        print(f"Splitting audio into {args.split_length}s segments...")
        segments = split_audio(processed_path, args.split_length, args.split_overlap)

        results = []
        for i, segment_path in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)}: {segment_path}")

            # Process each segment
            segment_result = transcriber.process_audio(
                audio_path=segment_path,
                language=args.language,
                audio_threshold=args.threshold,
                diarize=args.diarize,
                num_speakers=args.num_speakers,
            )

            # Add segment info
            segment_result["segment"] = {
                "index": i,
                "path": segment_path,
                "total_segments": len(segments),
            }

            results.append(segment_result)

        # Save segment results individually
        base_path = os.path.splitext(output_path)[0]
        for i, result in enumerate(results):
            segment_output = f"{base_path}_segment_{i+1}.json"
            transcriber.save_result(result, segment_output)

        # Merge results
        merged_result = merge_segment_results(results)
        transcriber.save_result(merged_result, output_path)

        # Save text output if requested
        if args.text_output:
            text_content = transcriber.get_plain_transcript(merged_result)
            with open(text_output, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"Formatted transcript saved to: {text_output}")

        # Save formatted text if requested
        if args.format:
            formatted_text = transcriber.get_formatted_transcript(
                merged_result, args.format
            )
            with open(text_output, "w", encoding="utf-8") as f:
                f.write(formatted_text)
            print(f"Formatted transcript saved to: {text_output}")

        print(f"Merged transcript saved to: {output_path}")
    else:
        # Process the entire file
        result = transcriber.process_audio(
            audio_path=processed_path,
            language=args.language,
            audio_threshold=args.threshold,
            diarize=args.diarize,
            num_speakers=args.num_speakers,
        )

        # Save result
        transcriber.save_result(result, output_path)
        print(f"Transcript saved to: {output_path}")

        # Save text output if requested
        if args.text_output:
            text_content = transcriber.get_plain_transcript(result)
            with open(text_output, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"Formatted transcript saved to: {text_output}")

        # Save formatted text if requested
        if args.format:
            formatted_text = transcriber.get_formatted_transcript(result, args.format)
            with open(text_output, "w", encoding="utf-8") as f:
                f.write(formatted_text)
            print(f"Formatted transcript saved to: {text_output}")


def merge_segment_results(segment_results):
    """Merge results from multiple segments into a single output"""
    if not segment_results:
        return {
            "segments": [],
            "audio_events": [],
            "integrated_transcript": {"plain_text": "", "rich_text": []},
        }

    # Start with the first segment
    merged = {
        "segments": [],
        "audio_events": [],
        "integrated_transcript": {"plain_text": "", "rich_text": []},
    }

    # Track time offset for each segment
    time_offset = 0

    for result in segment_results:
        # Adjust timestamps in segments
        for segment in result.get("segments", []):
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += time_offset
            adjusted_segment["end"] += time_offset

            if "words" in adjusted_segment:
                for word in adjusted_segment["words"]:
                    word["start"] += time_offset
                    word["end"] += time_offset

            merged["segments"].append(adjusted_segment)

        # Adjust timestamps in audio events
        for event in result.get("audio_events", []):
            adjusted_event = event.copy()
            adjusted_event["start"] += time_offset
            adjusted_event["end"] += time_offset
            merged["audio_events"].append(adjusted_event)

        # Get the maximum time from this segment
        max_time = 0
        for segment in result.get("segments", []):
            max_time = max(max_time, segment.get("end", 0))

        for event in result.get("audio_events", []):
            max_time = max(max_time, event.get("end", 0))

        # Update offset for next segment (accounting for overlap)
        time_offset += max_time

        # Append to integrated transcript
        if "integrated_transcript" in result:
            if merged["integrated_transcript"]["plain_text"]:
                merged["integrated_transcript"]["plain_text"] += " "
            merged["integrated_transcript"]["plain_text"] += result[
                "integrated_transcript"
            ].get("plain_text", "")

            merged["integrated_transcript"]["rich_text"].extend(
                result["integrated_transcript"].get("rich_text", [])
            )

    return merged


if __name__ == "__main__":
    main()
