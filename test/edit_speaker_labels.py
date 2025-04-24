#!/usr/bin/env python
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonata.utils.transcript_editor import TranscriptEditor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edit speaker labels in transcription results"
    )
    parser.add_argument("input", help="Input JSON transcript file")
    parser.add_argument(
        "-o", "--output", help="Output file path (defaults to input_edited.json)"
    )

    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument(
        "--list", action="store_true", help="List all speakers and structure"
    )
    operations.add_argument(
        "--rename", nargs=2, metavar=("OLD", "NEW"), help="Rename a speaker"
    )
    operations.add_argument(
        "--merge",
        nargs="+",
        metavar="SPEAKER",
        help="Merge speakers (requires --target)",
    )
    operations.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Reassign speakers in time range (requires --assign)",
    )
    operations.add_argument(
        "--segments",
        nargs="+",
        type=int,
        metavar="IDX",
        help="Reassign speakers for specific segments (requires --assign)",
    )

    parser.add_argument("--target", help="Target speaker name for merge operation")
    parser.add_argument(
        "--assign", help="Speaker to assign for reassignment operations"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return 1

    # Set default output if not specified
    if not args.output:
        base_name, ext = os.path.splitext(args.input)
        args.output = f"{base_name}_edited{ext}"

    # Initialize editor
    try:
        editor = TranscriptEditor(args.input)
        print(f"Loaded transcript file: {args.input}")

        # Get current speakers for display and validation
        speakers = editor.get_speakers()
        print(f"Found {len(speakers)} speakers: {', '.join(speakers)}")

        # List structure (analyze mode)
        if args.list:
            structure = editor.get_transcript_structure()
            print("\nTranscript structure:")
            print(f"- Has raw ASR data: {structure['has_raw_asr']}")
            print(f"- Has integrated transcript: {structure['has_integrated']}")

            if structure["has_raw_asr"] and "segments_count" in structure:
                print(f"- Segments count: {structure['segments_count']}")

            if structure["has_integrated"] and "words_count" in structure:
                print(f"- Words count: {structure['words_count']}")

            print("\nNo modifications made.")
            return 0

        # Rename speaker
        elif args.rename:
            old_name, new_name = args.rename
            if old_name not in speakers:
                print(f"Error: Speaker '{old_name}' not found")
                return 1

            changes = editor.rename_speaker(old_name, new_name)
            print(f"Renamed speaker '{old_name}' to '{new_name}' ({changes} instances)")

        # Merge speakers
        elif args.merge:
            if not args.target:
                print(
                    "Error: --merge requires --target to specify the target speaker name"
                )
                return 1

            missing = [s for s in args.merge if s not in speakers]
            if missing:
                print(f"Error: These speakers were not found: {', '.join(missing)}")
                return 1

            changes = editor.merge_speakers(args.merge, args.target)
            print(
                f"Merged speakers {', '.join(args.merge)} into '{args.target}' ({changes} changes)"
            )

        # Reassign by time range
        elif args.time_range:
            if not args.assign:
                print("Error: --time-range requires --assign to specify the speaker")
                return 1

            start, end = args.time_range
            changes = editor.reassign_speaker_by_time(start, end, args.assign)
            print(
                f"Assigned speaker '{args.assign}' to content between {start}s and {end}s ({changes} changes)"
            )

        # Reassign by segments
        elif args.segments:
            if not args.assign:
                print("Error: --segments requires --assign to specify the speaker")
                return 1

            changes = editor.reassign_speaker_by_segments(args.segments, args.assign)
            print(
                f"Assigned speaker '{args.assign}' to segments {args.segments} ({changes} changes)"
            )

        # Save the edited transcript
        editor.save_transcript(args.output)
        print(f"Saved edited transcript to: {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
