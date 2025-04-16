#!/usr/bin/env python
"""
SONATA Usage Example

This script demonstrates how to use the SONATA package in your Python code.
"""
import os
import json
from sonata.core.transcriber import IntegratedTranscriber
from sonata.utils.audio import convert_audio_file


def simple_example(input_file, output_file=None):
    """
    Demonstrate basic usage of SONATA for transcription.

    Args:
        input_file: Path to audio file
        output_file: Path to save JSON result (default: derived from input filename)
    """
    # Derive output filename if not provided
    if not output_file:
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{input_basename}_output.json"

    # Initialize the transcriber with default settings
    # You can customize the model, device, etc.
    transcriber = IntegratedTranscriber(
        asr_model="large-v3",  # Options: tiny, base, small, medium, large-v1, large-v2, large-v3
        device="cpu",  # Use "cuda" for GPU acceleration
        compute_type="float32",  # Precision type
    )

    # Process the audio file
    result = transcriber.process_audio(
        input_file,
        language="en",  # Language code: en, fr, de, es, etc.
        emotive_threshold=0.5,  # Threshold for emotive event detection
    )

    # Save the result to JSON
    transcriber.save_result(result, output_file)

    # Get a formatted transcript
    formatted_text = transcriber.get_formatted_transcript(result)

    # Return results
    return {
        "result": result,
        "output_file": output_file,
        "formatted_text": formatted_text,
    }


def extract_transcript_only(input_file):
    """
    Extract only the plain text transcript from an audio file.

    Args:
        input_file: Path to audio file

    Returns:
        str: The transcribed text
    """
    transcriber = IntegratedTranscriber()
    result = transcriber.process_audio(input_file)
    return result["integrated_transcript"]["plain_text"]


def process_with_timestamp_extraction(input_file):
    """
    Process audio and extract words with their timestamps.

    Args:
        input_file: Path to audio file

    Returns:
        list: Words with timestamps
    """
    transcriber = IntegratedTranscriber()
    result = transcriber.process_audio(input_file)

    # Extract words with timestamps
    words_with_timestamps = []
    for item in result["integrated_transcript"]["rich_text"]:
        if item["type"] == "word":
            words_with_timestamps.append(
                {"word": item["content"], "start": item["start"], "end": item["end"]}
            )

    return words_with_timestamps


if __name__ == "__main__":
    # Example file path - replace with your own audio file
    audio_file = "../sonata/data/podcast_bobbylee.wav"

    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} does not exist.")
        print("Please modify the script to point to a valid audio file.")
        exit(1)

    print("Running SONATA example...")

    # Run the simple example
    result = simple_example(audio_file)

    # Print output information
    print(f"Transcription saved to: {result['output_file']}")
    print("\nSample of formatted text:")
    print(result["formatted_text"][:500] + "...\n")

    # Extract just the transcript
    plain_text = extract_transcript_only(audio_file)
    print("Plain text transcript (first 200 chars):")
    print(plain_text[:200] + "...\n")

    # Extract words with timestamps
    words = process_with_timestamp_extraction(audio_file)
    print(f"Extracted {len(words)} words with timestamps")
    print("First 5 words with timestamps:")
    for i, word in enumerate(words[:5]):
        print(f"  {word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")

    print("\nExample completed successfully!")
