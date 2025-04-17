#!/usr/bin/env python
import os
import sys
import json
import argparse
import tempfile
import wave
import subprocess
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from sonata.core.transcriber import IntegratedTranscriber
from sonata.constants import (
    FORMAT_DEFAULT,
    FORMAT_CONCISE,
    FORMAT_EXTENDED,
    LanguageCode,
    FormatType,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SONATA Inference Tool")
    parser.add_argument("input", help="Path to input audio file (.wav format)")
    parser.add_argument("-o", "--output", help="Path to output JSON file")
    parser.add_argument(
        "-l",
        "--language",
        default=LanguageCode.ENGLISH.value,
        choices=[lang.value for lang in LanguageCode],
        help=f"Language code (default: {LanguageCode.ENGLISH.value}, options: {', '.join([lang.value for lang in LanguageCode])})",
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
        "-f",
        "--format",
        choices=[format_type.value for format_type in FormatType],
        default=FormatType.DEFAULT.value,
        help="Format for text output (default: default, options: concise, extended)",
    )
    parser.add_argument(
        "--text-output",
        help="Path to save formatted transcript text file",
        default=None,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply preprocessing to the audio file",
    )
    parser.add_argument(
        "--split", action="store_true", help="Split long audio files before processing"
    )
    parser.add_argument(
        "--split-length",
        type=int,
        default=30,
        help="Length in seconds of each split (default: 30)",
    )
    parser.add_argument(
        "--split-overlap",
        type=int,
        default=5,
        help="Overlap in seconds between splits (default: 5)",
    )
    return parser.parse_args()


def convert_to_wav(input_path):
    """Convert audio file to WAV format."""
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
        print(f"Error converting file to WAV: {e}")
        return input_path


def trim_silence(input_path):
    """Trim silence from the beginning and end of the audio file."""
    output_path = tempfile.mktemp(suffix=".wav")
    try:
        # Use ffmpeg to detect and trim silence
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_path,
                "-af",
                "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,aformat=dblp,areverse",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"Trimmed silence from {input_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error trimming silence: {e}")
        return input_path


def split_audio(input_path, split_length, split_overlap):
    """Split audio file into smaller chunks with overlap."""
    # Get audio duration
    with wave.open(input_path, "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)

    # If audio is shorter than split length, return original
    if duration <= split_length:
        return [input_path]

    # Calculate number of splits
    num_splits = max(
        1, int((duration - split_overlap) / (split_length - split_overlap))
    )
    split_files = []

    for i in range(num_splits):
        start_time = i * (split_length - split_overlap)
        end_time = min(start_time + split_length, duration)

        # Create output file for this split
        split_path = tempfile.mktemp(suffix=f"_split_{i}.wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-ss",
                    str(start_time),
                    "-to",
                    str(end_time),
                    "-c",
                    "copy",
                    split_path,
                ],
                check=True,
                capture_output=True,
            )
            split_files.append(split_path)
            print(
                f"Created split {i+1}/{num_splits}: {start_time:.1f}s to {end_time:.1f}s"
            )
        except subprocess.CalledProcessError as e:
            print(f"Error creating split {i}: {e}")

    return split_files


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

    # Set up text output path
    text_output = args.text_output
    if text_output is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        text_output = f"{input_basename}_transcript.txt"

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize transcriber
    print(f"Initializing transcriber with {args.model} model on {args.device}...")

    # Set up comprehensive warning suppression
    original_level = logging.getLogger().level
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        # Temporarily suppress all logging
        logging.getLogger().setLevel(logging.ERROR)

        # Suppress PyTorch Lightning and other package warnings
        # Redirect both stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            transcriber = IntegratedTranscriber(
                asr_model=args.model, device=args.device
            )
    finally:
        # Restore original logging level
        logging.getLogger().setLevel(original_level)

    # Prepare input file - preprocessing
    processed_file = args.input
    temp_files = []

    if args.preprocess:
        print("Preprocessing audio...")
        processed_file = convert_to_wav(processed_file)
        if processed_file != args.input:
            temp_files.append(processed_file)

        processed_file = trim_silence(processed_file)
        if processed_file != args.input:
            temp_files.append(processed_file)

    # Pass language explicitly only if provided
    kwargs = {}
    if args.language:
        kwargs["language"] = args.language
        print(f"Using language: {args.language}")

    # Set batch size
    kwargs["batch_size"] = args.batch_size
    print(f"Using batch size: {args.batch_size}")

    # Process audio
    # Combine both approaches to support splitting and batch size control
    if args.split:
        print(
            f"Splitting audio into {args.split_length}s segments with {args.split_overlap}s overlap..."
        )
        split_files = split_audio(processed_file, args.split_length, args.split_overlap)

        # Process each split and combine results
        all_results = []
        for i, split_file in enumerate(split_files):
            print(f"Processing split {i+1}/{len(split_files)}: {split_file}")

            # Configure batch size handling
            transcriber.asr.process_audio = (
                lambda audio_path, language, batch_size=args.batch_size: (
                    transcriber.asr._process_audio_with_batch_size(
                        audio_path, language, batch_size
                    )
                )
            )

            result = transcriber.process_audio(split_file, **kwargs)
            all_results.append(result)
            temp_files.append(split_file)

        # TODO: Implement proper combining of split results
        # For now, just use the first result
        result = all_results[0]
    else:
        print(f"Processing audio file: {processed_file}")

        # Modify ASR processor to use the specified batch size
        transcriber.asr.process_audio = (
            lambda audio_path, language, batch_size=args.batch_size: (
                transcriber.asr._process_audio_with_batch_size(
                    audio_path, language, batch_size
                )
            )
        )

        # Add the method to ASRProcessor class dynamically
        def _process_audio_with_batch_size(self, audio_path, language, batch_size):
            """Wrapper method that ensures batch_size is correctly used."""
            if self.model is None or self.current_language != language:
                try:
                    self.load_models(language_code=language)
                except Exception as e:
                    print(
                        f"Warning: Could not load alignment model for {language}. Falling back to transcription without alignment."
                    )
                    if self.model is None:
                        # Set up comprehensive warning suppression
                        original_level = logging.getLogger().level
                        stdout_buffer = io.StringIO()
                        stderr_buffer = io.StringIO()

                        try:
                            # Temporarily suppress all logging
                            logging.getLogger().setLevel(logging.ERROR)

                            # Redirect both stdout and stderr
                            with redirect_stdout(stdout_buffer), redirect_stderr(
                                stderr_buffer
                            ):
                                self.model = whisperx.load_model(
                                    self.model_name,
                                    self.device,
                                    compute_type=self.compute_type,
                                )
                        finally:
                            # Restore original logging level
                            logging.getLogger().setLevel(original_level)

            # Transcribe with whisperx
            audio = whisperx.load_audio(audio_path)

            # Use the specified batch size
            result = self.model.transcribe(
                audio, batch_size=batch_size, language=language
            )

            # Align timestamps if alignment model is available
            if self.align_model is not None:
                try:
                    result = whisperx.align(
                        result["segments"],
                        self.align_model,
                        self.align_metadata,
                        audio,
                        self.device,
                    )
                except Exception as e:
                    print(
                        f"Warning: Alignment failed. Using original timestamps. Error: {e}"
                    )

            return result

        # Add method to ASRProcessor instance
        import types
        import whisperx

        transcriber.asr._process_audio_with_batch_size = types.MethodType(
            _process_audio_with_batch_size, transcriber.asr
        )

        # Process audio with our wrapper that ensures batch_size is correctly used
        result = transcriber.process_audio(
            audio_path=processed_file, language=args.language
        )

    # Save results
    transcriber.save_result(result, args.output)
    print(f"Transcription saved to {args.output}")

    # Generate formatted text file
    print(f"Generating {args.format} format transcript...")
    formatted_text = transcriber.get_formatted_transcript(
        result, format_type=args.format
    )

    print(f"Saving formatted transcript to: {text_output}")
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
