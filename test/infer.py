#!/usr/bin/env python
import os
import sys
import json
import argparse
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
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for ASR processing (default: 16)",
    )
    return parser.parse_args()


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
    transcriber = IntegratedTranscriber(asr_model=args.model, device=args.device)

    # Process audio
    print(f"Processing audio file: {args.input}")
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
                    self.model = whisperx.load_model(
                        self.model_name, self.device, compute_type=self.compute_type
                    )

        # Transcribe with whisperx
        audio = whisperx.load_audio(audio_path)

        # Use the specified batch size
        result = self.model.transcribe(audio, batch_size=batch_size, language=language)

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
    result = transcriber.process_audio(audio_path=args.input, language=args.language)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
