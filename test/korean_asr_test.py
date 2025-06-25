#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sonata.models.korean_asr import KoreanASRModel
from sonata.core.asr import ASRProcessor


def test_korean_asr_model():
    """Test the Korean ASR model directly."""
    print("Testing Korean ASR Model...")

    try:
        model = KoreanASRModel()
        print("✓ Korean ASR model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load Korean ASR model: {e}")
        return False


def test_asr_processor_with_korean():
    """Test the ASR processor with Korean language."""
    print("\nTesting ASR Processor with Korean language...")

    try:
        processor = ASRProcessor()
        print("✓ ASR processor initialized successfully")

        # Test Korean language detection
        is_korean = processor._is_korean_language("ko")
        print(f"✓ Korean language detection: {is_korean}")

        korean_model = processor._get_korean_model()
        if korean_model and korean_model is not False:
            print("✓ Korean model loaded in ASR processor")
            return True
        else:
            print("✗ Korean model not available in ASR processor")
            return False

    except Exception as e:
        print(f"✗ Failed to initialize ASR processor: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Korean ASR integration")
    parser.add_argument("--audio", type=str, help="Path to Korean audio file to test")
    args = parser.parse_args()

    print("=" * 50)
    print("Korean ASR Model Integration Test")
    print("=" * 50)

    # Test 1: Korean ASR Model
    test1_passed = test_korean_asr_model()

    # Test 2: ASR Processor
    test2_passed = test_asr_processor_with_korean()

    # Test 3: Audio processing (if audio file provided)
    test3_passed = True
    if args.audio and os.path.exists(args.audio):
        print(f"\nTesting audio processing with: {args.audio}")
        try:
            processor = ASRProcessor()
            result = processor.process_audio(
                args.audio, language="ko", show_progress=True
            )
            print(f"✓ Audio processed successfully. Result segments: {len(result)}")
            if result:
                print(f"First segment: {result[0]}")
            test3_passed = True
        except Exception as e:
            print(f"✗ Audio processing failed: {e}")
            test3_passed = False

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Korean ASR Model: {'PASS' if test1_passed else 'FAIL'}")
    print(f"ASR Processor: {'PASS' if test2_passed else 'FAIL'}")
    if args.audio:
        print(f"Audio Processing: {'PASS' if test3_passed else 'FAIL'}")
    print("=" * 50)

    overall_pass = test1_passed and test2_passed and test3_passed
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
