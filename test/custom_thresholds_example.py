#!/usr/bin/env python
"""
Example script demonstrating the use of custom audio event thresholds in SONATA.

This example shows how to define and apply custom audio event thresholds,
which can be useful for adjusting detection sensitivity for different 
audio events based on your specific needs.
"""

import os
import sys
import logging
from typing import Dict, List

# Add the parent directory to the path to import sonata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonata.core.audio_event_detector import AudioEventDetector
from sonata.core.transcript import AudioEvent
from sonata.constants import AudioEventType, AUDIO_EVENT_THRESHOLDS


def print_events(events: List[AudioEvent], title: str = "Detected Events"):
    """Helper function to print detected events"""
    print(f"\n{title}:")
    print("-" * 50)

    # Sort events by start time
    sorted_events = sorted(events, key=lambda e: e.start_time)

    for event in sorted_events:
        # Format with 2 decimal places for better readability
        start = f"{event.start_time:.2f}"
        end = f"{event.end_time:.2f}"
        conf = f"{event.confidence:.2f}"
        print(f"{event.type}: {start}s - {end}s (confidence: {conf})")

    print(f"\nTotal events: {len(events)}")


def main():
    """Run example with custom audio event thresholds"""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Path to an audio file for testing
    # This should be replaced with a path to your own audio file
    audio_file = "data/sample.wav"
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please update the script with a valid audio file path.")
        return

    print(f"Processing audio file: {audio_file}")

    # First, detect events with default thresholds
    print("\nRunning detection with default thresholds...")
    detector = AudioEventDetector()
    default_events = detector.detect_events(audio_file)
    print_events(default_events, "Events with Default Thresholds")

    # Define custom thresholds for specific event types
    # This increases or decreases detection sensitivity for specific events
    custom_thresholds: Dict[str, float] = {
        # Increase detection sensitivity for laughter (lower threshold)
        "laughter": 0.05,  # Default is 0.1
        "giggle": 0.05,  # Default is 0.1
        # Decrease sensitivity for background sounds (higher threshold)
        "music": 0.7,  # Default would use general threshold of 0.5
        "noise": 0.7,
        # Custom threshold for specific events
        "cough": 0.2,  # Default is 0.15
        "sneeze": 0.2,  # Default is 0.15
    }

    print("\nRunning detection with custom thresholds...")
    custom_detector = AudioEventDetector(custom_thresholds=custom_thresholds)
    custom_events = custom_detector.detect_events(audio_file)
    print_events(custom_events, "Events with Custom Thresholds")

    # Print the thresholds being used for comparison
    print("\nCustom Threshold Settings:")
    print("-" * 50)
    for event_type, threshold in custom_thresholds.items():
        default = AUDIO_EVENT_THRESHOLDS.get(event_type, 0.5)  # 0.5 is general default
        print(f"{event_type}: {threshold} (default: {default})")


if __name__ == "__main__":
    main()
