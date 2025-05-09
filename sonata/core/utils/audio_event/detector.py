import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Set
from .event import AudioEvent
from .processor import AudioProcessor
from .models import AudiosetModelLoader
from .classifier import AudiosetClassifier


class AudioEventDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        label_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_vggish: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.processor = AudioProcessor()

        # Load model and labels
        self.model_loader = AudiosetModelLoader(
            model_path=model_path,
            label_path=label_path,
            device=self.device,
            use_vggish=use_vggish,
        )

        self.model = self.model_loader.load_model()
        self.labels = self.model_loader.load_labels()

        # Initialize classifier
        self.classifier = AudiosetClassifier(
            model=self.model, labels=self.labels, device=self.device
        )

    def detect_events(
        self,
        audio_file: str,
        threshold: float = 0.5,
        segment_size: float = 10.0,
        hop_size: float = 5.0,
        target_sr: int = 16000,
        target_events: Optional[Set[str]] = None,
    ) -> List[AudioEvent]:
        """Detect audio events in an audio file.

        Args:
            audio_file: Path to audio file
            threshold: Detection threshold
            segment_size: Size of processing segments in seconds
            hop_size: Hop size between segments in seconds
            target_sr: Target sample rate
            target_events: Set of event types to detect (if None, all are detected)

        Returns:
            List of detected AudioEvent objects
        """
        try:
            # Process audio file
            audio_data, sr = self.processor.load_audio(audio_file, target_sr=target_sr)

            # Get duration
            duration = len(audio_data) / sr

            # Process in segments for long files
            events = []
            segment_samples = int(segment_size * sr)
            hop_samples = int(hop_size * sr)

            # Create segments
            for start_sample in range(0, len(audio_data), hop_samples):
                end_sample = min(start_sample + segment_samples, len(audio_data))
                segment = audio_data[start_sample:end_sample]

                # Calculate segment time bounds
                start_time = start_sample / sr
                end_time = end_sample / sr

                # Detect events in segment
                events.extend(
                    self._detect_events_in_segment(
                        segment, sr, start_time, end_time, threshold, target_events
                    )
                )

            # Remove duplicates and merge overlapping events
            merged_events = self._merge_events(events)

            return merged_events

        except Exception as e:
            self.logger.error(f"Error detecting events: {str(e)}")
            return []

    def _detect_events_in_segment(
        self,
        audio_segment: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
        threshold: float,
        target_events: Optional[Set[str]] = None,
    ) -> List[AudioEvent]:
        """Detect events in a single audio segment.

        Args:
            audio_segment: Audio segment data
            sr: Sample rate
            start_time: Start time of segment in original audio
            end_time: End time of segment in original audio
            threshold: Detection threshold
            target_events: Set of event types to detect

        Returns:
            List of detected AudioEvent objects
        """
        events = []

        # Get probabilities from model
        probs = self.classifier.detect_from_array(audio_segment, sr=sr)

        if probs is None or len(probs) == 0:
            return events

        # Extract events above threshold
        for class_idx, prob in enumerate(probs[0]):
            if prob >= threshold and class_idx < len(self.labels):
                # Get class name from index - safer approach to avoid index errors
                class_names = [k for k, v in self.labels.items() if v == class_idx]
                if not class_names:
                    continue  # Skip if no matching class name found

                class_name = class_names[0]

                # Skip if not in target events
                if target_events and class_name not in target_events:
                    continue

                # Create event
                event = AudioEvent(
                    event_type=class_name,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=float(prob),
                )
                events.append(event)

        return events

    def _merge_events(self, events: List[AudioEvent]) -> List[AudioEvent]:
        """Merge overlapping events of the same type.

        Args:
            events: List of detected events

        Returns:
            List of merged events
        """
        if not events:
            return []

        # Sort events by start time, then by end time
        sorted_events = sorted(events, key=lambda e: (e.start_time, e.end_time))
        merged = []

        # Group by event type
        event_type_groups = {}
        for event in sorted_events:
            if event.event_type not in event_type_groups:
                event_type_groups[event.event_type] = []
            event_type_groups[event.event_type].append(event)

        # Process each event type separately
        for event_type, event_group in event_type_groups.items():
            if not event_group:
                continue

            current = event_group[0]

            for i in range(1, len(event_group)):
                next_event = event_group[i]

                # Check for overlap
                if next_event.start_time <= current.end_time:
                    # Merge events
                    current.end_time = max(current.end_time, next_event.end_time)
                    current.confidence = max(current.confidence, next_event.confidence)
                else:
                    # No overlap, add current to merged and start new current
                    merged.append(current)
                    current = next_event

            # Add the last event
            merged.append(current)

        return sorted(merged, key=lambda e: e.start_time)
