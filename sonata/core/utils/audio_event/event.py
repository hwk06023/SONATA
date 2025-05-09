from dataclasses import dataclass


@dataclass
class AudioEvent:
    """Represents a detected audio event with timing and confidence information."""

    type: str
    start_time: float
    end_time: float
    confidence: float

    def to_dict(self):
        """Convert the event to a dictionary format."""
        return {
            "type": self.type,
            "start": self.start_time,
            "end": self.end_time,
            "confidence": self.confidence,
        }

    def to_tag(self):
        """Convert the event to a simple tag format."""
        return f"[{self.type}]"

    def get_duration(self):
        """Get the duration of the event in seconds."""
        return self.end_time - self.start_time

    def overlap(self, other):
        """Calculate overlap with another event."""
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_duration = overlap_end - overlap_start
        return overlap_duration

    def iou(self, other):
        """Calculate Intersection over Union with another event of the same type."""
        if self.type != other.type:
            return 0.0

        overlap_duration = self.overlap(other)

        if overlap_duration <= 0:
            return 0.0

        union_duration = (
            (self.end_time - self.start_time)
            + (other.end_time - other.start_time)
            - overlap_duration
        )

        if union_duration <= 0:
            return 0.0

        return overlap_duration / union_duration
