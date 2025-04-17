"""
Constants for SONATA project.
"""
from enum import Enum, auto


class LanguageCode(str, Enum):
    """ISO 639-1 language codes for supported languages."""

    ENGLISH = "en"
    KOREAN = "ko"
    CHINESE = "zh"
    JAPANESE = "ja"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"


class FormatType(str, Enum):
    """Transcript format types."""

    CONCISE = "concise"  # Simple text with emotive tags
    DEFAULT = "default"  # Text with timestamps
    EXTENDED = "extended"  # With confidence scores


class EmotiveEventType(str, Enum):
    """Types of emotive events that can be detected."""

    LAUGH = "laugh"
    SIGH = "sigh"
    INHALE = "inhale"
    GROAN = "groan"
    COUGH = "cough"
    SNEEZE = "sneeze"
    SNIFFLE = "sniffle"
    SURPRISE = "surprise"
    YAWN = "yawn"
    WHIMPER = "whimper"


# Threshold values
EMOTIVE_THRESHOLD = 0.3  # Default threshold for detecting emotive events

# Default settings
DEFAULT_LANGUAGE = LanguageCode.ENGLISH.value
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "float32"

# Split settings
DEFAULT_SPLIT_LENGTH = 30  # Length of split segments in seconds
DEFAULT_SPLIT_OVERLAP = 5  # Overlap between split segments in seconds

# Format types for backwards compatibility
FORMAT_CONCISE = FormatType.CONCISE.value
FORMAT_DEFAULT = FormatType.DEFAULT.value
FORMAT_EXTENDED = FormatType.EXTENDED.value

# Mapping between AudioSet classes and our emotive tags
EMOTIVE_CLASS_MAPPING = {
    16: "laugh",  # Laughter
    18: "laugh",  # Giggle
    21: "laugh",  # Chuckle, chortle
    24: "whimper",  # Whimper
    26: "sigh",  # Sigh
    38: "groan",  # Groan
    44: "inhale",  # Gasp
    47: "cough",  # Cough
    49: "sneeze",  # Sneeze
    50: "sniffle",  # Sniff
}
