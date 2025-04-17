"""
Constants for SONATA project.
"""
# Thresholds
EMOTIVE_THRESHOLD = 0.5  # Default threshold for detecting emotive events

# Default settings
DEFAULT_LANGUAGE = "en"
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "float32"

# Format types
FORMAT_CONCISE = "concise"
FORMAT_DEFAULT = "default"
FORMAT_EXTENDED = "extended"

# Split settings
DEFAULT_SPLIT_LENGTH = 30  # Length of split segments in seconds
DEFAULT_SPLIT_OVERLAP = 5  # Overlap between split segments in seconds

# Emotive event types
EMOTIVE_TYPES = [
    "laugh",
    "whimper",
    "sigh",
    "groan",
    "inhale",
    "cough",
    "sneeze",
    "sniffle",
]

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
