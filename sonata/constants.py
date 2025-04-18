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

    CONCISE = "concise"  # Simple text with audio event tags
    DEFAULT = "default"  # Text with timestamps
    EXTENDED = "extended"  # With confidence scores


class AudioEventType(str, Enum):
    """Types of audio events that can be detected based on AudioSet ontology."""

    # Human sounds
    SPEECH = "speech"
    MALE_SPEECH = "male_speech"
    FEMALE_SPEECH = "female_speech"
    CHILD_SPEECH = "child_speech"
    CONVERSATION = "conversation"
    LAUGHTER = "laughter"
    BABY_LAUGHTER = "baby_laughter"
    GIGGLE = "giggle"
    CHUCKLE = "chuckle"
    CRYING = "crying"
    BABY_CRY = "baby_cry"
    WHIMPER = "whimper"
    SIGH = "sigh"
    WHISPERING = "whispering"
    SCREAMING = "screaming"
    SHOUTING = "shouting"

    # Breathing and vocal sounds
    BREATHING = "breathing"
    COUGH = "cough"
    SNEEZE = "sneeze"
    SNIFF = "sniff"
    SNORING = "snoring"
    GASP = "gasp"
    PANT = "pant"
    WHEEZE = "wheeze"
    THROAT_CLEARING = "throat_clearing"
    BURP = "burp"
    HICCUP = "hiccup"

    # Physical sounds
    CLAPPING = "clapping"
    FINGER_SNAPPING = "finger_snapping"
    FOOTSTEPS = "footsteps"
    HEARTBEAT = "heartbeat"

    # Animal sounds
    ANIMAL = "animal"
    DOG = "dog"
    BARK = "bark"
    CAT = "cat"
    MEOW = "meow"
    HORSE = "horse"
    COW = "cow"
    PIG = "pig"
    BIRD = "bird"
    BIRD_VOCALIZATION = "bird_vocalization"

    # Music related
    MUSIC = "music"
    MUSICAL_INSTRUMENT = "musical_instrument"
    SINGING = "singing"
    PIANO = "piano"
    GUITAR = "guitar"
    DRUM = "drum"
    VIOLIN = "violin"

    # Environmental sounds
    WIND = "wind"
    RAIN = "rain"
    THUNDER = "thunder"
    WATER = "water"
    STREAM = "stream"
    OCEAN = "ocean"
    FIRE = "fire"

    # Mechanical and electronic
    ENGINE = "engine"
    VEHICLE = "vehicle"
    CAR = "car"
    AIRPLANE = "airplane"
    TRAIN = "train"
    SIREN = "siren"
    ALARM = "alarm"
    TELEPHONE = "telephone"
    BELL = "bell"

    # Domestic sounds
    DOOR = "door"
    DOORBELL = "doorbell"
    KNOCK = "knock"
    TYPING = "typing"
    KEYBOARD = "keyboard"
    MICROWAVE = "microwave"

    # Miscellaneous
    SILENCE = "silence"
    NOISE = "noise"
    EXPLOSION = "explosion"
    GUNSHOT = "gunshot"
    CRASH = "crash"
    BREAKING = "breaking"


# Threshold values
AUDIO_EVENT_THRESHOLD = 0.5  # Default threshold for detecting audio events

# Event-specific thresholds for audio detection
AUDIO_EVENT_THRESHOLDS = {
    # Human vocal sounds - need lower thresholds to detect alongside speech
    "laughter": 0.1,
    "baby_laughter": 0.1,
    "giggle": 0.1,
    "chuckle": 0.1,
    "crying": 0.1,
    "baby_cry": 0.1,
    "whimper": 0.1,
    "sigh": 0.15,
    "breathing": 0.15,
    "cough": 0.15,
    "sneeze": 0.15,
    "sniff": 0.15,
    "burp": 0.15,
    "hiccup": 0.15,
    "throat_clearing": 0.2,
    "gasp": 0.3,
    "pant": 0.3,
    "wheeze": 0.3,
    # Prominent sounds should use standard threshold (these will use the default AUDIO_EVENT_THRESHOLD)
}

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

# Mapping between AudioSet classes and our friendly tag names
AUDIOSET_CLASS_MAPPING = {
    0: "speech",  # Speech
    1: "male_speech",  # Male speech, man speaking
    2: "female_speech",  # Female speech, woman speaking
    3: "child_speech",  # Child speech, kid speaking
    4: "conversation",  # Conversation
    5: "narration",  # Narration, monologue
    16: "laughter",  # Laughter
    17: "baby_laughter",  # Baby laughter
    18: "giggle",  # Giggle
    21: "chuckle",  # Chuckle, chortle
    22: "crying",  # Crying, sobbing
    23: "baby_cry",  # Baby cry, infant cry
    24: "whimper",  # Whimper
    26: "sigh",  # Sigh
    38: "groan",  # Groan
    40: "whistling",  # Whistling
    41: "breathing",  # Breathing
    42: "wheeze",  # Wheeze
    43: "snoring",  # Snoring
    44: "gasp",  # Gasp
    45: "pant",  # Pant
    47: "cough",  # Cough
    48: "throat_clearing",  # Throat clearing
    49: "sneeze",  # Sneeze
    50: "sniff",  # Sniff
    53: "footsteps",  # Walk, footsteps
    61: "hands",  # Hands
    62: "finger_snapping",  # Finger snapping
    63: "clapping",  # Clapping
    64: "heartbeat",  # Heart sounds, heartbeat
    66: "cheering",  # Cheering
    67: "applause",  # Applause
    72: "animal",  # Animal
    74: "dog",  # Dog
    75: "bark",  # Bark
    81: "cat",  # Cat
    83: "meow",  # Meow
    87: "horse",  # Horse
    90: "cattle",  # Cattle, bovinae
    91: "moo",  # Moo
    97: "sheep",  # Sheep
    100: "chicken",  # Chicken, rooster
    112: "bird_vocalization",  # Bird vocalization, bird call, bird song
    115: "pigeon",  # Pigeon, dove
    137: "music",  # Music
    138: "musical_instrument",  # Musical instrument
    139: "string_instrument",  # Plucked string instrument
    140: "guitar",  # Guitar
    152: "piano",  # Piano
    161: "percussion",  # Percussion
    162: "drum_kit",  # Drum kit
    164: "drum",  # Drum
    189: "violin",  # Bowed string instrument
    190: "violin",  # String section
    191: "violin",  # Violin, fiddle
    195: "wind_instrument",  # Wind instrument, woodwind instrument
    196: "flute",  # Flute
    200: "bell",  # Bell
    283: "wind",  # Wind
    284: "rustling_leaves",  # Rustling leaves
    286: "thunderstorm",  # Thunderstorm
    287: "thunder",  # Thunder
    288: "water",  # Water
    289: "rain",  # Rain
    290: "raindrop",  # Raindrop
    291: "rain_on_surface",  # Rain on surface
    292: "stream",  # Stream
    293: "waterfall",  # Waterfall
    294: "ocean",  # Ocean
    295: "waves",  # Waves, surf
    298: "fire",  # Fire
    300: "vehicle",  # Vehicle
    306: "motor_vehicle",  # Motor vehicle (road)
    307: "car",  # Car
    322: "emergency_vehicle",  # Emergency vehicle
    323: "police_car",  # Police car (siren)
    324: "ambulance",  # Ambulance (siren)
    325: "fire_truck",  # Fire engine, fire truck (siren)
    329: "train",  # Train
    333: "subway",  # Subway, metro, underground
    335: "aircraft",  # Aircraft
    340: "airplane",  # Fixed-wing aircraft, airplane
    343: "engine",  # Engine
    354: "door",  # Door
    356: "doorbell",  # Doorbell
    359: "slam",  # Slam
    360: "knock",  # Knock
    368: "microwave",  # Microwave oven
    376: "vacuum",  # Vacuum cleaner
    386: "keyboard",  # Computer keyboard
    388: "alarm",  # Alarm
    389: "telephone",  # Telephone
    390: "telephone_bell",  # Telephone bell ringing
    396: "alarm_clock",  # Alarm clock
    397: "siren",  # Siren
    399: "buzzer",  # Buzzer
    400: "smoke_detector",  # Smoke detector, smoke alarm
    401: "fire_alarm",  # Fire alarm
    426: "explosion",  # Explosion
    427: "gunshot",  # Gunshot, gunfire
    458: "arrow",  # Arrow
    467: "bang",  # Bang
    469: "smash",  # Smash, crash
    470: "breaking",  # Breaking
    500: "silence",  # Silence
    504: "sound_effect",  # Sound effect
    513: "noise",  # Noise
}
