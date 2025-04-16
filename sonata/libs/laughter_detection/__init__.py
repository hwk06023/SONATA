"""
Laughter detection module based on the work at https://github.com/jrgillick/laughter-detection
Modified for integration with SONATA
"""

from .laugh_segmenter import segment_laughs, segment_laugh_with_model
