import re
from sonata.constants import LanguageCode


def clean_text_for_language(text, language_code):
    """Clean text based on language-specific rules.

    Args:
        text (str): Input text to clean
        language_code (str): ISO language code (e.g., "en", "ko", "zh")

    Returns:
        list: List of cleaned words with empty words filtered out
    """
    words = text.split(" ")
    cleaned_words = []

    if language_code == LanguageCode.KOREAN.value:
        # Korean: Keep only Hangul characters and numbers
        cleaned_words = list(map(lambda x: re.sub(r"[^가-힣0-9]", "", x), words))
    elif language_code == LanguageCode.CHINESE.value:
        # Chinese: Keep only Chinese characters and numbers
        cleaned_words = list(
            map(lambda x: re.sub(r"[^\u4e00-\u9fff0-9]", "", x), words)
        )
    elif language_code == LanguageCode.JAPANESE.value:
        # Japanese: Keep only Japanese characters (Hiragana, Katakana, Kanji) and numbers
        cleaned_words = list(
            map(
                lambda x: re.sub(
                    r"[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff0-9]", "", x
                ),
                words,
            )
        )
    else:
        # Default for Latin-based languages: Keep only alphanumeric characters
        cleaned_words = list(map(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x), words))

    # Filter out empty words
    return [w for w in cleaned_words if w]
