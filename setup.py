from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 직접 요구 사항 지정
requirements = [
    "numpy>=1.20.0",
    "torch>=1.10.0",
    "torchaudio>=0.10.0",
    "transformers>=4.25.1",
    "whisperx>=3.1.0",
    "librosa>=0.9.0",
    "pydub>=0.25.1",
    "scipy>=1.7.0",
    "soundfile>=0.10.3",
    "tqdm>=4.62.0",
    "matplotlib>=3.5.0",
    "scikit-learn>=1.0.0",
    "huggingface_hub>=0.12.0",
    "hf_xet>=0.1.0",
]

setup(
    name="sonata-asr",
    version="0.0.1",
    author="hwk06023",
    author_email="hwk06023@github.com",
    description="SONATA: SOund and Narrative Advanced Transcription Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hwk06023/SONATA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sonata-asr=sonata.main:main",
        ],
    },
)
