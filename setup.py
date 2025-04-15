from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="sonata",
    version="0.1.0",
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
            "sonata=sonata.main:main",
        ],
    },
)
