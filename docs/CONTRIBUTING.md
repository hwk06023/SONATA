---
layout: default
title: Contributing Guidelines
nav_order: 9
---

# Contributing to SONATA

Thank you for your interest in contributing to SONATA! This document provides guidelines and instructions to help you get started.

> "Great software is built with the contributions of many."

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## Code of Conduct

By participating in this project, you agree to abide by the [SONATA Code of Conduct](https://github.com/hwk06023/SONATA/blob/main/CODE_OF_CONDUCT.md). We expect all contributors to be respectful, inclusive, and considerate to create a positive environment for everyone.

## Ways to Contribute

There are many ways to contribute to SONATA, and all contributions are valuable:

- **Code Contributions**: Implement new features, fix bugs, or improve performance
- **Documentation**: Improve or clarify existing documentation, add examples
- **Bug Reports**: Submit detailed bug reports with reproduction steps
- **Feature Requests**: Suggest new features or improvements
- **Testing**: Help test the software on different platforms and configurations
- **Community Support**: Answer questions and help others in discussions

You don't need to be an expert in ASR systems to contribute! Even small improvements are welcome and appreciated.

## Getting Started

### First-time Contributors

If you're making your first contribution, we recommend looking for issues labeled [good first issue](https://github.com/hwk06023/SONATA/labels/good%20first%20issue) for beginner-friendly tasks.

### Development Environment Setup

1. **Fork the Repository**:
   * Go to [https://github.com/hwk06023/SONATA](https://github.com/hwk06023/SONATA)
   * Click the "Fork" button in the upper right corner
   * This creates your own copy of the repository under your GitHub account

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/SONATA.git
   cd SONATA
   git remote add upstream https://github.com/hwk06023/SONATA.git
   ```

3. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Development Dependencies**:
   ```bash
   pip install -e ".[dev]"
   # or if that doesn't work:
   pip install -e .
   pip install pytest pytest-cov black isort flake8
   ```

5. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-you-are-fixing
   ```

### Development Workflow

1. Make your changes, following the coding standards
2. Write or update tests as needed
3. Ensure all tests pass
4. Update documentation if necessary
5. Submit a pull request

## Code Contribution Guidelines

### Coding Standards

SONATA follows these coding standards:

- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- **Docstrings**: [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- **Type Hints**: Include type hints for function parameters and return values 
- **Imports**: Organize imports using `isort`
- **Formatting**: Use `black` for automatic code formatting

Example of proper function formatting:

```python
def process_audio(
    audio_path: str, 
    language: Optional[str] = None,
    deep_detect: bool = False
) -> Dict[str, Any]:
    """
    Process audio file and return transcription results.
    
    Args:
        audio_path: Path to the audio file to process
        language: Language code (e.g., 'en', 'ko', 'zh')
        deep_detect: Whether to use deep detection for audio events
        
    Returns:
        Dictionary containing transcription results
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If processing fails
    """
    # Implementation here
    return results
```

### Testing

All new features and bug fixes should include appropriate tests:

- Unit tests should be added in the `tests/` directory
- Tests should verify both correct behavior and proper error handling
- Run the tests locally before submitting a PR:

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_specific_module.py

# Run with coverage report
pytest --cov=sonata tests/
```

## Pull Request Process

1. **Create a Pull Request (PR)** from your forked repository.
2. **PR Description**: Include a clear description of the changes and reference any related issues.
3. **Checks**: Ensure all CI checks pass (tests, linting, etc.).
4. **Review**: Address any feedback from maintainers.
5. **Merge**: Once approved, a maintainer will merge your PR.

### Pull Request Title and Description Guidelines

- Use a clear, descriptive title that summarizes the changes
- Include the issue number if applicable (e.g., "Fix ASR error handling #42")
- In the description, explain:
  - What changes you made
  - Why you made them
  - How to test them
  - Any potential side effects

### Pull Request Checklist

Before submitting, please ensure:

- [ ] Code follows project style guidelines
- [ ] Tests are added/updated and all tests pass
- [ ] Documentation is updated if needed
- [ ] Commits are properly formatted and descriptive
- [ ] You've signed the Contributor License Agreement (if required)

## Documentation Contributions

Documentation is crucial for SONATA's usability. When contributing to documentation:

1. Follow the Jekyll format used in the docs directory
2. Use clear, concise language
3. Include code examples where appropriate
4. Keep the target audience in mind (researchers, developers, etc.)

## Bug Reports and Feature Requests

### Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Detailed steps to reproduce the bug
- Expected behavior vs. actual behavior
- System information (OS, Python version, etc.)
- Code samples or error messages
- If possible, a minimal reproduction case

You can get system information with:
```bash
python -c "import platform, sys; print(f'OS: {platform.system()} {platform.release()}, Python: {sys.version}')"
```

### Feature Requests

For feature requests, include:

- A clear description of the feature
- The motivation behind the feature (why it's useful)
- Examples of how it would be used
- References to similar features in other projects (if applicable)

## Working with Audio Event Detection

When contributing to audio event detection features:

1. Test with diverse audio samples
2. Document thresholds and parameters
3. Consider performance implications
4. Ensure compatibility with existing detection models

## Working with Speaker Diarization

When improving the speaker diarization system:

1. Test with multi-speaker recordings
2. Consider edge cases (overlapping speech, background noise)
3. Document any changes to parameters or models
4. Compare performance metrics for different approaches

## Versioning and Releases

SONATA follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward-compatible manner
- **PATCH** version for backward-compatible bug fixes

## License

By contributing to SONATA, you agree that your contributions will be licensed under the project's [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Recognition

All contributors are recognized in the project. We value every contribution, regardless of size!

---

Thank you for contributing to SONATA! Your efforts help improve speech recognition for everyone. If you have any questions about contributing, please reach out to the maintainers. 