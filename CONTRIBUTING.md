# Contributing to Brayam Pineda ML Package

Thank you for your interest in contributing to the Brayam Pineda ML Package! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment
4. Install the package in development mode

```bash
git clone https://github.com/your-username/brayam-pineda-ml.git
cd brayam-pineda-ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Development Setup

The package uses several development tools:

- **pytest**: For running tests
- **black**: For code formatting
- **ruff**: For linting
- **mypy**: For type checking

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
```

### Linting

```bash
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Format and lint your code
6. Commit your changes with a descriptive message

## Pull Request Process

1. Update the README.md if needed
2. Update the CHANGELOG.md with your changes
3. Ensure the test suite passes
4. Request review from maintainers

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

## Testing

- Write tests for all new functionality
- Ensure test coverage remains high
- Use descriptive test names
- Test both success and failure cases

## Documentation

- Update docstrings for any modified functions
- Update README.md if adding new features
- Include examples in docstrings

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release on GitHub
4. The GitHub Action will automatically publish to PyPI

## Questions?

If you have questions about contributing, please open an issue on GitHub.
