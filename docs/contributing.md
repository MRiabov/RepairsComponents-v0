# Contributing to RepairsComponents

Thank you for your interest in contributing to RepairsComponents! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/RepairsComponents-v0.git
   cd RepairsComponents-v0
   ```
3. **Set up the development environment**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install the package in development mode with all dependencies
   pip install -e ".[dev]"
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. **Make your changes** following the code style guidelines

3. **Run tests** to ensure nothing is broken:
   ```bash
   pytest
   ```

4. **Commit your changes** with a descriptive message:
   ```bash
   git add .
   git commit -m "Add your commit message here"
   ```

5. **Push your changes** to your fork:
   ```bash
   git push origin your-branch-name
   ```

6. **Open a Pull Request** from your fork to the main repository

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for static type checking
- **pylint** for code quality

Run all linters and formatters:

```bash
# Format code
black .

# Sort imports
isort .
# Check types
mypy .
# Lint code
pylint src/
```

Or use the convenience script:

```bash
./scripts/lint.sh
```

## Testing

We use `pytest` for testing. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=repairs_components

# Run a specific test file
pytest tests/test_file.py

# Run a specific test
pytest tests/test_file.py::test_function_name
```

### Writing Tests

- Test files should be in the `tests` directory
- Test files should be named `test_*.py`
- Use descriptive test function names
- Each test should test a single piece of functionality
- Use fixtures for common setup/teardown

## Documentation

We use Sphinx with MyST (Markdown) for documentation.

### Building the Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements-docs.txt

# Build the docs
cd docs
make html

# Open the documentation
open _build/html/index.html  # On macOS
# or
xdg-open _build/html/index.html  # On Linux
```

### Writing Documentation

- Use Markdown (`.md`) for all documentation
- Follow the existing style and structure
- Document all public APIs
- Include examples where helpful
- Update the documentation when making API changes

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/)
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you

## Reporting Issues

When reporting issues, please include the following:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, package versions)
6. Any relevant error messages or logs

## Feature Requests

We welcome feature requests! Please include:

1. A clear and descriptive title
2. A description of the problem you're trying to solve
3. Any alternative solutions you've considered
4. Additional context or examples

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
