.PHONY: help docs html livehtml test lint typecheck clean dev install

# Default target when `make` is run without arguments
help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install    Install the package and development dependencies"
	@echo "  dev        Install in development mode with all dependencies"
	@echo "  docs       Build the documentation"
	@echo "  html       Alias for 'docs'"
	@echo "  livehtml   Build and serve docs with live reload"
	@echo "  test       Run tests"
	@echo "  lint       Run code style checks"
	@echo "  typecheck  Run static type checking"
	@echo "  clean      Remove build artifacts"

# Install the package and development dependencies
install:
	pip install -e .[dev]

# Install in development mode with all dependencies
dev: install
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt
	pre-commit install

# Build the documentation
docs html:
	@echo "Building documentation..."
	cd docs && make html

# Build and serve docs with live reload
livehtml:
	@echo "Starting live documentation server..."
	cd docs && sphinx-autobuild -b html --watch ../src . _build/html

# Run tests
test:
	@pytest tests/ -v --cov=repairs_components --cov-report=term-missing

# Run code style checks
lint:
	@echo "Running code style checks..."
	black --check .
	isort --check-only .
	pylint src/

# Run static type checking
typecheck:
	@echo "Running static type checking..."
	mypy .

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .mypy_cache/
	rm -rf docs/_build/ docs/api/generated/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name ".pytest_cache" -delete
dev-deps:
	pip install -r requirements-dev.txt

# Format code
format:
	black src/ tests/ examples/
	isort src/ tests/ examples/


# Lint code
lint:
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/
	mypy src/ tests/ examples/
	pylint src/ tests/ examples/
