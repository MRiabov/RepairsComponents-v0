.PHONY: docs test clean

# Build the documentation
docs:
	@echo "Building documentation..."
	@sphinx-build -b html docs/source docs/build

# Run tests
test:
	@pytest tests/

# Clean build artifacts
clean:
	@rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .mypy_cache/
	@rm -rf docs/build/ docs/source/api/generated/

# Install the package in development mode
dev:
	pip install -e .[dev]

# Install development dependencies
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
