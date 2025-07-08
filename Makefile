.PHONY: train submit clean lint format test

# Train the model
train:
	python src/train.py

# Create submission
submit:
	python src/submit.py

# Clean up generated files
clean:
	rm -rf outputs/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install dependencies
install:
	pip install -r requirements.lock

# Setup development environment
setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.lock

# Lint code with ruff
lint:
	ruff .

# Format code with black
format:
	black .

# Run tests with pytest
test:
	pytest -q
 