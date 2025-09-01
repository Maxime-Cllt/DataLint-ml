#!/bin/bash

# This script initializes the development environment using uv.

# Exit immediately if a command exits with a non-zero status
set -e

PYTHON_VERSION="3.12"

# Ensure we're using Python 3.12 (uv will create the venv automatically)
echo "Setting up virtual environment with Python $PYTHON_VERSION"
uv venv --python $PYTHON_VERSION

# Install dependencies from pyproject.toml
echo "Installing dependencies"
uv sync

# (Optional) lock dependencies explicitly (uv maintains a uv.lock file)
echo "Locking dependencies"
uv lock
