#!/bin/bash


# This script initializes the development environment using Poetry.

# Exit immediately if a command exits with a non-zero status
set -e

# Set up a new Poetry environment with Python 3.12
poetry env use 3.12

# Install dependencies
poetry install

# Sync
poetry sync

# Lock the dependencies
poetry lock