#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# File        : setup_venv.sh
# Created     : 2025-03-27
# Description : This is a script to setup the virtual environment for the Quark project.
# =============================================================================

# Set environment name and Python version (if needed)
VENV_DIR="venv"
PYTHON_CMD="python3"  # Change to python3.x if necessary

# Check if Python is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD is not installed. Please install Python and try again."
    exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists in $VENV_DIR. Skipping creation."
    source $VENV_DIR/bin/activate
else
    # Create virtual environment
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade
echo "Upgrading pip..."
pip install --upgrade pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing the pinned dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt" || { echo "Error: dependency installation failed."; exit 1; }

# Print success message
echo "Virtual environment setup complete. Activate it using:"
echo "source $VENV_DIR/bin/activate"

mkdir ../gguf_files
mkdir ../q4nx_files
