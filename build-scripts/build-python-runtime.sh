#!/bin/bash
# Build Python venv for bundling with Electron app
# This script creates a self-contained Python environment

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PROJECT_ROOT/python"
RESOURCES_DIR="$PROJECT_ROOT/resources"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        PLATFORM="mac-arm64"
    else
        PLATFORM="mac-x64"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux-x64"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PLATFORM="win-x64"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Building Python venv for platform: $PLATFORM"

# Create resources directory structure
RUNTIME_DIR="$RESOURCES_DIR/python-runtime/$PLATFORM"
mkdir -p "$RUNTIME_DIR"

# Clean old venv if exists
if [ -d "$RUNTIME_DIR/bin" ] || [ -d "$RUNTIME_DIR/Scripts" ]; then
    echo "Cleaning old venv..."
    rm -rf "$RUNTIME_DIR"/*
fi

# Create fresh venv
echo "Creating Python virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    python -m venv "$RUNTIME_DIR"
    PYTHON_EXE="$RUNTIME_DIR/Scripts/python.exe"
    PIP_EXE="$RUNTIME_DIR/Scripts/pip.exe"
else
    python3 -m venv "$RUNTIME_DIR"
    PYTHON_EXE="$RUNTIME_DIR/bin/python3"
    PIP_EXE="$RUNTIME_DIR/bin/pip3"
fi

# Upgrade pip
echo "Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
"$PIP_EXE" install -r "$PYTHON_DIR/requirements.txt"

# Copy application files to venv
echo "Copying application files..."
APP_DIR="$RUNTIME_DIR/app"
mkdir -p "$APP_DIR"
cp "$PYTHON_DIR/msk2k_audio_qso_server_Q12.py" "$APP_DIR/"
cp "$PYTHON_DIR/msk2k_complete.py" "$APP_DIR/"
cp "$PYTHON_DIR/msk2k_audio_qso_ui_Q12.html" "$APP_DIR/"

# Create version file
echo "v0.2.0" > "$RUNTIME_DIR/runtime_version.txt"

# List installed packages
echo "Installed packages:"
"$PIP_EXE" list

echo ""
echo "✅ Python venv built successfully!"
echo "   Platform: $PLATFORM"
echo "   Location: $RUNTIME_DIR"
echo ""
