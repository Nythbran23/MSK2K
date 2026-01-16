#!/bin/bash
# MSK2K Electron Setup Script

set -e

echo "======================================"
echo "MSK2K Electron Setup"
echo "======================================"
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    echo "Please install Node.js from: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✓ Node.js: $NODE_VERSION"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found"
    exit 1
fi

NPM_VERSION=$(npm --version)
echo "✓ npm: $NPM_VERSION"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo "Please install Python 3.9 or later"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python: $PYTHON_VERSION"

echo ""
echo "Installing Node.js dependencies..."
npm install

echo ""
echo "Installing Python dependencies..."
pip3 install -r python/requirements.txt

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Run the application:"
echo "  npm start          # Development mode"
echo ""
echo "Build the application:"
echo "  npm run dist       # Current platform"
echo "  npm run dist:mac   # macOS"
echo "  npm run dist:win   # Windows"
echo "  npm run dist:linux # Linux"
echo ""
