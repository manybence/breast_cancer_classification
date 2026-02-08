#!/usr/bin/env bash

set -o errexit
set -o pipefail

pause_on_error () {
  echo
  echo "❌ An error occurred."
  echo "Press ENTER to exit..."
  read
}

trap pause_on_error ERR

echo "=================================="
echo " Setting up Python environment"
echo "=================================="

echo "Creating virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
source .venv/Scripts/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "=================================="
echo " ✅ Environment setup complete!"
echo "=================================="
echo
echo "To activate the environment later:"
echo "  source .venv/Scripts/activate"
echo
echo "Press ENTER to close..."
read