#!/bin/bash

# Script to start the Onsei API for pitch accent comparison

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ONSEI_DIR="$SCRIPT_DIR/OnseiModified"

echo "=========================================="
echo "Onsei API Startup Script"
echo "=========================================="
echo ""

# Check if OnseiModified directory exists
if [ ! -d "$ONSEI_DIR" ]; then
    echo "❌ Error: OnseiModified directory not found!"
    echo ""
    echo "Please clone the repository first:"
    echo "  cd $SCRIPT_DIR"
    echo "  git clone https://github.com/derekvawdrey/OnseiModified.git"
    echo ""
    exit 1
fi

echo "Found OnseiModified directory ✓"
echo ""

cd "$ONSEI_DIR"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker and try again."
    echo ""
    exit 1
fi

echo "Docker is running ✓"
echo ""

# Check if onsei image exists, build if not
if ! docker images | grep -q "^onsei "; then
    echo "Building onsei base image..."
    docker build -t onsei .
    echo ""
fi

# Check if onsei-api image exists, build if not
if ! docker images | grep -q "^onsei-api "; then
    echo "Building onsei-api image..."
    docker build -f Dockerfile.api -t onsei-api .
    echo ""
fi

# Check if API is already running
if curl -s http://127.0.0.1:8000/ > /dev/null 2>&1; then
    echo "⚠️  API is already running on port 8000"
    echo ""
    read -p "Do you want to restart it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing container..."
        docker ps -q --filter "ancestor=onsei-api" | xargs -r docker stop
        docker ps -aq --filter "ancestor=onsei-api" | xargs -r docker rm
    else
        echo "Using existing API instance."
        exit 0
    fi
fi

echo "Starting Onsei API..."
echo ""
echo "The API will be available at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the API"
echo ""
echo "=========================================="
echo ""

# Run the API container
# Note: Using -p for port mapping (works on all platforms)
# --network=host only works properly on Linux
docker run -p 8000:8000 onsei-api

