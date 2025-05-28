#!/bin/bash

# Master Orchestrator Server Startup
echo "🚀 Starting Master Orchestrator..."

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source master-orchestrator-env/bin/activate

# Verify FastAPI is available
python -c "import fastapi; print('✅ FastAPI available')" || {
    echo "❌ FastAPI not found. Installing..."
    pip install fastapi uvicorn
}

# Start the server
echo "🌐 Starting server on http://localhost:8000"
python fixed_start.py