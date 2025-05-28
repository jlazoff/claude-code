#!/bin/bash

# Master Orchestrator Live Server Startup
echo "🚀 Starting Master Orchestrator Live System"
echo "==========================================="

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source master-orchestrator-env/bin/activate

# Check if required packages are installed
python -c "import websockets, watchdog; print('✅ Live dependencies available')" || {
    echo "📦 Installing live dependencies..."
    pip install websockets watchdog
}

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "live_server.py" 2>/dev/null || true
pkill -f "simple_server.py" 2>/dev/null || true

# Start the live server
echo "🌐 Starting Live Server with hot reload..."
echo "📡 WebSocket server will run on port 8001"
echo "🌍 HTTP server will run on port 8000"
echo ""
echo "✨ Features enabled:"
echo "   • Hot reload without page refresh"
echo "   • Real-time code generation"
echo "   • Continuous deployment"
echo "   • Live activity feed"
echo "   • Zero-downtime updates"
echo ""
echo "🌐 Access your live dashboard at: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start the live server
python live_server.py