#!/bin/bash

# Master Orchestrator Launch Script
echo "🚀 Master Orchestrator Launch"
echo "=============================="

cd "$(dirname "$0")"

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Master Orchestrator already running!"
    echo "🌐 Dashboard: http://localhost:8000"
    echo "📊 API Status: http://localhost:8000/api/status"
    exit 0
fi

# Start the server
echo "🚀 Starting Master Orchestrator server..."
echo "🌐 Dashboard will be available at: http://localhost:8000"
echo "📊 API endpoints will be available at: http://localhost:8000/api/"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Launch in foreground for interactive use
exec python3 simple_server.py