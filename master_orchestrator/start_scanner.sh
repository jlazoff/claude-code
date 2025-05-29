#!/bin/bash
# Quick start script for Continuous Repository Scanner

echo "Starting Continuous Repository Scanner..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "scanner_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv scanner_venv
fi

# Activate virtual environment
source scanner_venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_scanner.txt

# Run initial scan
echo "Running initial repository scan..."
python continuous_repo_scanner.py --base-path /Users/jlazoff/Documents/GitHub --once

# Check if scan was successful
if [ $? -eq 0 ]; then
    echo "Initial scan complete!"
    echo ""
    echo "Generated files:"
    echo "  - tool_catalog.json - Complete tool catalog"
    echo "  - tool_catalog_report.md - Analysis report"
    echo "  - architecture_diagram.md - System architecture"
    echo ""
    
    # Ask if user wants to generate deployment strategies
    read -p "Generate deployment strategies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Generating deployment strategies..."
        python deployment_testing_strategy.py
        echo "Deployment strategies generated in deployment_strategies/"
    fi
    
    # Ask if user wants to start continuous scanning
    read -p "Start continuous scanning? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting continuous scanner (Ctrl+C to stop)..."
        python continuous_repo_scanner.py --base-path /Users/jlazoff/Documents/GitHub
    fi
else
    echo "Scan failed. Check repo_scanner.log for details."
    exit 1
fi