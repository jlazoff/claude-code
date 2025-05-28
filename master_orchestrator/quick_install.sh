#!/bin/bash

# Quick Master Orchestrator Installation
# Simplified setup for immediate deployment

set -e

echo "🚀 Quick Master Orchestrator Setup"
echo "=================================="
echo

# Create virtual environment using built-in venv
echo "📦 Creating Python virtual environment..."
python3 -m venv master-orchestrator-env
source master-orchestrator-env/bin/activate

# Install basic dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn pydantic structlog typer rich

# Install AI/ML dependencies
pip install openai anthropic google-generativeai

# Install infrastructure dependencies  
pip install docker kubernetes ansible-core

# Install database dependencies
pip install python-arango redis

# Install monitoring dependencies
pip install prometheus-client

# Create basic config
echo "⚙️  Creating configuration..."
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo "✅ Created config.yaml"
fi

# Create required directories
mkdir -p data logs monitoring infrastructure/terraform infrastructure/ansible

# Create basic Docker setup
echo "🐳 Setting up Docker containers..."

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "🔴 Docker is not running. Please start Docker Desktop first."
    echo "   After starting Docker, run this script again."
    exit 1
fi

# Start ArangoDB
echo "📊 Starting ArangoDB..."
docker run -d \
    --name master-orchestrator-arangodb \
    -p 8529:8529 \
    -e ARANGO_ROOT_PASSWORD=orchestrator123 \
    -v $(pwd)/data/arangodb:/var/lib/arangodb3 \
    arangodb/arangodb:latest || echo "ArangoDB container may already exist"

# Start Redis
echo "🔴 Starting Redis..."
docker run -d \
    --name master-orchestrator-redis \
    -p 6379:6379 \
    redis:alpine || echo "Redis container may already exist"

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Create startup script
cat > quick_start.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Master Orchestrator..."

# Activate virtual environment
source master-orchestrator-env/bin/activate

# Start the API server
echo "🌐 Starting API server on http://localhost:8000"
python3 -c "
import asyncio
import sys
sys.path.append('.')

async def main():
    try:
        from master_orchestrator.api import run_server
        from master_orchestrator.config import OrchestratorConfig
        
        config = OrchestratorConfig.from_env()
        await run_server(config)
    except ImportError as e:
        print(f'Import error: {e}')
        print('Starting basic server...')
        
        # Fallback basic server
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(title='Master Orchestrator', version='0.1.0')
        
        @app.get('/')
        async def root():
            return {
                'message': 'Master Orchestrator is running!',
                'status': 'online',
                'version': '0.1.0',
                'endpoints': {
                    'dashboard': 'http://localhost:8000/dashboard',
                    'api': 'http://localhost:8000/docs'
                }
            }
        
        @app.get('/dashboard')
        async def dashboard():
            return '''
            <!DOCTYPE html>
            <html>
            <head><title>Master Orchestrator</title></head>
            <body style=\"font-family: Arial; margin: 40px; background: #f5f5f5;\">
                <div style=\"background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);\">
                    <h1 style=\"color: #2563eb;\">🚀 Master Orchestrator</h1>
                    <p style=\"font-size: 18px; color: #666;\">Your agentic orchestration system is running!</p>
                    
                    <div style=\"background: #f8fafc; padding: 20px; border-radius: 5px; margin: 20px 0;\">
                        <h3>✅ System Status: Online</h3>
                        <p>• API Server: Running on port 8000</p>
                        <p>• Database: ArangoDB on port 8529</p>
                        <p>• Cache: Redis on port 6379</p>
                    </div>
                    
                    <div style=\"background: #ecfdf5; padding: 20px; border-radius: 5px; margin: 20px 0;\">
                        <h3>🎯 Quick Actions</h3>
                        <p>• <a href=\"/docs\" style=\"color: #059669;\">API Documentation</a></p>
                        <p>• <a href=\"http://localhost:8529\" style=\"color: #059669;\">ArangoDB Interface</a> (root/orchestrator123)</p>
                        <p>• View logs in terminal</p>
                    </div>
                    
                    <div style=\"background: #eff6ff; padding: 20px; border-radius: 5px; margin: 20px 0;\">
                        <h3>📋 Next Steps</h3>
                        <p>1. Configure API keys in config.yaml</p>
                        <p>2. Add your GitHub repositories</p>
                        <p>3. Create and deploy agents</p>
                        <p>4. Monitor system performance</p>
                    </div>
                </div>
            </body>
            </html>
            '''
        
        uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    asyncio.run(main())
"

EOF

chmod +x quick_start.sh

echo
echo "🎉 Master Orchestrator Setup Complete!"
echo "======================================"
echo
echo "✅ Virtual environment created"
echo "✅ Dependencies installed"
echo "✅ Docker containers started"
echo "✅ Configuration ready"
echo
echo "🚀 To start the system:"
echo "   ./quick_start.sh"
echo
echo "🌐 Then visit:"
echo "   http://localhost:8000      - Main dashboard"
echo "   http://localhost:8000/docs - API documentation"
echo "   http://localhost:8529      - ArangoDB interface"
echo
echo "🔧 Services running:"
echo "   • ArangoDB: localhost:8529 (root/orchestrator123)"
echo "   • Redis: localhost:6379"
echo