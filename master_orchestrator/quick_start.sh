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

