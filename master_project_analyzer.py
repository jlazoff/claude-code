#!/usr/bin/env python3
"""
Master Project Analyzer - Analyzes ChatGPT conversations and GitHub repositories
to create a comprehensive agentic orchestration system.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import re

def analyze_chatgpt_conversations(file_path):
    """Analyze ChatGPT conversation export for project insights."""
    print(f"Analyzing ChatGPT conversations from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        project_keywords = set()
        tech_stack = Counter()
        ai_topics = Counter()
        
        # Keywords to identify project-related conversations
        project_terms = ['project', 'architecture', 'system', 'build', 'create', 'develop', 'implement']
        tech_terms = ['kubernetes', 'docker', 'ansible', 'terraform', 'ray', 'airflow', 'vllm', 'arangodb', 'neo4j', 'dspy', 'pydantic', 'rag', 'llm', 'ai', 'ml', 'agent']
        
        # Process conversations
        for conv_id, conversation in data.items():
            if not isinstance(conversation, dict):
                continue
                
            title = conversation.get('title', '').lower()
            messages = conversation.get('mapping', {})
            
            # Extract text content from messages
            text_content = []
            for msg_id, msg_data in messages.items():
                if msg_data and 'message' in msg_data and msg_data['message']:
                    content = msg_data['message'].get('content')
                    if content and 'parts' in content:
                        for part in content['parts']:
                            if isinstance(part, str):
                                text_content.append(part.lower())
            
            full_text = ' '.join(text_content)
            
            # Count technical terms
            for term in tech_terms:
                count = full_text.count(term)
                if count > 0:
                    tech_stack[term] += count
            
            # Identify project-related conversations
            if any(term in title or term in full_text for term in project_terms):
                conversations.append({
                    'title': conversation.get('title', 'Untitled'),
                    'create_time': conversation.get('create_time'),
                    'update_time': conversation.get('update_time'),
                    'length': len(full_text),
                    'tech_mentions': {term: full_text.count(term) for term in tech_terms if full_text.count(term) > 0}
                })
        
        return {
            'total_conversations': len(data),
            'project_conversations': len(conversations),
            'top_tech_stack': tech_stack.most_common(20),
            'recent_projects': sorted(conversations, key=lambda x: x.get('update_time', 0), reverse=True)[:10]
        }
        
    except Exception as e:
        print(f"Error analyzing conversations: {e}")
        return None

def analyze_github_repositories(github_path):
    """Analyze GitHub repositories for capabilities and architecture."""
    repos = {}
    
    for repo_dir in Path(github_path).iterdir():
        if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
            repo_info = analyze_single_repo(repo_dir)
            repos[repo_dir.name] = repo_info
    
    return repos

def analyze_single_repo(repo_path):
    """Analyze a single repository for its capabilities."""
    info = {
        'path': str(repo_path),
        'technologies': [],
        'capabilities': [],
        'config_files': [],
        'entry_points': []
    }
    
    # Check for common files
    for file_name in ['package.json', 'pyproject.toml', 'requirements.txt', 'Cargo.toml', 'go.mod', 'Dockerfile', 'docker-compose.yml', 'kubernetes', 'k8s']:
        if (repo_path / file_name).exists():
            info['config_files'].append(file_name)
    
    # Check for entry points
    for file_name in ['main.py', 'app.py', 'server.py', 'index.js', 'main.js', 'main.go', 'main.rs']:
        if (repo_path / file_name).exists():
            info['entry_points'].append(file_name)
    
    # Determine technologies based on files
    if any(f.endswith('.py') for f in os.listdir(repo_path) if os.path.isfile(repo_path / f)):
        info['technologies'].append('Python')
    if any(f.endswith('.js') or f.endswith('.ts') for f in os.listdir(repo_path) if os.path.isfile(repo_path / f)):
        info['technologies'].append('JavaScript/TypeScript')
    if any(f.endswith('.go') for f in os.listdir(repo_path) if os.path.isfile(repo_path / f)):
        info['technologies'].append('Go')
    if any(f.endswith('.rs') for f in os.listdir(repo_path) if os.path.isfile(repo_path / f)):
        info['technologies'].append('Rust')
    
    # Read README for capabilities
    readme_path = repo_path / 'README.md'
    if readme_path.exists():
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read().lower()
                
            # Extract capabilities from README
            capability_keywords = ['api', 'web', 'agent', 'llm', 'ai', 'ml', 'database', 'ui', 'frontend', 'backend', 'microservice', 'service']
            for keyword in capability_keywords:
                if keyword in readme_content:
                    info['capabilities'].append(keyword)
        except:
            pass
    
    return info

def create_master_architecture():
    """Create the master project architecture."""
    architecture = {
        'foundation_layer': {
            'infrastructure': ['Kubernetes', 'Docker/Podman', 'Terraform'],
            'orchestration': ['Ansible', 'Ray', 'Airflow 3'],
            'networking': ['Thunderbolt', '10GB Mesh', 'L3 Network']
        },
        'data_layer': {
            'knowledge_graph': ['ArangoDB (Primary)', 'Neo4J (Secondary)'],
            'storage': ['Iceberg', 'Synology NAS (1PB)', 'Asustor FlashGen SSD NAS'],
            'streaming': ['Flink', 'Ray']
        },
        'ai_layer': {
            'models': ['vLLM', 'vLLM-d', 'Local Models'],
            'providers': ['OpenAI', 'Google Gemini', 'Anthropic Claude'],
            'frameworks': ['DSPY', 'Pydantic', 'RAG', 'MCMC', 'RL', 'GNN']
        },
        'agent_layer': {
            'frameworks': ['AutoGPT', 'MetaGPT', 'CrewAI', 'Langroid', 'Letta'],
            'tools': ['MCP', 'AG-UI', 'Magentic-UI', 'OpenHands', 'Jarvis']
        },
        'interface_layer': {
            'web_ui': ['Magentic-UI', 'Custom Dashboard'],
            'cli': ['Claude Code', 'Custom CLI Tools'],
            'apis': ['REST APIs', 'GraphQL', 'MCP Servers']
        },
        'hardware_layer': {
            'local': ['2x Mac Studio (512GB)', '2x Mac Mini M4 Max (64GB)', 'MacBook Pro M4 Max (128GB)'],
            'storage': ['1PB Synology NAS', 'Asustor FlashGen 12 Pro Gen 2'],
            'networking': ['Thunderbolt Network', '10GB L3 Mesh']
        }
    }
    
    return architecture

def main():
    """Main analysis function."""
    print("🚀 Master Project Analyzer Starting...")
    
    # Analyze ChatGPT conversations
    chatgpt_file = "/Users/jlazoff/Documents/conversations.json"
    print("\n📊 Analyzing ChatGPT Conversations...")
    conv_analysis = analyze_chatgpt_conversations(chatgpt_file)
    
    if conv_analysis:
        print(f"✅ Found {conv_analysis['total_conversations']} total conversations")
        print(f"✅ Identified {conv_analysis['project_conversations']} project-related conversations")
        print("\n🔧 Top Technologies Mentioned:")
        for tech, count in conv_analysis['top_tech_stack'][:10]:
            print(f"  - {tech}: {count} mentions")
    
    # Analyze GitHub repositories
    github_path = "/Users/jlazoff/Documents/GitHub"
    print(f"\n📁 Analyzing GitHub Repositories in {github_path}...")
    repo_analysis = analyze_github_repositories(github_path)
    
    print(f"✅ Found {len(repo_analysis)} repositories")
    
    # Group repositories by type
    agent_repos = []
    ui_repos = []
    infrastructure_repos = []
    data_repos = []
    
    for name, info in repo_analysis.items():
        if any(cap in ['agent', 'ai', 'llm'] for cap in info['capabilities']):
            agent_repos.append(name)
        elif any(cap in ['ui', 'frontend', 'web'] for cap in info['capabilities']):
            ui_repos.append(name)
        elif any(cap in ['api', 'backend', 'service'] for cap in info['capabilities']):
            infrastructure_repos.append(name)
        elif any(cap in ['database'] for cap in info['capabilities']):
            data_repos.append(name)
    
    print(f"\n🤖 Agent Repositories: {len(agent_repos)}")
    for repo in agent_repos[:5]:
        print(f"  - {repo}")
    
    print(f"\n🖥️  UI Repositories: {len(ui_repos)}")
    for repo in ui_repos:
        print(f"  - {repo}")
    
    print(f"\n🏗️  Infrastructure Repositories: {len(infrastructure_repos)}")
    for repo in infrastructure_repos[:5]:
        print(f"  - {repo}")
    
    # Create master architecture
    print("\n🏛️  Master Architecture Design:")
    architecture = create_master_architecture()
    
    for layer_name, layer_config in architecture.items():
        print(f"\n📦 {layer_name.replace('_', ' ').title()}:")
        for category, items in layer_config.items():
            print(f"  🔹 {category.title()}: {', '.join(items)}")
    
    # Generate next steps
    print("\n🎯 Immediate Next Steps:")
    print("1. ✅ Repository Analysis Complete")
    print("2. 🔄 ChatGPT Analysis Complete") 
    print("3. 🏗️  Create Foundation Infrastructure (Kubernetes + Docker)")
    print("4. 📊 Setup ArangoDB Knowledge Graph")
    print("5. 🤖 Implement DSPY/Pydantic Agent Framework")
    print("6. 🖥️  Deploy Unified UI Dashboard")
    print("7. 🔗 Connect Hardware Network")
    print("8. 📈 Create Airflow DAGs for Orchestration")
    
    return {
        'conversations': conv_analysis,
        'repositories': repo_analysis,
        'architecture': architecture
    }

if __name__ == "__main__":
    results = main()