#!/usr/bin/env python3
"""
Conversation Project Initiator
Extracts and implements projects from ChatGPT conversations.json files
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib

from unified_config import SecureConfigManager
from parallel_llm_orchestrator import ParallelLLMOrchestrator
from github_integration import AutomatedDevelopmentWorkflow
from frontend_orchestrator import FrontendOrchestrator

class ConversationProjectInitiator:
    """Extract and implement projects from conversation data"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.llm_orchestrator = ParallelLLMOrchestrator()
        self.dev_workflow = AutomatedDevelopmentWorkflow()
        self.frontend = FrontendOrchestrator()
        self.projects_extracted = []
        self.implementation_results = []
        
    async def initialize(self):
        """Initialize all components"""
        await self.config.initialize()
        await self.llm_orchestrator.initialize()
        await self.dev_workflow.initialize()
        await self.frontend.initialize()
        
        logging.info("Conversation Project Initiator initialized")
        
    async def process_conversations_file(self, file_path: str) -> Dict[str, Any]:
        """Process conversations.json file and extract projects"""
        try:
            # Load conversations data
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations_data = json.load(f)
                
            # Extract projects from conversations
            projects = await self._extract_projects_from_conversations(conversations_data)
            
            # Prioritize and filter projects
            prioritized_projects = await self._prioritize_projects(projects)
            
            # Implement top priority projects
            implementation_results = []
            for project in prioritized_projects[:10]:  # Implement top 10 projects
                logging.info(f"Implementing project: {project['name']}")
                result = await self._implement_project(project)
                implementation_results.append(result)
                
            return {
                "success": True,
                "total_projects_found": len(projects),
                "projects_implemented": len(implementation_results),
                "projects": projects,
                "implementations": implementation_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error processing conversations file: {e}")
            return {"success": False, "error": str(e)}
            
    async def _extract_projects_from_conversations(self, conversations_data: Union[Dict, List]) -> List[Dict[str, Any]]:
        """Extract project ideas from conversation data"""
        projects = []
        
        # Handle different conversation data formats
        conversations = []
        if isinstance(conversations_data, dict):
            if 'conversations' in conversations_data:
                conversations = conversations_data['conversations']
            elif 'data' in conversations_data:
                conversations = conversations_data['data']
            else:
                conversations = [conversations_data]
        elif isinstance(conversations_data, list):
            conversations = conversations_data
            
        for conversation in conversations:
            try:
                # Extract messages from conversation
                messages = self._extract_messages(conversation)
                
                # Analyze messages for project mentions
                conversation_projects = await self._analyze_messages_for_projects(messages)
                projects.extend(conversation_projects)
                
            except Exception as e:
                logging.warning(f"Error processing conversation: {e}")
                continue
                
        # Remove duplicates and merge similar projects
        unique_projects = await self._deduplicate_projects(projects)
        
        return unique_projects
        
    def _extract_messages(self, conversation: Dict[str, Any]) -> List[str]:
        """Extract text messages from conversation structure"""
        messages = []
        
        # Handle different conversation structures
        if 'mapping' in conversation:
            # ChatGPT export format
            for node_id, node in conversation['mapping'].items():
                if 'message' in node and node['message']:
                    message = node['message']
                    if 'content' in message and 'parts' in message['content']:
                        for part in message['content']['parts']:
                            if isinstance(part, str) and len(part.strip()) > 20:
                                messages.append(part.strip())
        elif 'messages' in conversation:
            # Direct messages format
            for message in conversation['messages']:
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    if isinstance(content, str) and len(content.strip()) > 20:
                        messages.append(content.strip())
                elif isinstance(message, str) and len(message.strip()) > 20:
                    messages.append(message.strip())
        elif isinstance(conversation, str):
            # Single message
            messages.append(conversation)
            
        return messages
        
    async def _analyze_messages_for_projects(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Analyze messages to identify project mentions"""
        projects = []
        
        # Combine all messages for analysis
        full_text = ' '.join(messages)
        
        # Use LLM to extract project ideas
        project_extraction_prompt = f"""
Analyze this conversation text and extract specific software project ideas, features, or applications mentioned:

Text: {full_text[:8000]}  # Limit to avoid token limits

Extract projects that have:
1. Clear technical implementation possibilities
2. Specific functionality described
3. Potential business or utility value
4. Feasible scope for development

For each project found, provide:
- name: Short descriptive name
- description: What the project does
- category: Type of application (web app, API, tool, etc.)
- complexity: low, medium, or high
- technologies: Suggested tech stack
- features: Key features list
- priority: high, medium, or low based on utility and feasibility

Return as JSON array of project objects.
"""
        
        try:
            result = await self.llm_orchestrator.generate_code_parallel(
                project_extraction_prompt, 
                "consensus"
            )
            
            if result.get("success") and result.get("merged_code"):
                # Try to extract JSON from the response
                response_text = result["merged_code"]
                projects_data = self._extract_json_from_text(response_text)
                
                if projects_data and isinstance(projects_data, list):
                    for project_data in projects_data:
                        if isinstance(project_data, dict) and 'name' in project_data:
                            project = {
                                "id": hashlib.md5(project_data['name'].encode()).hexdigest()[:8],
                                "name": project_data.get('name', 'Unnamed Project'),
                                "description": project_data.get('description', ''),
                                "category": project_data.get('category', 'general'),
                                "complexity": project_data.get('complexity', 'medium'),
                                "technologies": project_data.get('technologies', []),
                                "features": project_data.get('features', []),
                                "priority": project_data.get('priority', 'medium'),
                                "source": "conversation_analysis",
                                "extracted_at": datetime.now().isoformat()
                            }
                            projects.append(project)
                            
        except Exception as e:
            logging.error(f"Error in LLM project extraction: {e}")
            
        # Fallback: Use pattern matching for common project indicators
        if not projects:
            projects.extend(self._pattern_based_project_extraction(full_text))
            
        return projects
        
    def _extract_json_from_text(self, text: str) -> Optional[Union[Dict, List]]:
        """Extract JSON data from text response"""
        try:
            # Look for JSON blocks
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*([\[{].*?[}\]])\s*```',
                r'(\[[\s\S]*?\])',
                r'(\{[\s\S]*?\})'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
                        
            # Try parsing the entire text as JSON
            return json.loads(text)
            
        except Exception as e:
            logging.debug(f"Could not extract JSON from text: {e}")
            return None
            
    def _pattern_based_project_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Extract projects using pattern matching as fallback"""
        projects = []
        
        # Common project indicators
        project_patterns = [
            r'(?:build|create|develop|make)\s+(?:a|an)\s+([^.!?]+?)(?:\.|!|\?|$)',
            r'(?:app|application|tool|system|platform)\s+(?:that|to|for)\s+([^.!?]+?)(?:\.|!|\?|$)',
            r'(?:website|dashboard|interface)\s+(?:for|to)\s+([^.!?]+?)(?:\.|!|\?|$)',
            r'(?:API|service|microservice)\s+(?:that|to|for)\s+([^.!?]+?)(?:\.|!|\?|$)'
        ]
        
        for pattern in project_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                project_desc = match.group(1).strip()
                if len(project_desc) > 10 and len(project_desc) < 200:
                    project = {
                        "id": hashlib.md5(project_desc.encode()).hexdigest()[:8],
                        "name": project_desc[:50] + "..." if len(project_desc) > 50 else project_desc,
                        "description": project_desc,
                        "category": "general",
                        "complexity": "medium",
                        "technologies": [],
                        "features": [],
                        "priority": "medium",
                        "source": "pattern_matching",
                        "extracted_at": datetime.now().isoformat()
                    }
                    projects.append(project)
                    
        return projects[:5]  # Limit to 5 pattern-based projects
        
    async def _deduplicate_projects(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and merge similar projects"""
        if not projects:
            return []
            
        # Group similar projects by name similarity
        groups = []
        for project in projects:
            added_to_group = False
            for group in groups:
                if self._are_projects_similar(project, group[0]):
                    group.append(project)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([project])
                
        # Merge each group into a single project
        unique_projects = []
        for group in groups:
            if len(group) == 1:
                unique_projects.append(group[0])
            else:
                merged_project = await self._merge_similar_projects(group)
                unique_projects.append(merged_project)
                
        return unique_projects
        
    def _are_projects_similar(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> bool:
        """Check if two projects are similar"""
        name1 = project1.get('name', '').lower()
        name2 = project2.get('name', '').lower()
        desc1 = project1.get('description', '').lower()
        desc2 = project2.get('description', '').lower()
        
        # Simple similarity check
        name_similarity = len(set(name1.split()) & set(name2.split())) / max(len(name1.split()), len(name2.split()), 1)
        desc_similarity = len(set(desc1.split()) & set(desc2.split())) / max(len(desc1.split()), len(desc2.split()), 1)
        
        return name_similarity > 0.5 or desc_similarity > 0.3
        
    async def _merge_similar_projects(self, projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar projects into one comprehensive project"""
        if not projects:
            return {}
            
        # Use the first project as base
        merged = projects[0].copy()
        
        # Combine features and technologies
        all_features = []
        all_technologies = []
        
        for project in projects:
            all_features.extend(project.get('features', []))
            all_technologies.extend(project.get('technologies', []))
            
        merged['features'] = list(set(all_features))
        merged['technologies'] = list(set(all_technologies))
        
        # Combine descriptions
        descriptions = [p.get('description', '') for p in projects if p.get('description')]
        if descriptions:
            merged['description'] = ' | '.join(descriptions[:3])  # Limit to avoid too long
            
        # Set highest priority
        priorities = [p.get('priority', 'medium') for p in projects]
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        highest_priority = max(priorities, key=lambda p: priority_order.get(p, 2))
        merged['priority'] = highest_priority
        
        return merged
        
    async def _prioritize_projects(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize projects based on various factors"""
        if not projects:
            return []
            
        # Score each project
        for project in projects:
            score = 0
            
            # Priority score
            priority_scores = {'high': 30, 'medium': 20, 'low': 10}
            score += priority_scores.get(project.get('priority', 'medium'), 20)
            
            # Complexity score (lower complexity = higher score for immediate implementation)
            complexity_scores = {'low': 25, 'medium': 15, 'high': 5}
            score += complexity_scores.get(project.get('complexity', 'medium'), 15)
            
            # Feature count score
            features_count = len(project.get('features', []))
            score += min(features_count * 2, 20)  # Cap at 20
            
            # Technology familiarity score
            common_technologies = ['python', 'javascript', 'react', 'fastapi', 'flask', 'sqlite', 'postgresql']
            tech_score = sum(1 for tech in project.get('technologies', []) 
                           if any(common_tech in tech.lower() for common_tech in common_technologies))
            score += min(tech_score * 3, 15)
            
            # Description completeness
            desc_length = len(project.get('description', ''))
            if desc_length > 50:
                score += 10
            elif desc_length > 20:
                score += 5
                
            project['priority_score'] = score
            
        # Sort by priority score (descending)
        return sorted(projects, key=lambda p: p.get('priority_score', 0), reverse=True)
        
    async def _implement_project(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Implement a specific project"""
        try:
            logging.info(f"Starting implementation of: {project['name']}")
            
            # Generate detailed implementation plan
            implementation_plan = await self._generate_implementation_plan(project)
            
            # Generate code for the project
            code_result = await self._generate_project_code(project, implementation_plan)
            
            # Create project structure
            project_structure = await self._create_project_structure(project, code_result)
            
            # Generate tests
            test_result = await self._generate_project_tests(project, code_result)
            
            # Create documentation
            docs_result = await self._generate_project_documentation(project, code_result)
            
            # Create deployment configuration
            deployment_config = await self._generate_deployment_config(project)
            
            # Register project with frontend
            await self._register_project_with_frontend(project, project_structure)
            
            return {
                "success": True,
                "project": project,
                "implementation_plan": implementation_plan,
                "code": code_result,
                "structure": project_structure,
                "tests": test_result,
                "documentation": docs_result,
                "deployment": deployment_config,
                "implemented_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error implementing project {project['name']}: {e}")
            return {
                "success": False,
                "project": project,
                "error": str(e),
                "attempted_at": datetime.now().isoformat()
            }
            
    async def _generate_implementation_plan(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed implementation plan"""
        plan_prompt = f"""
Create a detailed implementation plan for this project:

Project: {project['name']}
Description: {project['description']}
Category: {project['category']}
Complexity: {project['complexity']}
Features: {project.get('features', [])}
Technologies: {project.get('technologies', [])}

Generate a comprehensive implementation plan including:
1. Architecture overview
2. Technology stack recommendations
3. File structure
4. Implementation phases
5. Testing strategy
6. Deployment approach
7. Timeline estimation

Return as structured JSON.
"""
        
        result = await self.llm_orchestrator.generate_code_parallel(plan_prompt, "comprehensive")
        
        if result.get("success"):
            plan_data = self._extract_json_from_text(result["merged_code"])
            if plan_data:
                return plan_data
                
        # Fallback plan
        return {
            "architecture": "microservice",
            "technology_stack": ["Python", "FastAPI", "SQLite", "React"],
            "phases": ["setup", "core_implementation", "testing", "deployment"],
            "estimated_timeline": f"{project.get('complexity', 'medium')}_complexity_project"
        }
        
    async def _generate_project_code(self, project: Dict[str, Any], implementation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete project code"""
        code_prompt = f"""
Generate complete, production-ready code for this project:

Project: {project['name']}
Description: {project['description']}
Features: {project.get('features', [])}
Implementation Plan: {json.dumps(implementation_plan, indent=2)}

Create a complete application with:
1. Main application code
2. API endpoints (if applicable)
3. Database models
4. Frontend components (if applicable)
5. Configuration files
6. Requirements/dependencies
7. Error handling and logging
8. Security best practices

Generate modular, well-documented, production-ready code.
"""
        
        result = await self.llm_orchestrator.generate_code_parallel(code_prompt, "comprehensive")
        
        return {
            "success": result.get("success", False),
            "main_code": result.get("merged_code", ""),
            "providers_used": result.get("source_providers", []),
            "generation_metadata": result.get("execution_info", {})
        }
        
    async def _create_project_structure(self, project: Dict[str, Any], code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create physical project structure on disk"""
        try:
            # Create project directory
            project_dir = Path("projects") / project['id']
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create main application file
            main_file = project_dir / "main.py"
            with open(main_file, 'w') as f:
                f.write(code_result.get("main_code", "# Generated project code\nprint('Hello from generated project!')"))
                
            # Create requirements.txt
            requirements_file = project_dir / "requirements.txt"
            with open(requirements_file, 'w') as f:
                f.write("fastapi\nuvicorn\nsqlalchemy\npydantic\naiofiles\n")
                
            # Create README.md
            readme_file = project_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(f"""# {project['name']}

{project['description']}

## Generated Project

This project was automatically generated by Master Orchestrator.

### Features
{chr(10).join(f"- {feature}" for feature in project.get('features', []))}

### Technologies
{chr(10).join(f"- {tech}" for tech in project.get('technologies', []))}

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

Generated at: {datetime.now().isoformat()}
""")
            
            # Create configuration
            config_file = project_dir / "config.py"
            with open(config_file, 'w') as f:
                f.write("""# Configuration for generated project
import os

DEBUG = True
HOST = os.getenv('HOST', 'localhost')
PORT = int(os.getenv('PORT', 8000))
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
""")
            
            return {
                "success": True,
                "project_directory": str(project_dir),
                "files_created": [
                    str(main_file),
                    str(requirements_file),
                    str(readme_file),
                    str(config_file)
                ]
            }
            
        except Exception as e:
            logging.error(f"Error creating project structure: {e}")
            return {"success": False, "error": str(e)}
            
    async def _generate_project_tests(self, project: Dict[str, Any], code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tests for the project"""
        test_prompt = f"""
Generate comprehensive tests for this project:

Project: {project['name']}
Code: {code_result.get('main_code', '')[:2000]}

Create:
1. Unit tests for all functions
2. Integration tests for API endpoints
3. End-to-end tests for main workflows
4. Test configuration and fixtures

Use pytest framework and include proper mocking where needed.
"""
        
        result = await self.llm_orchestrator.generate_code_parallel(test_prompt, "best_practices")
        
        if result.get("success"):
            # Save test file
            project_dir = Path("projects") / project['id']
            test_file = project_dir / "test_main.py"
            
            with open(test_file, 'w') as f:
                f.write(result.get("merged_code", "# Generated tests\nimport pytest\n\ndef test_placeholder():\n    assert True"))
                
            return {
                "success": True,
                "test_file": str(test_file),
                "test_code": result.get("merged_code", "")
            }
            
        return {"success": False, "error": "Test generation failed"}
        
    async def _generate_project_documentation(self, project: Dict[str, Any], code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project documentation"""
        docs_prompt = f"""
Generate comprehensive documentation for this project:

Project: {project['name']}
Description: {project['description']}
Features: {project.get('features', [])}

Create:
1. API documentation (if applicable)
2. User guide
3. Developer guide
4. Installation instructions
5. Configuration guide

Make it professional and complete.
"""
        
        result = await self.llm_orchestrator.generate_code_parallel(docs_prompt, "comprehensive")
        
        if result.get("success"):
            # Save documentation
            project_dir = Path("projects") / project['id']
            docs_dir = project_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            docs_file = docs_dir / "README.md"
            with open(docs_file, 'w') as f:
                f.write(result.get("merged_code", f"# {project['name']} Documentation\n\nGenerated documentation."))
                
            return {
                "success": True,
                "docs_directory": str(docs_dir),
                "documentation": result.get("merged_code", "")
            }
            
        return {"success": False, "error": "Documentation generation failed"}
        
    async def _generate_deployment_config(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment configuration"""
        project_dir = Path("projects") / project['id']
        
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
"""
        
        dockerfile = project_dir / "Dockerfile"
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
            
        # Create docker-compose.yml
        compose_content = f"""
version: '3.8'

services:
  {project['id']}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./data:/app/data
"""
        
        compose_file = project_dir / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_content)
            
        return {
            "success": True,
            "dockerfile": str(dockerfile),
            "compose_file": str(compose_file)
        }
        
    async def _register_project_with_frontend(self, project: Dict[str, Any], project_structure: Dict[str, Any]) -> None:
        """Register project with frontend orchestrator"""
        try:
            project_info = {
                "id": project['id'],
                "name": project['name'],
                "description": project['description'],
                "category": project['category'],
                "status": "implemented",
                "created_at": datetime.now().isoformat(),
                "files": project_structure.get("files_created", []),
                "deployments": []
            }
            
            self.frontend.active_projects[project['id']] = project_info
            
            # Broadcast project creation
            await self.frontend._broadcast_websocket({
                "type": "project_implemented",
                "project": project_info
            })
            
        except Exception as e:
            logging.error(f"Error registering project with frontend: {e}")

async def main():
    """Main function for testing conversation project initiator"""
    logging.basicConfig(level=logging.INFO)
    
    initiator = ConversationProjectInitiator()
    await initiator.initialize()
    
    # Test with sample conversation data
    sample_conversations = {
        "conversations": [
            {
                "mapping": {
                    "1": {
                        "message": {
                            "content": {
                                "parts": [
                                    "I want to build a real-time chat application with WebSocket support and user authentication. It should have rooms, private messaging, and file sharing capabilities."
                                ]
                            }
                        }
                    }
                }
            },
            {
                "mapping": {
                    "2": {
                        "message": {
                            "content": {
                                "parts": [
                                    "Create a task management system with Kanban boards, time tracking, and team collaboration features. Should integrate with calendar and have mobile app support."
                                ]
                            }
                        }
                    }
                }
            }
        ]
    }
    
    # Save sample data for testing
    with open("sample_conversations.json", 'w') as f:
        json.dump(sample_conversations, f, indent=2)
        
    # Process the conversations
    result = await initiator.process_conversations_file("sample_conversations.json")
    
    print(f"Processing result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())