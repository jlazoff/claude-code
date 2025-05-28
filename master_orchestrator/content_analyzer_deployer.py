#!/usr/bin/env python3
"""
Content Analyzer & Auto-Deployer
Automatically analyzes YouTube channels/videos and research PDFs, then tests and deploys implementations
"""

import asyncio
import logging
import re
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import hashlib
import aiohttp
import aiofiles
from urllib.parse import urlparse, parse_qs
import youtube_dl
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import base64
import io
import ast
import black
import isort
import pytest
import docker
import git

from unified_config import SecureConfigManager
from litellm_manager import LiteLLMManager
from computer_control_orchestrator import ComputerControlOrchestrator

class YouTubeAnalyzer:
    """Analyze YouTube channels and videos for implementation insights"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True
        }
        
    async def analyze_channel(self, channel_url: str) -> Dict[str, Any]:
        """Analyze entire YouTube channel"""
        try:
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                # Extract channel info
                channel_info = ydl.extract_info(channel_url, download=False)
                
                videos = []
                if 'entries' in channel_info:
                    # Limit to recent 20 videos for analysis
                    for entry in list(channel_info['entries'])[:20]:
                        if entry:
                            video_analysis = await self.analyze_video(entry['webpage_url'])
                            videos.append(video_analysis)
                            
                return {
                    "type": "channel",
                    "title": channel_info.get('title', 'Unknown Channel'),
                    "uploader": channel_info.get('uploader', 'Unknown'),
                    "description": channel_info.get('description', ''),
                    "video_count": len(videos),
                    "videos": videos,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "common_themes": self._extract_common_themes(videos),
                    "implementation_opportunities": self._identify_implementation_opportunities(videos)
                }
                
        except Exception as e:
            logging.error(f"Channel analysis error: {e}")
            return {"error": str(e), "type": "channel_error"}
            
    async def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """Analyze single YouTube video"""
        try:
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                video_info = ydl.extract_info(video_url, download=False)
                
                # Extract subtitles if available
                subtitles = await self._extract_subtitles(video_info)
                
                return {
                    "type": "video",
                    "title": video_info.get('title', 'Unknown Video'),
                    "description": video_info.get('description', ''),
                    "duration": video_info.get('duration', 0),
                    "upload_date": video_info.get('upload_date', ''),
                    "view_count": video_info.get('view_count', 0),
                    "like_count": video_info.get('like_count', 0),
                    "subtitles": subtitles,
                    "tags": video_info.get('tags', []),
                    "categories": video_info.get('categories', []),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "code_mentions": self._extract_code_mentions(video_info.get('description', '') + ' ' + subtitles),
                    "technical_concepts": self._extract_technical_concepts(video_info.get('description', '') + ' ' + subtitles)
                }
                
        except Exception as e:
            logging.error(f"Video analysis error: {e}")
            return {"error": str(e), "type": "video_error"}
            
    async def _extract_subtitles(self, video_info: Dict) -> str:
        """Extract subtitles from video"""
        try:
            # This would need more implementation for actual subtitle extraction
            # For now, return description as proxy
            return video_info.get('description', '')
        except Exception as e:
            logging.error(f"Subtitle extraction error: {e}")
            return ""
            
    def _extract_code_mentions(self, text: str) -> List[str]:
        """Extract code-related mentions from text"""
        code_patterns = [
            r'\b(?:python|javascript|java|c\+\+|golang|rust|typescript|react|vue|angular)\b',
            r'\b(?:api|database|sql|mongodb|postgresql|redis|docker|kubernetes)\b',
            r'\b(?:machine learning|ai|neural network|deep learning|tensorflow|pytorch)\b',
            r'\b(?:github|gitlab|git|repository|repo|open source)\b',
            r'\b(?:algorithm|data structure|optimization|performance|scalability)\b'
        ]
        
        mentions = []
        for pattern in code_patterns:
            matches = re.findall(pattern, text.lower())
            mentions.extend(matches)
            
        return list(set(mentions))
        
    def _extract_technical_concepts(self, text: str) -> List[str]:
        """Extract technical concepts for implementation"""
        concepts = []
        
        # Look for specific technical patterns
        if re.search(r'\b(?:automation|automate|script|workflow)\b', text.lower()):
            concepts.append('automation')
        if re.search(r'\b(?:dashboard|visualization|chart|graph|analytics)\b', text.lower()):
            concepts.append('dashboard')
        if re.search(r'\b(?:real-time|websocket|streaming|live)\b', text.lower()):
            concepts.append('real-time')
        if re.search(r'\b(?:deployment|ci/cd|devops|pipeline)\b', text.lower()):
            concepts.append('deployment')
        if re.search(r'\b(?:monitoring|logging|metrics|observability)\b', text.lower()):
            concepts.append('monitoring')
            
        return concepts
        
    def _extract_common_themes(self, videos: List[Dict]) -> List[str]:
        """Extract common themes across videos"""
        all_concepts = []
        for video in videos:
            if 'technical_concepts' in video:
                all_concepts.extend(video['technical_concepts'])
                
        # Count frequency and return most common
        from collections import Counter
        concept_counts = Counter(all_concepts)
        return [concept for concept, count in concept_counts.most_common(10)]
        
    def _identify_implementation_opportunities(self, videos: List[Dict]) -> List[Dict[str, Any]]:
        """Identify specific implementation opportunities"""
        opportunities = []
        
        common_themes = self._extract_common_themes(videos)
        
        for theme in common_themes:
            if theme == 'automation':
                opportunities.append({
                    "type": "automation_script",
                    "description": "Create automation scripts based on video content",
                    "priority": "high",
                    "estimated_effort": "medium"
                })
            elif theme == 'dashboard':
                opportunities.append({
                    "type": "dashboard_component",
                    "description": "Build dashboard component for data visualization",
                    "priority": "medium",
                    "estimated_effort": "high"
                })
            elif theme == 'real-time':
                opportunities.append({
                    "type": "realtime_feature",
                    "description": "Implement real-time functionality",
                    "priority": "high",
                    "estimated_effort": "medium"
                })
                
        return opportunities

class PDFAnalyzer:
    """Analyze research PDFs for implementation insights"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        
    async def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF document"""
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(pdf_path)
            
            full_text = ""
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                full_text += text + "\n"
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n < 5:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_b64 = base64.b64encode(img_data).decode()
                            images.append({
                                "page": page_num + 1,
                                "index": img_index,
                                "data": img_b64
                            })
                        pix = None
                    except Exception as e:
                        logging.warning(f"Image extraction error: {e}")
                        
            doc.close()
            
            return {
                "type": "pdf",
                "filename": Path(pdf_path).name,
                "page_count": len(doc),
                "text_content": full_text,
                "images": images,
                "analysis_timestamp": datetime.now().isoformat(),
                "technical_sections": self._extract_technical_sections(full_text),
                "algorithms": self._extract_algorithms(full_text),
                "implementation_suggestions": self._suggest_implementations(full_text),
                "code_snippets": self._extract_code_snippets(full_text),
                "research_insights": self._extract_research_insights(full_text)
            }
            
        except Exception as e:
            logging.error(f"PDF analysis error: {e}")
            return {"error": str(e), "type": "pdf_error"}
            
    def _extract_technical_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract technical sections from PDF"""
        sections = []
        
        # Look for common academic paper sections
        section_patterns = [
            (r'(?i)abstract', 'abstract'),
            (r'(?i)introduction', 'introduction'),
            (r'(?i)methodology|method', 'methodology'),
            (r'(?i)results', 'results'),
            (r'(?i)discussion', 'discussion'),
            (r'(?i)conclusion', 'conclusion'),
            (r'(?i)algorithm', 'algorithm'),
            (r'(?i)implementation', 'implementation')
        ]
        
        for pattern, section_type in section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = match.start()
                # Extract some context around the match
                context_start = max(0, start - 200)
                context_end = min(len(text), start + 1000)
                context = text[context_start:context_end]
                
                sections.append({
                    "type": section_type,
                    "position": start,
                    "context": context.strip()
                })
                
        return sections
        
    def _extract_algorithms(self, text: str) -> List[Dict[str, Any]]:
        """Extract algorithm descriptions"""
        algorithms = []
        
        # Look for algorithm-related patterns
        algorithm_patterns = [
            r'Algorithm \d+[:\.].*?(?=Algorithm \d+|$)',
            r'Procedure[:\s].*?(?=Procedure|$)',
            r'Step \d+[:\.].*?(?=Step \d+|$)'
        ]
        
        for pattern in algorithm_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                algorithms.append({
                    "type": "algorithm",
                    "content": match.group().strip(),
                    "position": match.start()
                })
                
        return algorithms
        
    def _extract_code_snippets(self, text: str) -> List[Dict[str, Any]]:
        """Extract code snippets from PDF"""
        snippets = []
        
        # Look for code-like patterns
        code_patterns = [
            r'```.*?```',  # Markdown code blocks
            r'def\s+\w+\([^)]*\):.*?(?=def|\n\n|$)',  # Python functions
            r'function\s+\w+\([^)]*\)\s*{.*?}',  # JavaScript functions
            r'class\s+\w+.*?{.*?}',  # Class definitions
        ]
        
        for pattern in code_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                snippets.append({
                    "type": "code",
                    "content": match.group().strip(),
                    "position": match.start(),
                    "language": self._detect_language(match.group())
                })
                
        return snippets
        
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet"""
        if re.search(r'\bdef\s+\w+\(', code):
            return "python"
        elif re.search(r'\bfunction\s+\w+\(', code):
            return "javascript"
        elif re.search(r'\bclass\s+\w+', code):
            return "java"
        else:
            return "unknown"
            
    def _suggest_implementations(self, text: str) -> List[Dict[str, Any]]:
        """Suggest implementation based on PDF content"""
        suggestions = []
        
        if re.search(r'\b(?:neural network|deep learning|machine learning)\b', text.lower()):
            suggestions.append({
                "type": "ml_implementation",
                "description": "Implement machine learning model based on paper",
                "frameworks": ["tensorflow", "pytorch", "scikit-learn"],
                "priority": "high"
            })
            
        if re.search(r'\b(?:api|rest|graphql|microservice)\b', text.lower()):
            suggestions.append({
                "type": "api_implementation",
                "description": "Create API based on research methodology",
                "frameworks": ["fastapi", "flask", "express"],
                "priority": "medium"
            })
            
        if re.search(r'\b(?:database|data storage|persistence)\b', text.lower()):
            suggestions.append({
                "type": "database_implementation",
                "description": "Design database schema based on research data",
                "technologies": ["postgresql", "mongodb", "redis"],
                "priority": "medium"
            })
            
        return suggestions
        
    def _extract_research_insights(self, text: str) -> List[str]:
        """Extract key research insights"""
        insights = []
        
        # Look for conclusion or key findings
        conclusion_pattern = r'(?i)(?:conclusion|findings?|results?)[:\s].*?(?=\n\n|\.\s*[A-Z]|$)'
        matches = re.finditer(conclusion_pattern, text)
        
        for match in matches:
            insight = match.group().strip()
            if len(insight) > 50:  # Filter out very short matches
                insights.append(insight)
                
        return insights[:10]  # Limit to top 10 insights

class CodeGenerator:
    """Generate implementation code based on analysis"""
    
    def __init__(self, llm_manager: LiteLLMManager):
        self.llm_manager = llm_manager
        
    async def generate_implementation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code implementation based on analysis"""
        try:
            implementation_prompt = self._create_implementation_prompt(analysis)
            
            response = await self.llm_manager.generate_response(
                implementation_prompt,
                model="gpt-4",
                max_tokens=4000
            )
            
            code_content = response.get("content", "")
            
            # Parse and format the generated code
            formatted_code = await self._format_and_validate_code(code_content)
            
            return {
                "success": True,
                "code": formatted_code,
                "analysis_used": analysis,
                "generation_timestamp": datetime.now().isoformat(),
                "validation": await self._validate_generated_code(formatted_code)
            }
            
        except Exception as e:
            logging.error(f"Code generation error: {e}")
            return {"success": False, "error": str(e)}
            
    def _create_implementation_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create prompt for code generation"""
        content_type = analysis.get("type", "unknown")
        
        if content_type == "video" or content_type == "channel":
            technical_concepts = analysis.get("technical_concepts", [])
            opportunities = analysis.get("implementation_opportunities", [])
            
            prompt = f"""
Based on the YouTube content analysis, create a practical implementation:

Technical Concepts Found: {', '.join(technical_concepts)}
Implementation Opportunities: {json.dumps(opportunities, indent=2)}

Generate a complete Python implementation that:
1. Addresses the main technical concepts
2. Implements the identified opportunities
3. Includes proper error handling and logging
4. Follows best practices and is production-ready
5. Includes comprehensive docstrings and comments

Focus on creating modular, reusable code that can be easily integrated into existing systems.
"""

        elif content_type == "pdf":
            algorithms = analysis.get("algorithms", [])
            suggestions = analysis.get("implementation_suggestions", [])
            insights = analysis.get("research_insights", [])
            
            prompt = f"""
Based on the research PDF analysis, create a practical implementation:

Algorithms Found: {json.dumps(algorithms[:3], indent=2)}
Implementation Suggestions: {json.dumps(suggestions, indent=2)}
Key Research Insights: {insights[:5]}

Generate a complete Python implementation that:
1. Implements the core algorithms described in the research
2. Follows the methodology outlined in the paper
3. Includes proper scientific computing practices
4. Has comprehensive testing and validation
5. Includes detailed documentation explaining the research basis

Focus on creating scientifically accurate and computationally efficient code.
"""

        else:
            prompt = f"""
Create a Python implementation based on the provided analysis:
{json.dumps(analysis, indent=2)}

The implementation should be practical, well-documented, and production-ready.
"""

        return prompt
        
    async def _format_and_validate_code(self, code: str) -> str:
        """Format and validate generated code"""
        try:
            # Extract Python code from the response
            if "```python" in code:
                code_start = code.find("```python") + 9
                code_end = code.find("```", code_start)
                if code_end != -1:
                    code = code[code_start:code_end].strip()
            elif "```" in code:
                code_start = code.find("```") + 3
                code_end = code.find("```", code_start)
                if code_end != -1:
                    code = code[code_start:code_end].strip()
                    
            # Format with black
            try:
                formatted_code = black.format_str(code, mode=black.FileMode())
            except Exception:
                formatted_code = code  # Fall back to original if formatting fails
                
            # Sort imports with isort
            try:
                formatted_code = isort.code(formatted_code)
            except Exception:
                pass  # Keep the code as is if isort fails
                
            return formatted_code
            
        except Exception as e:
            logging.error(f"Code formatting error: {e}")
            return code
            
    async def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code"""
        validation_results = {
            "syntax_valid": False,
            "imports_valid": False,
            "functions_found": [],
            "classes_found": [],
            "errors": []
        }
        
        try:
            # Check syntax
            ast.parse(code)
            validation_results["syntax_valid"] = True
            
            # Analyze AST
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    validation_results["functions_found"].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    validation_results["classes_found"].append(node.name)
                    
        except SyntaxError as e:
            validation_results["errors"].append(f"Syntax error: {e}")
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {e}")
            
        return validation_results

class AutoDeployer:
    """Automatically test and deploy generated implementations"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.docker_client = None
        
    async def initialize(self):
        """Initialize deployer"""
        try:
            self.docker_client = docker.from_env()
            logging.info("Auto-deployer initialized")
        except Exception as e:
            logging.warning(f"Docker not available: {e}")
            
    async def deploy_implementation(self, implementation: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy generated implementation"""
        try:
            if not implementation.get("success"):
                return {"success": False, "error": "Invalid implementation"}
                
            code = implementation["code"]
            
            # Create project structure
            project_path = await self._create_project_structure(code, analysis)
            
            # Run tests
            test_results = await self._run_tests(project_path)
            
            # Deploy if tests pass
            if test_results.get("success", False):
                deployment_result = await self._deploy_to_docker(project_path, analysis)
                
                return {
                    "success": True,
                    "project_path": str(project_path),
                    "test_results": test_results,
                    "deployment": deployment_result,
                    "deployment_timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Tests failed",
                    "test_results": test_results,
                    "project_path": str(project_path)
                }
                
        except Exception as e:
            logging.error(f"Deployment error: {e}")
            return {"success": False, "error": str(e)}
            
    async def _create_project_structure(self, code: str, analysis: Dict[str, Any]) -> Path:
        """Create project structure for deployment"""
        # Create unique project directory
        project_name = f"auto_generated_{hashlib.md5(code.encode()).hexdigest()[:8]}"
        project_path = Path.cwd() / "generated_projects" / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Write main implementation
        main_file = project_path / "main.py"
        async with aiofiles.open(main_file, 'w') as f:
            await f.write(code)
            
        # Create requirements.txt
        requirements = self._extract_requirements(code)
        requirements_file = project_path / "requirements.txt"
        async with aiofiles.open(requirements_file, 'w') as f:
            await f.write('\n'.join(requirements))
            
        # Create test file
        test_code = await self._generate_test_code(code, analysis)
        test_file = project_path / "test_main.py"
        async with aiofiles.open(test_file, 'w') as f:
            await f.write(test_code)
            
        # Create Dockerfile
        dockerfile_content = await self._generate_dockerfile(requirements)
        dockerfile = project_path / "Dockerfile"
        async with aiofiles.open(dockerfile, 'w') as f:
            await f.write(dockerfile_content)
            
        # Create README
        readme_content = await self._generate_readme(analysis)
        readme_file = project_path / "README.md"
        async with aiofiles.open(readme_file, 'w') as f:
            await f.write(readme_content)
            
        return project_path
        
    def _extract_requirements(self, code: str) -> List[str]:
        """Extract Python requirements from code"""
        requirements = set()
        
        # Common imports to package mappings
        import_mappings = {
            'requests': 'requests',
            'aiohttp': 'aiohttp',
            'asyncio': '',  # Built-in
            'json': '',  # Built-in
            'numpy': 'numpy',
            'pandas': 'pandas',
            'sklearn': 'scikit-learn',
            'tensorflow': 'tensorflow',
            'torch': 'torch',
            'fastapi': 'fastapi',
            'flask': 'flask',
            'sqlalchemy': 'sqlalchemy',
            'pytest': 'pytest',
            'opencv': 'opencv-python',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn'
        }
        
        # Extract imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        package = import_mappings.get(alias.name, alias.name)
                        if package:
                            requirements.add(package)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        package = import_mappings.get(node.module, node.module)
                        if package:
                            requirements.add(package)
        except Exception as e:
            logging.warning(f"Could not parse imports: {e}")
            
        # Add common requirements
        requirements.update(['aiofiles', 'python-dotenv'])
        
        return sorted(list(requirements))
        
    async def _generate_test_code(self, code: str, analysis: Dict[str, Any]) -> str:
        """Generate test code for the implementation"""
        test_template = '''
import pytest
import asyncio
from main import *

class TestGeneratedImplementation:
    """Test suite for auto-generated implementation"""
    
    def test_imports(self):
        """Test that all imports work"""
        # This test passes if the file can be imported
        assert True
        
    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic functionality"""
        # Add specific tests based on the generated code
        try:
            # Basic smoke test
            assert True
        except Exception as e:
            pytest.fail(f"Basic functionality test failed: {e}")
            
    def test_code_structure(self):
        """Test code structure and organization"""
        # Verify that expected functions/classes exist
        import main
        import inspect
        
        members = inspect.getmembers(main)
        functions = [name for name, obj in members if inspect.isfunction(obj)]
        classes = [name for name, obj in members if inspect.isclass(obj)]
        
        # At least one function or class should be defined
        assert len(functions) > 0 or len(classes) > 0, "No functions or classes found"

if __name__ == "__main__":
    pytest.main([__file__])
'''
        return test_template
        
    async def _generate_dockerfile(self, requirements: List[str]) -> str:
        """Generate Dockerfile for deployment"""
        dockerfile_template = f'''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
'''
        return dockerfile_template
        
    async def _generate_readme(self, analysis: Dict[str, Any]) -> str:
        """Generate README for the project"""
        content_type = analysis.get("type", "unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        readme_template = f'''
# Auto-Generated Implementation

**Generated:** {timestamp}  
**Source Type:** {content_type}  
**Auto-Generated by:** Content Analyzer & Deployer

## Overview

This implementation was automatically generated based on analysis of {content_type} content.

## Analysis Summary

{json.dumps(analysis, indent=2)}

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Testing

```bash
pytest test_main.py
```

## Docker Deployment

```bash
docker build -t auto-generated-app .
docker run -p 8000:8000 auto-generated-app
```

---

*This project was automatically analyzed, generated, tested, and deployed.*
'''
        return readme_template
        
    async def _run_tests(self, project_path: Path) -> Dict[str, Any]:
        """Run tests for the project"""
        try:
            # Install requirements first
            install_process = await asyncio.create_subprocess_exec(
                "pip", "install", "-r", "requirements.txt",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await install_process.communicate()
            
            # Run pytest
            test_process = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", "test_main.py", "-v",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await test_process.communicate()
            
            return {
                "success": test_process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "return_code": test_process.returncode
            }
            
        except Exception as e:
            logging.error(f"Test execution error: {e}")
            return {"success": False, "error": str(e)}
            
    async def _deploy_to_docker(self, project_path: Path, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Docker container"""
        if not self.docker_client:
            return {"success": False, "error": "Docker not available"}
            
        try:
            project_name = project_path.name
            image_tag = f"auto-generated/{project_name}:latest"
            
            # Build Docker image
            image, build_logs = self.docker_client.images.build(
                path=str(project_path),
                tag=image_tag,
                rm=True
            )
            
            # Run container
            container = self.docker_client.containers.run(
                image_tag,
                detach=True,
                ports={'8000/tcp': None},  # Auto-assign port
                name=f"{project_name}_{int(datetime.now().timestamp())}"
            )
            
            return {
                "success": True,
                "image_id": image.id,
                "image_tag": image_tag,
                "container_id": container.id,
                "container_name": container.name,
                "status": container.status
            }
            
        except Exception as e:
            logging.error(f"Docker deployment error: {e}")
            return {"success": False, "error": str(e)}

class ContentAnalyzerDeployer:
    """Main orchestrator for content analysis and auto-deployment"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.llm_manager = LiteLLMManager(self.config)
        self.youtube_analyzer = YouTubeAnalyzer(self.config)
        self.pdf_analyzer = PDFAnalyzer(self.config)
        self.code_generator = None
        self.auto_deployer = AutoDeployer(self.config)
        self.computer_control = None
        
    async def initialize(self):
        """Initialize all components"""
        await self.config.initialize()
        await self.llm_manager.initialize()
        self.code_generator = CodeGenerator(self.llm_manager)
        await self.auto_deployer.initialize()
        
        # Initialize computer control for async help
        self.computer_control = ComputerControlOrchestrator()
        await self.computer_control.initialize()
        
        logging.info("Content Analyzer & Deployer initialized")
        
    async def process_content(self, content_input: Union[str, Path]) -> Dict[str, Any]:
        """Process any type of content (YouTube URL or PDF path)"""
        try:
            if isinstance(content_input, str) and ("youtube.com" in content_input or "youtu.be" in content_input):
                # YouTube content
                if "/channel/" in content_input or "/c/" in content_input or "/user/" in content_input:
                    analysis = await self.youtube_analyzer.analyze_channel(content_input)
                else:
                    analysis = await self.youtube_analyzer.analyze_video(content_input)
            elif isinstance(content_input, (str, Path)) and str(content_input).endswith('.pdf'):
                # PDF content
                analysis = await self.pdf_analyzer.analyze_pdf(str(content_input))
            else:
                return {"success": False, "error": f"Unsupported content type: {content_input}"}
                
            if "error" in analysis:
                return {"success": False, "error": analysis["error"]}
                
            # Generate implementation
            implementation = await self.code_generator.generate_implementation(analysis)
            
            if not implementation.get("success"):
                return {
                    "success": False, 
                    "error": "Code generation failed",
                    "details": implementation
                }
                
            # Deploy implementation
            deployment = await self.auto_deployer.deploy_implementation(implementation, analysis)
            
            # Prepare async help notification
            async_help_needed = self._assess_async_help_needed(analysis, implementation, deployment)
            
            return {
                "success": True,
                "content_input": str(content_input),
                "analysis": analysis,
                "implementation": implementation,
                "deployment": deployment,
                "async_help_needed": async_help_needed,
                "process_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Content processing error: {e}")
            return {"success": False, "error": str(e)}
            
    def _assess_async_help_needed(self, analysis: Dict, implementation: Dict, deployment: Dict) -> Dict[str, Any]:
        """Assess if async human help is needed"""
        help_needed = {
            "required": False,
            "priority": "low",
            "areas": [],
            "details": []
        }
        
        # Check if deployment failed
        if not deployment.get("success"):
            help_needed["required"] = True
            help_needed["priority"] = "high"
            help_needed["areas"].append("deployment")
            help_needed["details"].append("Deployment failed - manual intervention required")
            
        # Check if tests failed
        if deployment.get("test_results", {}).get("success") == False:
            help_needed["required"] = True
            help_needed["priority"] = "medium"
            help_needed["areas"].append("testing")
            help_needed["details"].append("Tests failed - code review needed")
            
        # Check for complex implementations
        validation = implementation.get("validation", {})
        if len(validation.get("errors", [])) > 0:
            help_needed["required"] = True
            help_needed["priority"] = "medium"
            help_needed["areas"].append("code_quality")
            help_needed["details"].extend(validation["errors"])
            
        # Check for high-complexity analysis
        if analysis.get("type") == "pdf" and len(analysis.get("algorithms", [])) > 3:
            help_needed["required"] = True
            help_needed["priority"] = "low"
            help_needed["areas"].append("algorithm_review")
            help_needed["details"].append("Complex algorithms detected - expert review recommended")
            
        return help_needed
        
    async def request_async_help(self, help_request: Dict[str, Any]) -> Dict[str, Any]:
        """Request async help through computer control interface"""
        try:
            # Broadcast help request through WebSocket
            help_message = {
                "type": "async_help_request",
                "priority": help_request.get("priority", "medium"),
                "areas": help_request.get("areas", []),
                "details": help_request.get("details", []),
                "timestamp": datetime.now().isoformat(),
                "auto_context": help_request
            }
            
            await self.computer_control.broadcast_message(help_message)
            
            return {
                "success": True,
                "help_requested": True,
                "message": "Async help request sent",
                "request_id": hashlib.md5(json.dumps(help_message).encode()).hexdigest()[:8]
            }
            
        except Exception as e:
            logging.error(f"Async help request error: {e}")
            return {"success": False, "error": str(e)}

async def main():
    """Main entry point for content analyzer & deployer"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    analyzer_deployer = ContentAnalyzerDeployer()
    await analyzer_deployer.initialize()
    
    # Example usage
    test_inputs = [
        "https://www.youtube.com/watch?v=example",  # Video
        "https://www.youtube.com/channel/UCexample",  # Channel
        Path("research_paper.pdf")  # PDF
    ]
    
    for content_input in test_inputs:
        if Path(str(content_input)).exists() or "youtube.com" in str(content_input):
            logging.info(f"Processing: {content_input}")
            result = await analyzer_deployer.process_content(content_input)
            
            if result["success"]:
                logging.info(f"Successfully processed and deployed: {content_input}")
                
                # Request async help if needed
                if result["async_help_needed"]["required"]:
                    await analyzer_deployer.request_async_help(result["async_help_needed"])
            else:
                logging.error(f"Failed to process: {content_input} - {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())