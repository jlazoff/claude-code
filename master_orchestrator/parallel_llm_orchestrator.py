#!/usr/bin/env python3
"""
Parallel LLM Orchestrator with Google Vertex AI, OpenAI, and Grok
Self-generating code analysis and merging system with Agent Garden integration
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import hashlib
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.cloud.aiplatform as aiplatform
from google.cloud import aiplatform_v1
from google.oauth2 import service_account
import openai
import requests
from dataclasses import dataclass, asdict
import difflib
import ast
import black
import isort

from unified_config import SecureConfigManager
from litellm_manager import LiteLLMManager

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    provider: str
    model: str
    content: str
    tokens_used: int
    latency_ms: float
    confidence_score: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class CodeAnalysis:
    """Code analysis result"""
    file_path: str
    language: str
    complexity_score: float
    quality_score: float
    suggestions: List[str]
    optimizations: List[str]
    security_issues: List[str]
    performance_issues: List[str]
    generated_improvements: str

class VertexAIManager:
    """Google Cloud Vertex AI and Agent Garden integration"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.project_id = None
        self.location = "us-central1"
        self.client = None
        self.credentials = None
        
    async def initialize(self):
        """Initialize Vertex AI connection"""
        try:
            # Get credentials from config
            vertex_config = self.config.get_api_key('vertex_ai')
            if isinstance(vertex_config, dict):
                self.project_id = vertex_config.get('project_id')
                credentials_path = vertex_config.get('credentials_path')
                
                if credentials_path and Path(credentials_path).exists():
                    self.credentials = service_account.Credentials.from_service_account_file(
                        credentials_path
                    )
                else:
                    # Try environment variable
                    self.credentials = service_account.Credentials.from_service_account_info(
                        vertex_config
                    )
                    
            # Initialize AI Platform
            aiplatform.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials
            )
            
            self.client = aiplatform_v1.PredictionServiceClient(credentials=self.credentials)
            logging.info(f"Vertex AI initialized for project: {self.project_id}")
            
        except Exception as e:
            logging.error(f"Vertex AI initialization error: {e}")
            raise
            
    async def generate_code_with_agent_garden(self, prompt: str, agent_type: str = "code_generator") -> LLMResponse:
        """Generate code using Agent Garden agents"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Use Gemini Pro for code generation
            model = aiplatform.GenerativeModel("gemini-1.5-pro")
            
            # Enhanced prompt for Agent Garden integration
            enhanced_prompt = f"""
You are an expert {agent_type} agent from Google's Agent Garden.
Your task: {prompt}

Requirements:
1. Generate production-ready, optimized code
2. Include comprehensive error handling
3. Follow Python best practices and PEP 8
4. Add detailed docstrings and type hints
5. Consider performance and scalability
6. Include security best practices
7. Make code self-documenting and maintainable

Generate the complete implementation with proper structure and organization.
"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(enhanced_prompt)
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            # Extract content
            content = response.text if hasattr(response, 'text') else str(response)
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(content)
            
            return LLMResponse(
                provider="vertex_ai",
                model="gemini-1.5-pro",
                content=content,
                tokens_used=len(content.split()),  # Approximate
                latency_ms=latency_ms,
                confidence_score=confidence,
                metadata={
                    "agent_type": agent_type,
                    "project_id": self.project_id,
                    "location": self.location
                }
            )
            
        except Exception as e:
            logging.error(f"Vertex AI generation error: {e}")
            return LLMResponse(
                provider="vertex_ai",
                model="gemini-1.5-pro",
                content="",
                tokens_used=0,
                latency_ms=0,
                confidence_score=0.0,
                error=str(e)
            )
            
    async def analyze_code_with_agents(self, code: str, file_path: str) -> CodeAnalysis:
        """Analyze code using specialized Agent Garden agents"""
        try:
            analysis_prompt = f"""
As a code analysis agent from Agent Garden, perform comprehensive analysis of this code:

File: {file_path}
Code:
```
{code}
```

Provide detailed analysis covering:
1. Code complexity and maintainability
2. Quality score (0-100)
3. Specific improvement suggestions
4. Performance optimization opportunities
5. Security vulnerability assessment
6. Best practice compliance

Return structured analysis with actionable recommendations.
"""
            
            response = await self.generate_code_with_agent_garden(
                analysis_prompt, 
                "code_analyzer"
            )
            
            # Parse analysis (simplified - would need more sophisticated parsing)
            return CodeAnalysis(
                file_path=file_path,
                language=self._detect_language(file_path),
                complexity_score=self._extract_complexity_score(response.content),
                quality_score=self._extract_quality_score(response.content),
                suggestions=self._extract_suggestions(response.content),
                optimizations=self._extract_optimizations(response.content),
                security_issues=self._extract_security_issues(response.content),
                performance_issues=self._extract_performance_issues(response.content),
                generated_improvements=""
            )
            
        except Exception as e:
            logging.error(f"Code analysis error: {e}")
            return CodeAnalysis(
                file_path=file_path,
                language="unknown",
                complexity_score=0.0,
                quality_score=0.0,
                suggestions=[],
                optimizations=[],
                security_issues=[],
                performance_issues=[],
                generated_improvements=""
            )
            
    def _calculate_confidence(self, content: str) -> float:
        """Calculate confidence score based on response quality"""
        factors = []
        
        # Check for code blocks
        if "```" in content:
            factors.append(0.3)
            
        # Check for proper structure
        if any(keyword in content.lower() for keyword in ["class", "def", "import", "return"]):
            factors.append(0.3)
            
        # Check for documentation
        if any(keyword in content for keyword in ['"""', "'''", "#"]):
            factors.append(0.2)
            
        # Check length (reasonable response)
        if 100 < len(content) < 5000:
            factors.append(0.2)
            
        return min(sum(factors), 1.0)
        
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
        return mapping.get(ext, 'unknown')
        
    def _extract_complexity_score(self, content: str) -> float:
        """Extract complexity score from analysis"""
        # Simplified extraction - would need better parsing
        import re
        match = re.search(r'complexity[:\s]*(\d+(?:\.\d+)?)', content.lower())
        return float(match.group(1)) if match else 50.0
        
    def _extract_quality_score(self, content: str) -> float:
        """Extract quality score from analysis"""
        import re
        match = re.search(r'quality[:\s]*(\d+(?:\.\d+)?)', content.lower())
        return float(match.group(1)) if match else 75.0
        
    def _extract_suggestions(self, content: str) -> List[str]:
        """Extract improvement suggestions"""
        # Simplified extraction
        suggestions = []
        lines = content.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['suggest', 'recommend', 'improve', 'consider']):
                suggestions.append(line.strip())
        return suggestions[:10]  # Limit to top 10
        
    def _extract_optimizations(self, content: str) -> List[str]:
        """Extract optimization opportunities"""
        optimizations = []
        lines = content.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['optimize', 'performance', 'faster', 'efficient']):
                optimizations.append(line.strip())
        return optimizations[:10]
        
    def _extract_security_issues(self, content: str) -> List[str]:
        """Extract security issues"""
        issues = []
        lines = content.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['security', 'vulnerability', 'exploit', 'unsafe']):
                issues.append(line.strip())
        return issues[:10]
        
    def _extract_performance_issues(self, content: str) -> List[str]:
        """Extract performance issues"""
        issues = []
        lines = content.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['slow', 'bottleneck', 'memory', 'cpu', 'latency']):
                issues.append(line.strip())
        return issues[:10]

class OpenAIManager:
    """Enhanced OpenAI integration with advanced code capabilities"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.client = None
        
    async def initialize(self):
        """Initialize OpenAI client"""
        try:
            api_key = self.config.get_api_key('openai')
            self.client = openai.AsyncOpenAI(api_key=api_key)
            logging.info("OpenAI client initialized")
        except Exception as e:
            logging.error(f"OpenAI initialization error: {e}")
            raise
            
    async def generate_code_with_codex(self, prompt: str, model: str = "gpt-4-turbo") -> LLMResponse:
        """Generate code using OpenAI's advanced models"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            enhanced_prompt = f"""
You are an expert software engineer using OpenAI Codex capabilities.
Task: {prompt}

Requirements:
1. Write production-ready, well-documented code
2. Include comprehensive error handling and logging
3. Follow industry best practices and design patterns
4. Optimize for performance and maintainability
5. Include unit tests where appropriate
6. Use type hints and proper documentation
7. Consider edge cases and error scenarios

Provide complete, runnable code with clear explanations.
"""
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert software engineer with deep knowledge of best practices, design patterns, and optimization techniques."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                provider="openai",
                model=model,
                content=content,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                confidence_score=0.9,  # High confidence for GPT-4
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": asdict(response.usage)
                }
            )
            
        except Exception as e:
            logging.error(f"OpenAI generation error: {e}")
            return LLMResponse(
                provider="openai",
                model=model,
                content="",
                tokens_used=0,
                latency_ms=0,
                confidence_score=0.0,
                error=str(e)
            )

class GrokManager:
    """Grok (X.AI) integration for additional perspectives"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.api_key = None
        self.base_url = "https://api.x.ai/v1"
        
    async def initialize(self):
        """Initialize Grok client"""
        try:
            self.api_key = self.config.get_api_key('grok')
            logging.info("Grok client initialized")
        except Exception as e:
            logging.error(f"Grok initialization error: {e}")
            raise
            
    async def generate_code_with_grok(self, prompt: str, model: str = "grok-beta") -> LLMResponse:
        """Generate code using Grok's capabilities"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            enhanced_prompt = f"""
You are Grok, an AI with wit and technical expertise.
Task: {prompt}

Provide innovative, efficient code solutions with:
1. Creative approaches to common problems
2. Performance-optimized implementations
3. Robust error handling
4. Clear documentation and comments
5. Consideration of edge cases
6. Modern Python practices and patterns

Be thorough and practical in your implementation.
"""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are Grok, a witty and technically excellent AI assistant focused on delivering high-quality code solutions."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                "model": model,
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 4000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status == 200:
                content = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens", 0)
                
                return LLMResponse(
                    provider="grok",
                    model=model,
                    content=content,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    confidence_score=0.85,  # High confidence for Grok
                    metadata={
                        "usage": result.get("usage", {}),
                        "finish_reason": result["choices"][0].get("finish_reason")
                    }
                )
            else:
                raise Exception(f"Grok API error: {result}")
                
        except Exception as e:
            logging.error(f"Grok generation error: {e}")
            return LLMResponse(
                provider="grok",
                model=model,
                content="",
                tokens_used=0,
                latency_ms=0,
                confidence_score=0.0,
                error=str(e)
            )

class CodeMerger:
    """Intelligent code merging and optimization"""
    
    def __init__(self):
        self.merger_strategies = {
            "best_practices": self._merge_best_practices,
            "performance": self._merge_performance_optimized,
            "comprehensive": self._merge_comprehensive,
            "consensus": self._merge_by_consensus
        }
        
    async def merge_code_responses(self, responses: List[LLMResponse], strategy: str = "comprehensive") -> Dict[str, Any]:
        """Merge multiple LLM code responses into optimal solution"""
        try:
            if not responses or all(r.error for r in responses):
                return {"success": False, "error": "No valid responses to merge"}
                
            # Filter out error responses
            valid_responses = [r for r in responses if not r.error and r.content]
            
            if not valid_responses:
                return {"success": False, "error": "No valid code responses"}
                
            # Extract code blocks from responses
            code_blocks = []
            for response in valid_responses:
                blocks = self._extract_code_blocks(response.content)
                code_blocks.extend([
                    {
                        "provider": response.provider,
                        "model": response.model,
                        "code": block,
                        "confidence": response.confidence_score,
                        "latency": response.latency_ms
                    }
                    for block in blocks
                ])
                
            if not code_blocks:
                return {"success": False, "error": "No code blocks found in responses"}
                
            # Apply merging strategy
            merger_func = self.merger_strategies.get(strategy, self._merge_comprehensive)
            merged_result = await merger_func(code_blocks, valid_responses)
            
            # Format and validate merged code
            formatted_code = await self._format_and_validate(merged_result["code"])
            
            return {
                "success": True,
                "merged_code": formatted_code,
                "strategy_used": strategy,
                "source_providers": [r.provider for r in valid_responses],
                "confidence_scores": [r.confidence_score for r in valid_responses],
                "merge_metadata": merged_result.get("metadata", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Code merging error: {e}")
            return {"success": False, "error": str(e)}
            
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from response content"""
        import re
        
        # Find code blocks marked with ```
        pattern = r'```(?:python|py)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            return [match.strip() for match in matches]
            
        # If no code blocks found, try to extract code-like content
        lines = content.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ']):
                in_code = True
            if in_code:
                code_lines.append(line)
                
        if code_lines:
            return ['\n'.join(code_lines)]
            
        return [content]  # Return full content as fallback
        
    async def _merge_best_practices(self, code_blocks: List[Dict], responses: List[LLMResponse]) -> Dict[str, Any]:
        """Merge based on best practices adherence"""
        # Score each code block for best practices
        scored_blocks = []
        
        for block in code_blocks:
            score = self._score_best_practices(block["code"])
            block["best_practices_score"] = score
            scored_blocks.append(block)
            
        # Select highest scoring block as base
        best_block = max(scored_blocks, key=lambda x: x["best_practices_score"])
        
        # Enhance with elements from other blocks
        enhanced_code = await self._enhance_with_best_elements(best_block["code"], scored_blocks)
        
        return {
            "code": enhanced_code,
            "metadata": {
                "base_provider": best_block["provider"],
                "best_practices_score": best_block["best_practices_score"],
                "enhancement_sources": [b["provider"] for b in scored_blocks if b != best_block]
            }
        }
        
    async def _merge_performance_optimized(self, code_blocks: List[Dict], responses: List[LLMResponse]) -> Dict[str, Any]:
        """Merge with focus on performance optimization"""
        # Analyze performance characteristics
        performance_scores = []
        
        for block in code_blocks:
            score = self._score_performance(block["code"])
            block["performance_score"] = score
            performance_scores.append(score)
            
        # Select most performant as base
        best_performance = max(code_blocks, key=lambda x: x["performance_score"])
        
        return {
            "code": best_performance["code"],
            "metadata": {
                "base_provider": best_performance["provider"],
                "performance_score": best_performance["performance_score"],
                "optimization_focus": "performance"
            }
        }
        
    async def _merge_comprehensive(self, code_blocks: List[Dict], responses: List[LLMResponse]) -> Dict[str, Any]:
        """Comprehensive merge considering all factors"""
        # Score each block on multiple dimensions
        for block in code_blocks:
            block["scores"] = {
                "best_practices": self._score_best_practices(block["code"]),
                "performance": self._score_performance(block["code"]),
                "readability": self._score_readability(block["code"]),
                "completeness": self._score_completeness(block["code"]),
                "security": self._score_security(block["code"])
            }
            
            # Calculate weighted composite score
            weights = {
                "best_practices": 0.25,
                "performance": 0.20,
                "readability": 0.20,
                "completeness": 0.20,
                "security": 0.15
            }
            
            block["composite_score"] = sum(
                block["scores"][dimension] * weight
                for dimension, weight in weights.items()
            )
            
        # Select best overall block
        best_block = max(code_blocks, key=lambda x: x["composite_score"])
        
        # Enhance with best elements from other blocks
        enhanced_code = await self._comprehensive_enhancement(best_block["code"], code_blocks)
        
        return {
            "code": enhanced_code,
            "metadata": {
                "base_provider": best_block["provider"],
                "composite_score": best_block["composite_score"],
                "dimension_scores": best_block["scores"],
                "enhancement_applied": True
            }
        }
        
    async def _merge_by_consensus(self, code_blocks: List[Dict], responses: List[LLMResponse]) -> Dict[str, Any]:
        """Merge by finding consensus among responses"""
        # Find common patterns and structures
        common_elements = self._find_common_elements(code_blocks)
        
        # Build consensus-based implementation
        consensus_code = await self._build_consensus_code(code_blocks, common_elements)
        
        return {
            "code": consensus_code,
            "metadata": {
                "strategy": "consensus",
                "common_elements": len(common_elements),
                "participating_providers": list(set(b["provider"] for b in code_blocks))
            }
        }
        
    def _score_best_practices(self, code: str) -> float:
        """Score code for best practices adherence"""
        score = 0.0
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 20
            
        # Check for type hints
        if '->' in code or ': ' in code:
            score += 15
            
        # Check for proper imports
        if code.strip().startswith(('import ', 'from ')):
            score += 10
            
        # Check for error handling
        if 'try:' in code and 'except' in code:
            score += 20
            
        # Check for logging
        if 'logging' in code or 'logger' in code:
            score += 10
            
        # Check for classes and functions
        if 'class ' in code:
            score += 10
        if 'def ' in code:
            score += 10
            
        # Check for comments
        if '#' in code:
            score += 5
            
        return min(score, 100.0)
        
    def _score_performance(self, code: str) -> float:
        """Score code for performance characteristics"""
        score = 0.0
        
        # Check for async usage
        if 'async ' in code or 'await ' in code:
            score += 25
            
        # Check for efficient data structures
        if any(ds in code for ds in ['set(', 'dict(', 'defaultdict', 'deque']):
            score += 20
            
        # Check for list comprehensions
        if '[' in code and 'for ' in code and ' in ' in code:
            score += 15
            
        # Check for generators
        if 'yield' in code:
            score += 15
            
        # Check for caching
        if any(cache in code for cache in ['cache', 'lru_cache', 'memoize']):
            score += 15
            
        # Penalize for inefficient patterns
        if 'for ' in code and '.append(' in code:
            score -= 5  # Could use list comprehension
            
        return max(min(score, 100.0), 0.0)
        
    def _score_readability(self, code: str) -> float:
        """Score code for readability"""
        lines = code.split('\n')
        score = 0.0
        
        # Check line length
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        if avg_line_length < 80:
            score += 20
            
        # Check for meaningful variable names
        if any(len(word) > 3 for word in code.split() if word.isidentifier()):
            score += 20
            
        # Check for proper spacing
        if ' = ' in code and ' == ' in code:
            score += 15
            
        # Check for docstrings and comments
        comment_ratio = sum(1 for line in lines if line.strip().startswith('#')) / len(lines) if lines else 0
        score += comment_ratio * 30
        
        return min(score, 100.0)
        
    def _score_completeness(self, code: str) -> float:
        """Score code for completeness"""
        score = 0.0
        
        # Check for main function or class
        if 'def main(' in code or 'class ' in code:
            score += 30
            
        # Check for imports
        if any(imp in code for imp in ['import ', 'from ']):
            score += 20
            
        # Check for error handling
        if 'try:' in code:
            score += 20
            
        # Check for return statements
        if 'return ' in code:
            score += 15
            
        # Check for if __name__ == "__main__"
        if '__name__' in code and '__main__' in code:
            score += 15
            
        return min(score, 100.0)
        
    def _score_security(self, code: str) -> float:
        """Score code for security practices"""
        score = 100.0  # Start with perfect score
        
        # Penalize for security anti-patterns
        security_issues = [
            'eval(',
            'exec(',
            'os.system(',
            'subprocess.call(',
            'shell=True',
            'input(' # Raw input without validation
        ]
        
        for issue in security_issues:
            if issue in code:
                score -= 20
                
        # Reward for security best practices
        if any(practice in code for practice in ['validate', 'sanitize', 'escape']):
            score += 10
            
        return max(min(score, 100.0), 0.0)
        
    async def _enhance_with_best_elements(self, base_code: str, all_blocks: List[Dict]) -> str:
        """Enhance base code with best elements from other blocks"""
        enhanced = base_code
        
        # Add missing imports
        all_imports = set()
        for block in all_blocks:
            imports = self._extract_imports(block["code"])
            all_imports.update(imports)
            
        base_imports = set(self._extract_imports(base_code))
        missing_imports = all_imports - base_imports
        
        if missing_imports:
            import_lines = '\n'.join(missing_imports)
            enhanced = f"{import_lines}\n\n{enhanced}"
            
        return enhanced
        
    async def _comprehensive_enhancement(self, base_code: str, all_blocks: List[Dict]) -> str:
        """Apply comprehensive enhancements from all blocks"""
        enhanced = base_code
        
        # Collect all best practices from all blocks
        best_practices = []
        for block in all_blocks:
            if block["scores"]["best_practices"] > 80:
                best_practices.append(block["code"])
                
        # Apply enhancements (simplified implementation)
        enhanced = await self._merge_imports(enhanced, all_blocks)
        enhanced = await self._merge_error_handling(enhanced, all_blocks)
        enhanced = await self._merge_documentation(enhanced, all_blocks)
        
        return enhanced
        
    async def _build_consensus_code(self, code_blocks: List[Dict], common_elements: List[str]) -> str:
        """Build code based on consensus among all responses"""
        # Start with most common structure
        base_structure = max(code_blocks, key=lambda x: x["confidence"])["code"]
        
        # Integrate common elements
        for element in common_elements:
            if element not in base_structure:
                base_structure = f"{base_structure}\n{element}"
                
        return base_structure
        
    def _find_common_elements(self, code_blocks: List[Dict]) -> List[str]:
        """Find common code elements across all blocks"""
        all_functions = []
        all_imports = []
        
        for block in code_blocks:
            code = block["code"]
            all_functions.extend(self._extract_functions(code))
            all_imports.extend(self._extract_imports(code))
            
        # Find elements that appear in multiple blocks
        from collections import Counter
        
        function_counts = Counter(all_functions)
        import_counts = Counter(all_imports)
        
        common_functions = [func for func, count in function_counts.items() if count > 1]
        common_imports = [imp for imp, count in import_counts.items() if count > 1]
        
        return common_functions + common_imports
        
    def _extract_functions(self, code: str) -> List[str]:
        """Extract function definitions from code"""
        import re
        pattern = r'def\s+(\w+)\s*\([^)]*\):'
        return re.findall(pattern, code)
        
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code"""
        lines = code.split('\n')
        imports = []
        for line in lines:
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                imports.append(line)
        return imports
        
    async def _merge_imports(self, base_code: str, all_blocks: List[Dict]) -> str:
        """Merge import statements from all blocks"""
        all_imports = set()
        for block in all_blocks:
            imports = self._extract_imports(block["code"])
            all_imports.update(imports)
            
        # Add missing imports to base code
        base_imports = set(self._extract_imports(base_code))
        missing_imports = all_imports - base_imports
        
        if missing_imports:
            import_section = '\n'.join(sorted(missing_imports))
            # Insert at the beginning
            lines = base_code.split('\n')
            # Find where to insert (after existing imports or at top)
            insert_pos = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith(('import ', 'from ', '#', '"""', "'''")):
                    insert_pos = i
                    break
                    
            lines.insert(insert_pos, import_section)
            return '\n'.join(lines)
            
        return base_code
        
    async def _merge_error_handling(self, base_code: str, all_blocks: List[Dict]) -> str:
        """Enhance error handling based on all blocks"""
        # Find best error handling patterns
        best_error_handling = ""
        best_score = 0
        
        for block in all_blocks:
            if 'try:' in block["code"] and 'except' in block["code"]:
                score = block["code"].count('except')
                if score > best_score:
                    best_score = score
                    # Extract error handling pattern (simplified)
                    best_error_handling = block["code"]
                    
        # Apply error handling improvements (simplified)
        if best_error_handling and 'try:' not in base_code:
            # Wrap main logic in try-except (very simplified)
            base_code = f"try:\n    {base_code.replace(chr(10), chr(10) + '    ')}\nexcept Exception as e:\n    logging.error(f'Error: {{e}}')\n    raise"
            
        return base_code
        
    async def _merge_documentation(self, base_code: str, all_blocks: List[Dict]) -> str:
        """Enhance documentation based on all blocks"""
        # Find best documented version
        best_docs = ""
        best_doc_score = 0
        
        for block in all_blocks:
            doc_score = block["code"].count('"""') + block["code"].count("'''") + block["code"].count('#')
            if doc_score > best_doc_score:
                best_doc_score = doc_score
                best_docs = block["code"]
                
        # Apply documentation improvements (simplified)
        if best_docs and ('"""' in best_docs or "'''" in best_docs):
            # Extract and apply docstrings (very simplified)
            import re
            docstring_pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'
            docstrings = re.findall(docstring_pattern, best_docs, re.DOTALL)
            
            if docstrings and '"""' not in base_code and "'''" not in base_code:
                # Add main docstring at the beginning
                base_code = f'"""{docstrings[0].strip('"""').strip("'''").strip()}"""\n\n{base_code}'
                
        return base_code
        
    async def _format_and_validate(self, code: str) -> str:
        """Format and validate the merged code"""
        try:
            # Format with black
            formatted = black.format_str(code, mode=black.FileMode())
            
            # Sort imports with isort
            formatted = isort.code(formatted)
            
            # Validate syntax
            ast.parse(formatted)
            
            return formatted
            
        except Exception as e:
            logging.warning(f"Code formatting/validation error: {e}")
            return code  # Return original if formatting fails

class ParallelLLMOrchestrator:
    """Main orchestrator for parallel LLM code generation and analysis"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.vertex_ai = VertexAIManager(self.config)
        self.openai_manager = OpenAIManager(self.config)
        self.grok_manager = GrokManager(self.config)
        self.code_merger = CodeMerger()
        self.execution_history = []
        
    async def initialize(self):
        """Initialize all LLM providers"""
        await self.config.initialize()
        
        # Initialize providers in parallel
        init_tasks = [
            self.vertex_ai.initialize(),
            self.openai_manager.initialize(),
            self.grok_manager.initialize()
        ]
        
        # Execute with error handling
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Log initialization results
        providers = ["Vertex AI", "OpenAI", "Grok"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.warning(f"{providers[i]} initialization failed: {result}")
            else:
                logging.info(f"{providers[i]} initialized successfully")
                
        logging.info("Parallel LLM Orchestrator initialized")
        
    async def generate_code_parallel(self, prompt: str, merge_strategy: str = "comprehensive") -> Dict[str, Any]:
        """Generate code using all providers in parallel and merge results"""
        try:
            # Start parallel code generation
            tasks = [
                self.vertex_ai.generate_code_with_agent_garden(prompt, "code_generator"),
                self.openai_manager.generate_code_with_codex(prompt),
                self.grok_manager.generate_code_with_grok(prompt)
            ]
            
            # Execute in parallel with timeout
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0  # 2 minute timeout
            )
            
            # Filter valid responses
            valid_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logging.error(f"Provider {i} failed: {response}")
                elif isinstance(response, LLMResponse) and not response.error:
                    valid_responses.append(response)
                    
            if not valid_responses:
                return {
                    "success": False,
                    "error": "All providers failed to generate code",
                    "provider_errors": [str(r) if isinstance(r, Exception) else r.error for r in responses]
                }
                
            # Merge responses
            merge_result = await self.code_merger.merge_code_responses(valid_responses, merge_strategy)
            
            # Record execution history
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "providers_used": [r.provider for r in valid_responses],
                "merge_strategy": merge_strategy,
                "success": merge_result["success"],
                "total_tokens": sum(r.tokens_used for r in valid_responses),
                "avg_latency": sum(r.latency_ms for r in valid_responses) / len(valid_responses)
            }
            self.execution_history.append(execution_record)
            
            # Enhance result with execution info
            if merge_result["success"]:
                merge_result.update({
                    "execution_info": execution_record,
                    "provider_responses": [
                        {
                            "provider": r.provider,
                            "model": r.model,
                            "confidence": r.confidence_score,
                            "tokens": r.tokens_used,
                            "latency_ms": r.latency_ms
                        }
                        for r in valid_responses
                    ]
                })
                
            return merge_result
            
        except asyncio.TimeoutError:
            logging.error("Code generation timed out")
            return {"success": False, "error": "Generation timed out after 2 minutes"}
        except Exception as e:
            logging.error(f"Parallel code generation error: {e}")
            return {"success": False, "error": str(e)}
            
    async def analyze_code_parallel(self, code: str, file_path: str) -> Dict[str, Any]:
        """Analyze code using all providers in parallel"""
        try:
            # Start parallel analysis
            tasks = [
                self.vertex_ai.analyze_code_with_agents(code, file_path),
                self._analyze_with_openai(code, file_path),
                self._analyze_with_grok(code, file_path)
            ]
            
            # Execute in parallel
            analyses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine analysis results
            combined_analysis = await self._combine_analyses(analyses, file_path)
            
            return combined_analysis
            
        except Exception as e:
            logging.error(f"Parallel code analysis error: {e}")
            return {"success": False, "error": str(e)}
            
    async def _analyze_with_openai(self, code: str, file_path: str) -> CodeAnalysis:
        """Analyze code using OpenAI"""
        prompt = f"""
Analyze this code for quality, performance, and best practices:

File: {file_path}
Code:
```
{code}
```

Provide detailed analysis including:
1. Code quality score (0-100)
2. Complexity assessment
3. Performance bottlenecks
4. Security vulnerabilities
5. Specific improvement suggestions
6. Optimization opportunities
"""
        
        response = await self.openai_manager.generate_code_with_codex(prompt)
        
        # Parse response into CodeAnalysis (simplified)
        return CodeAnalysis(
            file_path=file_path,
            language=self.vertex_ai._detect_language(file_path),
            complexity_score=self.vertex_ai._extract_complexity_score(response.content),
            quality_score=self.vertex_ai._extract_quality_score(response.content),
            suggestions=self.vertex_ai._extract_suggestions(response.content),
            optimizations=self.vertex_ai._extract_optimizations(response.content),
            security_issues=self.vertex_ai._extract_security_issues(response.content),
            performance_issues=self.vertex_ai._extract_performance_issues(response.content),
            generated_improvements=""
        )
        
    async def _analyze_with_grok(self, code: str, file_path: str) -> CodeAnalysis:
        """Analyze code using Grok"""
        prompt = f"""
Analyze this code with your witty and thorough perspective:

File: {file_path}
Code:
```
{code}
```

Provide frank analysis covering:
1. What's good and what's not
2. Performance issues and fixes
3. Security concerns
4. Readability and maintainability
5. Clever optimizations you'd suggest
"""
        
        response = await self.grok_manager.generate_code_with_grok(prompt)
        
        # Parse response into CodeAnalysis (simplified)
        return CodeAnalysis(
            file_path=file_path,
            language=self.vertex_ai._detect_language(file_path),
            complexity_score=self.vertex_ai._extract_complexity_score(response.content),
            quality_score=self.vertex_ai._extract_quality_score(response.content),
            suggestions=self.vertex_ai._extract_suggestions(response.content),
            optimizations=self.vertex_ai._extract_optimizations(response.content),
            security_issues=self.vertex_ai._extract_security_issues(response.content),
            performance_issues=self.vertex_ai._extract_performance_issues(response.content),
            generated_improvements=""
        )
        
    async def _combine_analyses(self, analyses: List[Union[CodeAnalysis, Exception]], file_path: str) -> Dict[str, Any]:
        """Combine multiple code analyses into comprehensive result"""
        valid_analyses = [a for a in analyses if isinstance(a, CodeAnalysis)]
        
        if not valid_analyses:
            return {"success": False, "error": "No valid analyses produced"}
            
        # Aggregate scores and suggestions
        avg_complexity = sum(a.complexity_score for a in valid_analyses) / len(valid_analyses)
        avg_quality = sum(a.quality_score for a in valid_analyses) / len(valid_analyses)
        
        all_suggestions = []
        all_optimizations = []
        all_security_issues = []
        all_performance_issues = []
        
        for analysis in valid_analyses:
            all_suggestions.extend(analysis.suggestions)
            all_optimizations.extend(analysis.optimizations)
            all_security_issues.extend(analysis.security_issues)
            all_performance_issues.extend(analysis.performance_issues)
            
        # Remove duplicates while preserving order
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        unique_optimizations = list(dict.fromkeys(all_optimizations))
        unique_security_issues = list(dict.fromkeys(all_security_issues))
        unique_performance_issues = list(dict.fromkeys(all_performance_issues))
        
        return {
            "success": True,
            "file_path": file_path,
            "language": valid_analyses[0].language,
            "average_complexity_score": avg_complexity,
            "average_quality_score": avg_quality,
            "combined_suggestions": unique_suggestions,
            "combined_optimizations": unique_optimizations,
            "combined_security_issues": unique_security_issues,
            "combined_performance_issues": unique_performance_issues,
            "provider_count": len(valid_analyses),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    async def optimize_existing_code(self, file_path: str) -> Dict[str, Any]:
        """Optimize existing code file using parallel LLM analysis"""
        try:
            # Read existing code
            async with aiofiles.open(file_path, 'r') as f:
                original_code = await f.read()
                
            # Analyze current code
            analysis = await self.analyze_code_parallel(original_code, file_path)
            
            if not analysis["success"]:
                return analysis
                
            # Generate optimized version
            optimization_prompt = f"""
Based on this analysis, optimize the code in {file_path}:

Current Issues:
- Complexity Score: {analysis['average_complexity_score']}/100
- Quality Score: {analysis['average_quality_score']}/100
- Suggestions: {analysis['combined_suggestions'][:5]}
- Performance Issues: {analysis['combined_performance_issues'][:3]}
- Security Issues: {analysis['combined_security_issues'][:3]}

Original Code:
```python
{original_code}
```

Generate an optimized version that addresses all identified issues while maintaining functionality.
"""
            
            optimization_result = await self.generate_code_parallel(
                optimization_prompt, 
                merge_strategy="performance"
            )
            
            if optimization_result["success"]:
                # Write optimized code to a new file
                optimized_path = f"{file_path}.optimized"
                async with aiofiles.open(optimized_path, 'w') as f:
                    await f.write(optimization_result["merged_code"])
                    
                return {
                    "success": True,
                    "original_file": file_path,
                    "optimized_file": optimized_path,
                    "analysis": analysis,
                    "optimization": optimization_result,
                    "improvements": {
                        "complexity_addressed": len(analysis['combined_suggestions']),
                        "performance_fixes": len(analysis['combined_performance_issues']),
                        "security_fixes": len(analysis['combined_security_issues'])
                    }
                }
            else:
                return optimization_result
                
        except Exception as e:
            logging.error(f"Code optimization error: {e}")
            return {"success": False, "error": str(e)}
            
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about LLM usage and performance"""
        if not self.execution_history:
            return {"total_executions": 0}
            
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h["success"])
        
        # Provider usage stats
        provider_usage = {}
        for history in self.execution_history:
            for provider in history["providers_used"]:
                provider_usage[provider] = provider_usage.get(provider, 0) + 1
                
        # Performance stats
        total_tokens = sum(h.get("total_tokens", 0) for h in self.execution_history)
        avg_latency = sum(h.get("avg_latency", 0) for h in self.execution_history) / total_executions
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions * 100,
            "provider_usage": provider_usage,
            "total_tokens_used": total_tokens,
            "average_latency_ms": avg_latency,
            "recent_executions": self.execution_history[-10:]  # Last 10
        }

async def main():
    """Main entry point for parallel LLM orchestrator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = ParallelLLMOrchestrator()
    await orchestrator.initialize()
    
    # Example usage
    test_prompt = """
Create a FastAPI application with the following features:
1. Real-time WebSocket chat
2. JWT authentication
3. SQLAlchemy database integration
4. Comprehensive error handling
5. API documentation with Swagger
6. Health check endpoints
7. Rate limiting
8. Caching with Redis
9. Background task processing
10. Full test coverage
"""
    
    logging.info("Generating code with all providers in parallel...")
    result = await orchestrator.generate_code_parallel(test_prompt)
    
    if result["success"]:
        logging.info("Code generation successful!")
        print(f"Generated {len(result['merged_code'])} characters of code")
        print(f"Used providers: {result['source_providers']}")
        print(f"Merge strategy: {result['strategy_used']}")
        
        # Save generated code
        output_file = Path("generated_fastapi_app.py")
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(result["merged_code"])
        logging.info(f"Code saved to {output_file}")
        
    else:
        logging.error(f"Code generation failed: {result.get('error')}")
        
    # Display statistics
    stats = orchestrator.get_execution_statistics()
    logging.info(f"Execution statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())