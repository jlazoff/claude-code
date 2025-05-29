#!/usr/bin/env python3
"""
YouTube GitHub Extractor
Extracts GitHub repositories from YouTube channels and integrates them into the system
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from youtube_cli_integration import YouTubeCLIIntegration
from github_discovery_orchestrator import GitHubDiscoveryOrchestrator
from local_agentic_framework import LocalAgenticFramework, KnowledgeGraphNode, DataLakeRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeGitHubExtractor:
    """Extracts GitHub repositories from YouTube channel videos"""
    
    def __init__(self, framework: LocalAgenticFramework, github_orchestrator: GitHubDiscoveryOrchestrator):
        self.framework = framework
        self.github_orchestrator = github_orchestrator
        self.youtube_cli = YouTubeCLIIntegration()
        
        self.foundation_dir = Path("foundation_data")
        self.youtube_repos_dir = self.foundation_dir / "youtube_discovered_repos"
        self.analysis_dir = self.foundation_dir / "youtube_github_analysis"
        
        # Create directories
        for dir_path in [self.youtube_repos_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # GitHub patterns for extraction
        self.github_patterns = [
            r'github\.com/([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)',
            r'https?://github\.com/([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)',
            r'git clone.*github\.com[:/]([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)',
            r'@([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_\.]+)',  # Handle @user/repo format
            r'(?:repo|repository|code).*?([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)',
        ]
        
        # AI/ML specific repository patterns
        self.ai_repo_patterns = [
            r'(langchain|llamaindex|transformers|openai|anthropic)',
            r'(agent|autonomous|multi-agent|agentic)',
            r'(llm|gpt|claude|gemini|mistral)',
            r'(machine.learning|deep.learning|neural|ai)',
            r'(workflow|automation|orchestration)'
        ]
        
        logger.info("YouTube GitHub Extractor initialized")

    async def analyze_channel_for_repos(self, channel_url: str) -> Dict[str, Any]:
        """Analyze YouTube channel and extract all GitHub repositories"""
        logger.info(f"🔍 Analyzing channel for GitHub repos: {channel_url}")
        
        start_time = time.time()
        
        # Step 1: Extract channel videos
        logger.info("📺 Extracting channel videos...")
        channel_analysis = await self.youtube_cli.analyze_channel(channel_url)
        
        if "error" in channel_analysis:
            return {"error": channel_analysis["error"]}
        
        videos = channel_analysis.get("videos", [])
        logger.info(f"✅ Found {len(videos)} videos to analyze")
        
        # Step 2: Extract GitHub repos from each video
        all_repos = set()
        video_repo_mapping = {}
        
        for i, video in enumerate(videos):
            logger.info(f"🔎 Processing video {i+1}/{len(videos)}: {video.get('title', 'Unknown')[:50]}...")
            
            repos = await self.extract_repos_from_video(video)
            all_repos.update(repos)
            
            if repos:
                video_repo_mapping[video['video_id']] = {
                    'title': video['title'],
                    'repos': list(repos),
                    'url': f"https://www.youtube.com/watch?v={video['video_id']}"
                }
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Step 3: Validate and analyze discovered repositories
        logger.info(f"🔍 Validating {len(all_repos)} discovered repositories...")
        validated_repos = []
        
        for repo_path in all_repos:
            repo_info = await self.validate_and_analyze_repo(repo_path)
            if repo_info:
                validated_repos.append(repo_info)
        
        # Step 4: Store results
        analysis_results = {
            "channel_info": channel_analysis.get("channel_info", {}),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_videos_analyzed": len(videos),
            "total_repos_discovered": len(all_repos),
            "validated_repos": len(validated_repos),
            "execution_time": time.time() - start_time,
            "video_repo_mapping": video_repo_mapping,
            "discovered_repositories": validated_repos
        }
        
        # Save analysis
        analysis_file = self.analysis_dir / f"youtube_github_analysis_{int(time.time())}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Store in knowledge graph
        await self.store_analysis_in_knowledge_graph(analysis_results)
        
        logger.info(f"✅ Channel analysis complete: {len(validated_repos)} repos validated in {analysis_results['execution_time']:.2f}s")
        
        return analysis_results

    async def extract_repos_from_video(self, video: Dict[str, Any]) -> List[str]:
        """Extract GitHub repository references from a single video using yt-dlp"""
        repos = set()
        
        # Get video details using yt-dlp command line
        video_details = await self.get_video_details_with_ytdlp(video['video_id'])
        
        # Combine all text sources
        text_sources = [
            video_details.get('title', ''),
            video_details.get('description', ''),
            video_details.get('transcript', ''),
            video.get('title', ''),
            video.get('description', ''),
            video.get('transcript', '')
        ]
        
        full_text = ' '.join(text_sources)
        
        # Extract using GitHub patterns
        for pattern in self.github_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Handle grouped matches
                    if len(match) == 2:
                        repo_path = f"{match[0]}/{match[1]}"
                    else:
                        repo_path = match[0]
                else:
                    repo_path = match
                
                # Clean up repo path
                repo_path = repo_path.strip('/')
                if '/' in repo_path and len(repo_path.split('/')) == 2:
                    repos.add(repo_path)
        
        # Additional extraction using AI/ML context
        ai_repos = await self.extract_ai_repos_from_context(full_text)
        repos.update(ai_repos)
        
        return list(repos)

    async def get_video_details_with_ytdlp(self, video_id: str) -> Dict[str, Any]:
        """Get video details using yt-dlp command line tool"""
        try:
            # Use yt-dlp to get video metadata and subtitles
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--write-subs",
                "--write-auto-subs", 
                "--sub-lang", "en",
                "--skip-download",
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                video_data = json.loads(stdout.decode())
                
                # Try to get transcript from subtitle files
                transcript = await self.extract_transcript_from_ytdlp(video_id)
                
                return {
                    "title": video_data.get("title", ""),
                    "description": video_data.get("description", ""),
                    "transcript": transcript,
                    "duration": video_data.get("duration", 0),
                    "view_count": video_data.get("view_count", 0),
                    "upload_date": video_data.get("upload_date", ""),
                    "tags": video_data.get("tags", [])
                }
            else:
                logger.warning(f"yt-dlp failed for video {video_id}: {stderr.decode()}")
                return {}
                
        except Exception as e:
            logger.warning(f"Error getting video details with yt-dlp for {video_id}: {e}")
            return {}

    async def extract_transcript_from_ytdlp(self, video_id: str) -> str:
        """Extract transcript from yt-dlp generated subtitle files"""
        try:
            # Look for subtitle files
            possible_files = [
                f"{video_id}.en.vtt",
                f"{video_id}.en.srt",
                f"{video_id}.vtt",
                f"{video_id}.srt"
            ]
            
            for filename in possible_files:
                if Path(filename).exists():
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Clean subtitle content
                    transcript = self.clean_subtitle_content(content)
                    
                    # Clean up file
                    Path(filename).unlink()
                    
                    return transcript
            
            return ""
            
        except Exception as e:
            logger.warning(f"Error extracting transcript for {video_id}: {e}")
            return ""

    def clean_subtitle_content(self, content: str) -> str:
        """Clean subtitle content to extract just text"""
        lines = content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip VTT headers, timestamps, and empty lines
            if (line.startswith('WEBVTT') or 
                line.startswith('NOTE') or 
                '-->' in line or 
                line.isdigit() or 
                not line):
                continue
            
            # Remove HTML tags and formatting
            line = re.sub(r'<[^>]+>', '', line)
            line = re.sub(r'\{[^}]+\}', '', line)
            line = re.sub(r'\[[^\]]+\]', '', line)
            line = re.sub(r'\([^)]+\)', '', line)
            
            # Clean whitespace
            line = ' '.join(line.split())
            
            if line and line not in text_lines:
                text_lines.append(line)
        
        return ' '.join(text_lines)

    async def extract_ai_repos_from_context(self, text: str) -> List[str]:
        """Extract AI/ML repositories based on context clues"""
        repos = set()
        
        # Look for common AI/ML repository names in context
        common_ai_repos = [
            "langchain-ai/langchain",
            "run-llama/llama_index",
            "huggingface/transformers",
            "openai/openai-python",
            "anthropics/anthropic-sdk-python",
            "microsoft/autogen",
            "joaomdmoura/crewAI",
            "assafelovic/gpt-researcher",
            "microsoft/semantic-kernel",
            "stanfordnlp/dspy",
            "SweAgents/SweAgent",
            "all-hands-ai/OpenHands",
            "paul-gauthier/aider",
            "princeton-nlp/tree-of-thought-llm",
            "hwchase17/langchain",
            "jerryjliu/llama_index"
        ]
        
        # Check if any common repos are mentioned
        text_lower = text.lower()
        for repo in common_ai_repos:
            repo_name = repo.split('/')[-1].lower()
            if repo_name in text_lower or repo.lower() in text_lower:
                repos.add(repo)
        
        # Extract based on keywords
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Look for lines that mention code/repo/github with AI terms
            if any(keyword in line_lower for keyword in ['github', 'repository', 'repo', 'code', 'git clone']):
                if any(ai_term in line_lower for ai_term in ['agent', 'llm', 'ai', 'gpt', 'claude', 'openai']):
                    # Try to extract repo from this context
                    potential_matches = re.findall(r'([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)', line)
                    for match in potential_matches:
                        if '/' in match and len(match.split('/')) == 2:
                            repos.add(match)
        
        return list(repos)

    async def validate_and_analyze_repo(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Validate that a repository exists and analyze it"""
        
        # Clean repo path
        repo_path = repo_path.strip().strip('/')
        if not repo_path or '/' not in repo_path:
            return None
        
        try:
            # Check if repository exists on GitHub
            github_url = f"https://api.github.com/repos/{repo_path}"
            response = requests.get(github_url, timeout=10)
            
            if response.status_code == 200:
                repo_data = response.json()
                
                # Extract relevant information
                repo_info = {
                    "full_name": repo_data["full_name"],
                    "name": repo_data["name"],
                    "description": repo_data.get("description", ""),
                    "language": repo_data.get("language", ""),
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "topics": repo_data.get("topics", []),
                    "created_at": repo_data.get("created_at", ""),
                    "updated_at": repo_data.get("updated_at", ""),
                    "html_url": repo_data["html_url"],
                    "clone_url": repo_data["clone_url"],
                    "size": repo_data.get("size", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None,
                    "validation_status": "valid",
                    "ai_ml_relevance": self.assess_ai_ml_relevance(repo_data)
                }
                
                logger.info(f"✅ Validated repo: {repo_path} ({repo_info['stars']} stars)")
                return repo_info
                
            else:
                logger.warning(f"❌ Repository not found: {repo_path}")
                return None
                
        except Exception as e:
            logger.warning(f"❌ Error validating repository {repo_path}: {e}")
            return None

    def assess_ai_ml_relevance(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how relevant a repository is to AI/ML"""
        relevance_score = 0.0
        indicators = []
        
        # Check description
        description = (repo_data.get("description", "") or "").lower()
        ai_keywords = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'transformer', 'llm', 'large language model', 'gpt', 'claude', 'openai',
            'agent', 'autonomous', 'nlp', 'computer vision', 'reinforcement learning'
        ]
        
        for keyword in ai_keywords:
            if keyword in description:
                relevance_score += 0.1
                indicators.append(f"description_contains_{keyword.replace(' ', '_')}")
        
        # Check topics
        topics = repo_data.get("topics", [])
        ai_topics = [
            'artificial-intelligence', 'machine-learning', 'deep-learning', 'neural-networks',
            'transformers', 'llm', 'gpt', 'openai', 'agents', 'nlp', 'computer-vision'
        ]
        
        for topic in topics:
            if topic.lower() in ai_topics:
                relevance_score += 0.2
                indicators.append(f"topic_{topic}")
        
        # Check language (some languages are more common for AI/ML)
        language = repo_data.get("language", "").lower()
        ai_languages = ['python', 'jupyter notebook', 'r', 'scala', 'julia']
        if language in ai_languages:
            relevance_score += 0.1
            indicators.append(f"language_{language}")
        
        # Check repository name
        repo_name = repo_data.get("name", "").lower()
        if any(keyword in repo_name for keyword in ['ai', 'ml', 'agent', 'llm', 'gpt', 'neural', 'transformer']):
            relevance_score += 0.2
            indicators.append("name_contains_ai_terms")
        
        return {
            "relevance_score": min(1.0, relevance_score),
            "indicators": indicators,
            "is_ai_ml_related": relevance_score > 0.3
        }

    async def store_analysis_in_knowledge_graph(self, analysis: Dict[str, Any]):
        """Store analysis results in knowledge graph"""
        
        # Store channel analysis
        channel_node = KnowledgeGraphNode(
            node_id=f"youtube_channel_analysis_{int(time.time())}",
            node_type="youtube_channel_analysis",
            content=analysis,
            metadata={
                "source": "youtube_github_extractor",
                "channel": analysis.get("channel_info", {}).get("channel", "unknown"),
                "repos_discovered": analysis["total_repos_discovered"],
                "repos_validated": analysis["validated_repos"]
            },
            created_at=analysis["analysis_timestamp"],
            updated_at=analysis["analysis_timestamp"]
        )
        
        await self.framework.store_knowledge_node(channel_node)
        
        # Store individual repositories
        for repo in analysis["discovered_repositories"]:
            repo_node = KnowledgeGraphNode(
                node_id=f"youtube_discovered_repo_{repo['full_name'].replace('/', '_')}",
                node_type="github_repository",
                content=repo,
                metadata={
                    "source": "youtube_discovery",
                    "ai_ml_related": repo["ai_ml_relevance"]["is_ai_ml_related"],
                    "relevance_score": repo["ai_ml_relevance"]["relevance_score"],
                    "stars": repo["stars"]
                },
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            await self.framework.store_knowledge_node(repo_node)

    async def download_and_integrate_repositories(self, repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Download and integrate discovered repositories into the system"""
        logger.info(f"📥 Downloading and integrating {len(repositories)} repositories...")
        
        integration_results = {
            "total_repositories": len(repositories),
            "downloaded": [],
            "failed": [],
            "integrated": [],
            "containerized": []
        }
        
        for repo in repositories:
            repo_name = repo["full_name"]
            logger.info(f"📦 Processing repository: {repo_name}")
            
            try:
                # Download repository
                download_result = await self.download_repository(repo)
                if download_result["status"] == "success":
                    integration_results["downloaded"].append(repo_name)
                    
                    # Integrate with GitHub orchestrator
                    await self.integrate_with_github_orchestrator(repo, download_result["local_path"])
                    integration_results["integrated"].append(repo_name)
                    
                    # Attempt containerization if relevant
                    if repo["ai_ml_relevance"]["is_ai_ml_related"]:
                        container_result = await self.create_container_environment(repo, download_result["local_path"])
                        if container_result:
                            integration_results["containerized"].append(repo_name)
                
                else:
                    integration_results["failed"].append({
                        "repository": repo_name,
                        "error": download_result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {repo_name}: {e}")
                integration_results["failed"].append({
                    "repository": repo_name,
                    "error": str(e)
                })
            
            # Rate limiting
            await asyncio.sleep(2)
        
        # Save integration results
        results_file = self.analysis_dir / f"integration_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(integration_results, f, indent=2, default=str)
        
        logger.info(f"✅ Integration complete: {len(integration_results['downloaded'])} downloaded, {len(integration_results['integrated'])} integrated")
        
        return integration_results

    async def download_repository(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Download a repository to local storage"""
        repo_name = repo["full_name"]
        clone_url = repo["clone_url"]
        
        # Create local path
        safe_name = repo_name.replace("/", "_")
        local_path = self.youtube_repos_dir / safe_name
        
        try:
            # Clone repository
            if local_path.exists():
                # Update existing repository
                cmd = ["git", "pull"]
                result = subprocess.run(cmd, cwd=local_path, capture_output=True, text=True, timeout=300)
            else:
                # Clone new repository
                cmd = ["git", "clone", clone_url, str(local_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Downloaded: {repo_name}")
                return {
                    "status": "success",
                    "local_path": local_path,
                    "repository": repo_name
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "repository": repo_name
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "Download timeout",
                "repository": repo_name
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "repository": repo_name
            }

    async def integrate_with_github_orchestrator(self, repo: Dict[str, Any], local_path: Path):
        """Integrate repository with GitHub orchestrator"""
        
        # Create repository object for GitHub orchestrator
        from github_discovery_orchestrator import GitHubRepository
        
        github_repo = GitHubRepository(
            name=repo["name"],
            full_name=repo["full_name"],
            url=repo["html_url"],
            description=repo["description"],
            language=repo["language"],
            stars=repo["stars"],
            forks=repo["forks"],
            topics=repo["topics"],
            framework_type="ai_ml" if repo["ai_ml_relevance"]["is_ai_ml_related"] else "general"
        )
        
        # Analyze repository structure
        analyzed_repo = await self.github_orchestrator.analyze_repository(github_repo)
        
        # Store in orchestrator's discovered repos
        self.github_orchestrator.discovered_repos[repo["full_name"]] = analyzed_repo

    async def create_container_environment(self, repo: Dict[str, Any], local_path: Path) -> Optional[Dict[str, Any]]:
        """Create container environment for AI/ML repositories"""
        
        try:
            # Create repository object for containerization
            from github_discovery_orchestrator import GitHubRepository
            
            github_repo = GitHubRepository(
                name=repo["name"],
                full_name=repo["full_name"],
                url=repo["html_url"],
                description=repo["description"],
                language=repo["language"],
                stars=repo["stars"],
                forks=repo["forks"],
                topics=repo["topics"],
                framework_type="ai_ml"
            )
            
            # Check for containerization files
            dockerfile_exists = (local_path / "Dockerfile").exists()
            requirements_exists = (local_path / "requirements.txt").exists()
            package_json_exists = (local_path / "package.json").exists()
            
            github_repo.has_dockerfile = dockerfile_exists
            github_repo.has_requirements = requirements_exists
            github_repo.has_package_json = package_json_exists
            
            # Only containerize if it looks promising
            if dockerfile_exists or requirements_exists or package_json_exists:
                container_env = await self.github_orchestrator.create_container_environment(github_repo)
                return container_env.model_dump() if container_env else None
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to containerize {repo['full_name']}: {e}")
            return None

async def main():
    """Test the YouTube GitHub Extractor"""
    
    # Initialize framework and orchestrators
    framework = LocalAgenticFramework()
    await asyncio.sleep(5)  # Wait for framework initialization
    
    github_orchestrator = GitHubDiscoveryOrchestrator(framework)
    extractor = YouTubeGitHubExtractor(framework, github_orchestrator)
    
    # Analyze the specified channel
    channel_url = "https://www.youtube.com/@AIAgentsStudio/videos"
    
    print(f"🔍 Analyzing YouTube channel: {channel_url}")
    print("="*60)
    
    # Step 1: Extract repositories from channel
    analysis_results = await extractor.analyze_channel_for_repos(channel_url)
    
    if "error" in analysis_results:
        print(f"❌ Analysis failed: {analysis_results['error']}")
        return
    
    print(f"📊 Analysis Results:")
    print(f"   Videos Analyzed: {analysis_results['total_videos_analyzed']}")
    print(f"   Repositories Discovered: {analysis_results['total_repos_discovered']}")
    print(f"   Repositories Validated: {analysis_results['validated_repos']}")
    print(f"   Execution Time: {analysis_results['execution_time']:.2f}s")
    
    print(f"\n🏆 Top Discovered Repositories:")
    sorted_repos = sorted(
        analysis_results['discovered_repositories'], 
        key=lambda x: x['stars'], 
        reverse=True
    )
    
    for i, repo in enumerate(sorted_repos[:10]):
        ai_indicator = "🤖" if repo["ai_ml_relevance"]["is_ai_ml_related"] else "📄"
        print(f"   {i+1}. {ai_indicator} {repo['full_name']} ({repo['stars']} ⭐)")
        print(f"      {repo['description'][:80]}...")
    
    # Step 2: Download and integrate repositories
    if analysis_results['discovered_repositories']:
        print(f"\n📥 Downloading and integrating repositories...")
        
        # Focus on AI/ML related repos and high-star repos
        repos_to_integrate = [
            repo for repo in analysis_results['discovered_repositories']
            if repo["ai_ml_relevance"]["is_ai_ml_related"] or repo["stars"] > 100
        ][:10]  # Limit to top 10
        
        print(f"   Selected {len(repos_to_integrate)} repositories for integration")
        
        integration_results = await extractor.download_and_integrate_repositories(repos_to_integrate)
        
        print(f"\n✅ Integration Results:")
        print(f"   Downloaded: {len(integration_results['downloaded'])}")
        print(f"   Integrated: {len(integration_results['integrated'])}")
        print(f"   Containerized: {len(integration_results['containerized'])}")
        print(f"   Failed: {len(integration_results['failed'])}")
        
        if integration_results['failed']:
            print(f"\n❌ Failed Repositories:")
            for failure in integration_results['failed']:
                print(f"   • {failure['repository']}: {failure['error']}")
    
    print(f"\n🎯 YouTube GitHub extraction and integration complete!")
    print(f"   All repositories are now available in: {extractor.youtube_repos_dir}")
    print(f"   Analysis results saved in: {extractor.analysis_dir}")

if __name__ == "__main__":
    asyncio.run(main())