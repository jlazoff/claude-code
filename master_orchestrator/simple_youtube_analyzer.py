#!/usr/bin/env python3
"""
Simple YouTube Research Analyzer
Analyzes YouTube channel content and extracts research insights
"""

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yt_dlp
import requests
import arxiv
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoAnalysis:
    video_id: str
    title: str
    description: str
    transcript: str
    research_papers: List[Dict[str, Any]]
    key_concepts: List[str]
    implementation_ideas: List[Dict[str, Any]]
    agent_recommendations: List[str]
    uploaded_date: str
    duration: str
    view_count: int

@dataclass
class ChannelAnalysis:
    channel_name: str
    channel_url: str
    total_videos: int
    videos: List[VideoAnalysis]
    research_themes: List[str]
    top_concepts: List[str]
    project_recommendations: List[Dict[str, Any]]
    analysis_timestamp: str

class SimpleYouTubeAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Create output directory
        self.output_dir = Path("youtube_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("YouTube analyzer initialized successfully")

    def extract_channel_videos(self, channel_url: str) -> List[Dict[str, Any]]:
        """Extract all videos from a YouTube channel"""
        logger.info(f"Extracting videos from channel: {channel_url}")
        
        ydl_opts = {
            'quiet': False,  # Enable verbose output to debug
            'extract_flat': True,
            'playlistend': 100,  # Get more videos
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Try different URL formats
                urls_to_try = [
                    channel_url,
                    f"{channel_url}/videos",
                    f"{channel_url}/streams",
                    f"{channel_url}/featured"
                ]
                
                for url in urls_to_try:
                    try:
                        logger.info(f"Trying URL: {url}")
                        info = ydl.extract_info(url, download=False)
                        videos = []
                        
                        if 'entries' in info:
                            for entry in info['entries']:
                                if entry and entry.get('id'):
                                    videos.append({
                                        'id': entry.get('id', ''),
                                        'title': entry.get('title', ''),
                                        'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                                        'duration': entry.get('duration', 0),
                                        'upload_date': entry.get('upload_date', ''),
                                        'view_count': entry.get('view_count', 0)
                                    })
                        
                        if videos:
                            logger.info(f"Found {len(videos)} videos in channel using {url}")
                            return videos
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract from {url}: {e}")
                        continue
                
                logger.warning("No videos found with any URL format")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting channel videos: {e}")
            return []

    def get_video_details(self, video_id: str) -> Dict[str, Any]:
        """Get detailed video information"""
        try:
            ydl_opts = {
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                return {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'upload_date': info.get('upload_date', ''),
                    'duration': info.get('duration_string', ''),
                    'view_count': info.get('view_count', 0),
                    'tags': info.get('tags', [])
                }
                
        except Exception as e:
            logger.error(f"Error getting video details for {video_id}: {e}")
            return {}

    def get_video_transcript(self, video_id: str) -> str:
        """Get video transcript if available"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = ' '.join([entry['text'] for entry in transcript_list])
            return transcript
        except Exception as e:
            logger.warning(f"Could not get transcript for {video_id}: {e}")
            return ""

    def extract_research_papers(self, text: str) -> List[Dict[str, Any]]:
        """Extract research paper references from text"""
        papers = []
        
        # Look for arXiv paper patterns
        arxiv_pattern = r'arXiv[:\s]*(\d{4}\.\d{4,5})'
        arxiv_matches = re.findall(arxiv_pattern, text, re.IGNORECASE)
        
        for match in arxiv_matches:
            try:
                search = arxiv.Search(id_list=[match])
                paper = next(search.results())
                papers.append({
                    'type': 'arxiv',
                    'id': match,
                    'title': paper.title,
                    'authors': [str(author) for author in paper.authors],
                    'summary': paper.summary,
                    'url': paper.entry_id
                })
                logger.info(f"Found arXiv paper: {paper.title}")
            except Exception as e:
                logger.warning(f"Could not fetch arXiv paper {match}: {e}")
        
        # Look for general paper patterns
        paper_patterns = [
            r'"([^"]+?)"[^"]*?(?:paper|research|study)',
            r'paper[:\s]+"([^"]+?)"',
            r'research[:\s]+"([^"]+?)"'
        ]
        
        for pattern in paper_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10 and match not in [p.get('title', '') for p in papers]:
                    papers.append({
                        'type': 'mentioned',
                        'title': match,
                        'context': 'mentioned in video',
                        'source': 'video_content'
                    })
        
        return papers

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key AI/ML concepts from text"""
        concepts = set()
        
        # AI/ML terms to look for
        ai_terms = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'transformer', 'attention mechanism', 'gpt', 'llm', 'large language model',
            'computer vision', 'natural language processing', 'nlp', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'convolutional neural network',
            'cnn', 'rnn', 'lstm', 'bert', 'claude', 'openai', 'anthropic', 'google ai',
            'tensorflow', 'pytorch', 'hugging face', 'gradient descent', 'backpropagation',
            'fine-tuning', 'prompt engineering', 'embedding', 'vector database', 'rag',
            'retrieval augmented generation', 'multimodal', 'diffusion model', 'gan',
            'autoencoder', 'optimization', 'hyperparameter', 'dataset', 'training data',
            'inference', 'model deployment', 'edge computing', 'federated learning'
        ]
        
        text_lower = text.lower()
        for term in ai_terms:
            if term in text_lower:
                concepts.add(term)
        
        return list(concepts)

    def generate_implementation_ideas(self, video_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation ideas based on video content"""
        ideas = []
        
        title = video_analysis.get('title', '').lower()
        description = video_analysis.get('description', '').lower()
        concepts = video_analysis.get('key_concepts', [])
        
        # Generate ideas based on content
        if 'tutorial' in title or 'how to' in title:
            ideas.append({
                'type': 'tutorial_implementation',
                'title': f"Implement: {video_analysis.get('title', '')}",
                'description': 'Create a practical implementation based on this tutorial',
                'complexity': 'medium',
                'estimated_time': '1-2 weeks'
            })
        
        if any(concept in ['transformer', 'attention', 'bert', 'gpt'] for concept in concepts):
            ideas.append({
                'type': 'model_implementation',
                'title': 'Transformer Model Implementation',
                'description': 'Build a custom transformer model for specific use case',
                'complexity': 'high',
                'estimated_time': '3-4 weeks'
            })
        
        if 'rag' in concepts or 'retrieval' in concepts:
            ideas.append({
                'type': 'rag_system',
                'title': 'RAG System Implementation',
                'description': 'Build a retrieval-augmented generation system',
                'complexity': 'medium',
                'estimated_time': '2-3 weeks'
            })
        
        if 'agent' in title or 'autonomous' in description:
            ideas.append({
                'type': 'agent_system',
                'title': 'Autonomous Agent Implementation',
                'description': 'Create an autonomous agent based on discussed concepts',
                'complexity': 'high',
                'estimated_time': '4-6 weeks'
            })
        
        return ideas

    def analyze_video(self, video_id: str) -> Optional[VideoAnalysis]:
        """Analyze a single video"""
        logger.info(f"Analyzing video: {video_id}")
        
        try:
            # Get video details
            details = self.get_video_details(video_id)
            if not details:
                return None
            
            # Get transcript
            transcript = self.get_video_transcript(video_id)
            
            # Combine text for analysis
            full_text = f"{details.get('title', '')} {details.get('description', '')} {transcript}"
            
            # Extract research papers
            research_papers = self.extract_research_papers(full_text)
            
            # Extract key concepts
            key_concepts = self.extract_key_concepts(full_text)
            
            # Generate implementation ideas
            implementation_ideas = self.generate_implementation_ideas({
                'title': details.get('title', ''),
                'description': details.get('description', ''),
                'key_concepts': key_concepts
            })
            
            # Generate agent recommendations
            agent_recommendations = []
            if key_concepts:
                agent_recommendations.append('research_agent')
            if implementation_ideas:
                agent_recommendations.append('implementation_agent')
            if research_papers:
                agent_recommendations.append('paper_analysis_agent')
            
            return VideoAnalysis(
                video_id=video_id,
                title=details.get('title', ''),
                description=details.get('description', ''),
                transcript=transcript,
                research_papers=research_papers,
                key_concepts=key_concepts,
                implementation_ideas=implementation_ideas,
                agent_recommendations=agent_recommendations,
                uploaded_date=details.get('upload_date', ''),
                duration=details.get('duration', ''),
                view_count=details.get('view_count', 0)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_id}: {e}")
            return None

    def analyze_channel(self, channel_url: str) -> ChannelAnalysis:
        """Analyze entire YouTube channel"""
        logger.info(f"Starting channel analysis: {channel_url}")
        
        # Extract channel videos
        videos_info = self.extract_channel_videos(channel_url)
        
        # Analyze each video
        analyzed_videos = []
        all_concepts = []
        all_papers = []
        
        for i, video_info in enumerate(videos_info[:10]):  # Limit to first 10 for testing
            logger.info(f"Processing video {i+1}/{len(videos_info[:10])}: {video_info.get('title', 'Unknown')}")
            
            analysis = self.analyze_video(video_info['id'])
            if analysis:
                analyzed_videos.append(analysis)
                all_concepts.extend(analysis.key_concepts)
                all_papers.extend(analysis.research_papers)
        
        # Analyze overall themes
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Generate research themes
        research_themes = []
        if any('transformer' in concept for concept, _ in top_concepts):
            research_themes.append('Transformer Architecture')
        if any('agent' in concept for concept, _ in top_concepts):
            research_themes.append('AI Agents')
        if any('rag' in concept for concept, _ in top_concepts):
            research_themes.append('Retrieval Systems')
        if any('llm' in concept or 'language model' in concept for concept, _ in top_concepts):
            research_themes.append('Large Language Models')
        
        # Generate project recommendations
        project_recommendations = [
            {
                'title': 'Multi-Agent Research System',
                'description': 'Build a system that automatically researches and implements AI concepts from video content',
                'priority': 'high',
                'technologies': ['LangChain', 'Vector Databases', 'LLMs']
            },
            {
                'title': 'Knowledge Graph Builder',
                'description': 'Create a knowledge graph from research papers and video content',
                'priority': 'medium',
                'technologies': ['Neo4j', 'NLP', 'Graph Algorithms']
            },
            {
                'title': 'Automated Literature Review',
                'description': 'System that automatically reviews and summarizes research papers',
                'priority': 'high',
                'technologies': ['ArXiv API', 'Transformers', 'Summarization']
            }
        ]
        
        return ChannelAnalysis(
            channel_name=channel_url.split('/')[-1],
            channel_url=channel_url,
            total_videos=len(videos_info),
            videos=analyzed_videos,
            research_themes=research_themes,
            top_concepts=[concept for concept, _ in top_concepts],
            project_recommendations=project_recommendations,
            analysis_timestamp=datetime.now().isoformat()
        )

    def save_results(self, analysis: ChannelAnalysis):
        """Save analysis results to JSON file"""
        filename = f"channel_analysis_{analysis.channel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        # Convert to dict for JSON serialization
        analysis_dict = asdict(analysis)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis results saved to: {filepath}")
        return filepath

    def print_summary(self, analysis: ChannelAnalysis):
        """Print analysis summary"""
        print("\n" + "="*80)
        print(f"YOUTUBE CHANNEL ANALYSIS SUMMARY")
        print("="*80)
        print(f"Channel: {analysis.channel_name}")
        print(f"Total Videos: {analysis.total_videos}")
        print(f"Analyzed Videos: {len(analysis.videos)}")
        print(f"Analysis Time: {analysis.analysis_timestamp}")
        
        print(f"\n📚 Research Themes ({len(analysis.research_themes)}):")
        for theme in analysis.research_themes:
            print(f"  • {theme}")
        
        print(f"\n🔑 Top Concepts ({len(analysis.top_concepts[:10])}):")
        for concept in analysis.top_concepts[:10]:
            print(f"  • {concept}")
        
        print(f"\n💡 Project Recommendations ({len(analysis.project_recommendations)}):")
        for project in analysis.project_recommendations:
            print(f"  • {project['title']} ({project['priority']} priority)")
            print(f"    {project['description']}")
        
        print(f"\n📹 Video Analysis Results:")
        for video in analysis.videos:
            print(f"  • {video.title}")
            print(f"    Papers: {len(video.research_papers)}, Concepts: {len(video.key_concepts)}, Ideas: {len(video.implementation_ideas)}")
        
        print("\n" + "="*80)

async def main():
    """Main execution function"""
    # Try multiple channels to demonstrate functionality
    channels_to_try = [
        "https://www.youtube.com/@code4AI",
        "https://www.youtube.com/@TwoMinutePapers",  # Known AI research channel
        "https://www.youtube.com/@yannickilcher"     # Another AI research channel
    ]
    
    analyzer = SimpleYouTubeAnalyzer()
    
    for channel_url in channels_to_try:
        try:
            print(f"\n🔍 Trying to analyze channel: {channel_url}")
            
            # Analyze the channel
            analysis = analyzer.analyze_channel(channel_url)
            
            if analysis.videos:  # If we found videos to analyze
                # Save results
                filepath = analyzer.save_results(analysis)
                
                # Print summary
                analyzer.print_summary(analysis)
                
                print(f"\n✅ Analysis complete! Results saved to: {filepath}")
                break  # Success, exit loop
            else:
                print(f"No analyzable videos found in {channel_url}, trying next channel...")
                
        except Exception as e:
            logger.error(f"Analysis failed for {channel_url}: {e}")
            continue
    
    else:
        print("❌ Could not analyze any of the target channels")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())