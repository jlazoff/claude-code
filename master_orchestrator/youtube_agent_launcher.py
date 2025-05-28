#!/usr/bin/env python3
"""
YouTube Agent Launcher
Deploys and manages YouTube research agents with foundation integration
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from simple_youtube_analyzer import SimpleYouTubeAnalyzer, ChannelAnalysis

# Foundation integration
try:
    from unified_config import SecureConfigManager
    from knowledge_orchestrator import KnowledgeOrchestrator
    from enterprise_agent_ecosystem import EnterpriseAgentEcosystem
except ImportError as e:
    logging.warning(f"Foundation modules not fully available: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeAgentLauncher:
    def __init__(self):
        self.config = None
        self.knowledge_orchestrator = None
        self.agent_ecosystem = None
        self.analyzer = SimpleYouTubeAnalyzer()
        
        # Foundation directories
        self.foundation_dir = Path("foundation_data")
        self.results_dir = self.foundation_dir / "youtube_analysis"
        self.agents_dir = self.foundation_dir / "active_agents"
        self.monitoring_dir = self.foundation_dir / "monitoring"
        
        # Create directories
        for dir_path in [self.foundation_dir, self.results_dir, self.agents_dir, self.monitoring_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info("YouTube Agent Launcher initialized")

    async def initialize_foundation(self):
        """Initialize foundation components"""
        try:
            # Initialize configuration
            try:
                self.config = SecureConfigManager()
                logger.info("✅ SecureConfigManager initialized")
            except Exception as e:
                logger.warning(f"SecureConfigManager unavailable: {e}")
            
            # Initialize knowledge orchestrator
            try:
                self.knowledge_orchestrator = KnowledgeOrchestrator()
                logger.info("✅ Knowledge Orchestrator initialized")
            except Exception as e:
                logger.warning(f"Knowledge Orchestrator unavailable: {e}")
            
            # Initialize agent ecosystem
            try:
                self.agent_ecosystem = EnterpriseAgentEcosystem()
                logger.info("✅ Enterprise Agent Ecosystem initialized")
            except Exception as e:
                logger.warning(f"Agent Ecosystem unavailable: {e}")
                
        except Exception as e:
            logger.error(f"Foundation initialization failed: {e}")

    def save_to_foundation(self, analysis: ChannelAnalysis, analysis_type: str = "youtube_channel"):
        """Save analysis results to foundation with structured metadata"""
        timestamp = datetime.now().isoformat()
        
        # Create structured result
        foundation_result = {
            "metadata": {
                "type": analysis_type,
                "timestamp": timestamp,
                "channel": analysis.channel_name,
                "source": analysis.channel_url,
                "analyzer_version": "1.0",
                "total_videos": analysis.total_videos,
                "analyzed_videos": len(analysis.videos)
            },
            "analysis": analysis.__dict__,
            "insights": {
                "research_themes": analysis.research_themes,
                "top_concepts": analysis.top_concepts[:10],
                "project_recommendations": analysis.project_recommendations,
                "paper_count": sum(len(video.research_papers) for video in analysis.videos),
                "concept_diversity": len(set(concept for video in analysis.videos for concept in video.key_concepts))
            },
            "actionable_items": self.generate_actionable_items(analysis)
        }
        
        # Save to foundation
        filename = f"youtube_analysis_{analysis.channel_name}_{timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(foundation_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis saved to foundation: {filepath}")
        return filepath

    def generate_actionable_items(self, analysis: ChannelAnalysis) -> List[Dict[str, Any]]:
        """Generate actionable items from analysis"""
        actionable_items = []
        
        # Research paper implementation projects
        papers = [paper for video in analysis.videos for paper in video.research_papers]
        if papers:
            actionable_items.append({
                "type": "research_implementation",
                "title": "Implement Research Papers",
                "description": f"Implement {len(papers)} research papers found in videos",
                "priority": "high",
                "estimated_effort": "4-6 weeks",
                "resources_needed": ["GPU compute", "Development environment", "Research access"],
                "papers": papers[:5]  # Top 5 papers
            })
        
        # Concept-based projects
        concept_counts = {}
        for video in analysis.videos:
            for concept in video.key_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_concepts:
            actionable_items.append({
                "type": "concept_exploration",
                "title": "Deep Dive into Top Concepts",
                "description": f"Explore and implement projects around {', '.join([c[0] for c in top_concepts])}",
                "priority": "medium",
                "estimated_effort": "2-3 weeks",
                "concepts": top_concepts
            })
        
        # Implementation ideas aggregation
        all_ideas = [idea for video in analysis.videos for idea in video.implementation_ideas]
        if all_ideas:
            actionable_items.append({
                "type": "implementation_projects",
                "title": "Execute Implementation Ideas",
                "description": f"Execute {len(all_ideas)} implementation ideas generated from videos",
                "priority": "high",
                "estimated_effort": "8-12 weeks",
                "ideas": all_ideas
            })
        
        return actionable_items

    async def launch_youtube_agent(self, channel_url: str, continuous: bool = True) -> Dict[str, Any]:
        """Launch YouTube agent for specific channel"""
        logger.info(f"🚀 Launching YouTube agent for: {channel_url}")
        
        # Perform initial analysis
        analysis = self.analyzer.analyze_channel(channel_url)
        
        # Save to foundation
        foundation_path = self.save_to_foundation(analysis)
        
        # Create agent configuration
        agent_config = {
            "agent_id": f"youtube_agent_{analysis.channel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "youtube_research_agent",
            "target_channel": channel_url,
            "channel_name": analysis.channel_name,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_analysis": foundation_path.name,
            "monitoring": {
                "check_interval": "1h" if continuous else "manual",
                "auto_analyze_new_videos": continuous,
                "save_to_foundation": True
            },
            "metrics": {
                "videos_analyzed": len(analysis.videos),
                "papers_found": sum(len(video.research_papers) for video in analysis.videos),
                "concepts_extracted": len(set(concept for video in analysis.videos for concept in video.key_concepts)),
                "ideas_generated": sum(len(video.implementation_ideas) for video in analysis.videos)
            }
        }
        
        # Save agent configuration
        agent_file = self.agents_dir / f"{agent_config['agent_id']}.json"
        with open(agent_file, 'w') as f:
            json.dump(agent_config, f, indent=2, default=str)
        
        logger.info(f"✅ YouTube agent deployed: {agent_config['agent_id']}")
        return {
            "agent_config": agent_config,
            "analysis": analysis,
            "foundation_path": str(foundation_path)
        }

    def create_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard"""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "active_agents": [],
            "recent_analyses": [],
            "system_metrics": {},
            "actionable_items": []
        }
        
        # Load active agents
        for agent_file in self.agents_dir.glob("*.json"):
            try:
                with open(agent_file) as f:
                    agent_config = json.load(f)
                    dashboard_data["active_agents"].append(agent_config)
            except Exception as e:
                logger.warning(f"Could not load agent config {agent_file}: {e}")
        
        # Load recent analyses
        analysis_files = sorted(self.results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        for analysis_file in analysis_files[:10]:  # Last 10 analyses
            try:
                with open(analysis_file) as f:
                    analysis = json.load(f)
                    dashboard_data["recent_analyses"].append({
                        "filename": analysis_file.name,
                        "timestamp": analysis.get("metadata", {}).get("timestamp"),
                        "channel": analysis.get("metadata", {}).get("channel"),
                        "videos_analyzed": analysis.get("metadata", {}).get("analyzed_videos", 0),
                        "insights": analysis.get("insights", {})
                    })
            except Exception as e:
                logger.warning(f"Could not load analysis {analysis_file}: {e}")
        
        # Aggregate actionable items
        for analysis_file in analysis_files[:5]:  # Last 5 analyses
            try:
                with open(analysis_file) as f:
                    analysis = json.load(f)
                    actionable_items = analysis.get("actionable_items", [])
                    dashboard_data["actionable_items"].extend(actionable_items)
            except Exception as e:
                logger.warning(f"Could not extract actionable items from {analysis_file}: {e}")
        
        # Save dashboard data
        dashboard_file = self.monitoring_dir / f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return dashboard_data

    async def run_continuous_monitoring(self, agents: List[Dict[str, Any]]):
        """Run continuous monitoring for active agents"""
        logger.info("🔄 Starting continuous monitoring...")
        
        while True:
            try:
                for agent in agents:
                    if agent.get("monitoring", {}).get("auto_analyze_new_videos"):
                        channel_url = agent.get("target_channel")
                        if channel_url:
                            logger.info(f"Checking for updates: {agent.get('channel_name')}")
                            # Re-analyze channel for new content
                            analysis = self.analyzer.analyze_channel(channel_url)
                            self.save_to_foundation(analysis, "continuous_monitoring")
                
                # Update dashboard data
                self.create_monitoring_dashboard_data()
                
                # Wait before next check (1 hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

async def main():
    """Main execution function"""
    launcher = YouTubeAgentLauncher()
    
    # Initialize foundation
    await launcher.initialize_foundation()
    
    # Target channels
    target_channels = [
        "https://www.youtube.com/@code4AI",
        "https://www.youtube.com/@TwoMinutePapers"  # Fallback for demonstration
    ]
    
    launched_agents = []
    
    for channel_url in target_channels:
        try:
            print(f"\n🎯 Launching agent for: {channel_url}")
            
            # Launch agent
            result = await launcher.launch_youtube_agent(channel_url, continuous=True)
            launched_agents.append(result["agent_config"])
            
            print(f"✅ Agent launched successfully!")
            print(f"   Agent ID: {result['agent_config']['agent_id']}")
            print(f"   Videos analyzed: {result['agent_config']['metrics']['videos_analyzed']}")
            print(f"   Papers found: {result['agent_config']['metrics']['papers_found']}")
            print(f"   Concepts extracted: {result['agent_config']['metrics']['concepts_extracted']}")
            print(f"   Foundation path: {result['foundation_path']}")
            
        except Exception as e:
            logger.error(f"Failed to launch agent for {channel_url}: {e}")
            continue
    
    if launched_agents:
        # Create initial dashboard data
        dashboard_data = launcher.create_monitoring_dashboard_data()
        print(f"\n📊 Dashboard data created with {len(dashboard_data['active_agents'])} active agents")
        print(f"📈 Recent analyses: {len(dashboard_data['recent_analyses'])}")
        print(f"🎯 Actionable items: {len(dashboard_data['actionable_items'])}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"YOUTUBE AGENT DEPLOYMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Active Agents: {len(launched_agents)}")
        print(f"Foundation Directory: {launcher.foundation_dir}")
        print(f"Results Directory: {launcher.results_dir}")
        print(f"Monitoring Directory: {launcher.monitoring_dir}")
        
        print(f"\n🔄 To run continuous monitoring, use:")
        print(f"   python3 youtube_agent_launcher.py --continuous")
        
        print(f"\n🌐 Frontend will be available at: http://localhost:8000")
        print(f"   Use the monitoring dashboard to track agent progress")
        
        # Option to run continuous monitoring
        run_continuous = input("\nRun continuous monitoring now? (y/n): ").lower().strip() == 'y'
        if run_continuous:
            await launcher.run_continuous_monitoring(launched_agents)
    
    else:
        print("❌ No agents were successfully launched")

if __name__ == "__main__":
    if "--continuous" in sys.argv:
        # Run in continuous mode for background service
        launcher = YouTubeAgentLauncher()
        asyncio.run(launcher.run_continuous_monitoring([]))
    else:
        asyncio.run(main())