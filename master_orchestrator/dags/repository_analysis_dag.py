"""
Airflow DAG for Automated Repository Analysis
Orchestrates continuous analysis of all repositories in the ecosystem
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable

import structlog

logger = structlog.get_logger()

# Default arguments for the DAG
default_args = {
    'owner': 'master-orchestrator',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'repository_analysis_workflow',
    default_args=default_args,
    description='Automated repository analysis and knowledge graph updates',
    schedule_interval=timedelta(hours=6),  # Run every 6 hours
    catchup=False,
    max_active_runs=1,
    tags=['analysis', 'repositories', 'knowledge-graph'],
)


def discover_repositories(**context) -> Dict[str, Any]:
    """Discover all repositories in the GitHub directory."""
    import os
    from pathlib import Path
    
    github_path = Path("/Users/jlazoff/Documents/GitHub")
    discovered_repos = []
    
    if github_path.exists():
        for repo_dir in github_path.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                discovered_repos.append({
                    'name': repo_dir.name,
                    'path': str(repo_dir),
                    'size': sum(f.stat().st_size for f in repo_dir.rglob('*') if f.is_file()),
                    'last_modified': max(f.stat().st_mtime for f in repo_dir.rglob('*') if f.is_file())
                })
    
    logger.info(f"Discovered {len(discovered_repos)} repositories")
    
    # Store in XCom for downstream tasks
    context['task_instance'].xcom_push(key='discovered_repos', value=discovered_repos)
    return {'repository_count': len(discovered_repos)}


def analyze_repository(repo_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single repository."""
    from pathlib import Path
    import json
    import hashlib
    
    repo_path = Path(repo_info['path'])
    repo_name = repo_info['name']
    
    logger.info(f"Analyzing repository: {repo_name}")
    
    analysis = {
        'name': repo_name,
        'path': str(repo_path),
        'timestamp': datetime.utcnow().isoformat(),
        'languages': [],
        'technologies': [],
        'capabilities': [],
        'file_count': 0,
        'size_bytes': 0
    }
    
    try:
        # Basic file analysis
        file_extensions = {}
        file_count = 0
        total_size = 0
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_count += 1
                try:
                    total_size += file_path.stat().st_size
                    ext = file_path.suffix.lower()
                    if ext:
                        file_extensions[ext] = file_extensions.get(ext, 0) + 1
                except:
                    pass
        
        analysis['file_count'] = file_count
        analysis['size_bytes'] = total_size
        
        # Language detection
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        for ext, count in file_extensions.items():
            if ext in language_map and count > 0:
                analysis['languages'].append(language_map[ext])
        
        # Technology detection
        tech_indicators = {
            'docker': ['dockerfile', 'docker-compose'],
            'kubernetes': ['k8s', 'kubernetes'],
            'terraform': ['.tf'],
            'nodejs': ['package.json'],
            'python': ['requirements.txt', 'pyproject.toml'],
            'ai': ['model', 'agent', 'llm'],
            'web': ['html', 'css', 'frontend']
        }
        
        all_files = [f.name.lower() for f in repo_path.rglob('*') if f.is_file()]
        
        for tech, indicators in tech_indicators.items():
            if any(any(indicator in filename for filename in all_files) for indicator in indicators):
                analysis['technologies'].append(tech)
        
        # Capability detection from README
        readme_content = ""
        for readme_name in ['README.md', 'README.txt', 'README']:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                try:
                    readme_content = readme_path.read_text(encoding='utf-8').lower()
                    break
                except:
                    continue
        
        capability_keywords = {
            'api': ['api', 'rest', 'graphql'],
            'agent': ['agent', 'autonomous'],
            'web': ['web', 'frontend', 'ui'],
            'cli': ['cli', 'command line']
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in readme_content for keyword in keywords):
                analysis['capabilities'].append(capability)
        
        logger.info(f"Analysis complete for {repo_name}: {len(analysis['languages'])} languages, {len(analysis['technologies'])} technologies")
        
    except Exception as e:
        logger.error(f"Analysis failed for {repo_name}: {e}")
        analysis['error'] = str(e)
    
    return analysis


def batch_analyze_repositories(**context) -> Dict[str, Any]:
    """Analyze all discovered repositories in batches."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Get discovered repositories from XCom
    discovered_repos = context['task_instance'].xcom_pull(
        task_ids='discover_repositories',
        key='discovered_repos'
    )
    
    if not discovered_repos:
        logger.warning("No repositories found to analyze")
        return {'analyzed_count': 0}
    
    # Analyze repositories in parallel
    analyses = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(analyze_repository, repo_info)
            for repo_info in discovered_repos
        ]
        
        for future in futures:
            try:
                analysis = future.result(timeout=300)  # 5 minute timeout per repo
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Repository analysis failed: {e}")
    
    # Store analyses in XCom
    context['task_instance'].xcom_push(key='repository_analyses', value=analyses)
    
    logger.info(f"Completed analysis of {len(analyses)} repositories")
    return {'analyzed_count': len(analyses)}


def update_knowledge_graph(**context) -> Dict[str, Any]:
    """Update the knowledge graph with repository analyses."""
    
    # Get analyses from XCom
    analyses = context['task_instance'].xcom_pull(
        task_ids='batch_analyze_repositories',
        key='repository_analyses'
    )
    
    if not analyses:
        logger.warning("No analyses found to update knowledge graph")
        return {'updated_count': 0}
    
    # In a real implementation, this would connect to ArangoDB
    # For now, we'll simulate the update
    updated_count = 0
    
    for analysis in analyses:
        try:
            # Simulate knowledge graph update
            logger.info(f"Updating knowledge graph for {analysis['name']}")
            
            # Create project node
            project_data = {
                'name': analysis['name'],
                'path': analysis['path'],
                'languages': analysis['languages'],
                'technologies': analysis['technologies'],
                'capabilities': analysis['capabilities'],
                'file_count': analysis['file_count'],
                'size_bytes': analysis['size_bytes'],
                'last_analyzed': analysis['timestamp']
            }
            
            # Create relationships
            for tech in analysis['technologies']:
                # Create technology relationships
                pass
            
            for capability in analysis['capabilities']:
                # Create capability relationships
                pass
            
            updated_count += 1
            
        except Exception as e:
            logger.error(f"Failed to update knowledge graph for {analysis['name']}: {e}")
    
    logger.info(f"Updated knowledge graph for {updated_count} repositories")
    return {'updated_count': updated_count}


def generate_analysis_report(**context) -> Dict[str, Any]:
    """Generate a comprehensive analysis report."""
    
    # Get analyses from XCom
    analyses = context['task_instance'].xcom_pull(
        task_ids='batch_analyze_repositories',
        key='repository_analyses'
    )
    
    if not analyses:
        return {'report': 'No data available'}
    
    # Generate report
    total_repos = len(analyses)
    total_files = sum(a.get('file_count', 0) for a in analyses)
    total_size = sum(a.get('size_bytes', 0) for a in analyses)
    
    # Technology breakdown
    tech_count = {}
    for analysis in analyses:
        for tech in analysis.get('technologies', []):
            tech_count[tech] = tech_count.get(tech, 0) + 1
    
    # Language breakdown
    lang_count = {}
    for analysis in analyses:
        for lang in analysis.get('languages', []):
            lang_count[lang] = lang_count.get(lang, 0) + 1
    
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'summary': {
            'total_repositories': total_repos,
            'total_files': total_files,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
        },
        'technologies': dict(sorted(tech_count.items(), key=lambda x: x[1], reverse=True)),
        'languages': dict(sorted(lang_count.items(), key=lambda x: x[1], reverse=True)),
        'repositories': analyses
    }
    
    # Save report to file
    import json
    from pathlib import Path
    
    reports_dir = Path('/tmp/master-orchestrator-reports')
    reports_dir.mkdir(exist_ok=True)
    
    report_file = reports_dir / f"repository_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Analysis report saved to {report_file}")
    
    # Store summary in XCom
    context['task_instance'].xcom_push(key='analysis_report', value=report['summary'])
    
    return report['summary']


# Task definitions
discover_task = PythonOperator(
    task_id='discover_repositories',
    python_callable=discover_repositories,
    dag=dag,
)

analyze_task = PythonOperator(
    task_id='batch_analyze_repositories',
    python_callable=batch_analyze_repositories,
    dag=dag,
)

update_kg_task = PythonOperator(
    task_id='update_knowledge_graph',
    python_callable=update_knowledge_graph,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_analysis_report',
    python_callable=generate_analysis_report,
    dag=dag,
)

# Cleanup old reports
cleanup_task = BashOperator(
    task_id='cleanup_old_reports',
    bash_command='find /tmp/master-orchestrator-reports -name "*.json" -mtime +7 -delete',
    dag=dag,
)

# Task dependencies
discover_task >> analyze_task >> update_kg_task >> report_task >> cleanup_task

# Optional: Add data quality checks
def data_quality_check(**context) -> bool:
    """Perform data quality checks on the analysis results."""
    
    analyses = context['task_instance'].xcom_pull(
        task_ids='batch_analyze_repositories',
        key='repository_analyses'
    )
    
    if not analyses:
        return False
    
    # Check for minimum data quality
    valid_analyses = [a for a in analyses if 'error' not in a and a.get('file_count', 0) > 0]
    quality_ratio = len(valid_analyses) / len(analyses)
    
    logger.info(f"Data quality check: {quality_ratio:.2%} of analyses are valid")
    
    return quality_ratio >= 0.8  # Require 80% valid analyses


quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag,
)

# Insert quality check before knowledge graph update
analyze_task >> quality_check_task >> update_kg_task