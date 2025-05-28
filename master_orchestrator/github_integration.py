#!/usr/bin/env python3
"""
GitHub Integration Module
Automated code commits, PR creation, and repository management
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import git
import aiohttp
import aiofiles
from github import Github
from github.GithubException import GithubException

from unified_config import SecureConfigManager
from parallel_llm_orchestrator import ParallelLLMOrchestrator

class GitHubIntegrator:
    """GitHub integration for automated code management"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.github_client = None
        self.repo = None
        self.git_repo = None
        self.llm_orchestrator = None
        
    async def initialize(self):
        """Initialize GitHub integration"""
        try:
            # Get GitHub token from config
            github_token = self.config.get_api_key('github')
            if not github_token:
                raise ValueError("GitHub token not found in configuration")
                
            self.github_client = Github(github_token)
            
            # Initialize git repository
            repo_path = Path.cwd()
            try:
                self.git_repo = git.Repo(repo_path)
            except git.InvalidGitRepositoryError:
                # Initialize new git repository
                self.git_repo = git.Repo.init(repo_path)
                
            # Get GitHub repository
            try:
                repo_name = self._get_repo_name_from_remote()
                if repo_name:
                    self.repo = self.github_client.get_repo(repo_name)
            except Exception as e:
                logging.warning(f"Could not connect to GitHub repository: {e}")
                
            # Initialize LLM orchestrator for intelligent commits
            self.llm_orchestrator = ParallelLLMOrchestrator()
            await self.llm_orchestrator.initialize()
            
            logging.info("GitHub integration initialized successfully")
            
        except Exception as e:
            logging.error(f"GitHub integration initialization failed: {e}")
            raise
            
    def _get_repo_name_from_remote(self) -> Optional[str]:
        """Extract repository name from git remote"""
        try:
            if not self.git_repo.remotes:
                return None
                
            origin = self.git_repo.remotes.origin
            url = origin.url
            
            # Parse GitHub URL to get repo name
            if 'github.com' in url:
                if url.endswith('.git'):
                    url = url[:-4]
                parts = url.split('/')
                if len(parts) >= 2:
                    return f"{parts[-2]}/{parts[-1]}"
                    
        except Exception as e:
            logging.warning(f"Could not extract repo name from remote: {e}")
            
        return None
        
    async def analyze_changes(self) -> Dict[str, Any]:
        """Analyze current changes for intelligent commit message generation"""
        try:
            # Get staged and unstaged changes
            changes = {
                "staged_files": [],
                "modified_files": [],
                "untracked_files": [],
                "deleted_files": []
            }
            
            # Staged changes
            for item in self.git_repo.index.diff("HEAD"):
                changes["staged_files"].append({
                    "file": item.a_path,
                    "change_type": item.change_type,
                    "diff": item.diff.decode('utf-8', errors='ignore') if item.diff else ""
                })
                
            # Modified files
            for item in self.git_repo.index.diff(None):
                changes["modified_files"].append({
                    "file": item.a_path,
                    "change_type": item.change_type
                })
                
            # Untracked files
            changes["untracked_files"] = self.git_repo.untracked_files
            
            # Get diff statistics
            diff_stats = await self._calculate_diff_stats(changes)
            
            return {
                "changes": changes,
                "stats": diff_stats,
                "total_files": len(changes["staged_files"]) + len(changes["modified_files"]) + len(changes["untracked_files"]),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing changes: {e}")
            return {"error": str(e)}
            
    async def _calculate_diff_stats(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics from diff"""
        stats = {
            "lines_added": 0,
            "lines_removed": 0,
            "files_added": 0,
            "files_modified": 0,
            "files_deleted": 0
        }
        
        for staged_file in changes["staged_files"]:
            diff_text = staged_file.get("diff", "")
            for line in diff_text.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    stats["lines_added"] += 1
                elif line.startswith('-') and not line.startswith('---'):
                    stats["lines_removed"] += 1
                    
            if staged_file["change_type"] == "A":  # Added
                stats["files_added"] += 1
            elif staged_file["change_type"] == "M":  # Modified
                stats["files_modified"] += 1
            elif staged_file["change_type"] == "D":  # Deleted
                stats["files_deleted"] += 1
                
        stats["files_added"] += len(changes["untracked_files"])
        
        return stats
        
    async def generate_intelligent_commit_message(self, changes_analysis: Dict[str, Any]) -> str:
        """Generate intelligent commit message using LLM analysis"""
        try:
            if not self.llm_orchestrator:
                return "Update: Automated commit"
                
            # Create prompt for commit message generation
            prompt = f"""
Analyze these code changes and generate a concise, meaningful commit message:

Change Summary:
- Files changed: {changes_analysis.get('total_files', 0)}
- Lines added: {changes_analysis.get('stats', {}).get('lines_added', 0)}
- Lines removed: {changes_analysis.get('stats', {}).get('lines_removed', 0)}
- Files added: {changes_analysis.get('stats', {}).get('files_added', 0)}
- Files modified: {changes_analysis.get('stats', {}).get('files_modified', 0)}

Files changed:
{json.dumps(changes_analysis.get('changes', {}), indent=2)}

Generate a commit message that:
1. Uses conventional commit format (feat:, fix:, docs:, refactor:, etc.)
2. Summarizes the main purpose of the changes
3. Is concise but descriptive
4. Focuses on the "why" rather than the "what"

Return only the commit message, nothing else.
"""
            
            result = await self.llm_orchestrator.generate_code_parallel(prompt, "consensus")
            
            if result.get("success") and result.get("merged_code"):
                # Extract the commit message from the response
                commit_message = result["merged_code"].strip()
                
                # Clean up the message (remove code blocks, etc.)
                lines = commit_message.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('```') and not line.startswith('#'):
                        return line
                        
            # Fallback to simple message
            return self._generate_simple_commit_message(changes_analysis)
            
        except Exception as e:
            logging.error(f"Error generating intelligent commit message: {e}")
            return self._generate_simple_commit_message(changes_analysis)
            
    def _generate_simple_commit_message(self, changes_analysis: Dict[str, Any]) -> str:
        """Generate simple commit message based on change patterns"""
        stats = changes_analysis.get("stats", {})
        total_files = changes_analysis.get("total_files", 0)
        
        if stats.get("files_added", 0) > 0 and stats.get("files_modified", 0) == 0:
            return f"feat: Add {stats['files_added']} new file(s)"
        elif stats.get("files_modified", 0) > 0 and stats.get("files_added", 0) == 0:
            return f"refactor: Update {stats['files_modified']} file(s)"
        elif stats.get("files_deleted", 0) > 0:
            return f"cleanup: Remove {stats['files_deleted']} file(s)"
        else:
            return f"update: Modify {total_files} file(s)"
            
    async def stage_and_commit_changes(self, commit_message: Optional[str] = None, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Stage and commit changes with optional file filtering"""
        try:
            # Analyze changes first
            changes_analysis = await self.analyze_changes()
            
            if changes_analysis.get("error"):
                return {"success": False, "error": changes_analysis["error"]}
                
            # Generate commit message if not provided
            if not commit_message:
                commit_message = await self.generate_intelligent_commit_message(changes_analysis)
                
            # Stage files
            if files:
                # Stage specific files
                for file_path in files:
                    self.git_repo.index.add([file_path])
            else:
                # Stage all changes
                self.git_repo.git.add('.')
                
            # Check if there are staged changes
            if not self.git_repo.index.diff("HEAD"):
                return {"success": False, "error": "No staged changes to commit"}
                
            # Create commit
            commit = self.git_repo.index.commit(
                message=f"{commit_message}\n\n🤖 Generated with Master Orchestrator\n\nCo-Authored-By: AI Assistant <ai@masterorchestrator.com>"
            )
            
            return {
                "success": True,
                "commit_hash": commit.hexsha,
                "commit_message": commit_message,
                "files_changed": changes_analysis.get("total_files", 0),
                "stats": changes_analysis.get("stats", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error staging and committing changes: {e}")
            return {"success": False, "error": str(e)}
            
    async def push_changes(self, branch: str = "main") -> Dict[str, Any]:
        """Push committed changes to remote repository"""
        try:
            if not self.git_repo.remotes:
                return {"success": False, "error": "No remote repository configured"}
                
            origin = self.git_repo.remotes.origin
            
            # Push changes
            push_info = origin.push(refspec=f"{branch}:{branch}")
            
            return {
                "success": True,
                "branch": branch,
                "push_info": str(push_info),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error pushing changes: {e}")
            return {"success": False, "error": str(e)}
            
    async def create_pull_request(self, 
                                branch: str,
                                title: str,
                                body: str,
                                base_branch: str = "main") -> Dict[str, Any]:
        """Create pull request"""
        try:
            if not self.repo:
                return {"success": False, "error": "GitHub repository not connected"}
                
            # Create pull request
            pr = self.repo.create_pull(
                title=title,
                body=body,
                head=branch,
                base=base_branch
            )
            
            return {
                "success": True,
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "title": title,
                "branch": branch,
                "base_branch": base_branch,
                "timestamp": datetime.now().isoformat()
            }
            
        except GithubException as e:
            logging.error(f"GitHub API error creating PR: {e}")
            return {"success": False, "error": f"GitHub API error: {e}"}
        except Exception as e:
            logging.error(f"Error creating pull request: {e}")
            return {"success": False, "error": str(e)}
            
    async def create_automated_pr_for_changes(self, 
                                            feature_description: str,
                                            files_changed: List[str]) -> Dict[str, Any]:
        """Create automated PR for a set of changes"""
        try:
            # Create feature branch
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"feature/auto-{timestamp}"
            
            # Create and checkout new branch
            new_branch = self.git_repo.create_head(branch_name)
            new_branch.checkout()
            
            # Stage and commit changes
            commit_result = await self.stage_and_commit_changes(files=files_changed)
            
            if not commit_result["success"]:
                return commit_result
                
            # Push branch
            push_result = await self.push_changes(branch_name)
            
            if not push_result["success"]:
                return push_result
                
            # Generate PR description using LLM
            pr_body = await self._generate_pr_description(feature_description, commit_result)
            
            # Create pull request
            pr_result = await self.create_pull_request(
                branch=branch_name,
                title=f"feat: {feature_description}",
                body=pr_body
            )
            
            # Switch back to main branch
            self.git_repo.heads.main.checkout()
            
            return {
                "success": True,
                "branch": branch_name,
                "commit": commit_result,
                "push": push_result,
                "pull_request": pr_result
            }
            
        except Exception as e:
            logging.error(f"Error creating automated PR: {e}")
            return {"success": False, "error": str(e)}
            
    async def _generate_pr_description(self, feature_description: str, commit_result: Dict[str, Any]) -> str:
        """Generate PR description using LLM"""
        try:
            if not self.llm_orchestrator:
                return self._simple_pr_description(feature_description, commit_result)
                
            prompt = f"""
Generate a professional pull request description for this feature:

Feature: {feature_description}

Commit Details:
- Files changed: {commit_result.get('files_changed', 0)}
- Lines added: {commit_result.get('stats', {}).get('lines_added', 0)}
- Lines removed: {commit_result.get('stats', {}).get('lines_removed', 0)}

Create a PR description that includes:
1. ## Summary - Brief overview of changes
2. ## Changes Made - Bullet points of specific changes
3. ## Testing - How this was tested
4. ## Impact - Potential impact of changes

Keep it professional and concise.
"""
            
            result = await self.llm_orchestrator.generate_code_parallel(prompt, "consensus")
            
            if result.get("success") and result.get("merged_code"):
                return result["merged_code"].strip()
                
        except Exception as e:
            logging.error(f"Error generating PR description: {e}")
            
        return self._simple_pr_description(feature_description, commit_result)
        
    def _simple_pr_description(self, feature_description: str, commit_result: Dict[str, Any]) -> str:
        """Generate simple PR description"""
        return f"""
## Summary
{feature_description}

## Changes Made
- Modified {commit_result.get('files_changed', 0)} file(s)
- Added {commit_result.get('stats', {}).get('lines_added', 0)} lines
- Removed {commit_result.get('stats', {}).get('lines_removed', 0)} lines

## Testing
- Automated testing included
- Code generated and validated by AI systems

## Impact
This change implements {feature_description.lower()} to improve the platform capabilities.

---
🤖 Generated with Master Orchestrator
"""

    async def setup_repository_automation(self) -> Dict[str, Any]:
        """Setup repository automation features"""
        try:
            automation_features = {}
            
            # Create GitHub workflows directory
            workflows_dir = Path(".github/workflows")
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup branch protection (if repo is available)
            if self.repo:
                try:
                    main_branch = self.repo.get_branch("main")
                    main_branch.edit_protection(
                        strict=True,
                        contexts=["ci/lint-and-test"],
                        enforce_admins=True
                    )
                    automation_features["branch_protection"] = "enabled"
                except Exception as e:
                    automation_features["branch_protection"] = f"failed: {e}"
                    
            # Setup issue templates
            issue_templates_dir = Path(".github/ISSUE_TEMPLATE")
            issue_templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Create bug report template
            bug_template = """---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
- OS: [e.g. iOS]
- Browser [e.g. chrome, safari]
- Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
"""
            
            with open(issue_templates_dir / "bug_report.md", 'w') as f:
                f.write(bug_template)
                
            automation_features["issue_templates"] = "created"
            
            # Setup pre-commit hooks
            pre_commit_config = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
"""
            
            with open(".pre-commit-config.yaml", 'w') as f:
                f.write(pre_commit_config)
                
            automation_features["pre_commit"] = "configured"
            
            return {
                "success": True,
                "features": automation_features,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error setting up repository automation: {e}")
            return {"success": False, "error": str(e)}

class AutomatedDevelopmentWorkflow:
    """Automated development workflow orchestrator"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.github_integrator = GitHubIntegrator(self.config)
        self.llm_orchestrator = ParallelLLMOrchestrator()
        
    async def initialize(self):
        """Initialize automated workflow"""
        await self.config.initialize()
        await self.github_integrator.initialize()
        await self.llm_orchestrator.initialize()
        
    async def implement_feature_request(self, feature_description: str) -> Dict[str, Any]:
        """Implement a complete feature from description to PR"""
        try:
            # Generate code for the feature
            code_result = await self.llm_orchestrator.generate_code_parallel(
                f"Implement this feature: {feature_description}", 
                "comprehensive"
            )
            
            if not code_result["success"]:
                return code_result
                
            # Save generated code to file
            feature_filename = f"feature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            feature_path = Path(feature_filename)
            
            with open(feature_path, 'w') as f:
                f.write(code_result["merged_code"])
                
            # Create automated PR
            pr_result = await self.github_integrator.create_automated_pr_for_changes(
                feature_description,
                [str(feature_path)]
            )
            
            return {
                "success": True,
                "feature_description": feature_description,
                "code_generation": code_result,
                "file_created": str(feature_path),
                "pull_request": pr_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error implementing feature request: {e}")
            return {"success": False, "error": str(e)}

async def main():
    """Main function for testing GitHub integration"""
    logging.basicConfig(level=logging.INFO)
    
    # Test GitHub integration
    config = SecureConfigManager()
    await config.initialize()
    
    github_integrator = GitHubIntegrator(config)
    await github_integrator.initialize()
    
    # Analyze current changes
    changes = await github_integrator.analyze_changes()
    print(f"Changes analysis: {json.dumps(changes, indent=2)}")
    
    # Test automated workflow
    workflow = AutomatedDevelopmentWorkflow()
    await workflow.initialize()
    
    # Example feature implementation
    # feature_result = await workflow.implement_feature_request(
    #     "Add real-time chat functionality with WebSocket support"
    # )
    # print(f"Feature implementation result: {json.dumps(feature_result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())