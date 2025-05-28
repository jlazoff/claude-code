# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the public documentation and issue tracking repository for Claude Code, an agentic coding CLI tool. The actual source code is maintained privately by Anthropic and distributed as the npm package `@anthropic-ai/claude-code`.

## Key Architecture

- **Distribution**: NPM package requiring Node.js 18+
- **Authentication**: OAuth integration with Claude Max or Anthropic Console accounts
- **Models**: Supports Sonnet 4 and Opus 4 with configurable model settings
- **MCP Support**: Model Context Protocol server integration with SSE configurations
- **Real-time Interaction**: Users can send messages while Claude works to provide steering

## Available Commands

Since this is a documentation repository, there are no build/test/lint commands. The main workflow is:

```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Run in any project directory
claude
```

## Development Workflow

This repository uses GitHub Actions for automation:

- **Issue Triage**: Automatically triages new issues using Claude Code via `claude-issue-triage.yml`
- **Claude Assistant**: Responds to `@claude` mentions in issues, PRs, and comments via `claude.yml`

## Environment Configuration

Key environment variables for Claude Code users:
- `ANTHROPIC_MODEL`: Custom model selection
- `ANTHROPIC_SMALL_FAST_MODEL`: Fast model for quick operations
- `ANTHROPIC_LOG=debug`: Enable debug logging
- `DISABLE_INTERLEAVED_THINKING`: Opt out of interleaved thinking
- `BASH_DEFAULT_TIMEOUT_MS` / `BASH_MAX_TIMEOUT_MS`: Bash execution timeouts

## Bug Reporting

- Use `/bug` command within Claude Code for direct feedback
- File GitHub issues for public bug reports
- Feedback transcripts are retained for 30 days only

## Security & Privacy

- Vulnerability disclosure via HackerOne
- No model training on user feedback
- 30-day retention policy for sensitive data
- OAuth-based authentication flow