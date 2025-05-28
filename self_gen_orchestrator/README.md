# Self-Generating Orchestrator

This CLI loads a list of decision questions and interactively prompts the user to make selections.

## Quickstart

1. Install dependencies:

   ```bash
   pip install .[dev]
   ```

2. Prepare `questions.json`:

   See `questions.json` for an example.

3. Run:

   ```bash
   self-gen-orchestrator run --questions-file questions.json --output-file decisions.json
   ```