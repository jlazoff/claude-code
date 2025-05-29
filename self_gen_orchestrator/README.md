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

## Frontend Project Map

This project integrates the AG-UI Dojo frontend to visualize the project mind map generated from your ChatGPT conversations.

1. The AG-UI Dojo frontend is included here under `self_gen_orchestrator/frontend`.

2. Generate the projects JSON from your ChatGPT export:

   ```bash
   python3 scripts/digest_chatgpt_map.py \
     --json-path ~/Downloads/conversations.json \
     --projects-output self_gen_orchestrator/frontend/public/projects.json
   ```

3. Start the AG-UI frontend:

   ```bash
   cd self_gen_orchestrator/frontend
   pnpm install
   pnpm dev
   ```

The frontend will be available at http://localhost:3000 and will display the mind map of projects.