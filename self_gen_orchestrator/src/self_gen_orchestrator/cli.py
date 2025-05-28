"""
CLI for the Self-Generating Agentic Orchestrator.
"""
import asyncio
import json
from pathlib import Path

import typer

app = typer.Typer(help="Self-generating agentic orchestration CLI.")


@app.command()
def run(
    questions_file: Path = typer.Option(
        Path("questions.json"),
        "--questions-file", "-q",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON file containing questions.",
    ),
    output_file: Path = typer.Option(
        Path("decisions.json"),
        "--output-file", "-o",
        writable=True,
        dir_okay=False,
        help="Path to write out user decisions.",
    ),
):
    """
    Run the interactive decision prompt for provided questions.
    """
    asyncio.run(_run_async(questions_file, output_file))


async def _run_async(questions_file: Path, output_file: Path):
    typer.echo(f"Loading questions from {questions_file}")
    questions = json.loads(questions_file.read_text())
    decisions: list[dict] = []

    for q in questions:
        typer.echo("\n" + "-" * 40)
        typer.secho(f"Question: {q.get('question', '')}", fg=typer.colors.GREEN, bold=True)
        reason = q.get("reason")
        if reason:
            typer.echo(f"Reason: {reason}")
        data = q.get("data")
        if data:
            typer.echo("Data:")
            typer.echo(json.dumps(data, indent=2))
        options: list[str] = q.get("options", [])
        choice = await asyncio.to_thread(
            typer.prompt,
            "Select an option",
            typer.Choice(options),
        )
        decisions.append({"id": q.get("id"), "choice": choice})

    typer.echo(f"\nSaving decisions to {output_file}")
    output_file.write_text(json.dumps(decisions, indent=2))
    typer.secho("Done.", fg=typer.colors.BLUE, bold=True)


def main():
    app()


if __name__ == "__main__":
    main()