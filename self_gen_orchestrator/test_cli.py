import typer
app = typer.Typer()
@app.command()
def run():
    typer.echo("hi")

def main():
    app()

if __name__ == "__main__":
    main()
