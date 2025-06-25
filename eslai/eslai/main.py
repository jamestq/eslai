from typer import Typer

app = Typer()

@app.command()
def hello(name: str = "World"):
    """Say hello to NAME."""
    print(f"Hello, {name}!")


if __name__ == "__main__":
    app()