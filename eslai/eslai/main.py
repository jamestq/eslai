import typer, os, shutil
from typer import Typer, Argument, Option
from enum import Enum
from typing_extensions import Annotated
from pathlib import Path

class FunctionName(str, Enum):
    extract = "extract"
    transform = "transform"
    load = "load"

app = Typer()

def extract(file_list_path: Path, file_parent_path: Path, output_path: Path):
    os.makedirs(output_path, exist_ok=True)
    with open(file_list_path, 'r') as file:        
        files = file.readlines()
        for file in files:      
            file = Path(file.strip())
            file_path = file_parent_path / file
            copied_file_path = output_path / file
            shutil.copy(file_path, copied_file_path)           # Placeholder for actual extraction lgic
    
# 
@app.command()
def execute(    
    function: Annotated[FunctionName, Option("--function", "-f", case_sensitive=False)],
    file_list_path: Annotated[Path, Option("--file-extract", "-e", help="Path to the file list", dir_okay=True, file_okay=True)] = None,
    file_parent_path: Annotated[Path, Option("--file-parent", "-p", help="Parent path for files", dir_okay=True, file_okay=False)] = os.getcwd(),
    output_path: Annotated[Path, Option("--output", "-o", help="Output path for results", dir_okay=True, file_okay=False)] = os.getcwd(),
):
    try:
        match function:
            case FunctionName.extract:
                if file_list_path is None:
                    raise ValueError("File list path must be provided for extraction.")
                typer.echo(file_list_path)
                extract(file_list_path, file_parent_path, output_path)                
            case FunctionName.transform:
                pass
            case FunctionName.load:
                pass
    except ValueError as e:
        typer.echo(f"Error running function {function}", err=True)

if __name__ == "__main__":
    app()