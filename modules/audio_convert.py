import ffmpeg, csv, json, warnings, os
from pathlib import Path

def convert_audio_to_wav(audio_file: str) -> None:
    try:
        orig_output_file = audio_file.split(".")[0] + ".wav"
        output_file = orig_output_file.replace("archive", "converted")        
        out, _ = (
            ffmpeg.input(audio_file)
            .output(output_file, format="wav")            
            .run(capture_stdout=True)
        )
    except Exception as e:
        print(e)    


def row_reader(reader: csv.DictReader, root: Path) -> list[str]:
    audio_files: list[str] = []
    for row in reader:
        if row["accent"]:
            audio_files.append(f"{root / row["filename"]}")
    return audio_files

def extract_csv_data(cv_route: str, root: Path) -> list[str]:
    csv_file = root / f"{cv_route}.csv"
    root = root / cv_route
    try:
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            print(reader.fieldnames)
            return row_reader(reader, root)            
    except Exception as e:
        print(e)
    
def main():
    cwd = Path.cwd()
    audio_paths = cwd / "data" / "data" / "audio_paths.txt"
    with open(audio_paths, "r") as file:
        audio_files = [line.strip() for line in file]
    for file_path in audio_files:
        try:
            convert_audio_to_wav(file_path)
        except Exception as e:
            warnings.warn(e)    
        
    # audio_files: dict[str] = {}    
    # root: Path = Path.home() / "Downloads" / "archive"
    # file_folders = [
    #     "cv-valid-train",
    #     "cv-valid-test",
    #     "cv-valid-dev",
    # ]
    # for folder in file_folders:
    #     route: Path = Path.home() / "Downloads" / "converted" / folder / folder
    #     route.mkdir(parents=True, exist_ok=True)
    # try:
    #     for folder in file_folders:
    #         audio_files[folder] = extract_csv_data(folder, root)            
    #     with open("audio_sources.json", "w") as f:
    #         json.dump(audio_files, f, indent=4)
    #     for folder, file_list in audio_files.items():
    #         file_list = audio_files[folder]
    #         for file in file_list:
    #             convert_audio_to_wav(file)
    # except Exception as e:
    #     warnings.warn(e)

if __name__ == "__main__":
    main()