from modules.whisper import run_whisper
from shiny import render, ui, reactive, Inputs, Outputs, Session, App
from shiny.types import FileInfo
from shinymedia import audio_spinner

app_ui = ui.page_fluid(
    ui.panel_title("Whisper Transcriber"),
    # ui.input_slider("n", "N", 0, 100, 20)
    ui.input_file(
        id="audio1",
        label="Audio for transcription",
        multiple=False
    ),
    ui.output_ui("show_audio_spinner"),
    ui.output_text_verbatim("txt"),
)

def server(input: Inputs, output: Outputs, session: Session):    
    @reactive.calc
    def process_audio():
        inputs: list[FileInfo] | None = input.audio1()  
        file = inputs[0] if inputs else None
        print(file)
        if file is None or not file["datapath"]:
            return f"No file has been uploaded"
        try:
            text = run_whisper(file["datapath"])
            return text
        except Exception as e:
            return f"An error occured while processing the file: {e} - {file}"

    @render.text
    def txt():
        text = process_audio()
        return text
    
    @render.ui
    @reactive.calc
    def show_audio_spinner():
        inputs: list[FileInfo] | None = input.audio1()  
        file = inputs[0] if inputs else None
        try:
            if file is None:
                return "Upload an audio file"            
            return audio_spinner(
                src=file["datapath"],
                width="125px",
                height="125px",
                autoplay=False,
            )
        except Exception as e:
            return f"An error occured while processing the file: {e} - {file}"
        

app = App(app_ui, server, debug=True)