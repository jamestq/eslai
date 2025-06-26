import pandas as pd
import numpy as np
import typer, evaluate
from datasets import load_dataset, Audio, ClassLabel, load_from_disk
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import glob, os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pathlib import Path

app = typer.Typer()

class AudioProcessor:
    def __init__(self, input_file_path: Path, create_analysis_data: bool = False):
        self.input_file_path = input_file_path
        self.audio = AudioSegment.from_file(input_file_path)
        self.create_analysis_data = create_analysis_data

    def split_audio_by_silence(self, min_silence_len, threshold):
        return detect_nonsilent(self.audio, min_silence_len=min_silence_len, silence_thresh=threshold)

    def save_audio(self, audio, output_path):
        audio.export(output_path, format="mp3")
        print(f"Saved audio to {output_path}")

    def save_timeline_to_text(self, timeline_data, output_path):
        with open(output_path, 'w') as file:
            file.write("[")
            for start, end in timeline_data:
                file.write(f"({start}, {end}), ")
            file.write("]")
        print(f"Saved timeline data to {output_path}")

    def is_fully_silent(self, min_silence_len=100, threshold=-30):
        # Detect non-silent parts
        non_silent_segments = self.split_audio_by_silence(min_silence_len, threshold)
        # If the list of non-silent segments is empty, the audio is fully silent
        return not non_silent_segments
    
    def process_audio(self, min_silence_len=100, threshold=-30, output_folder: Path = None):
        try:
            print("Processing audio...")
            non_silent_parts = self.split_audio_by_silence(min_silence_len, threshold)
            audio_silent = AudioSegment.empty()
            audio_non_silent = AudioSegment.empty()
            silent_parts_times = []
            non_silent_parts_times = []

            for i, (start_time, end_time) in enumerate(non_silent_parts):
                audio_non_silent += self.audio[start_time:end_time]
                non_silent_parts_times.append((start_time, end_time))

                if i == 0:
                    silent_start_time = 0
                else:
                    silent_start_time = non_silent_parts[i - 1][1]
                silent_end_time = start_time

                audio_silent += self.audio[silent_start_time:silent_end_time]
                silent_parts_times.append((silent_start_time, silent_end_time))

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)            
            filename = Path(self.input_file_path.name)        

            output_non_silent_path = output_folder / filename# Save the audio segments and timeline
            self.save_audio(audio_non_silent, output_non_silent_path)

            if self.create_analysis_data:
                additionaL_data = output_folder / "additional_data"
                os.makedirs(additionaL_data, exist_ok=True)
                output_silent_path = additionaL_data / filename
                silent_txt_path = additionaL_data / f"{self.input_file_path.stem}_silent_parts.txt"
                non_silent_txt_path = additionaL_data / f"{self.input_file_path.stem}_non_silent_parts.txt"    
                self.save_audio(audio_silent, output_silent_path)            
                self.save_timeline_to_text(silent_parts_times, silent_txt_path)
                self.save_timeline_to_text(non_silent_parts_times, non_silent_txt_path)

        except Exception as e:
            print("An error occurred:", str(e))

def load_audio(dataset):
    return {"path": [x['path'] for x in dataset['path']]}

def get_accent_classes(dataset):
    """
    Extract unique accent classes from the dataset.
    """
    accent_classes = dataset.unique("reduced_accents")
    accent_classes = sorted(accent_classes)
    accent_classes = ClassLabel(names=accent_classes)
    dataset = dataset.cast_column("reduced_accents", accent_classes)
    return dataset

def get_label_dicts(labels):
    """
    Create label2id and id2label mappings from the list of labels.
    """
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

def get_audio_features(row, feature_extractor):    
    audio_arrays = [x["array"] for x in row["path"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True
    )
    return inputs

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

@app.command()
def audio_remove_silence(
        input_dir: str,
        output_dir: str,
        min_silence_len: int = 100,
        silence_thresh: int = -30,
        create_analysis_data: bool = False
    ):
    """
    Remove silence from audio files in a specified directory.
    """
    # This function is a placeholder for the actual implementation
    for file in glob.glob(f'{input_dir}/*.mp3', recursive=False):
        audio = AudioProcessor(Path(file), create_analysis_data=create_analysis_data)
        audio.process_audio(
            min_silence_len=min_silence_len, 
            threshold=silence_thresh, 
            output_folder=Path(output_dir),             
        )    

@app.command()
def process_audio(
    input_csv: str,
    audio_file_location: str,
    output_dir: str
):
    """
    Process audio files from a CSV file and save them in a format compatible with Hugging Face datasets.
    """
    # Load the dataset from the CSV file
    dataset = load_dataset("csv", sep="\t", data_files=input_csv, split="train")
    dataset = dataset.map(lambda row: {"path": f"{audio_file_location}/{row['path']}"})   
    # Cast the 'path' column to Audio type with a sampling rate of 16000
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))    
    # Map the audio file paths to the correct location         
    # Load audio data into the dataset
    dataset = dataset.map(load_audio, batched=True)
    # Extract accent classes and cast the 'accents' column
    dataset = get_accent_classes(dataset)    
    # Save the processed dataset to disk    
    dataset.save_to_disk(output_dir)    
    print(f"Dataset successfully saved to {output_dir}")

@app.command()
def feature_extraction(
    input_encoded_dataset: str,
    model_name: str = "facebook/wav2vec2-large-xlsr-53",
    output_dir: str = "featured_extracted"
):
    """
    Extract audio features from the encoded dataset using a specified model.
    """
    dataset = load_from_disk(input_encoded_dataset)    
    # Create label2id and id2label mappings    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # Preprocess the dataset to extract audio features
    feature_extracted = dataset.map(lambda row: get_audio_features(row, feature_extractor), remove_columns="path", batched=True)
    feature_extracted = feature_extracted.rename_column("reduced_accents", "label")
    feature_extracted.save_to_disk(output_dir)
    print(f"Feature extracted dataset successfully saved to {output_dir}")

@app.command()
def train_model(
    feature_extracted_dataset: str,
    model_name: str = "facebook/wav2vec2-large-xlsr-53",
    output_dir: str = "model_output"
):
    dataset = load_from_disk(feature_extracted_dataset)
    label2id, id2label = get_label_dicts(dataset.features["label"].names)
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",    
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,  # Define your compute_metrics function if needed
    )
    trainer.train()


if __name__ == "__main__":
    app()




