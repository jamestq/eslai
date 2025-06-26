import pandas as pd
from datasets import load_dataset, Audio

CSV_PATH = "data/train.tsv"

df = pd.read_csv(CSV_PATH, sep="\t")
# Remove all columns except 'path', 'sentence', 'age', 'gender', 'accents'
df = df[['path', 'sentence', 'age', 'gender', 'accents']]
# Filter out rows where 'accents' is NaN or empty
df_filtered = df[df['accents'].notna() & (df['accents'] != '') & df['gender'].notna() & df['age'].notna()]

df_filtered['path'] = "dataset/" + df_filtered['path'].astype(str)
output_path = "data/train_clean.csv"
df_filtered.to_csv(output_path, index=False)

output_path = "data/train_clean.csv"
df_filtered.to_csv(output_path, index=False)

output_unzip = "data/train_unzip.csv"
df_files = df_filtered['path']
df_files.to_csv(output_unzip, index=False, header=False)

# Load the dataset and save it in a format compatible with Hugging Face datasets
input_path = "data/train_clean.csv"
dataset = load_dataset("csv", data_files=input_path, split="train")
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

def load_audio(dataset):
    return {"path": [x['path'] for x in dataset['path']]}

dataset = dataset.map(load_audio, batched=True)
dataset.save_to_disk("data/encoding")

CSV_PATH = "data/train_clean.csv"
OUTPUT_PATH = "data/encoding"

from datasets import load_dataset, load_from_disk, Audio
# Load the dataset and save it in a format compatible with Hugging Face datasets
dataset = load_dataset("csv", data_files=CSV_PATH, split="train")
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

def load_audio(dataset):
    return {"path": [x['path'] for x in dataset['path']]}

dataset = dataset.map(load_audio, batched=True)
dataset.save_to_disk(OUTPUT_PATH)