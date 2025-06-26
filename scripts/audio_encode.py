from datasets import load_from_disk, ClassLabel
from transformers import AutoFeatureExtractor

DATASET_PATH = "data/encoding"

# Load the dataset from disk and cast the 'accents' column to ClassLabel
dataset = load_from_disk(DATASET_PATH)
accent_classes = dataset.unique("accents")
accent_classes = sorted(accent_classes)
accent_classes = ClassLabel(names=accent_classes)
dataset = dataset.cast_column("accents", accent_classes)
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

# Create label2id and id2label mappings
labels = dataset_split["train"].features["accents"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Preprocess the dataset to extract audio features
def preprocess_function(examples):    
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True
    )
    return inputs

encoded_dataset = dataset_split.map(preprocess_function, remove_columns="path", batched=True)
encoded_dataset = encoded_dataset.rename_column("accents", "label")

# Save the processed dataset to disk
OUTPUT_DIR = "data/processed"
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

encoded_dataset.save_to_disk(OUTPUT_DIR)
print(f"Dataset successfully saved to {OUTPUT_DIR}")