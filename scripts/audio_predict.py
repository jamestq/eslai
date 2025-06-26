import pandas as pd
import numpy as np
import typer, evaluate
from datasets import load_dataset, Audio, ClassLabel, load_from_disk
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def get_label_dicts(labels):
    """
    Create label2id and id2label mappings from the list of labels.
    """
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

dataset = load_from_disk("accent_recognition/feature_extracted")
label2id, id2label = get_label_dicts(dataset.features["label"].names)
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
feature_extractor = AutoFeatureExtractor.from_pretrained(

)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)
training_args = TrainingArguments(
    output_dir="model_output",
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