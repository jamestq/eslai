import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import typer
from datasets import Audio, ClassLabel, load_dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import evaluate
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)

app = typer.Typer()


# --- Custom Model with Pooling ---
# To replicate the pooling strategies, we create a custom model class.
class Wav2Vec2ForAccentClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if self.pooling_mode == "mean":
            pooled_output = torch.mean(hidden_states, dim=1)
        elif self.pooling_mode == "sum":
            pooled_output = torch.sum(hidden_states, dim=1)
        elif self.pooling_mode == "max":
            pooled_output = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return torch.nn.utils.rnn.PackedSequence(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss=loss,
        )


# --- Custom Trainer for Dual Optimizers ---
# This is the core adaptation from the SpeechBrain script.
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # Separate parameters for the w2v2 model and the classification head
        w2v2_params = self.model.wav2vec2.parameters()
        classifier_params = self.model.classifier.parameters()

        # Create two separate optimizers
        self.optimizer = torch.optim.AdamW(
            classifier_params,
            lr=self.args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.wav2vec2_optimizer = torch.optim.AdamW(
            w2v2_params,
            lr=self.args.w2v2_learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # Create two separate schedulers using the Noam scheduler logic
        self.lr_scheduler = self.create_noam_scheduler(self.optimizer, self.args.warmup_steps)
        self.wav2vec2_lr_scheduler = self.create_noam_scheduler(
            self.wav2vec2_optimizer, self.args.warmup_steps
        )

    def create_noam_scheduler(self, optimizer, warmup_steps):
        """Creates a Noam learning rate scheduler."""

        def lr_lambda(current_step):
            current_step += 1  # 1-based step
            return (
                self.model.config.hidden_size ** (-0.5)
                * min(current_step ** (-0.5), current_step * warmup_steps ** (-1.5))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def training_step(self, model, inputs) -> torch.Tensor:
        """Perform a training step with two optimizers."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # Step both optimizers and schedulers
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.wav2vec2_optimizer.step()
        self.wav2vec2_lr_scheduler.step()
        self.wav2vec2_optimizer.zero_grad()

        return loss.detach() / self.args.gradient_accumulation_steps


# --- Data and Training Configuration ---
@dataclass
class DataTrainingArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    data_folder: str = field(metadata={"help": "The folder where the CSVs are stored."})
    pooling_mode: str = field(default="mean", metadata={"help": "Pooling mode: 'mean', 'sum', or 'max'."})
    max_duration_in_seconds: float = field(default=20.0, metadata={"help": "Filter audio files that are longer than this."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    w2v2_learning_rate: float = field(default=5e-5, metadata={"help": "The learning rate for the Wav2Vec2 model."})


# --- Main Training Function ---
@app.command()
def train(
    model_args_file: str = typer.Argument(..., help="Path to a json file with model and data arguments."),
    training_args_file: str = typer.Argument(..., help="Path to a json file with training arguments."),
):
    """
    Train an accent classification model using Hugging Face Transformers.
    This script uses two separate optimizers and Noam schedulers,
    adapting the strategy from the SpeechBrain recipe.
    """
    # 1. Parse arguments
    parser = HfArgumentParser((DataTrainingArguments, CustomTrainingArguments))
    model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(model_args_file))
    training_args.output_dir = training_args_file.split("/")[-2]

    # 2. Load data
    train_csv = os.path.join(model_args.data_folder, "train.csv")
    valid_csv = os.path.join(model_args.data_folder, "dev.csv")
    test_csv = os.path.join(model_args.data_folder, "test.csv")

    dataset = load_dataset(
        "csv",
        data_files={"train": train_csv, "validation": valid_csv, "test": test_csv},
    )

    # 3. Prepare dataset
    # Get labels
    label_list = sorted(dataset["train"].unique("accent"))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    # Preprocessing function
    def preprocess_function(examples):
        audio_arrays = [read_audio(path) for path in examples["path"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * model_args.max_duration_in_seconds),
            truncation=True,
        )
        inputs["labels"] = [label2id[accent] for accent in examples["accent"]]
        return inputs

    # Apply preprocessing
    encoded_dataset = dataset.map(preprocess_function, remove_columns=["accent", "path"], batched=True)

    # 4. Load model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        pooling_mode=model_args.pooling_mode,
    )
    model = Wav2Vec2ForAccentClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True, # Allows loading pre-trained weights into our custom model
    )

    # Freeze the feature extractor if needed
    if training_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 5. Define metrics
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # 6. Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    # 7. Train
    trainer.train()

    # 8. Evaluate on test set
    logger.info("*** Test Evaluation ***")
    test_results = trainer.evaluate(encoded_dataset["test"])
    logger.info(test_results)


if __name__ == "__main__":
    app()