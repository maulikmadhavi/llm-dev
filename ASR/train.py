from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import os
import evaluate
from transformers import WhisperForConditionalGeneration

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import pandas as pd
from datasets import Dataset
from datasets import Audio
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from huggingface_hub import HfApi


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")


def prepare_dataset(examples):
    # compute log-Mel input features from input audio array 
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(
        audio["array"], sampling_rate=16000).input_features[0]
    del examples["audio"]
    sentences = examples["sentence"]

    # encode target text to label ids 
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["sentence"]
    return examples

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


## we will load the both of the data here.
train_df = pd.read_csv("data/train.csv")

## we will rename the columns as "audio", "sentence".
train_df.columns = ["audio", "sentence"]


## convert the pandas dataframes to dataset 
train_dataset = Dataset.from_pandas(train_df)

## convert the sample rate of every audio files using cast_column function
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = train_dataset.map(prepare_dataset, num_proc=1)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

## lets initiate the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)




metric = evaluate.load("wer")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-en",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    gradient_checkpointing=True,
    fp16=True,
    num_train_epochs=5,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",        # Save after each epoch
    logging_strategy="epoch",     # Log after each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    report_to=["tensorboard"],
    metric_for_best_model="wer",
    push_to_hub=False,
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

## start the model training
trainer.train()

## save the model
trainer.save_model("whisper-base-en-finetuned")



api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/mnt/d/llm-devs/ASR/whisper-base-en-finetuned",
    repo_id="MaulikMadhavi/lapel_mic_whisper_base_finetuned",
    repo_type="model",
)